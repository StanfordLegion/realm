# Copyright 2025 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Auto-loaded by gdb (via the .debug_gdb_scripts section in librealm) to make
# Realm's user-level threads visible to the debugger.
#
# Provides:
#   - `thread apply all bt` (also `backtrace`/`where`/`info stack`, with any
#     bt options) prints the usual per-thread backtraces and then the stack
#     of every PARKED user thread.  A RUNNING user thread is already visible
#     as its host pthread's stack, so it is never repeated.
#   - `realm-thread` lists the user threads.  `realm-thread <ptr>` shows a
#     parked user thread's stack on the selected thread, so that `bt`,
#     `frame N`, `info locals` and `print VAR` (function calls included)
#     work on it, and `realm-thread back` undoes that.
#
# Nothing in the debuggee is modified to do this.  A Python unwinder makes
# the selected thread's innermost frame unwind into the parked user thread's
# saved context instead: frame #0 stays the thread's own frame and the user
# thread's stack starts at frame #1.  So there is nothing to undo before the
# program resumes, gdb detaches, or gdb exits, whichever way that happens
# (CLI, MI, batch mode, an IDE), and live processes and core files work the
# same way.  See "Views" below.
#
# Linux/FreeBSD on x86-64 or aarch64.  On other targets the script loads
# as a no-op.

import atexit
import gdb
import gdb.unwinder
import os
import platform
import re
import struct
import sys
import traceback
from gdb.FrameDecorator import FrameDecorator


# --------------------------------------------------------------------------
# Architecture abstraction
# --------------------------------------------------------------------------
#
# Each supported architecture provides unwind_regs_from_ctx(), which returns
# {gdb register name: value} for the PC, SP and every callee-saved register
# in a UserThread::ctx (a ucontext_t, laid out differently per CPU and OS),
# and names its PC and SP registers in pc_reg / sp_reg, and its general
# registers in gp_regs.
#
# 64-bit unsigned mask used to coerce gdb's signed-int reads into the
# unsigned address space the kernel saved.
_U64 = (1 << 64) - 1


class _Arch:
    pc_reg = None
    sp_reg = None
    gp_regs = ()
    # Registers that a parked user thread shares with whichever thread shows
    # it, which gdb may need to unwind its frames.  The view passes them
    # through from the host frame (those that exist).
    thread_regs = ()

    @staticmethod
    def matches():
        raise NotImplementedError

    @staticmethod
    def unwind_regs_from_ctx(ut):
        raise NotImplementedError


_X86_64_GP_REGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
                   "rip", "eflags")


class _X86_64Linux(_Arch):
    pc_reg = "rip"
    sp_reg = "rsp"
    gp_regs = _X86_64_GP_REGS
    # glibc x86-64 ucontext gregs[] indices (see /usr/include/sys/ucontext.h).
    _UNWIND_GREGS = (("r12", 4), ("r13", 5), ("r14", 6), ("r15", 7),
                     ("rbp", 10), ("rbx", 11), ("rsp", 15), ("rip", 16))

    @staticmethod
    def matches():
        return (platform.system() == "Linux"
                and platform.machine() in ("x86_64", "amd64"))

    @staticmethod
    def unwind_regs_from_ctx(ut):
        gregs = ut["ctx"]["uc_mcontext"]["gregs"]
        return {name: int(gregs[i]) & _U64
                for name, i in _X86_64Linux._UNWIND_GREGS}


class _X86_64FreeBSD(_Arch):
    pc_reg = "rip"
    sp_reg = "rsp"
    gp_regs = _X86_64_GP_REGS
    # FreeBSD mcontext_t (sys/x86/include/ucontext.h) uses named fields
    # (mc_rip, mc_rsp, ...), not a gregs[] array.

    @staticmethod
    def matches():
        return (platform.system() == "FreeBSD"
                and platform.machine() in ("x86_64", "amd64"))

    @staticmethod
    def unwind_regs_from_ctx(ut):
        mc = ut["ctx"]["uc_mcontext"]
        return {name: int(mc["mc_" + name]) & _U64
                for name in ("r12", "r13", "r14", "r15",
                             "rbp", "rbx", "rsp", "rip")}


# gdb needs the pointer-authentication masks to unwind return addresses
# signed with -mbranch-protection=pac-ret; tpidr, fpsr/fpcr and vg are
# per-thread too.  (Linux on aarch64 is tested in CI; FreeBSD isn't.)
_AARCH64_THREAD_REGS = ("pauth_dmask", "pauth_cmask", "tpidr", "fpsr", "fpcr", "vg")
_AARCH64_GP_REGS = tuple("x{}".format(i) for i in range(31)) + ("sp", "pc", "cpsr")


class _AArch64Linux(_Arch):
    pc_reg = "pc"
    sp_reg = "sp"
    gp_regs = _AARCH64_GP_REGS
    thread_regs = _AARCH64_THREAD_REGS
    # Linux/glibc aarch64 mcontext_t is `struct sigcontext` (see
    # linux/arch/arm64/include/uapi/asm/sigcontext.h):
    #     __u64 fault_address;
    #     __u64 regs[31];     // X0..X30
    #     __u64 sp;
    #     __u64 pc;
    #     __u64 pstate;
    # Saved by glibc's swapcontext(): sp holds the SP, and pc where the
    # context resumes: the return address (the LR swapcontext was called
    # with), or with glibc >= 2.39 a stub in swapcontext that returns to it
    # from x1, where it is saved instead (and pc in x30 too).

    @staticmethod
    def matches():
        return (platform.system() == "Linux"
                and platform.machine() in ("aarch64", "arm64"))

    @staticmethod
    def unwind_regs_from_ctx(ut):
        # x19..x28 are callee-saved; x29/x30 are FP/LR.
        mc = ut["ctx"]["uc_mcontext"]
        regs = mc["regs"]
        out = {"x{}".format(i): int(regs[i]) & _U64 for i in range(19, 31)}
        out["sp"] = int(mc["sp"]) & _U64
        out["pc"] = int(mc["pc"]) & _U64
        if _AArch64Linux._returns_to_x1(out["pc"]):
            # (As the stub does.)
            out["pc"] = out["x30"] = int(regs[1]) & _U64
        return out

    _stubs = {}

    @staticmethod
    def _returns_to_x1(pc):
        """Whether the code at `pc` is glibc's stub: maybe a hint (`nop`,
        `bti`), then `mov x30, x1; ret`.  (A64 code is little-endian.)"""
        key = (gdb.selected_inferior().pid, pc)
        if key not in _AArch64Linux._stubs:
            try:
                words = struct.unpack("<4I", bytes(
                    gdb.selected_inferior().read_memory(pc, 16)))
            except Exception:
                return False
            i = 0
            while i < 2 and words[i] & 0xfffff01f == 0xd503201f:  # HINT
                i += 1
            _AArch64Linux._stubs[key] = (words[i] == 0xaa0103fe  # mov x30, x1
                                         and words[i + 1] == 0xd65f03c0)  # ret
        return _AArch64Linux._stubs[key]


class _AArch64FreeBSD(_Arch):
    pc_reg = "pc"
    sp_reg = "sp"
    gp_regs = _AARCH64_GP_REGS
    thread_regs = _AARCH64_THREAD_REGS
    # FreeBSD aarch64 mcontext_t (sys/arm64/include/ucontext.h):
    #     struct gpregs {
    #         __register_t  gp_x[30];   // X0..X29
    #         __register_t  gp_lr;      // X30
    #         __register_t  gp_sp;
    #         __register_t  gp_elr;     // exception link register == PC
    #         ...
    #     };
    #     typedef struct __mcontext {
    #         struct gpregs  mc_gpregs;
    #         ...
    #     } mcontext_t;

    @staticmethod
    def matches():
        return (platform.system() == "FreeBSD"
                and platform.machine() in ("aarch64", "arm64"))

    @staticmethod
    def unwind_regs_from_ctx(ut):
        gpr = ut["ctx"]["uc_mcontext"]["mc_gpregs"]
        out = {"x{}".format(i): int(gpr["gp_x"][i]) & _U64 for i in range(19, 30)}
        out["x30"] = int(gpr["gp_lr"]) & _U64
        out["sp"] = int(gpr["gp_sp"]) & _U64
        out["pc"] = int(gpr["gp_elr"]) & _U64
        return out


def _detect_arch():
    if platform.system() not in ("Linux", "FreeBSD"):
        return None
    for cls in (_X86_64Linux, _X86_64FreeBSD, _AArch64Linux, _AArch64FreeBSD):
        if cls.matches():
            return cls
    return None


# Selected once at script load; None on unsupported platforms (the
# commands are not registered in that case).
_arch = _detect_arch()


def _supported():
    return _arch is not None


# --------------------------------------------------------------------------
# The user thread registry
# --------------------------------------------------------------------------

class _RegistryLoop(gdb.error):
    """The registry's links loop (a crashed process may leave it corrupt)."""


def _registry_head():
    try:
        return gdb.parse_and_eval("Realm::user_thread_registry_head")
    except gdb.error:
        return None


def _iter_user_threads():
    """Every registered Realm::UserThread* of the selected inferior.  Raises
    _RegistryLoop, after yielding the threads before the loop, if the links
    loop rather than end."""
    cur = _registry_head()
    if cur is None:
        return
    seen = set()
    while int(cur) != 0:
        if int(cur) in seen:
            raise _RegistryLoop("the user thread registry is corrupt: it loops "
                                "back to {:#x}".format(int(cur)))
        seen.add(int(cur))
        yield cur
        cur = cur["registry_next"]


def _is_parked(ut):
    # A user thread is "parked" when no host pthread is currently executing
    # it.  Running threads are already covered by the host pthread's
    # backtrace, so we skip them.
    try:
        return not bool(ut["running"])
    except gdb.error:
        return False


def _state_readable(ut):
    try:
        bool(ut["running"])
        return True
    except gdb.error:
        return False


def _find_user_thread(addr):
    for ut in _iter_user_threads():
        if int(ut) == addr:
            return ut
    return None


def _host_sps():
    """{thread number: SP} for every stopped thread of the selected
    inferior, and whether any of its threads is running (non-stop mode)."""
    sps = {}
    running = False
    with _KeepSelection():
        for t in gdb.selected_inferior().threads():
            if not t.is_valid():
                continue
            if not t.is_stopped():
                running = True
                continue
            try:
                t.switch()
                sps[t.num] = _frame_sp(gdb.newest_frame())
            except gdb.error:
                pass
    return sps, running


def _executing_on(ut, sps):
    """The thread (number) whose SP is on user thread `ut`'s stack, if any:
    that thread is executing `ut`, even if ut->running is still false (Realm
    sets and clears it around the context switch, not atomically with it)."""
    try:
        base = int(ut["stack_base"]) & _U64
        top = base + (int(ut["stack_size"]) & _U64)
    except gdb.error:
        return None
    for num, sp in sorted(sps.items()):
        if base <= sp < top:
            return num
    return None


def _host_thread_of(ut):
    """The gdb thread running user thread `ut`, if it can be determined
    (InferiorThread.handle() is gdb >= 8.3, and needs libthread_db)."""
    try:
        want = int(ut["host_pthread"]) & _U64
        with _KeepSelection():
            for t in gdb.selected_inferior().threads():
                if int.from_bytes(t.handle(), sys.byteorder) == want:
                    return t
    except Exception:
        pass
    return None


def _has_target():
    """True if there is a live process or a core file to look at."""
    try:
        return gdb.selected_inferior().pid != 0
    except gdb.error:
        return False


def _select_inferior(inferior):
    """Select a thread of `inferior`: the selected thread if it is one and
    is stopped, else a stopped one, else any.  False if it has no threads.
    (The caller restores the selection.)"""
    threads = [t for t in inferior.threads() if t.is_valid()]
    if not threads:
        return False
    selected = gdb.selected_thread()
    if (selected is not None and selected.is_valid() and selected.is_stopped()
            and selected.inferior.num == inferior.num):
        return True
    stopped = [t for t in threads if t.is_stopped()]
    (stopped or threads)[0].switch()
    return True


def _user_threads_elsewhere():
    """[(inferior number, [user thread address...])] for every other
    inferior with registered user threads."""
    found = []
    current = gdb.selected_inferior().num
    with _KeepSelection():
        for inferior in sorted(gdb.inferiors(), key=lambda inf: inf.num):
            if inferior.num == current or not _select_inferior(inferior):
                continue
            addrs = []
            try:
                for ut in _iter_user_threads():
                    addrs.append(int(ut))
            except gdb.error:
                pass
            if addrs:
                found.append((inferior.num, addrs))
    return found


# Parked user threads aren't shown while any thread of their program runs
# (see "Views"); every message saying so gives the reason and how to see
# them.

def _running_reason():
    if _parameter("non-stop", False):
        return "the program is in non-stop mode and some of its threads are running"
    return "the program is running"


def _stop_advice():
    # (Older gdbs reject `interrupt -a` in all-stop mode.)
    return "stop every thread ('{}')".format(
        "interrupt -a" if _parameter("non-stop", False) else "interrupt")


# --------------------------------------------------------------------------
# Frames, threads and settings
# --------------------------------------------------------------------------

def _selected_thread_num():
    try:
        t = gdb.selected_thread()
    except (gdb.error, SystemError):
        return None
    if t is None or not t.is_valid():
        return None
    return t.global_num


def _frame_sp(frame):
    return int(frame.read_register(_arch.sp_reg)) & _U64


def _innermost_real_frame():
    """The selected thread's innermost frame that isn't an inline frame,
    and its level."""
    frame = gdb.newest_frame()
    level = 0
    while frame.type() == gdb.INLINE_FRAME:
        older = frame.older()
        if older is None:
            break
        frame = older
        level += 1
    return frame, level


def _selected_level():
    try:
        sel = gdb.selected_frame()
        frame = gdb.newest_frame()
    except gdb.error:
        return 0
    level = 0
    while frame is not None:
        if frame == sel:
            return level
        frame = frame.older()
        level += 1
    return 0


def _select_level(level):
    frame = gdb.newest_frame()
    for _ in range(level):
        older = frame.older()
        if older is None:
            break
        frame = older
    frame.select()


class _KeepSelection(object):
    """Restore the selected thread and frame on exit, with the thread's view
    (if any) shown again.  Some gdbs (10, on core files at least) lose the
    selected frame when Inferior.threads() refreshes the thread list, and
    commands that switch threads without unwinding them leave a view
    unarmed on gdb < 11."""

    def __enter__(self):
        try:
            self.thread = gdb.selected_thread()
            self.level = _selected_level() if self.thread is not None else 0
        except gdb.error:
            self.thread = None
        return self

    def __exit__(self, *exc):
        try:
            if self.thread is not None and self.thread.is_valid():
                if gdb.selected_thread() != self.thread:
                    self.thread.switch()
                if self.thread.is_stopped():
                    _refresh_frames(self.level)
        except gdb.error:
            pass
        return False


def _parameter(name, default=None):
    try:
        return gdb.parameter(name)
    except (gdb.error, RuntimeError):
        return default


def _scheduler_locked():
    return _parameter("scheduler-locking") == "on"


def _notify(msg, later=False):
    """Print a message now, or (later=True, from event handlers that may run
    in the middle of other output) at the next prompt, event or realm-gdb
    command, or once the command is over (see "Views")."""
    if later:
        _messages.append(msg)
    else:
        print(msg)


def _flush_messages():
    while _messages:
        print(_messages.pop(0))


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------
#
# A view shows a parked user thread's stack on a gdb thread (the "host").
# The unwinder below claims the host thread's innermost real frame (the
# "host frame") and reports the parked user thread's saved context as its
# caller's registers, so frame #0 is the host frame and frames #1 and up are
# the parked stack.  A frame filter prints the host frame as a marker line
# in `bt` (see "the marker line").  Nothing in the inferior changes, so
# dropping a view only needs the frame cache rebuilt.
#
# gdb mustn't lose a parked frame while it is using it: gdb >= 13 aborts if
# a frame it was printing is gone when it looks for it again (after a
# function call made by a pretty-printer, for instance), and gdb 11 to 13
# abort if the selected frame is (see _selected_frame_loss_aborts).  The
# frames go once Realm resumes the parked thread, so parked threads are
# shown only while nothing in the program can resume them:
#
#   - A view is dropped as soon as its host thread runs again.  In all-stop
#     mode, the views in an inferior are dropped whenever it runs.  In
#     non-stop mode, parked threads are shown only while every thread of
#     the inferior is stopped, and its views are dropped once one runs.
#     Parked threads are never left out silently: where one isn't shown,
#     the user is told why and how to see it (_running_reason).
#
#   - Inferior calls made while a view is shown run on the host thread with
#     its real registers (gdb takes the stack pointer from frame #0), while
#     their arguments are evaluated in whatever parked frame is selected.
#     Normally every thread runs during a call, so calls made from a host
#     thread hold the other threads still (scheduler-locking), unless `set
#     realm-lock-calls off`.
#
#   - Calls made from other threads can still let a parked thread run, so
#     views are checked against the registry afterwards (_drop_stale_views),
#     and once the call's command is over against where every thread is
#     (see "Once a command is over"), and the unwinder only claims a frame
#     while the parked thread's context is unchanged.
#
# Finding the host frame:
#
#   gdb >= 11 tells an unwinder the pending frame's level.  The unwinder
#   claims the frame at the host frame's level if its PC and SP are still
#   the host frame's (they aren't during an inferior call, for instance).
#
#   Older gdbs don't, and reading a register of the pending frame hits an
#   internal error in gdb 9 ("get_frame_id: Assertion `fi->level == 0'
#   failed") when the frame just inside it is an inline frame.  So there a
#   view is "armed" when the host thread's frame cache is about to be
#   rebuilt from scratch, and the unwinder claims the first frame it is then
#   asked about, which is the innermost one.  gdb rebuilds the cache on
#   every thread switch (the unwinder notices those), after inferior calls,
#   and when memory or registers are written.  Wherever the host thread
#   might not be at the host frame, the view is "hidden" (not claimed at
#   all) until _rearm() has checked that it is: frame #0's own PC and SP are
#   always the thread's real ones, so that is a cheap check.  If the host
#   thread may have run (an unlocked inferior call on another thread), its
#   views are dropped instead, since that can't always be checked in time
#   (MI never shows a prompt).
#
# Forcing a return (`return`, MI's -exec-return) from a parked frame writes
# the parked thread's registers into the host thread, and a return value
# into whichever registers the ABI returns it in (floating-point and vector
# ones included).  The CLI command is refused (see "Refusing commands"
# below); MI's can't be.  So whenever the unwinder claims the host frame,
# which it does again after any change to the host thread's registers that
# gdb reports, it keeps a copy of them all (those gdb saves around an
# inferior call), and once the host thread turns up on the parked thread's
# stack, they are written back (_undo_forced_return) once the command is
# over (registers can't be written from an unwinder).  Whatever lets the
# program run or lets go of it before then has to wait for that: CLI
# commands that do (`continue`, `detach`, ..., and `interpreter-exec`, which
# can run MI commands that do) are hooked (see "Refusing commands"), and
# gdb exiting has an event (gdb >= 12; `quit` is hooked for older ones).  But
# nothing can make an MI command run before gdb has finished the one that
# forced the return (from Python's gdb.execute_mi, say) wait, nor a
# function call made before then (by a pretty-printer as MI prints the
# frame returned to, say): the user is told if the host thread then runs
# (_too_late_to_undo, _call_runs_undone) or gdb lets go of the program
# (_on_exited).  gdb doesn't report MI's -data-write-register-values (nor
# rebuild the frames after it), so a forced return after one undoes that
# change too, and the user is told it may have.
#
# Once a command is over:
#
#   Some things wait until gdb has finished the command it is running:
#   undoing a forced return, checking views against where every thread is
#   (that switches threads, which isn't safe during an inferior call), and
#   with gdb 11 to 13 dropping views that must outlive gdb looking for the
#   selected frame again (see _doom_views).  _catch_up does them at the
#   next prompt and realm-gdb command and, since an MI UI shows no prompt,
#   from an event posted for gdb to run once the command is over
#   (_post_catch_up).  But gdb 14 and older crash as they exit if such an
#   event hasn't run, as after the last command of `gdb -batch` or a
#   `quit` in a user-defined command: exit() destroys gdb's queue of
#   events after Python has been shut down, and destroying an event needs
#   Python.  (gdb 15 empties the queue first.)  So with gdb 12 to 14, if
#   the event still hasn't run once gdb is shutting Python down (after it
#   has killed or detached from every program and saved its history),
#   _exit_if_event_waiting has exit() end the process, with gdb's exit code
#   (from gdb.events.gdb_exiting), just before it would get to the queue:
#   which is after Python's shutdown, every exit handler registered in
#   Python included, and gdb's, but skips the C and C++ ones registered
#   before then (gdb's and its libraries' static destructors, which only
#   free memory, among them).  gdb 11 has no such event, so there an
#   event is posted only for a forced return or doomed views, and only
#   until gdb has shown a prompt (under MI, or before the first prompt): it
#   can still crash as it exits if one hasn't run (after it has detached),
#   and nothing undoes a forced return as `gdb -batch` ends or MI's
#   -gdb-exit exits.  Otherwise, under MI, gdb 11 waits for the next
#   realm-gdb command.
#
# A view is a dict:
#   thread:           global number of the host thread
#   thread_num:       its number within its inferior, for messages
#   thread_object:    its gdb.InferiorThread
#   inferior, pid:    its inferior and process, so that a view never
#                     outlives it
#   ut:               the UserThread*, as a gdb.Value
#   user_thread:      the same, as an int
#   stack:            (base, end) of the user thread's stack, or None
#   host_pc, host_sp: the host frame's PC and SP
#   host_level:       the host frame's level (> 0 below inline frames)
#   regs:             unwind_regs_from_ctx() of the parked user thread
#   values:           regs (plus _Arch.thread_regs) as gdb.Values
#   frame_id:         the host frame's ID while the view is shown
#   temporary:        True for the views `thread apply all bt` shows each
#                     parked thread with, just for its bt
#   saved_regs:       names of the host thread's registers to keep
#   host_regs:        {name: gdb.Value} of them, as last claimed
#   last:             what the unwinder was last asked about: (level, PC,
#                     SP) of a frame, or (level, None, None) of the host
#                     frame it claimed (gdb >= 11; see the unwinder)
#   undoing:          a forced return is waiting to be undone
#   doomed:           the number of the thread whose resuming means the
#                     view must go (see _doom_views; gdb 11 to 13 only)
#   armed, hidden:    see above (gdb < 11 only)
#   level:            the frame the user last had selected while the view
#                     was shown, to go back to if gdb loses it (gdb < 11)
#
# Views are kept per host thread, in a stack (realm-thread nests); only the
# top one is shown.  A view is dropped when its host thread runs again, or
# when its parked thread turns out to have run.
#
# (This script runs in gdb's __main__ namespace, and may be loaded again, by
# another copy of librealm or by hand: keep the state of the earlier load.)
_views = globals().get("_views", {})

# gdb thread the unwinder last ran for (gdb < 11).
_last_unwound_thread = globals().get("_last_unwound_thread")

# Inferior calls in progress, and for each, whether it lets the other
# threads run, its inferior, and whether every inferior runs; how many calls
# have let the other threads run; the scheduler-locking setting to restore
# after a call we locked.
_infcall_depth = globals().get("_infcall_depth", 0)
_calls = globals().get("_calls", [])
_unlocked_calls = globals().get("_unlocked_calls", 0)
_saved_scheduler_locking = globals().get("_saved_scheduler_locking")

# Inferiors whose threads have run since their views were last checked
# against where every thread is (see _drop_stale_views); that is done at the
# next prompt.
_hosts_unchecked = globals().get("_hosts_unchecked", set())

# Messages to print once gdb has finished what it is doing (see _notify).
_messages = globals().get("_messages", [])

# Views no longer shown with a forced return still to undo: their host
# thread hasn't run since, so it is still on the parked thread's stack.
_repairs = globals().get("_repairs", [])

# Whether an event posted for _catch_up is waiting to run, and whether gdb
# has shown a prompt yet (see "Views").
_catch_up_posted = globals().get("_catch_up_posted", False)
_prompt_seen = globals().get("_prompt_seen", False)

# Whether PendingFrame.level() exists (gdb >= 11).
_have_level = hasattr(getattr(gdb, "PendingFrame", None), "level")


def _gdb_major():
    m = re.match(r"\s*(\d+)\.", gdb.VERSION)
    return int(m.group(1)) if m else None


# gdb 11 to 13 abort ("get_selected_frame: Assertion `selected_frame !=
# NULL' failed") when, looking for the selected frame again after a
# command or event, they can't find it (and it isn't frame #0): which is
# what becomes of a selected parked frame once Realm resumes its thread.
# (Fixed in gdb 14; nothing in the API tells, so go by the version.)  So
# with those, views are dropped only once gdb has looked for the selected
# frame again (see _on_cont).
_selected_frame_loss_aborts = _gdb_major() in (11, 12, 13)

# gdb 14 and older crash as they exit if an event posted with gdb.post_event
# hasn't run yet (fixed in gdb 15; see "Views").
_posting_safe = (_gdb_major() or 0) >= 15
_have_exiting = hasattr(getattr(gdb, "events", None), "gdb_exiting")

# The exit code gdb is about to exit with (from gdb.events.gdb_exiting).
_exit_code = globals().get("_exit_code")


class _LockCalls(gdb.Parameter):
    """Whether function calls made from a thread showing a parked Realm user
    thread hold the other threads still (as with `set scheduler-locking on`),
    so that the parked thread stays where it is.  On by default.  A call
    that needs another thread to make progress then hangs (interrupt it
    with Ctrl-C), as it would with scheduler-locking on.  With it off, a
    call may let Realm resume the parked thread, whose frames then
    disappear: gdb 11 to 13 abort if one of them was selected, and gdb 13
    and later if one was being printed (by a pretty-printer that makes
    calls, say)."""

    set_doc = "Set whether calls made while showing a parked user thread hold other threads."
    show_doc = "Show whether calls made while showing a parked user thread hold other threads."

    def __init__(self):
        super(_LockCalls, self).__init__("realm-lock-calls", gdb.COMMAND_DATA,
                                         gdb.PARAM_BOOLEAN)
        self.value = True

    def get_set_string(self):
        return ""

    def get_show_string(self, svalue):
        return ("Function calls made while showing a parked Realm user thread "
                "hold the other threads: {}.".format(svalue))


def _top_view(thread_num):
    stack = _views.get(thread_num)
    if not stack:
        return None
    view = stack[-1]
    try:
        if gdb.selected_inferior().pid != view["pid"]:
            return None
    except gdb.error:
        return None
    return view


def _current_view():
    """The view on the selected thread, if any."""
    return _top_view(_selected_thread_num())


def _remove_view(view):
    """Stop showing a view.  A forced return it has seen is still undone
    (see _repairs)."""
    stack = _views.get(view["thread"], [])
    for i, v in enumerate(stack):
        if v is view:
            del stack[i]
            break
    if not stack:
        _views.pop(view["thread"], None)
    if view.get("undoing") and not any(v is view for v in _repairs):
        _repairs.append(view)
    _frames_changed()


def _frames_changed():
    """Tell gdb's DAP (gdb >= 14), if it is in use, that a view has come or
    gone: it keeps the frames, scopes and variables it has listed until the
    program runs, and would go on showing (or fail on) those of the view
    that was there before.  A client that wants to be told is told to fetch
    the stacks again."""
    for module, name in (("gdb.dap.frames", "_clear_frame_ids"),
                         ("gdb.dap.scopes", "clear_scopes"),
                         ("gdb.dap.varref", "clear_vars")):
        clear = getattr(sys.modules.get(module), name, None)
        if clear is not None:
            try:
                clear(None)
            except Exception:
                pass
    server = sys.modules.get("gdb.dap.server")
    config = getattr(getattr(server, "_server", None), "config", None)
    if isinstance(config, dict) and config.get("supportsInvalidatedEvent"):
        try:
            server.send_event("invalidated", {"areas": ["stacks"]})
        except Exception:
            pass


def _listed(view):
    return any(v is view for v in _views.get(view["thread"], []))


def _saved_reg_names(frame):
    """The registers to keep a copy of for _undo_forced_return: those gdb
    saves around an inferior call (gdb >= 10), else the general ones."""
    try:
        return tuple(r.name for r in frame.architecture().registers("save"))
    except Exception:
        return _arch.gp_regs


def _read_regs(frame, names):
    """{name: gdb.Value} of `frame`'s registers (a gdb.Frame or
    gdb.PendingFrame), leaving out any that can't be read."""
    out = {}
    for name in names:
        try:
            out[name] = frame.read_register(name)
        except Exception:
            pass
    return out


def _make_view(ut, temporary=False):
    frame, level = _innermost_real_frame()
    regs = _arch.unwind_regs_from_ctx(ut)
    values = {name: gdb.Value(value).cast(frame.read_register(name).type)
              for name, value in regs.items()}
    for name in _arch.thread_regs:
        try:
            v = frame.read_register(name)
            values[name] = gdb.Value(int(v) & _U64).cast(v.type)
        except Exception:
            pass  # not on this CPU or gdb
    try:
        base = int(ut["stack_base"]) & _U64
        stack = (base, base + (int(ut["stack_size"]) & _U64))
    except gdb.error:
        stack = None
    saved_regs = () if temporary else _saved_reg_names(frame)
    pc_type = frame.read_register(_arch.pc_reg).type
    # gdb stops unwinding at a caller whose frame ID is inner to (below)
    # this frame's, and the parked stack can be anywhere relative to the
    # host's.  So give the host frame an ID at the parked thread's SP,
    # which is just inside the parked thread's innermost frame.
    frame_id = _FrameId(values[_arch.sp_reg], gdb.Value(frame.pc()).cast(pc_type))
    inferior = gdb.selected_inferior()
    return {
        "thread": _selected_thread_num(),
        "thread_num": gdb.selected_thread().num,
        "thread_object": gdb.selected_thread(),
        "inferior": inferior.num,
        "pid": inferior.pid,
        "ut": ut,
        "user_thread": int(ut),
        "stack": stack,
        "host_pc": frame.pc(),
        "host_sp": _frame_sp(frame),
        "host_level": level,
        "regs": regs,
        "values": values,
        "frame_id": frame_id,
        "temporary": temporary,
        "saved_regs": saved_regs,
        "host_regs": _read_regs(gdb.newest_frame(), saved_regs),
        "last": None,
        "armed": False,
        "hidden": False,
        "level": None,
    }


def _context_unchanged(view):
    """Whether the parked thread is still parked with the context the view
    shows (if that can't be read, assume so)."""
    ut = view["ut"]
    try:
        return not bool(ut["running"]) and _arch.unwind_regs_from_ctx(ut) == view["regs"]
    except gdb.error:
        return True


def _is_host_frame(frame, view):
    """True for the host frame (and any inline frames at the same PC and SP,
    which share its registers)."""
    try:
        return frame.pc() == view["host_pc"] and _frame_sp(frame) == view["host_sp"]
    except gdb.error:
        return False


def _at_host(view):
    """Whether the host thread is at the host frame (frame #0's PC and SP are
    the thread's real registers, whatever the unwinder claims)."""
    try:
        return _is_host_frame(gdb.newest_frame(), view)
    except gdb.error:
        return False


def _is_parked_frame(frame, view):
    regs = view["regs"]
    return frame.pc() == regs[_arch.pc_reg] and _frame_sp(frame) == regs[_arch.sp_reg]


def _first_parked_frame(view):
    """The parked thread's innermost frame, if the view is shown: right
    outside the host frame."""
    try:
        frame = gdb.newest_frame()
        while frame is not None and _is_host_frame(frame, view):
            frame = frame.older()
        if frame is not None and _is_parked_frame(frame, view):
            return frame
    except gdb.error:
        pass
    return None


def _shown(view):
    return _first_parked_frame(view) is not None


def _parked_frames_anywhere(view, limit=64):
    """Whether any of the innermost frames is one of the parked thread's,
    shown or (should something have gone wrong) elsewhere."""
    try:
        frame = gdb.newest_frame()
        for _ in range(limit):
            if frame is None:
                return False
            if _is_parked_frame(frame, view):
                return True
            frame = frame.older()
    except gdb.error:
        pass
    return False


def _on_parked_stack(view, sp):
    stack = view["stack"]
    return stack is not None and stack[0] <= sp < stack[1]


def _on_parked_frame(view):
    """Whether the host thread itself (frame #0) is on the parked thread's
    stack: a forced return into its frames has happened."""
    try:
        return _on_parked_stack(view, _frame_sp(gdb.newest_frame()))
    except gdb.error:
        return False


def _raw(value):
    """A register's contents, as bytes (None if unavailable)."""
    try:
        if hasattr(value, "bytes"):  # gdb >= 15
            return bytes(value.bytes)
        size = value.type.sizeof
        vec = value.cast(gdb.lookup_type("unsigned char").vector(size - 1))
        return bytes(int(vec[i]) & 0xff for i in range(size))
    except Exception:
        return None


def _undo_forced_returns():
    """Undo the forced returns the unwinder has seen, and (where it doesn't
    look for them, gdb < 11) one on the selected thread."""
    while _repairs:
        _undo_forced_return(_repairs.pop(0))
    for stack in list(_views.values()):
        if stack and stack[-1].get("undoing"):
            _undo_forced_return(stack[-1])
    view = _current_view()
    if view is not None:
        _undo_forced_return(view)


def _undo_forced_return(view):
    """Write the host thread's registers back after a forced return (MI's
    -exec-return can't be refused) moved it onto the parked thread's
    stack."""
    view.pop("undoing", None)
    try:
        thread = view["thread_object"]
        if not thread.is_valid() or not thread.is_stopped():
            return
        if gdb.selected_thread() == thread:
            if _restore_host_regs(view):
                _refresh_frames("parked")
            return
        with _KeepSelection():
            thread.switch()
            _restore_host_regs(view)
    except (gdb.error, RuntimeError):
        pass


def _write_reg(name, value):
    """Set the selected thread's register `name` to `value` (a gdb.Value of
    its type, as read)."""
    # (A register is written in the selected frame, and gdb rebuilds the
    # frames after each write.)
    gdb.newest_frame().select()
    if hasattr(gdb, "set_convenience_variable"):  # gdb >= 8.3
        gdb.set_convenience_variable("_realm_gdb_reg", value)
        gdb.execute("set var ${} = $_realm_gdb_reg".format(name), to_string=True)
    else:
        gdb.execute("set var ${} = {}".format(name, int(value)), to_string=True)


def _restore_host_regs(view):
    """_undo_forced_return on the selected thread.  Returns whether there
    was anything to undo."""
    if (_listed(view) and _current_view() is not view) or not _on_parked_frame(view):
        return False
    failed = []
    view["restoring"] = True  # (the unwinder leaves host_regs alone)
    try:
        for name, value in dict(view["host_regs"]).items():
            want = _raw(value)
            try:
                if _raw(gdb.newest_frame().read_register(name)) == want:
                    continue
                _write_reg(name, value)
                if _raw(gdb.newest_frame().read_register(name)) != want:
                    failed.append(name)
            except Exception:
                failed.append(name)
    finally:
        view["restoring"] = False
        if hasattr(gdb, "set_convenience_variable"):
            gdb.set_convenience_variable("_realm_gdb_reg", None)
    if failed:
        print("realm-thread: a forced return (MI's -exec-return, which can't be "
              "refused) moved thread {} onto user thread {:#x}'s stack, and its "
              "registers {} could not be put back: resuming the program may crash "
              "it.".format(view["thread_num"], view["user_thread"],
                           ", ".join("$" + n for n in failed)))
    else:
        print("realm-thread: put thread {} back where it was: a forced return (MI's "
              "-exec-return, which can't be refused) had moved it onto user thread "
              "{:#x}'s stack.  (A change made to its registers with MI's "
              "-data-write-register-values while the user thread was shown, which "
              "gdb doesn't report, may have been undone too.)".format(
                  view["thread_num"], view["user_thread"]))
    return True


def _rearm(view):
    """gdb < 11: let the unwinder claim the host frame again if the host
    thread is there, else hide the view.  The caller rebuilds frames."""
    at_host = _at_host(view)
    view["hidden"] = not at_host
    view["armed"] = at_host


def _refresh_frames(select=None):
    """Rebuild the selected thread's frames, with its view (if any) applied
    if its thread is at the host frame.  Then select frame `select` (a
    level), or the parked thread's innermost frame ("parked"), or #0."""
    view = _current_view()
    if view is not None and not _have_level:
        _rearm(view)
    gdb.invalidate_cached_frames()
    _frames_changed()
    if select == "parked" and view is not None:
        frame = _first_parked_frame(view)
        if frame is not None:
            frame.select()
            return
    _select_level(select if isinstance(select, int) else 0)


def _not_started(view):
    """A user thread that has never run still holds makecontext()'s
    context, whose PC is its entry point rather than a return address.
    gdb would look up the caller frame at PC-1, inside whatever function
    precedes the entry point, so there is no meaningful stack to show.
    Returns the entry function's name for such a thread, else None."""
    pc = view["regs"][_arch.pc_reg]
    try:
        block = gdb.block_for_pc(pc)
    except RuntimeError:
        return None
    while block is not None and block.function is None:
        block = block.superblock
    if block is None or int(block.function.value().address) != pc:
        return None
    return block.function.print_name


class _FrameId(object):
    def __init__(self, sp, pc):
        self.sp = sp
        self.pc = pc


class _ParkedThreadUnwinder(gdb.unwinder.Unwinder):
    def __init__(self):
        super(_ParkedThreadUnwinder, self).__init__("realm-parked-thread")

    def __call__(self, pending_frame):
        global _last_unwound_thread
        try:
            num = _selected_thread_num()
            view = _top_view(num)
            if _have_level:
                if view is None:
                    return None
                level = pending_frame.level()
                pc = int(pending_frame.read_register(_arch.pc_reg)) & _U64
                sp = int(pending_frame.read_register(_arch.sp_reg)) & _U64
                claim = (level == view["host_level"] and pc == view["host_pc"]
                         and sp == view["host_sp"]
                         and (view.get("doomed") or _context_unchanged(view)))
                if not view["temporary"]:
                    _watch_host(view, pending_frame, level, pc, sp, claim)
                if not claim:
                    return None
            else:
                if num != _last_unwound_thread:
                    # A thread switch, so the frames are being rebuilt.
                    _last_unwound_thread = num
                    if view is not None:
                        view["armed"] = True
                if view is None or view["hidden"] or not view["armed"]:
                    return None
                view["armed"] = False
                if not view.get("doomed") and not _context_unchanged(view):
                    return None
            info = pending_frame.create_unwind_info(view["frame_id"])
            for name, value in view["values"].items():
                info.add_saved_register(name, value)
            return info
        except Exception:
            return None


def _watch_host(view, pending_frame, level, pc, sp, claim):
    """gdb >= 11, from the unwinder: keep a copy of the host thread's
    registers, and notice a forced return (see "Views").

    Each time gdb rebuilds a thread's frames it asks the unwinders about
    them innermost first, except for inline frames, so the first one it
    asks about is the innermost real frame, whose registers are the
    thread's own.  If that is the host frame, the view claims it, and keeps
    a copy of them.  If it is on the parked thread's stack, a forced return
    has put the thread there.  The first frame of a rebuild is one that
    doesn't carry on outwards from the frame asked about before: carrying on
    is going to a higher level and a higher (or the same) SP, and not to
    the same PC and SP.  The parked thread's own frames carry on from the
    host frame, wherever its stack is."""
    last = view["last"]
    view["last"] = (level, None, None) if claim else (level, pc, sp)
    if view.get("restoring"):
        return
    if claim:
        view["host_regs"] = _read_regs(pending_frame, view["saved_regs"])
        return
    carries_on = (last is not None and level > last[0]
                  and (last[1] is None or (sp >= last[2] and (pc, sp) != last[1:])))
    if not carries_on and not view.get("undoing") and _on_parked_stack(view, sp):
        view["undoing"] = True
        _post_catch_up()


# ---- the marker line --------------------------------------------------------
#
# Any enabled frame filter makes gdb print the whole backtrace, every parked
# frame included, with its Python frame printer instead of its own.  Mostly
# that changes only how it looks, but in three ways it matters.  First,
# that printer gets each frame's locals from its decorator, and gdb's
# FrameDecorator leaves out some that gdb's printer shows (artificial ones
# such as __func__, and ones that are optimized out or constant) and,
# before gdb 13, adds those of the function an inlined one is in; so the
# filter below lists them as gdb's printer does (_frame_locals).  And while
# the printer prints values from the filter's frames:
#
#   - Function calls in the program are refused (`set may-call-functions
#     off`).  A call (from a pretty-printer: libstdc++'s make some, for
#     std::error_code before gcc 12 and std::stringstream from gcc 14)
#     rebuilds gdb's frames, and the printer then drops frame numbers and,
#     before gdb 13, goes on using the frame it was printing after it has
#     been freed.  A value whose pretty-printer has a call refused is
#     printed without it, and a note says so (_CallsRefused).  gdb 8 has no
#     such setting, so there the marker line is drawn only while no
#     pretty-printers are in use but gdb's own.
#
#   - From gdb 10 it hands pretty-printers values that are entirely
#     optimized out, which most can't print (libstdc++'s raise an error);
#     gdb's own printer prints `<optimized out>` without asking them.  So a
#     lookup function that gdb consults before all others does the same
#     (_printing_lookup).
#
# The printer prints a frame's arguments and locals as it fetches them from
# the frame's decorator, so that is when (_printing), until it has printed
# the last frame.  gdb's DAP doesn't use the printer, and keeps frame
# iterators from one request to the next, so none of this applies to the
# frames it gets (see _for_frame_printer).

_SAFE_PRINTERS = ("builtin", "realm-gdb")

# Whether gdb can be told to refuse function calls (gdb >= 9).
_can_refuse_calls = _parameter("may-call-functions") is not None

# How many _printing calls are under way; the printer list _printing_lookup
# is in meanwhile; whether realm-gdb has turned may-call-functions off (kept
# across a reload, so that it is turned back on); whether a value has been
# printed without its pretty-printer since; and, for each iteration of the
# filter's frames by gdb's printer, whether it holds _printing on.
_printing_depth = 0
_printing_list = None
_calls_refused = globals().get("_calls_refused", False)
_printed_raw = False
_printing_holders = []


def _printers_safe():
    lists = [gdb.pretty_printers, gdb.current_progspace().pretty_printers]
    lists += [o.pretty_printers for o in gdb.objfiles()]
    for printers in lists:
        for p in printers:
            if getattr(p, "enabled", True) and getattr(p, "name", None) not in _SAFE_PRINTERS:
                return False
    return True


def _for_frame_printer():
    """Whether gdb's frame printer (`bt`, or MI's -stack-list-* with frame
    filters enabled) is the one asking for frames: it calls
    gdb.frames.execute_frame_filters."""
    frame = sys._getframe(1)
    while frame is not None:
        if (frame.f_code.co_name == "execute_frame_filters"
                and frame.f_globals.get("__name__") == "gdb.frames"):
            return True
        frame = frame.f_back
    return False


def _printing(items):
    """The arguments or locals of a frame being printed by gdb's frame
    printer, which prints each as it gets it: with what "the marker line"
    describes in place meanwhile."""
    if items is None:
        return None
    return _printing_items(items)


def _printing_items(items):
    try:
        _begin_printing()
        for item in items:
            yield item
    finally:
        _end_printing()


def _begin_printing():
    global _printing_depth, _printing_list, _calls_refused, _printed_raw
    _printing_depth += 1
    # (Stay this way until the printer has printed every frame.)
    for holder in _printing_holders:
        if not holder.holding:
            holder.holding = True
            _printing_depth += 1
    if _printing_list is not None:
        return
    _printed_raw = False
    try:
        objfiles = gdb.objfiles()
        if objfiles:
            _printing_list = objfiles[0].pretty_printers
        else:
            _printing_list = gdb.current_progspace().pretty_printers
        _printing_list.insert(0, _printing_lookup)
    except Exception:
        _printing_list = None
    if _can_refuse_calls and not _calls_refused and _parameter("may-call-functions"):
        try:
            gdb.execute("set may-call-functions off", to_string=True)
            _calls_refused = True
        except gdb.error:
            pass


def _end_printing():
    global _printing_depth, _printing_list, _printed_raw
    _printing_depth -= 1
    if _printing_depth > 0:
        return
    _printing_depth = 0
    if _printing_list is not None:
        while _printing_lookup in _printing_list:
            _printing_list.remove(_printing_lookup)
        _printing_list = None
    _allow_calls()
    if _printed_raw:
        _printed_raw = False
        print("realm-gdb: a pretty-printer above tried to call a function in the "
              "program, which realm-gdb doesn't allow while gdb prints a backtrace "
              "with a <parked Realm user thread> line (gdb mishandles calls then), "
              "so its value is shown without it.  To see it with its "
              "pretty-printer, select its frame ('frame N') and use 'print VAR', "
              "'info locals' or 'info args'.")


def _printing_over():
    """Once gdb has finished a command, make sure nothing of _printing is
    left in place (were an iteration of the filter's frames not closed)."""
    global _printing_depth
    del _printing_holders[:]
    if _printing_depth > 0:
        _printing_depth = 1
        _end_printing()
    else:
        _allow_calls()


def _allow_calls():
    """Turn may-call-functions back on, if realm-gdb turned it off and
    nothing is being printed."""
    global _calls_refused
    if _calls_refused and _printing_depth == 0:
        try:
            gdb.execute("set may-call-functions on", to_string=True)
            _calls_refused = False
        except gdb.error:
            pass


def _printing_lookup(val):
    """A pretty-printer lookup function that gdb consults before all others
    while its frame printer prints values from the frame filter's frames
    (see "the marker line")."""
    try:
        if _entirely_optimized_out(val, [_OPTIMIZED_OUT_BUDGET]):
            return _OptimizedOut(val)
    except Exception:
        pass  # (Can't tell: leave it to the others.)
    if not _calls_refused:
        return None
    try:
        printer = _find_printer(val)
    except gdb.error as e:
        if not _refused(e):
            raise
        return _WithoutPrinters(val)
    return None if printer is None else _CallsRefused(printer, val)


_printing_lookup.name = "realm-gdb"

# How many fields and elements of a value _entirely_optimized_out looks at.
_OPTIMIZED_OUT_BUDGET = 4096

_NOT_SCALAR = (gdb.TYPE_CODE_ARRAY, gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION,
               gdb.TYPE_CODE_SET, gdb.TYPE_CODE_STRING)
_REFERENCE = (gdb.TYPE_CODE_REF, getattr(gdb, "TYPE_CODE_RVALUE_REF", gdb.TYPE_CODE_REF))


def _entirely_optimized_out(val, budget):
    """Whether no part of `val` is there, as gdb's printer checks before it
    asks pretty-printers.  (Value.is_optimized_out is any part.)  Raises if
    that would take more than budget[0] fields and elements to find out, or
    can't be told."""
    if not val.is_optimized_out:
        return False
    t = val.type.strip_typedefs()
    if t.code in (gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION):
        parts = (val[f] for f in t.fields() if hasattr(f, "bitpos"))  # (not static)
    elif t.code == gdb.TYPE_CODE_ARRAY:
        low, high = t.range()
        parts = (val[i] for i in range(low, high + 1))
    elif t.code in _NOT_SCALAR:
        raise ValueError("can't tell")
    else:
        return True
    for part in parts:
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("too big to tell")
        if not _entirely_optimized_out(part, budget):
            return False
    return True


def _scalar_type(t):
    """gdb's val_print_scalar_type_p."""
    t = t.strip_typedefs()
    while t.code in _REFERENCE:
        t = t.target().strip_typedefs()
    return t.code not in _NOT_SCALAR


class _OptimizedOut(object):
    """Prints a value that is entirely optimized out as gdb's own printer
    does (in summary mode, as with `set print frame-arguments scalars`,
    `...` unless it is a scalar: gdb < 13 doesn't tell)."""

    def __init__(self, val):
        self._val = val

    def to_string(self):
        options = gdb.print_options() if hasattr(gdb, "print_options") else {}
        if options.get("summary") and not _scalar_type(self._val.type):
            return "..."
        return "<optimized out>"


def _refused(error):
    """Whether a gdb.error is gdb refusing a function call."""
    return "may-call-functions" in str(error)


def _find_printer(val):
    """The pretty-printer gdb would find for `val` (but for
    _printing_lookup), as find_pretty_printer does."""
    lists = [o.pretty_printers for o in gdb.objfiles()]
    lists += [gdb.current_progspace().pretty_printers, gdb.pretty_printers]
    for printers in lists:
        for lookup in list(printers):
            if lookup is _printing_lookup:
                continue
            if hasattr(lookup, "enabled") and not lookup.enabled:
                continue
            printer = lookup(val)
            if printer is not None:
                return printer
    return None


def _without_printers(val):
    """`val` as gdb prints it without pretty-printers."""
    global _printed_raw
    _printed_raw = True
    try:
        return val.format_string(raw=True)
    except gdb.error as e:
        return "<error: {}>".format(e)


class _WithoutPrinters(object):
    """Prints a value without pretty-printers."""

    def __init__(self, val):
        self._val = val

    def to_string(self):
        return _without_printers(self._val)


class _CallsRefused(object):
    """A pretty-printer, but if it has a function call refused, the value
    is printed without it (or, if that happens while it lists the value's
    children, the rest of them are left out)."""

    def __init__(self, printer, val):
        self._printer = printer
        self._val = val
        self._failed = False

    # (Has just the methods the printer has: gdb asks.)
    def __getattr__(self, name):
        attr = getattr(self._printer, name)
        if name == "to_string":
            return lambda: self._to_string(attr)
        if name == "children":
            return lambda: self._children(attr)
        return attr

    def _to_string(self, to_string):
        try:
            return to_string()
        except gdb.error as e:
            if not _refused(e):
                raise
            self._failed = True
            return _without_printers(self._val)

    def _children(self, children):
        global _printed_raw
        if self._failed:
            return
        try:
            for child in children():
                yield child
        except gdb.error as e:
            if not _refused(e):
                raise
            _printed_raw = True


class _ParkedThreadMarker(FrameDecorator):
    # (gdb's DAP needs every frame's address; gdb's frame printer leaves it
    # out of the marker line if there is none.)
    def __init__(self, base, user_thread_ptr, printer):
        super(_ParkedThreadMarker, self).__init__(base)
        self._user_thread_ptr = user_thread_ptr
        self._printer = printer

    def function(self):
        return "<parked Realm user thread {:#x}>".format(self._user_thread_ptr)

    def address(self):
        return None if self._printer else self._base.address()

    def filename(self):
        return None

    def line(self):
        return None

    def frame_args(self):
        return None

    def frame_locals(self):
        return None


class _PrintingHolder(object):
    """An iteration of the filter's frames by gdb's frame printer, which
    once it is printing values (_begin_printing) holds that on until it
    is done."""

    def __init__(self):
        self.holding = False


class _PrintedFrame(FrameDecorator):
    """A frame for gdb's frame printer (see "the marker line")."""

    def frame_args(self):
        return _printing(self._base.frame_args())

    def frame_locals(self):
        if _stock_locals(self._base):
            return _printing(_frame_locals(self.inferior_frame()))
        return _printing(self._base.frame_locals())

    def elided(self):
        elided = self._base.elided()
        if elided is None:
            return None
        return (_PrintedFrame(dec) for dec in elided)


def _function(method):
    return getattr(method, "__func__", method)


def _stock_locals(dec):
    """Whether gdb's FrameDecorator would list the locals of frame
    decorator `dec` itself (no frame filter's decorator lists them)."""
    while isinstance(dec, FrameDecorator):
        if _function(type(dec).frame_locals) is not _function(FrameDecorator.frame_locals):
            return False
        dec = getattr(dec, "_base", None)
    return not hasattr(dec, "frame_locals")


class _Local(object):
    """A local variable, for gdb's frame printer to read (as gdb's own
    printer does)."""

    def __init__(self, sym):
        self._sym = sym

    def symbol(self):
        return self._sym

    def value(self):
        return None


def _frame_locals(frame):
    """A frame's locals as gdb's own printer lists them: those of each
    block, from the innermost out to the function's own."""
    try:
        block = frame.block()
    except RuntimeError:
        return None
    found = []
    while block is not None and not (block.is_global or block.is_static):
        for sym in block:
            if (sym.is_variable or sym.is_constant) and not sym.is_argument:
                found.append(_Local(sym))
        if block.function is not None:
            break
        block = block.superblock
    return found


class _ParkedThreadFrameFilter(object):
    def __init__(self):
        self.name = "realm-parked-thread"
        self.priority = 100
        self._user_enabled = True

    # Reporting the filter enabled sends every `bt` through gdb's Python
    # frame printer, so do that only while a view is shown, and (gdb 8)
    # only when the printers are safe for it (see above).  gdb asks before
    # each backtrace, so also make sure calls are allowed again.
    @property
    def enabled(self):
        try:
            _allow_calls()
            view = _current_view()
            return view is not None and _marker_drawn() and _shown(view)
        except Exception:
            return False

    @enabled.setter
    def enabled(self, value):
        self._user_enabled = value

    def filter(self, frame_iter):
        try:
            view = _current_view()
            if view is None or not _shown(view):
                return frame_iter
            printer = _for_frame_printer()
        except Exception:
            return frame_iter
        return self._mark(iter(frame_iter), view, printer)

    @staticmethod
    def _mark(frame_iter, view, printer):
        # Collapse the host frame and any inline frames at the same spot
        # into one marker line, then pass the parked stack through.
        holder = _PrintingHolder()
        if printer:
            _printing_holders.append(holder)
        try:
            marked = False
            for dec in frame_iter:
                if _is_host_frame(dec.inferior_frame(), view):
                    if not marked:
                        marked = True
                        yield _ParkedThreadMarker(dec, view["user_thread"], printer)
                    continue
                yield _PrintedFrame(dec) if printer else dec
                break
            for dec in frame_iter:
                yield _PrintedFrame(dec) if printer else dec
        finally:
            if printer:
                _printing_holders[:] = [h for h in _printing_holders if h is not holder]
                if holder.holding:
                    _end_printing()


# (Keep the user's `disable frame-filter` across a reload.)
_frame_filter = _ParkedThreadFrameFilter()
_frame_filter._user_enabled = getattr(gdb.frame_filters.get("realm-parked-thread"),
                                      "_user_enabled", True)


def _marker_drawn():
    """Whether a shown view's frame #0 is drawn as the marker line."""
    return _frame_filter._user_enabled and (_can_refuse_calls or _printers_safe())


# ---- keeping views correct --------------------------------------------------

def _drop_stale_views(why, later=False, check_hosts=False, sps=None):
    """Stop showing any parked user thread (of the selected inferior) that
    has run since it was shown, and so parked again somewhere else or
    finished: its stack is no longer there.  That can happen while other
    threads run during an inferior call that isn't locked.  With
    check_hosts, also look for one that a thread is executing right now (see
    _executing_on), which takes every thread's SP: `sps`, or found by
    switching threads, so not from inside an inferior call.  Unless `later`
    (inside an inferior call, after which gdb rebuilds the frames itself),
    rebuild the selected thread's frames if its view went."""
    inferior = gdb.selected_inferior()
    if check_hosts:
        _hosts_unchecked.discard(inferior.num)
    views = [v for stack in _views.values() for v in stack
             if v["inferior"] == inferior.num]
    if not views:
        return
    if check_hosts:
        # (A forced return puts a host thread on the parked stack, which
        # would look like that.)
        _undo_forced_returns()
        if sps is None:
            sps = _host_sps()[0]
    else:
        sps = {}
    current = _current_view()
    for view in views:
        stale = view["pid"] != inferior.pid
        if not stale:
            try:
                ut = _find_user_thread(view["user_thread"])
                stale = (ut is None or not _is_parked(ut)
                         or _arch.unwind_regs_from_ctx(ut) != view["regs"]
                         or _executing_on(ut, sps) is not None)
            except gdb.error:
                # Can't tell (e.g. gdb 9 can't read memory through a running
                # thread in non-stop mode): leave it to the next check.
                continue
        if stale:
            _remove_view(view)
            _notify("realm-thread: user thread {:#x} {}; no longer showing "
                    "it.".format(view["user_thread"], why), later)
        elif check_hosts and _host_moved(view):
            _remove_view(view)
            _notify("realm-thread: thread {} has run since it was showing user "
                    "thread {:#x}; no longer showing it.".format(
                        view["thread_num"], view["user_thread"]), later)
    if not later and current is not None and current is not _current_view():
        t = gdb.selected_thread()
        if t is not None and t.is_stopped():
            _refresh_frames("parked")


def _host_moved(view):
    """Whether the host thread is no longer at the host frame (it may have
    run during a function call made on another thread).  Switches threads."""
    thread = view["thread_object"]
    try:
        if not thread.is_valid() or not thread.is_stopped():
            return True
        with _KeepSelection():
            thread.switch()
            return not _at_host(view)
    except (gdb.error, RuntimeError):
        return False


def _restore_scheduler_locking():
    global _saved_scheduler_locking
    if _saved_scheduler_locking is not None and _infcall_depth == 0:
        try:
            gdb.execute("set scheduler-locking " + _saved_scheduler_locking,
                        to_string=True)
        except gdb.error:
            pass
        _saved_scheduler_locking = None


def _on_cont(event):
    """A thread is running again, so the views on it are gone.  (Inferior
    calls also resume the program, but come back to the same place.)  In
    non-stop mode, Realm may now resume any parked thread of the resumed
    thread's process, so drop all the views in its inferior (with gdb 11 to
    13, only once gdb has finished: see _doom_views).  In all-stop mode an
    inferior's threads run together: drop all the views in each inferior
    that runs (every inferior, with schedule-multiple).  A forced return
    on a thread that stays stopped is still undone once the command is
    over; on one that runs it is too late, and the user is told so."""
    if _infcall_depth > 0:
        return
    _restore_scheduler_locking()
    _flush_messages()
    if not _views and not _repairs:
        return
    t = getattr(event, "inferior_thread", None)
    if t is not None:
        resumed = _views.pop(t.global_num, [])
        if resumed:
            print("realm-thread: thread {} is running; no longer showing parked "
                  "user threads on it.".format(t.num))
        _too_late_to_undo(resumed)
        others = [view for stack in _views.values() for view in stack
                  if view["inferior"] == t.inferior.num]
        if others and _selected_frame_loss_aborts:
            _doom_views(t.inferior.num, t.num)
        elif others:
            for view in others:
                _remove_view(view)
            print(_no_longer_showing(t.num, others))
            if _repairs:
                _post_catch_up()
        return
    # Which inferiors run?  gdb marks the threads it resumes as running
    # before it reports it.  (The selected inferior isn't always the one:
    # gdb 10 to 12 report resuming a new thread in one inferior with
    # another selected.)
    running = set()
    for stack in _views.values():
        for view in stack:
            if _thread_running(view["thread_object"]):
                running.add(view["inferior"])
    selected = gdb.selected_thread()
    if selected is not None and selected.is_valid() and selected.is_running():
        running.add(selected.inferior.num)
    dropped = False
    for stack in list(_views.values()):
        for view in list(stack):
            if view["inferior"] in running:
                _remove_view(view)
                dropped = True
    if dropped:
        print("realm-thread: the program is running; no longer showing parked "
              "user threads.")
    _too_late_to_undo()


def _too_late_to_undo(views=()):
    """Forget the forced returns still to undo (of `views`, and in
    _repairs) whose host thread is running now, from the parked thread's
    stack: saying so, since the program may crash."""
    for view in list(views) + list(_repairs):
        if not view.get("undoing") or not _thread_running(view["thread_object"]):
            continue
        view.pop("undoing", None)
        _repairs[:] = [v for v in _repairs if v is not view]
        print("realm-thread: thread {} is running from user thread {:#x}'s stack, "
              "where a forced return (MI's -exec-return) moved it, before realm-gdb "
              "could put it back: the program may crash.".format(
                  view["thread_num"], view["user_thread"]))


def _call_runs_undone(caller, others_run, inferior, every_inferior):
    """Before a function call made on thread `caller`: say so if it runs a
    thread that a forced return has moved onto a parked thread's stack,
    before that could be undone (see "Views").  The call's frames may
    overwrite the parked thread's.  One that runs on its own thread puts it
    back where it was, but one that runs while other threads do lets the
    thread run on from there, which is too late to undo."""
    pending = [stack[-1] for stack in _views.values()
               if stack and stack[-1].get("undoing")] + list(_repairs)
    for view in pending:
        if view["thread"] == caller:
            _notify("realm-thread: a function call is running on thread {} from "
                    "user thread {:#x}'s stack, where a forced return (MI's "
                    "-exec-return) moved it, before realm-gdb could put it back: it "
                    "may overwrite that user thread's frames, and the program may "
                    "crash.".format(view["thread_num"], view["user_thread"]),
                    later=True)
        elif others_run and (every_inferior or view["inferior"] == inferior):
            view.pop("undoing", None)
            _repairs[:] = [v for v in _repairs if v is not view]
            _notify("realm-thread: a function call let thread {} run from user "
                    "thread {:#x}'s stack, where a forced return (MI's -exec-return) "
                    "moved it, before realm-gdb could put it back: the program may "
                    "crash.".format(view["thread_num"], view["user_thread"]),
                    later=True)


def _doom_views(inferior, thread_num):
    """gdb 11 to 13 (see _selected_frame_loss_aborts), non-stop mode: thread
    `thread_num` of `inferior` is running, so Realm may resume any user
    thread shown on its other threads.  Stop showing them, but only once gdb
    has looked for the frame selected before this (after `thread apply N
    continue &`, say): until then the unwinder keeps claiming their frames
    as they were, so that gdb finds it.  That is once the command is over
    (_catch_up)."""
    doomed = [view for stack in _views.values() for view in stack
              if view["inferior"] == inferior and not view.get("doomed")]
    for view in doomed:
        view["doomed"] = thread_num
    if doomed:
        _post_catch_up()


def _drop_doomed_views():
    """Drop the views _doom_views has doomed (from _catch_up)."""
    selected = _selected_thread_num()
    reselect = False
    dropped = {}
    for stack in list(_views.values()):
        for view in list(stack):
            if view.get("doomed"):
                _remove_view(view)
                dropped.setdefault(view["doomed"], []).append(view)
                reselect = reselect or view["thread"] == selected
    thread = gdb.selected_thread()
    if reselect and thread is not None and thread.is_stopped():
        # (This forgets the frame gdb has been told to select again, if it
        # hasn't looked for it yet, as well as the one it found.)
        _refresh_frames(0)
    for thread_num, views in sorted(dropped.items()):
        print(_no_longer_showing(thread_num, views))


def _catch_up():
    """Do what has had to wait until gdb finished a command (see "Views"):
    undo forced returns, drop doomed views, and check the views of
    inferiors whose threads have run against where every thread is.
    (Undoing a forced return can make gdb look for the selected frame,
    which a doomed view is there for.)"""
    _printing_over()
    _undo_forced_returns()
    _drop_doomed_views()
    current = gdb.selected_inferior().num
    if current in _hosts_unchecked:
        _drop_stale_views("has run since", check_hosts=True)
    for inferior in gdb.inferiors():
        if inferior.num in _hosts_unchecked and inferior.num != current:
            with _KeepSelection():
                if _select_inferior(inferior):
                    _drop_stale_views("has run since", check_hosts=True)
                else:
                    _hosts_unchecked.discard(inferior.num)


def _post_catch_up(views_only=False):
    """Have _catch_up run once gdb has finished the command it is running,
    from an event posted for that if one may be (see "Views"), as well as
    at the next prompt or realm-gdb command.  `views_only`: nothing but
    what is shown is at stake (not the program's registers)."""
    global _catch_up_posted
    if _catch_up_posted:
        return
    if not _posting_safe and not _have_exiting:
        # gdb 11 (older ones have nothing to post).
        if views_only or _prompt_seen:
            return
    _catch_up_posted = True
    gdb.post_event(_posted_catch_up)


def _posted_catch_up():
    global _catch_up_posted
    _catch_up_posted = False
    try:
        # (gdb runs posted events while it waits for an inferior call to
        # finish; the call posts another once it has.)
        if _infcall_depth == 0:
            _flush_messages()
            _catch_up()
    except Exception:
        if os.environ.get("REALM_GDB_DEBUG"):
            traceback.print_exc()


def _no_longer_showing(thread_num, views):
    """Non-stop mode: thread `thread_num` is running, so `views` (on other
    threads) were dropped."""
    return ("realm-thread: thread {} is running (non-stop mode), so Realm may "
            "resume any parked user thread at any moment, which makes it unsafe "
            "to keep showing "
            "them: no longer showing {}.  To see one again, stop every thread "
            "('interrupt -a') and run 'realm-thread <ptr>' again.".format(
                thread_num, ", ".join(
                    "user thread {:#x} (on thread {})".format(
                        view["user_thread"], view["thread_num"])
                    for view in views)))


def _thread_running(thread):
    """Whether a thread is running (or gone)."""
    try:
        return not thread.is_valid() or thread.is_running()
    except RuntimeError:
        return True


def _on_stop(event):
    _flush_messages()
    if _parameter("non-stop", False):
        _hosts_unchecked.add(gdb.selected_inferior().num)
        _drop_stale_views("has run since")


def _on_exited(event):
    """A program has exited, or gdb has detached from or killed it (which
    has no exit code).  If that was before a forced return could be undone
    (see "Views"), say so: a detached program may crash."""
    global _infcall_depth
    _flush_messages()
    inferior = getattr(event, "inferior", None)
    for stack in list(_views.values()):
        for view in list(stack):
            if inferior is None or view["inferior"] == inferior.num:
                _remove_view(view)
    gone = [view for view in _repairs
            if inferior is None or view["inferior"] == inferior.num]
    _repairs[:] = [view for view in _repairs if not any(v is view for v in gone)]
    if not hasattr(event, "exit_code"):
        for view in gone:
            print("realm-thread: gdb has let go of process {} (detached from or "
                  "killed it) before realm-gdb could undo a forced return (MI's "
                  "-exec-return) that had moved its thread {} onto user thread "
                  "{:#x}'s stack: if it was detached, it may crash.".format(
                      view["pid"], view["thread_num"], view["user_thread"]))
    _infcall_depth = 0
    del _calls[:]
    _restore_scheduler_locking()


def _on_gdb_exiting(event):
    """gdb >= 12: undo a forced return that hasn't been yet (see
    "Views"), e.g. when `gdb -batch` is done, before gdb detaches."""
    global _exit_code
    _exit_code = getattr(event, "exit_code", None)
    _flush_messages()
    _undo_forced_returns()


def _exit_if_event_waiting():
    """Run by Python as gdb shuts it down, once gdb has killed or detached
    from every program and saved its history.  gdb 12 to 14 would crash as
    they exit if an event posted for _catch_up still hasn't run (see
    "Views"), when exit() gets to the C++ destructor of gdb's event queue.
    exit() runs C exit handlers newest first, before destructors registered
    earlier, so have it flush every stdio stream and end the process with
    gdb's exit code, once the rest of Python's shutdown and gdb's are done.
    (If this can't be arranged, gdb crashes as it would have.)"""
    if _posting_safe or not _catch_up_posted or _exit_code is None:
        return
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        at_exit = libc["__cxa_atexit"]
        at_exit.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        at_exit.restype = ctypes.c_int

        def address(name):
            return ctypes.cast(libc[name], ctypes.c_void_p).value

        # (Each is called with the second argument here as its first.)
        end, flush = address("_exit"), address("fflush")
        if at_exit(end, _exit_code & 0xff, None) == 0:
            at_exit(flush, None, None)
    except Exception:
        if os.environ.get("REALM_GDB_DEBUG"):
            traceback.print_exc()


def _call_thread_num(event):
    """The gdb thread an inferior call event is for, if it is the selected
    thread (as it is unless another thread stopped during the call).  Don't
    look through Inferior.threads(): that refreshes the thread list, which
    in the middle of an inferior call makes gdb 12 fail an internal
    assertion."""
    t = gdb.selected_thread()
    if t is None or not t.is_valid() or t.ptid != event.ptid:
        return None
    return t.global_num


def _call_lets_others_run():
    """Whether an inferior call lets the other threads run: in all-stop
    mode, unless scheduler-locking is on.  (In non-stop mode only the
    calling thread runs.)"""
    return not _parameter("non-stop", False) and not _scheduler_locked()


def _on_inferior_call(event):
    """Before a call made from a thread showing a parked thread: hold the
    other threads still (see "Views" above), and on gdb < 11 hide the view,
    since the thread's innermost frames are the call's for now.

    After any call: restore scheduler-locking.  If the call let the other
    threads run, drop the views that are no longer right: in the calling
    inferior, those whose parked thread has run (the rest are checked
    against where every thread is once the command is over); in other
    inferiors, if they ran too (schedule-multiple), all of them, since that
    can't be checked during the call.  On gdb < 11 also: once a call has returned to
    gdb's dummy frame on top of the host frame, gdb restores the thread's
    registers and rebuilds its frames (re-selecting the frame the user had
    selected), so show the view again; if the call stopped somewhere else
    (e.g. at a breakpoint), the view stays hidden and the user sees the real
    frames.  If the call let the other threads run, their views are
    dropped, since their host threads may have moved and that can't be
    checked now."""
    global _infcall_depth, _unlocked_calls, _saved_scheduler_locking
    if isinstance(event, getattr(gdb, "InferiorCallPreEvent", ())):
        _infcall_depth += 1
        view = _top_view(_call_thread_num(event))
        if (view is not None and _lock_calls.value and _call_lets_others_run()
                and _saved_scheduler_locking is None):
            _saved_scheduler_locking = _parameter("scheduler-locking")
            gdb.execute("set scheduler-locking on", to_string=True)
        others_run = _call_lets_others_run()
        if others_run:
            _unlocked_calls += 1
        _calls.append((others_run, gdb.selected_inferior().num,
                       bool(_parameter("schedule-multiple", False))))
        _call_runs_undone(_call_thread_num(event), *_calls[-1])
        if view is not None and not _have_level:
            view["hidden"] = True
        return
    if not isinstance(event, getattr(gdb, "InferiorCallPostEvent", ())):
        return
    _infcall_depth = max(0, _infcall_depth - 1)
    if _calls:
        others_run, inferior, every_inferior = _calls.pop()
    else:
        others_run, inferior, every_inferior = True, gdb.selected_inferior().num, True
    _restore_scheduler_locking()
    if others_run:
        for stack in list(_views.values()):
            for view in list(stack):
                if view["inferior"] != inferior and every_inferior:
                    _remove_view(view)
                    _notify("realm-thread: inferior {} ran during the function call; "
                            "no longer showing user thread {:#x} on its thread "
                            "{}.".format(view["inferior"], view["user_thread"],
                                         view["thread_num"]), later=True)
        _hosts_unchecked.add(inferior)
        _drop_stale_views("ran during the function call", later=True)
    num = _call_thread_num(event)
    if others_run and not _have_level:
        for other, stack in list(_views.items()):
            if other != num:
                for view in list(stack):
                    if view["inferior"] != inferior:
                        continue
                    _remove_view(view)
                    _notify("realm-thread: thread {} ran during the function call; "
                            "no longer showing user thread {:#x} on it.".format(
                                view["thread_num"], view["user_thread"]), later=True)
    # What waits until the command is over (see "Views") may have waited
    # for this call too.
    if _repairs or any(view.get("doomed") or view.get("undoing")
                       for stack in _views.values() for view in stack):
        _post_catch_up()
    elif _views and _hosts_unchecked:
        _post_catch_up(views_only=True)
    if _have_level:
        return
    view = _top_view(num)
    if view is None:
        return
    try:
        newest = gdb.newest_frame()
        returned = (newest.type() == gdb.DUMMY_FRAME
                    and newest.older() is not None
                    and _is_host_frame(newest.older(), view))
    except gdb.error:
        returned = False
    if returned:
        view["hidden"] = False
        view["armed"] = True


def _on_target_written(event):
    """gdb < 11: writing memory or registers makes gdb rebuild the frame
    cache (writing a register may also move the host thread).  The host
    thread's registers as written are the ones to keep (gdb >= 11 keeps
    them as it unwinds)."""
    if _have_level:
        return
    view = _current_view()
    if view is not None:
        if (isinstance(event, getattr(gdb, "RegisterChangedEvent", ()))
                and not view.get("restoring") and _at_host(view)):
            view["host_regs"] = _read_regs(gdb.newest_frame(), view["saved_regs"])
        _rearm(view)
        gdb.invalidate_cached_frames()


def _on_before_prompt():
    """Undo a forced return and check the views.  On gdb < 11 also show the
    selected thread's view again if gdb rebuilt its frames without the
    unwinder knowing (e.g. `thread apply 1 2 3 print 1`), going back to the
    frame the user had selected."""
    global _infcall_depth, _prompt_seen
    _prompt_seen = True
    _infcall_depth = 0
    del _calls[:]
    _restore_scheduler_locking()
    _flush_messages()
    _catch_up()
    if _parameter("non-stop", False):
        _drop_stale_views("has run since")
    view = _current_view()
    if view is None:
        return
    if not _context_unchanged(view):
        _drop_stale_views("has run since")
        return
    if _have_level:
        return
    if _shown(view):
        view["level"] = _selected_level()
        return
    if not _at_host(view):
        if not view["hidden"]:
            # e.g. the user moved the thread with `set $pc`.
            view["hidden"] = True
            gdb.invalidate_cached_frames()
        return
    _refresh_frames(view["level"] if view["level"] is not None else "parked")
    if not _shown(view):
        # E.g. the user disabled the unwinder.  Don't try again at every
        # prompt, losing the user's frame selection each time.
        _remove_view(view)
        _refresh_frames()
        print("realm-thread: cannot show user thread {:#x} (is the "
              "realm-parked-thread unwinder disabled?); no longer showing "
              "it.".format(view["user_thread"]))


def _guarded(handler):
    """Event handlers must never raise: gdb would print the exception at
    every prompt (e.g. while the selected thread is running).  Set
    REALM_GDB_DEBUG=1 in gdb's environment to see them anyway."""
    def run(*args):
        try:
            handler(*args)
        except Exception:
            if os.environ.get("REALM_GDB_DEBUG"):
                traceback.print_exc()
    return run


def _connect(registry_name, handler_name):
    """Connect a handler once per gdb session, even if this script is loaded
    again (by another copy of librealm, or by hand).  The connection looks
    the handler up by name, so the most recently loaded copy runs."""
    flag = "_realm_gdb_connected_" + registry_name
    registry = getattr(gdb.events, registry_name, None)
    if registry is None or getattr(gdb, flag, False):
        return
    namespace = globals()
    registry.connect(lambda *args: _guarded(namespace[handler_name])(*args))
    setattr(gdb, flag, True)


def _install_view_support():
    gdb.unwinder.register_unwinder(None, _ParkedThreadUnwinder(), replace=True)
    gdb.frame_filters["realm-parked-thread"] = _frame_filter
    _connect("cont", "_on_cont")
    _connect("stop", "_on_stop")
    _connect("exited", "_on_exited")
    _connect("inferior_call", "_on_inferior_call")
    _connect("memory_changed", "_on_target_written")
    _connect("register_changed", "_on_target_written")
    _connect("before_prompt", "_on_before_prompt")
    _connect("gdb_exiting", "_on_gdb_exiting")
    if _have_exiting and not _posting_safe and not getattr(gdb, "_realm_gdb_atexit", False):
        namespace = globals()
        atexit.register(lambda: namespace["_exit_if_event_waiting"]())
        gdb._realm_gdb_atexit = True


# --------------------------------------------------------------------------
# Refusing commands that would run the host thread
# --------------------------------------------------------------------------

# What each refused command would do while a parked user thread is shown on
# thread N.  `return` pops frames by writing the selected frame's caller's
# registers -- here, a parked thread's -- into thread N, and `jump` to a
# line of a parked frame runs that code on thread N's stack.  The rest act
# on thread N where it really is, which is harmless but not what the user
# is looking at.  (Measured with gdb 9.2 and 16.2.)
_CORRUPTS = ("would apply the parked user thread's frames to thread {thread} and "
             "may corrupt or crash the program")
_WRONG_THREAD = ("would act on thread {thread} itself, not on the parked user "
                 "thread, which can't run until Realm resumes it")
_REFUSED_COMMANDS = {
    "return": _CORRUPTS,
    "jump": _CORRUPTS,
    "step": _WRONG_THREAD, "stepi": _WRONG_THREAD,
    "next": _WRONG_THREAD, "nexti": _WRONG_THREAD,
    "until": _WRONG_THREAD, "advance": _WRONG_THREAD,
    "finish": ("would let the program run until the parked frame returns, "
               "whenever Realm resumes that user thread"),
    # record/replay variants (rr, gdb 'record full')
    "reverse-step": _WRONG_THREAD, "reverse-stepi": _WRONG_THREAD,
    "reverse-next": _WRONG_THREAD, "reverse-nexti": _WRONG_THREAD,
    "reverse-finish": _WRONG_THREAD,
}

# Commands that let the program run, or let go of it, which aren't refused
# but are hooked too, so that a forced return still to be undone is undone
# first (see "Views").  `interpreter-exec` can run MI commands that do.
# (gdb >= 12 tells us as it exits, before letting go of every program,
# however that happens; `quit` is hooked only for older ones.)
_UNDO_FIRST_COMMANDS = ("continue", "signal", "detach", "detach inferiors",
                        "disconnect", "interpreter-exec") + (
                            () if _have_exiting else ("quit",))


def _undo_first():
    """Before a command in _UNDO_FIRST_COMMANDS.  This must never fail: gdb
    doesn't run a command whose hook fails, `detach` and `quit` included.
    Nothing else that waits until a command is over is done here, since
    this may be in the middle of one (see _doom_views)."""
    try:
        _flush_messages()
        _undo_forced_returns()
    except Exception:
        if os.environ.get("REALM_GDB_DEBUG"):
            traceback.print_exc()


class RealmThreadGuard(gdb.Command):
    """realm-thread-guard COMMAND

    Used by realm-gdb.py's hook-COMMAND definitions: fails if the selected
    thread is showing a parked user thread, which stops COMMAND from
    running.  Stepping commands, `finish`, `jump` and `return` act on the
    selected thread itself, not on the parked user thread on show.  (MI's
    -exec-return bypasses CLI hooks; see _undo_forced_return.)  Commands
    that let the program run or let go of it, such as `continue`, `detach`
    and `quit`, are never refused, but are hooked too, since a forced
    return still to be undone must be undone before them."""

    def __init__(self):
        super(RealmThreadGuard, self).__init__("realm-thread-guard",
                                               gdb.COMMAND_MAINTENANCE)

    def invoke(self, arg, from_tty):
        cmd = arg.strip()
        if cmd not in _REFUSED_COMMANDS:
            _undo_first()
            return
        # First make sure what is on show is still right.
        _flush_messages()
        _catch_up()
        _drop_stale_views("has run since", check_hosts=True)
        # Only while parked frames are on show: when stopped inside an
        # inferior call made from a parked frame, `finish` and friends act
        # on the call's real frames, as usual.
        view = _current_view()
        if view is None or not _parked_frames_anywhere(view):
            return
        why = _REFUSED_COMMANDS[cmd].format(
            thread=gdb.selected_thread().num)
        raise gdb.GdbError(
            "realm-thread: refusing to '{}' while showing user thread {:#x}: it "
            "{}.  Run 'realm-thread back' first, or 'continue' to resume the "
            "program.".format(cmd, view["user_thread"], why))


def _install_guards():
    # This runs while librealm is being loaded, often in the middle of
    # `run`; a command that fails here makes gdb 12 return from `run`
    # without waiting for the program to stop.  So find the hooks that
    # already exist without asking for ones that don't.
    try:
        defined = gdb.execute("help user-defined", to_string=True)
    except gdb.error:
        defined = ""
    for c in sorted(_REFUSED_COMMANDS) + list(_UNDO_FIRST_COMMANDS):
        # (`detach inferiors` is hooked by `detach hook-inferiors`.)
        prefix, _, word = c.rpartition(" ")
        hook = (prefix + " " if prefix else "") + "hook-" + word
        # Leave the user's own hooks alone.
        if re.search(r"^{} -- ".format(re.escape(hook)), defined, re.M):
            try:
                existing = gdb.execute("show user " + hook, to_string=True)
            except gdb.error:
                existing = ""
            if "realm-thread-guard" not in existing:
                if c in _REFUSED_COMMANDS:
                    print("realm-gdb: you defined {}, so '{}' is not refused while "
                          "a parked Realm user thread is shown on a thread N.  There "
                          "it {}.".format(hook, c, _REFUSED_COMMANDS[c].format(
                              thread="N")))
                else:
                    print("realm-gdb: you defined {}, so a forced return (MI's "
                          "-exec-return) from a parked Realm user thread's frame "
                          "isn't undone before '{}' if both happen in one "
                          "command.".format(hook, c))
                continue
        try:
            gdb.execute("define {}\nrealm-thread-guard {}\nend\n".format(hook, c),
                        to_string=True)
        except gdb.error as e:
            print("realm-gdb: could not define {}: {}".format(hook, e))


# --------------------------------------------------------------------------
# thread apply all
# --------------------------------------------------------------------------

def _bt_args(command):
    """If `command` is a backtrace command, return its arguments, else
    None."""
    m = re.match(r"\s*(\S+)\s*(.*)$", command, re.S)
    if m is None:
        return None
    word, rest = m.group(1), m.group(2)
    if word in ("bt", "where") or (len(word) >= 2 and "backtrace".startswith(word)):
        return rest.strip()
    if word == "i" or (len(word) >= 3 and "info".startswith(word)):
        m = re.match(r"(\S+)\s*(.*)$", rest, re.S)
        if m is not None and "stack".startswith(m.group(1)):
            return m.group(2).strip()
    return None


# backtrace options that take a value, and the values boolean options may
# take; see `help backtrace` (gdb >= 9).
_BT_VALUE_OPTIONS = ("-entry-values", "-frame-arguments", "-frame-info")
_BT_BOOLEAN_VALUES = ("on", "off", "yes", "no", "enable", "disable", "1", "0")
_BT_QUALIFIERS = ("full", "no-filters", "hide")


def _split_bt_args(bt_args):
    """(options and qualifiers, count expression) of `bt` arguments."""
    tokens = bt_args.split()
    options = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _BT_QUALIFIERS:
            options.append(tok)
            i += 1
        elif tok == "--":
            options.append(tok)
            i += 1
            break
        elif re.match(r"-[A-Za-z]", tok):
            options.append(tok)
            i += 1
            if i < len(tokens) and (
                    (len(tok) > 2 and any(o.startswith(tok) for o in _BT_VALUE_OPTIONS))
                    or tokens[i] in _BT_BOOLEAN_VALUES):
                options.append(tokens[i])
                i += 1
        else:
            break
    return options, " ".join(tokens[i:])


def _parked_bt_args(options, count, view):
    """bt arguments that show `count` of the parked thread's frames: `bt N`
    counts from #0, and before the parked stack come the host frame and any
    inline frames inside it (one marker line, if that is drawn)."""
    if not count:
        return " ".join(options)
    n = int(gdb.parse_and_eval(count))
    if n > 0:
        filtered = (_marker_drawn() and "no-filters" not in options
                    and not any(len(o) > 2 and "-no-filters".startswith(o)
                                for o in options))
        n += 1 if filtered else view["host_level"] + 1
    return " ".join(options + [str(n)])


def _dump_parked_threads(bt_args):
    """Print the stack of every parked user thread of each inferior, each
    shown in turn on a stopped thread of it, in a section that begins and
    ends with a line of its own.  (The caller restores the selection.)"""
    for inferior in sorted(gdb.inferiors(), key=lambda inf: inf.num):
        if _select_inferior(inferior):
            _dump_inferior_parked_threads(bt_args)


def _dump_inferior_parked_threads(bt_args):
    """_dump_parked_threads for the selected inferior, on the selected
    thread."""
    where = ""
    if len(gdb.inferiors()) > 1:
        where = " of inferior {}".format(gdb.selected_inferior().num)
    end = "=== End of parked Realm user threads{} ===".format(where)
    others_running = False
    try:
        sps, others_running = _host_sps()
        parked = []
        loop = None
        try:
            for ut in _iter_user_threads():
                # (One whose state can't be read is listed too, with the
                # error it gets, rather than left out.)
                if _is_parked(ut) or not _state_readable(ut):
                    parked.append(ut)
        except _RegistryLoop as e:
            loop = e
        except gdb.error as e:
            if not parked:
                raise
            loop = "cannot read the rest of the Realm user thread registry: {}".format(e)
    except gdb.error as e:
        if others_running:
            # (gdb 9 can't read memory through a running thread.)
            _print_not_shown(where, None)
        else:
            print("\n=== Parked Realm user threads{} ===".format(where))
            print("realm-gdb: cannot read the Realm user thread registry: {}".format(e))
        print(end)
        return
    if not parked and loop is None:
        return
    host = gdb.selected_thread()
    if others_running or not host.is_stopped():
        _print_not_shown(where, len(parked))
        if loop is not None:
            print("realm-gdb: {}; there may be more parked threads.".format(loop))
        print(end)
        return
    print("\n=== Parked Realm user threads{} ({}), shown on thread {}: each stack "
          "starts at frame #1 (#0 is {}thread {}'s own frame) ===".format(
              where, len(parked), host.num,
              "a marker for " if _marker_drawn() else "", host.num))
    options, count = _split_bt_args(bt_args)
    try:
        for i, ut in enumerate(parked):
            print("\nUser thread {:#x}:".format(int(ut)))
            try:
                executing = _executing_on(ut, sps)
                if executing is not None:
                    print("  <running on thread {}: it is switching contexts>".format(
                        executing))
                    continue
                view = _make_view(ut, temporary=True)
                start = _not_started(view)
                if start is not None:
                    print("  <not started yet; will begin in {}>".format(start))
                    continue
                _views.setdefault(host.global_num, []).append(view)
                calls = _unlocked_calls
                # `set backtrace limit N` should show N parked frames, as
                # `bt N` does: allow for the frames before them.
                limit = _parameter("backtrace limit")
                try:
                    if limit:
                        gdb.execute("set backtrace limit {}".format(
                            limit + view["host_level"] + 1), to_string=True)
                    _refresh_frames("parked")
                    args = _parked_bt_args(options, count, view)
                    gdb.execute("bt " + args, from_tty=False)
                finally:
                    _remove_view(view)
                    if limit:
                        gdb.execute("set backtrace limit {}".format(limit), to_string=True)
            except gdb.error as e:
                print("  <bt failed: {}>".format(e))
                continue
            if _unlocked_calls != calls:
                if i + 1 < len(parked):
                    rest = len(parked) - i - 1
                    print("\n<A function call (by a pretty-printer?) let the program "
                          "run, so the other {} parked user thread{} may have moved: "
                          "not showing {}.  To see {}, {}rerun this "
                          "command.>".format(
                              rest, "s" if rest > 1 else "",
                              "them" if rest > 1 else "it",
                              "them" if rest > 1 else "it",
                              "" if _lock_calls.value else
                              "run 'set realm-lock-calls on' (which holds the "
                              "other threads still during such calls) and "))
                break
        if loop is not None:
            print("\nrealm-gdb: {}; there may be more parked threads.".format(loop))
        print("\n" + end)
    finally:
        _refresh_frames()


def _print_not_shown(where, count):
    print("\n=== Parked Realm user threads{}{}: not shown ===".format(
        where, "" if count is None else " ({})".format(count)))
    reason = _running_reason()
    print("{}{}, which makes it unsafe to show them: Realm may resume any of "
          "them at any moment.  To see them, {} and rerun this command.".format(
              reason[0].upper(), reason[1:], _stop_advice()))


class ThreadApplyAll(gdb.Command):
    """Apply a command to all threads, plus parked Realm user threads for bt.

    Usage: thread apply all [-ascending] [-q] [-c] [-s] [--] COMMAND

    Like gdb's own `thread apply all` (see `help thread apply`), which this
    replaces: the flags mean the same and threads are visited in descending
    order unless -ascending is given.  When COMMAND is `bt` (or `backtrace`,
    `where` or `info stack`, with any options), the stack of every parked
    Realm user thread of each inferior is printed afterwards, with the same
    options, between a "=== Parked Realm user threads ... ===" line and a
    "=== End of parked Realm user threads ===" line.  (While some threads
    of the program are running, as they can be in non-stop mode, it says
    how many there are and how to see them instead.)  If COMMAND fails on
    a thread, gdb stops there as usual (unless -c is given), and the
    command fails with gdb's error, but only after the parked threads."""

    _FLAGS = ("-ascending", "-c", "-q", "-s", "--")

    def __init__(self):
        super(ThreadApplyAll, self).__init__("thread apply all", gdb.COMMAND_STACK)

    @staticmethod
    def _split_flags(arg):
        """(flags, the command after them, verbatim)."""
        flags = []
        rest = arg.lstrip()
        while rest.startswith("-"):
            m = re.match(r"(\S+)\s*(.*)$", rest, re.S)
            flags.append(m.group(1))
            rest = m.group(2)
            if m.group(1) == "--":
                break
        return flags, rest

    def invoke(self, arg, from_tty):
        _flush_messages()
        _catch_up()
        with _KeepSelection():
            self._invoke(arg, from_tty)

    def _invoke(self, arg, from_tty):
        flags, rest = self._split_flags(arg.strip())
        if not rest:
            raise gdb.GdbError("Please specify a command at the end of "
                               "'thread apply all'")
        ascending = any(len(f) > 1 and "-ascending".startswith(f) for f in flags)
        flags = [f for f in flags if not (len(f) > 1 and "-ascending".startswith(f))]
        threads = []
        for inf in gdb.inferiors():
            threads += [(inf.num, t.num) for t in inf.threads()]
        threads.sort(reverse=not ascending)
        bt_args = _bt_args(rest)
        error = None
        try:
            if threads:
                ids = " ".join("{}.{}".format(i, t) for i, t in threads)
                gdb.execute("thread apply {} {}{}".format(
                    ids, "".join(f + " " for f in flags), rest), from_tty=from_tty)
        except gdb.error as e:
            # Show the parked threads even so (e.g. after a running thread in
            # non-stop mode), but not after a quit (e.g. 'q' at the pager),
            # which arrives as KeyboardInterrupt and so isn't caught here.
            error = e
        if bt_args is not None:
            _dump_parked_threads(bt_args)
        if error is not None:
            raise gdb.GdbError(str(error))

    def complete(self, text, word):
        # As gdb's own `thread apply ID... COMMAND` would, after the flags.
        # (gdb >= 13 first asks with word=None, to find the word breaks.)
        if word is None:
            word = re.split(r"[ \t]", text)[-1]
        flags, rest = self._split_flags(text)
        last = re.split(r"[ \t]", text)[-1]
        if rest == "" and last.startswith("-") and "--" not in flags[:-1]:
            # (gdb may break words at '-', so `word` can lack it.)
            return [f[len(last) - len(word):] for f in self._FLAGS
                    if f.startswith(last)]
        prefix = "thread apply 1 "
        try:
            out = gdb.execute("complete " + prefix + rest, to_string=True)
        except gdb.error:
            return []
        start = len(rest) - len(word)
        return [line[len(prefix):][start:] for line in out.splitlines()
                if line.startswith(prefix)]


# --------------------------------------------------------------------------
# realm-thread
# --------------------------------------------------------------------------

class RealmThread(gdb.Command):
    """realm-thread [<ptr> | back]

    With no argument, list the Realm user threads of the selected inferior.
    With a UserThread pointer (any expression), show that parked user
    thread's stack on the selected thread: frame #0 stays the thread's own
    frame (which backtraces show as a `<parked Realm user thread ...>`
    line; with gdb 8, only while no pretty-printers are loaded but gdb's
    own), and the user thread's stack starts at frame #1, so that `bt`,
    `frame N`, `info locals` and `print VAR` operate on it.  Calls nest.
    Pretty-printers can't call functions in the program while a backtrace
    with that line is printed (gdb mishandles it then): their values are
    shown without them, and `frame N` then `print VAR`, `info locals` or
    `info args` shows them as usual.
    `realm-thread back` (or `restore`) undoes the latest one.  Resuming the
    thread undoes all of them on it.  See also `show realm-lock-calls`.

    Parked user threads are only shown while nothing can resume them, since
    gdb may abort if their frames go while it uses them: in non-stop mode,
    only while every thread of the program is stopped (`interrupt -a`), and
    no longer once one runs.  Whenever one can't be shown, realm-thread
    says why and how to see it."""

    def __init__(self):
        super(RealmThread, self).__init__("realm-thread", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        _flush_messages()
        _catch_up()
        arg = arg.strip()
        if not arg:
            self._list()
        elif arg in ("back", "restore"):
            self._back()
        else:
            self._show_thread(arg)

    def _list(self):
        sps = {}
        others_running = False
        if _has_target():
            try:
                sps, others_running = _host_sps()
            except gdb.error:
                pass
            _drop_stale_views("has run since", check_hosts=True, sps=sps)
        inferior = gdb.selected_inferior().num
        shown = {}
        for stack in _views.values():
            for view in stack:
                if view["inferior"] == inferior:
                    shown.setdefault(view["user_thread"], []).append(view["thread_num"])
        found = False
        try:
            # (Read the registry through a stopped thread if there is one:
            # gdb 9 can't read memory through a running thread.)
            with _KeepSelection():
                if _has_target():
                    _select_inferior(gdb.selected_inferior())
                for ut in _iter_user_threads():
                    found = True
                    executing = _executing_on(ut, sps)
                    if bool(ut["running"]):
                        t = _host_thread_of(ut)
                        state = "RUNNING" + ("  (on thread {})".format(t.num) if t else "")
                    elif executing is not None:
                        state = "RUNNING  (on thread {}, switching contexts)".format(
                            executing)
                    elif int(ut) in shown:
                        state = "parked  (shown on thread {})".format(
                            ", ".join(str(n) for n in sorted(shown[int(ut)])))
                    else:
                        state = "parked"
                    print("  {:#x}  {}".format(int(ut), state))
        except _RegistryLoop as e:
            print("realm-thread: {}; the list above may be incomplete.".format(e))
        except gdb.error as e:
            if not others_running:
                raise gdb.GdbError("realm-thread: cannot read the Realm user "
                                   "thread registry: {}".format(e))
            print("realm-thread: cannot read the Realm user thread registry "
                  "while {}.  To list them, {} and rerun this command.".format(
                      _running_reason(), _stop_advice()))
        else:
            if not found:
                print("realm-thread: no Realm user threads in the registry.")
            elif others_running:
                print("realm-thread: {}, so these may change at any moment, and "
                      "it is unsafe to show a parked one until you {}.".format(
                          _running_reason(), _stop_advice()))
        if len(gdb.inferiors()) > 1:
            for num, addrs in _user_threads_elsewhere():
                print("realm-thread: inferior {} has {} Realm user thread{} too: "
                      "select it ('inferior {}') to list {}.".format(
                          num, len(addrs), "s" if len(addrs) > 1 else "", num,
                          "them" if len(addrs) > 1 else "it"))

    def _show_thread(self, expr):
        if not _has_target():
            raise gdb.GdbError("realm-thread: no process or core file.")
        thread = gdb.selected_thread()
        if thread is not None and thread.is_running():
            raise gdb.GdbError(
                "realm-thread: {}, which makes it unsafe to show a parked user "
                "thread: Realm may resume it at any moment.  To see one, {} and "
                "rerun this command.".format(_running_reason(), _stop_advice()))
        if thread is None or not thread.is_stopped():
            raise gdb.GdbError("realm-thread: the selected thread must be stopped.")
        try:
            addr = int(gdb.parse_and_eval(expr)) & _U64
        except gdb.error as e:
            raise gdb.GdbError("realm-thread: cannot use {!r} as a UserThread "
                               "pointer: {}".format(expr, e))
        try:
            ut = _find_user_thread(addr)
        except gdb.error as e:
            raise gdb.GdbError("realm-thread: cannot read the Realm user thread "
                               "registry: {}".format(e))
        if ut is None:
            for num, addrs in (_user_threads_elsewhere()
                               if len(gdb.inferiors()) > 1 else []):
                if addr in addrs:
                    raise gdb.GdbError(
                        "realm-thread: {:#x} is a user thread of inferior {}: "
                        "select it ('inferior {}') and rerun this command.".format(
                            addr, num, num))
            raise gdb.GdbError("realm-thread: {:#x} is not a Realm user thread "
                               "(run 'realm-thread' to list them).".format(addr))
        sps, others_running = _host_sps()
        executing = _executing_on(ut, sps)
        if not _is_parked(ut) or executing is not None:
            host = _host_thread_of(ut) if executing is None else None
            num = executing if executing is not None else (host.num if host else None)
            raise gdb.GdbError(
                "realm-thread: user thread {:#x} is running{}, so its stack is "
                "that thread's own.".format(
                    addr, " on thread {}".format(num) if num is not None else ""))
        if others_running:
            raise gdb.GdbError(
                "realm-thread: {}, which makes it unsafe to show user thread {:#x}: "
                "Realm may resume it at any moment.  To see it, {} and rerun this "
                "command.".format(_running_reason(), addr, _stop_advice()))
        try:
            view = _make_view(ut)
        except gdb.error as e:
            raise gdb.GdbError("realm-thread: cannot read the context of user "
                               "thread {:#x}: {}".format(addr, e))
        start = _not_started(view)
        if start is not None:
            raise gdb.GdbError("realm-thread: user thread {:#x} has not started "
                               "yet; it will begin in {}.".format(addr, start))
        _views.setdefault(view["thread"], []).append(view)
        _refresh_frames("parked")
        if not _shown(view):
            _remove_view(view)
            _refresh_frames("parked")
            limit = _parameter("backtrace limit")
            raise gdb.GdbError("realm-thread: cannot unwind user thread {:#x}{}.".format(
                addr, "" if not limit or limit > view["host_level"] + 1 else
                " within 'set backtrace limit {}'".format(limit)))
        print("realm-thread: showing user thread {:#x} on thread {}: its stack "
              "starts at frame #1 (#0 is {}thread {}'s own frame).".format(
                  addr, thread.num, "a marker for " if _marker_drawn() else "",
                  thread.num))
        print("Use 'bt', 'frame N', 'info locals', 'print VAR' as normal.  "
              "Run 'realm-thread back' to undo.")

    def _back(self):
        view = _current_view()
        if view is None:
            print("realm-thread: nothing to restore.")
            return
        _remove_view(view)
        _refresh_frames("parked")
        view = _current_view()
        if view is not None:
            print("realm-thread: back to user thread {:#x}.".format(view["user_thread"]))
        else:
            print("realm-thread: restored.")


_HELP_BANNER = """\
======================================================================
realm-gdb.py loaded.  Realm user-thread debugging commands:

  thread apply all bt       gdb's 'thread apply all bt', plus the stack
                            of every parked Realm user thread.  Also for
                            'backtrace', 'where' and 'info stack', and any
                            bt options (e.g. 'thread apply all bt full').

  realm-thread              List Realm user threads with their UserThread
                            pointers.

  realm-thread <ptr>        Show a parked user thread's stack on the
                            selected thread: frame #0 stays the thread's
                            own frame, and the user thread's stack starts
                            at frame #1.  Use 'bt', 'frame N', 'info
                            locals', 'print VAR' etc. as usual, including
                            function calls, which run on the selected
                            thread with the other threads held still
                            (see 'show realm-lock-calls').  Calls nest.

  realm-thread back         Undo the latest 'realm-thread <ptr>'.

  realm-help                Reprint this banner.

Nothing in the program is modified, so it is always safe to continue,
detach or quit.  Parked user threads are shown only while the program is
stopped (in non-stop mode, every thread of it), and no longer once it
runs again.  return and jump are refused while a user thread is
shown, since they could corrupt the program, and so are step, next,
finish, until and advance, which would act on the selected thread itself.
MI's -exec-return can't be refused; on gdb 11 and later its effect is
undone once gdb has finished the command, or before a continue, detach,
quit or the like in the same one.  (This may also undo register changes
made with MI's -data-write-register-values while the user thread is
shown.  An MI command in the same command, from Python's gdb.execute_mi
say, can let the program run or detach from it first: realm-gdb warns
that it may crash.)

Linux / FreeBSD on x86-64 or aarch64.  Source: src/realm/realm-gdb.py.
======================================================================"""


class RealmHelp(gdb.Command):
    """realm-help

    Print the realm-gdb.py command summary."""

    def __init__(self):
        super(RealmHelp, self).__init__("realm-help", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        print(_HELP_BANNER)


if _supported():
    ThreadApplyAll()
    RealmThread()
    RealmThreadGuard()
    RealmHelp()
    # (Keep the user's setting across a reload.)
    _lock_calls_value = getattr(globals().get("_lock_calls"), "value", True)
    _lock_calls = _LockCalls()
    _lock_calls.value = _lock_calls_value
    _install_guards()
    # Registering the unwinder rebuilds the frames; if this is a reload
    # while a parked thread is shown, keep the user's frame selected.
    _reload_level = _selected_level() if _current_view() is not None else None
    _install_view_support()
    if _reload_level is not None:
        _refresh_frames(_reload_level)
    print(_HELP_BANNER)
else:
    # Other platforms / architectures: do nothing rather than provide a
    # half-working command.
    pass
