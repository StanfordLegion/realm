# Copyright 2025 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Auto-loaded by gdb (via the .debug_gdb_scripts section in librealm) to make
# Realm's user-level threads visible to the debugger.
#
# Provides:
#   - An enhanced `thread apply` command.  `thread apply all bt` (and
#     `backtrace`/`where`) now also prints backtraces for every PARKED user
#     thread, in addition to the kernel-thread output you normally get.  A
#     currently-RUNNING user thread is already visible as the host pthread's
#     stack, so we never duplicate it.
#   - `realm-thread <ptr>` / `realm-thread back` -- redirect the currently
#     selected pthread's registers to a parked user thread's saved context so
#     that `frame N`, `info locals`, and `print VAR` work on its stack.  Run
#     `realm-thread back` to restore the original registers.
#   - `realm-thread` with no arguments lists parked user threads.
#
# Core files can't take register writes.  There, a Python unwinder makes
# the selected pthread's innermost frame unwind into the parked user
# thread's saved context instead: frame #0 becomes a placeholder and the
# user thread's stack starts at frame #1.  See "Read-only targets" below.
#
# Linux/FreeBSD on x86-64 or aarch64.  On other targets the script loads
# as a no-op.

import gdb
import gdb.unwinder
import platform
from gdb.FrameDecorator import FrameDecorator


# --------------------------------------------------------------------------
# Architecture abstraction
# --------------------------------------------------------------------------
#
# The minimum we need from each supported architecture is the ability to:
#   * read the currently-selected gdb thread's program counter, stack
#     pointer, and frame pointer (plus any other regs the unwinder needs
#     to round-trip cleanly);
#   * read those same registers out of a UserThread::ctx (which is a glibc
#     ucontext_t laid out differently per arch);
#   * write a register dict back to the currently-selected gdb thread.
#
# The rest of the script treats register dicts as opaque blobs that get
# passed from current_regs()/saved_regs_from_ctx() into set_regs().  The
# only constraint is that an arch's set_regs() must accept the same dict
# its own current_regs()/saved_regs_from_ctx() produced.
#
# For read-only targets, each arch also provides unwind_regs_from_ctx(),
# which returns {gdb register name: value} for the PC, SP and every
# callee-saved register in a UserThread::ctx, and names its PC and SP
# registers in pc_reg / sp_reg.
#
# 64-bit unsigned mask used to coerce gdb's signed-int reads into the
# unsigned address space the kernel saved.
_U64 = (1 << 64) - 1


class _Arch:
    name = "<abstract>"

    @staticmethod
    def matches():
        raise NotImplementedError

    @staticmethod
    def current_regs():
        raise NotImplementedError

    @staticmethod
    def saved_regs_from_ctx(ut):
        raise NotImplementedError

    @staticmethod
    def set_regs(regs):
        raise NotImplementedError

    @staticmethod
    def unwind_regs_from_ctx(ut):
        raise NotImplementedError


def _select_newest_frame():
    """`$pc`, `$sp` etc. name the *selected* frame's registers, and writing
    one in an outer frame overwrites its saved slot on the stack.  The
    thread's own registers are frame #0's, so select that before writing."""
    gdb.newest_frame().select()


# Per-CPU bases: current_regs() / set_regs() depend only on the CPU
# architecture (gdb register names are the same regardless of OS).
# OS-specific leaf subclasses below override matches() and
# saved_regs_from_ctx() because the ucontext_t layout is OS-specific:
#   * glibc Linux x86-64 uses an indexed gregs[] array;
#   * FreeBSD x86-64 uses named mc_rip / mc_rsp / mc_rbp fields;
#   * Linux aarch64 uses `struct sigcontext` with regs[31] / sp / pc;
#   * FreeBSD aarch64 uses mc_gpregs with gp_x[30] / gp_lr / gp_sp / gp_elr.


class _X86_64Base(_Arch):
    pc_reg = "rip"
    sp_reg = "rsp"

    @staticmethod
    def current_regs():
        t = gdb.selected_thread()
        f = gdb.newest_frame()
        return {
            "thread_num": t.num if (t is not None and t.is_valid()) else None,
            "rip": int(f.read_register("rip")) & _U64,
            "rsp": int(f.read_register("rsp")) & _U64,
            "rbp": int(f.read_register("rbp")) & _U64,
        }

    @staticmethod
    def set_regs(regs):
        _select_newest_frame()
        gdb.execute("set $rip = {:#x}".format(regs["rip"]), to_string=True)
        gdb.execute("set $rsp = {:#x}".format(regs["rsp"]), to_string=True)
        gdb.execute("set $rbp = {:#x}".format(regs["rbp"]), to_string=True)


class _X86_64Linux(_X86_64Base):
    name = "linux/x86_64"
    # glibc x86-64 ucontext gregs[] indices (see /usr/include/sys/ucontext.h).
    _REG_RBP = 10
    _REG_RSP = 15
    _REG_RIP = 16
    _UNWIND_GREGS = (("r12", 4), ("r13", 5), ("r14", 6), ("r15", 7),
                     ("rbp", 10), ("rbx", 11), ("rsp", 15), ("rip", 16))

    @staticmethod
    def matches():
        return (platform.system() == "Linux"
                and platform.machine() in ("x86_64", "amd64"))

    @staticmethod
    def saved_regs_from_ctx(ut):
        gregs = ut["ctx"]["uc_mcontext"]["gregs"]
        return {
            "rip": int(gregs[_X86_64Linux._REG_RIP]) & _U64,
            "rsp": int(gregs[_X86_64Linux._REG_RSP]) & _U64,
            "rbp": int(gregs[_X86_64Linux._REG_RBP]) & _U64,
        }

    @staticmethod
    def unwind_regs_from_ctx(ut):
        gregs = ut["ctx"]["uc_mcontext"]["gregs"]
        return {name: int(gregs[i]) & _U64
                for name, i in _X86_64Linux._UNWIND_GREGS}


class _X86_64FreeBSD(_X86_64Base):
    name = "freebsd/x86_64"
    # FreeBSD mcontext_t (sys/x86/include/ucontext.h) uses named fields,
    # not a gregs[] array.  The relevant fields are mc_rip, mc_rsp, mc_rbp.

    @staticmethod
    def matches():
        return (platform.system() == "FreeBSD"
                and platform.machine() in ("x86_64", "amd64"))

    @staticmethod
    def saved_regs_from_ctx(ut):
        mc = ut["ctx"]["uc_mcontext"]
        return {
            "rip": int(mc["mc_rip"]) & _U64,
            "rsp": int(mc["mc_rsp"]) & _U64,
            "rbp": int(mc["mc_rbp"]) & _U64,
        }

    @staticmethod
    def unwind_regs_from_ctx(ut):
        mc = ut["ctx"]["uc_mcontext"]
        return {name: int(mc["mc_" + name]) & _U64
                for name in ("r12", "r13", "r14", "r15",
                             "rbp", "rbx", "rsp", "rip")}


class _AArch64Base(_Arch):
    pc_reg = "pc"
    sp_reg = "sp"

    @staticmethod
    def current_regs():
        t = gdb.selected_thread()
        f = gdb.newest_frame()
        return {
            "thread_num": t.num if (t is not None and t.is_valid()) else None,
            "pc":  int(f.read_register("pc")) & _U64,
            "sp":  int(f.read_register("sp")) & _U64,
            "x29": int(f.read_register("x29")) & _U64,
            "x30": int(f.read_register("x30")) & _U64,
        }

    @staticmethod
    def set_regs(regs):
        _select_newest_frame()
        gdb.execute("set $pc  = {:#x}".format(regs["pc"]),  to_string=True)
        gdb.execute("set $sp  = {:#x}".format(regs["sp"]),  to_string=True)
        gdb.execute("set $x29 = {:#x}".format(regs["x29"]), to_string=True)
        gdb.execute("set $x30 = {:#x}".format(regs["x30"]), to_string=True)


class _AArch64Linux(_AArch64Base):
    name = "linux/aarch64"
    # Linux/glibc aarch64 mcontext_t is `struct sigcontext` (see
    # linux/arch/arm64/include/uapi/asm/sigcontext.h):
    #     __u64 fault_address;
    #     __u64 regs[31];     // X0..X30
    #     __u64 sp;
    #     __u64 pc;
    #     __u64 pstate;
    # Saved by glibc getcontext() at swapcontext time: pc holds the resume
    # address (i.e. the LR at the getcontext call site), sp holds the SP,
    # regs[29] holds the frame pointer, regs[30] holds the link register.
    _REG_FP = 29
    _REG_LR = 30

    @staticmethod
    def matches():
        return (platform.system() == "Linux"
                and platform.machine() in ("aarch64", "arm64"))

    @staticmethod
    def saved_regs_from_ctx(ut):
        mc = ut["ctx"]["uc_mcontext"]
        regs = mc["regs"]
        return {
            "pc":  int(mc["pc"]) & _U64,
            "sp":  int(mc["sp"]) & _U64,
            "x29": int(regs[_AArch64Linux._REG_FP]) & _U64,
            "x30": int(regs[_AArch64Linux._REG_LR]) & _U64,
        }

    @staticmethod
    def unwind_regs_from_ctx(ut):
        # x19..x28 are callee-saved; x29/x30 are FP/LR.
        mc = ut["ctx"]["uc_mcontext"]
        regs = mc["regs"]
        out = {"x{}".format(i): int(regs[i]) & _U64 for i in range(19, 31)}
        out["sp"] = int(mc["sp"]) & _U64
        out["pc"] = int(mc["pc"]) & _U64
        return out


class _AArch64FreeBSD(_AArch64Base):
    name = "freebsd/aarch64"
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
    _REG_FP = 29

    @staticmethod
    def matches():
        return (platform.system() == "FreeBSD"
                and platform.machine() in ("aarch64", "arm64"))

    @staticmethod
    def saved_regs_from_ctx(ut):
        gpr = ut["ctx"]["uc_mcontext"]["mc_gpregs"]
        return {
            "pc":  int(gpr["gp_elr"]) & _U64,
            "sp":  int(gpr["gp_sp"]) & _U64,
            "x29": int(gpr["gp_x"][_AArch64FreeBSD._REG_FP]) & _U64,
            "x30": int(gpr["gp_lr"]) & _U64,
        }

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


def _registry_head():
    try:
        h = gdb.parse_and_eval("Realm::user_thread_registry_head")
    except gdb.error:
        return None
    return h


def _iter_user_threads():
    h = _registry_head()
    if h is None:
        return
    cur = h
    while int(cur) != 0:
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


def _saved_regs(ut):
    return _arch.saved_regs_from_ctx(ut)


def _current_regs():
    """Snapshot the currently-selected thread's registers AND record which
    thread we snapshot from, so the restore can target the same thread
    even if the selected thread has changed in the meantime (e.g., during
    gdb teardown when `gdb.events.gdb_exiting` fires)."""
    return _arch.current_regs()


def _set_regs(regs):
    """Write the architecture-specific subset of registers (PC, SP, FP,
    and arch-dependent extras) to the currently-selected thread."""
    _arch.set_regs(regs)


def _apply_snapshot(entry):
    """Restore a snapshot taken by _current_regs().  Switches to the
    originating thread first when possible, so registers are always
    written back to the thread the snapshot came from -- never to whatever
    thread happens to be selected right now.  Returns True on success."""
    target_num = entry.get("thread_num")
    if target_num is not None:
        target = None
        try:
            for t in gdb.selected_inferior().threads():
                if t.num == target_num:
                    target = t
                    break
        except gdb.error:
            target = None
        if target is None or not target.is_valid():
            print(
                "realm-gdb: WARNING: cannot restore impersonation -- original "
                "thread (gdb id {}) is gone; skipping snapshot.".format(target_num)
            )
            return False
        try:
            target.switch()
        except gdb.error as e:
            print(
                "realm-gdb: WARNING: cannot switch to thread {} for restore: "
                "{}".format(target_num, e)
            )
            return False
    try:
        _set_regs(entry)
    except gdb.error as e:
        print("realm-gdb: WARNING: failed to write registers: {}".format(e))
        return False
    return True


def _write_regs_or_restore(regs, original):
    """Write `regs` to the selected thread.  If the target refuses (a core
    file can't take register writes), put back anything already written
    and return False so the caller can fall back to a virtual view."""
    try:
        _set_regs(regs)
        return True
    except gdb.error:
        try:
            _set_regs(original)
        except gdb.error:
            pass
        return False


# --------------------------------------------------------------------------
# Read-only targets (core files)
# --------------------------------------------------------------------------
#
# When registers can't be written, a "view" stands in for impersonation.
# The unwinder below makes the selected thread's innermost frame report
# the parked user thread's saved context as its caller's registers, so
# frame #0 is a placeholder and frames #1 and up are the parked stack.
# The frame filter prints the placeholder as a marker line in `bt`.
# Nothing in the inferior changes, so dropping the view only needs the
# frame cache flushed.
#
# The unwinder never reads registers.  In gdb 9, PendingFrame.read_register
# hits an internal error ("get_frame_id: Assertion `fi->level == 0'
# failed") when the frame just inside the pending one is an inline frame,
# which is common in both host and parked stacks.  Instead, the view is
# "armed" whenever the viewed thread's frame cache is rebuilt from scratch,
# and the unwinder claims the first frame it is asked about on that thread,
# which is its innermost non-inline frame.  Everything the unwinder returns
# is computed up front by _make_view().
#
# A view is a dict:
#   thread_num:       gdb thread whose innermost frame is the placeholder
#   host_pc, host_sp: that frame's real PC and SP, which is how the frame
#                     filter recognises it
#   regs:             unwind_regs_from_ctx() of the parked user thread
#   values:           regs as gdb.Values of each register's type
#   frame_id:         the placeholder frame's ID
#   user_thread_ptr:  the UserThread pointer (informational)
#   armed:            True until the unwinder claims the innermost frame
_view = None

# gdb thread the unwinder last ran for.  gdb rebuilds the frame cache on
# every thread switch, so a change means the viewed thread's frames are
# being rebuilt (see _ParkedThreadUnwinder).
_last_unwound_thread = None


def _make_view(ut):
    frame = gdb.newest_frame()
    regs = _arch.unwind_regs_from_ctx(ut)
    values = {name: gdb.Value(value).cast(frame.read_register(name).type)
              for name, value in regs.items()}
    pc_type = frame.read_register(_arch.pc_reg).type
    # gdb stops unwinding at a caller whose frame ID is inner to (below)
    # this frame's, and the parked stack can be anywhere relative to the
    # host's.  So give the placeholder an ID at the parked thread's SP,
    # which is just inside the parked thread's innermost frame.
    frame_id = _FrameId(values[_arch.sp_reg], gdb.Value(frame.pc()).cast(pc_type))
    return {
        "thread_num": gdb.selected_thread().num,
        "host_pc": frame.pc(),
        "host_sp": int(frame.read_register(_arch.sp_reg)) & _U64,
        "regs": regs,
        "values": values,
        "frame_id": frame_id,
        "user_thread_ptr": int(ut),
        "armed": False,
    }


def _view_applies(view):
    if view is None:
        return False
    t = gdb.selected_thread()
    return t is not None and t.num == view["thread_num"]


def _is_host_frame(frame, view):
    """True for the frame the view replaces (and any inline frames at the
    same PC and SP, which share its registers)."""
    try:
        return (frame.pc() == view["host_pc"]
                and int(frame.read_register(_arch.sp_reg)) & _U64 == view["host_sp"])
    except gdb.error:
        return False


def _first_parked_frame(view):
    """The parked thread's innermost frame, or None if the view isn't in
    effect in the current frame cache."""
    frame = gdb.newest_frame()
    while frame is not None and _is_host_frame(frame, view):
        frame = frame.older()
    if frame is None:
        return None
    regs = view["regs"]
    if (frame.pc() != regs[_arch.pc_reg]
            or int(frame.read_register(_arch.sp_reg)) & _U64 != regs[_arch.sp_reg]):
        return None
    return frame


def _set_view(view, select=False):
    """Make `view` (or None) the active view.  With select=True, also
    select the parked thread's innermost frame so that `info locals` and
    friends start there."""
    global _view
    if view is not None:
        view["armed"] = True
    _view = view
    gdb.invalidate_cached_frames()
    if not (select and _view_applies(view)):
        return
    frame = _first_parked_frame(view)
    if frame is not None:
        frame.select()


def _on_before_prompt():
    """Re-apply the view if something rebuilt the frame cache without a
    thread switch (e.g. `flushregs` or `set backtrace limit`)."""
    view = _view
    if not _view_applies(view):
        return
    try:
        if _first_parked_frame(view) is None:
            _set_view(view, select=True)
    except gdb.error:
        pass


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
        view = _view
        if view is None:
            return None
        t = gdb.selected_thread()
        num = t.num if t is not None else None
        if num != _last_unwound_thread:
            _last_unwound_thread = num
            if num == view["thread_num"]:
                view["armed"] = True
        if not view["armed"] or num != view["thread_num"]:
            return None
        view["armed"] = False
        info = pending_frame.create_unwind_info(view["frame_id"])
        for name, value in view["values"].items():
            info.add_saved_register(name, value)
        return info


class _ParkedThreadMarker(FrameDecorator):
    def __init__(self, base, user_thread_ptr):
        super(_ParkedThreadMarker, self).__init__(base)
        self._user_thread_ptr = user_thread_ptr

    def function(self):
        return "<parked Realm user thread {:#x}>".format(self._user_thread_ptr)

    def address(self):
        return None

    def filename(self):
        return None

    def line(self):
        return None

    def frame_args(self):
        return None

    def frame_locals(self):
        return None


class _ParkedThreadFrameFilter(object):
    def __init__(self):
        self.name = "realm-parked-thread"
        self.priority = 100
        self._user_enabled = True

    # Any enabled frame filter sends every `bt` through gdb's Python frame
    # printer, which formats frames differently.  Report enabled only while
    # a view applies so ordinary backtraces are left alone.
    @property
    def enabled(self):
        return self._user_enabled and _view_applies(_view)

    @enabled.setter
    def enabled(self, value):
        self._user_enabled = value

    def filter(self, frame_iter):
        view = _view
        if not _view_applies(view):
            return frame_iter
        return self._mark(iter(frame_iter), view)

    @staticmethod
    def _mark(frame_iter, view):
        # Collapse the host frame and any inline frames at the same spot
        # into one marker line, then pass the parked stack through.
        marked = False
        for dec in frame_iter:
            if _is_host_frame(dec.inferior_frame(), view):
                if not marked:
                    marked = True
                    yield _ParkedThreadMarker(dec, view["user_thread_ptr"])
                continue
            yield dec
            break
        for dec in frame_iter:
            yield dec


def _install_view_support():
    gdb.unwinder.register_unwinder(None, _ParkedThreadUnwinder(), replace=True)
    gdb.frame_filters["realm-parked-thread"] = _ParkedThreadFrameFilter()
    gdb.events.before_prompt.connect(_on_before_prompt)


# Stack of impersonation entries for nested `realm-thread`/`realm-thread back`.
# Each entry is a dict:
#   host_regs:        snapshot of the host pthread's registers BEFORE impersonation
#                     (used to restore on `realm-thread back`, quit, continue, etc.)
#   parked_regs:      the parked user thread's saved context (re-applied after an
#                     inferior function call, which temporarily swaps host_regs back)
#   user_thread_ptr:  the UserThread pointer the user named (informational)
#   view:             only on read-only targets: the view (see "Read-only
#                     targets") standing in for the register writes.  No
#                     registers were changed, so host_regs/parked_regs are
#                     not written back.
# Both host_regs and parked_regs are snapshots as returned by _current_regs:
# {thread_num, rip, rsp, rbp}.  thread_num identifies which gdb thread the
# entry applies to so restore writes to the right thread even if the
# selected thread has shifted.
_impersonate_stack = []


def _pop_impersonation():
    """Undo the newest impersonation entry.  Returns True on success."""
    top = _impersonate_stack[-1]
    if "view" in top:
        _impersonate_stack.pop()
        below = _impersonate_stack[-1] if _impersonate_stack else {}
        _set_view(below.get("view"), select=True)
        return True
    if not _apply_snapshot(top["host_regs"]):
        return False
    _impersonate_stack.pop()
    return True


# True while gdb is inside an inferior function call on the impersonated
# thread (the window between InferiorCallPreEvent and InferiorCallPostEvent).
# During this window the host's real registers have been written back so
# the call uses the host pthread's stack instead of the parked user thread's
# tiny mmap'd stack.  See _on_inferior_call_pre/_post.
_call_suspended = False


def _restore_all_impersonations():
    """Pop every outstanding realm-thread impersonation, writing each
    host_regs snapshot back to the thread it came from.

    Invoked from gdb hook- commands defined in _install_safety_hooks() so
    that the user can't accidentally `continue`, `thread N`, `quit`, etc.
    with clobbered registers.  Critically, this restores to the
    originating thread by number, not to whatever thread happens to be
    selected at restore time -- the selected thread can change between
    impersonation and restore (gdb teardown, stop events, IDE wrappers),
    and writing original-thread-A's registers onto thread-B would
    silently corrupt thread-B's state."""
    if not _impersonate_stack:
        return
    count = len(_impersonate_stack)
    saved_cur = None
    try:
        saved_cur = gdb.selected_thread()
    except gdb.error:
        pass
    while _impersonate_stack:
        if not _pop_impersonation():
            break
    # Best-effort: leave selected thread as it was before the restore.
    try:
        if saved_cur is not None and saved_cur.is_valid():
            saved_cur.switch()
    except gdb.error:
        pass
    print(
        "realm-thread: auto-restored {} impersonation(s) before resume/switch.".format(
            count
        )
    )


def _thread_for_ptid(ptid):
    """Find the gdb.InferiorThread with the given ptid tuple, or None."""
    try:
        for t in gdb.selected_inferior().threads():
            if t.ptid == ptid:
                return t
    except gdb.error:
        pass
    return None


def _on_inferior_call_pre(event):
    """Before gdb runs an inferior function call (e.g. `p foo()`,
    `call foo()`), if we're impersonating a parked user thread on the
    same thread that's about to run the call, swap the host pthread's
    real registers back in.  This is critical: a parked user thread's
    stack is a small mmap'd region with a red-zone page at the end, so
    a function called via gdb on the impersonated stack can easily
    overflow it, fault, and get caught by realm_freeze -- corrupting
    the host pthread's state in a way the regular safety net can't
    undo.  By running the call on the host's normal stack, this stays
    safe."""
    global _call_suspended
    if _call_suspended or not _impersonate_stack:
        return
    top = _impersonate_stack[-1]
    if "view" in top:
        # The host's registers were never replaced.
        return
    target_num = top["host_regs"].get("thread_num")
    if target_num is None:
        return
    call_thread = _thread_for_ptid(event.ptid)
    if call_thread is None or call_thread.num != target_num:
        return
    if not _apply_snapshot(top["host_regs"]):
        return
    _call_suspended = True


def _on_inferior_call_post(event):
    """After the inferior call returns, re-apply the parked user thread's
    saved context so the user's view of the impersonated stack is
    restored.  No-op if we weren't impersonating during the call."""
    global _call_suspended
    if not _call_suspended:
        return
    _call_suspended = False
    if not _impersonate_stack:
        return
    top = _impersonate_stack[-1]
    try:
        _apply_snapshot(top["parked_regs"])
    except gdb.error as e:
        print(
            "realm-gdb: WARNING: failed to re-impersonate after inferior "
            "call: {}".format(e)
        )


def _on_inferior_call(event):
    """Dispatcher: gdb.events.inferior_call fires with either a
    InferiorCallPreEvent or InferiorCallPostEvent."""
    if isinstance(event, getattr(gdb, "InferiorCallPreEvent", ())):
        _on_inferior_call_pre(event)
    elif isinstance(event, getattr(gdb, "InferiorCallPostEvent", ())):
        _on_inferior_call_post(event)


def _block_if_impersonating(cmd_name):
    """Refuse a single-thread step-like command while impersonating.

    Stepping while impersonating would step the host pthread, not the
    parked user thread whose stack we're inspecting -- gdb has no way to
    step a parked user thread because it isn't a real OS thread.  Rather
    than silently auto-restore and step the wrong thing, raise an error
    that aborts the command."""
    if _impersonate_stack:
        raise gdb.GdbError(
            "realm-gdb: refusing to '{}' while impersonating a parked user "
            "thread.  Single-thread step/next/finish/jump operate on the "
            "host pthread, not on the parked user thread whose stack you "
            "are inspecting (gdb has no way to step a parked user thread).  "
            "Run 'realm-thread back' to restore the host pthread first, or "
            "use 'continue' to resume the whole process.".format(cmd_name)
        )


def _install_safety_hooks():
    """Install gdb hook-<cmd> commands in two categories.

    Reject: single-thread step-like commands.  Aborted via gdb.GdbError
    when an impersonation is active, since they would step the host
    pthread rather than the user thread the user is looking at.

    Auto-restore: whole-process resume, thread switch, and gdb teardown
    commands.  These have well-defined semantics regardless of which
    user thread the host pthread "looks like" -- continue resumes
    everything, detach/quit walk away.  Restoring registers before they
    run is just bookkeeping so the inferior ends up in a clean state."""

    reject_cmds = (
        "step", "stepi", "next", "nexti", "finish",
        "until", "advance", "jump",
        # record/replay variants (rr, gdb 'record full'): same hazard --
        # they resume the inferior on a single thread.
        "reverse-step", "reverse-stepi",
        "reverse-next", "reverse-nexti",
        "reverse-finish",
    )
    for c in reject_cmds:
        try:
            gdb.execute(
                "define hook-{c}\n"
                "python _block_if_impersonating(\"{c}\")\n"
                "end\n".format(c=c),
                to_string=True,
            )
        except gdb.error as e:
            print("realm-gdb: WARNING: could not install hook-{}: {}".format(c, e))

    restore_cmds = (
        # whole-process resume
        "continue", "run", "start", "signal",
        # record/replay whole-process resume
        "reverse-continue",
        # thread switch (leaves impersonated regs stranded otherwise)
        "thread",
        # gdb teardown -- detach/quit write registers back to the inferior
        "detach", "quit",
    )
    for c in restore_cmds:
        try:
            gdb.execute(
                "define hook-{c}\n"
                "python _restore_all_impersonations()\n"
                "end\n".format(c=c),
                to_string=True,
            )
        except gdb.error as e:
            print("realm-gdb: WARNING: could not install hook-{}: {}".format(c, e))

    # Fallback for gdb exits that don't route through hook-quit.  Added in
    # gdb 12; ignore if missing on older builds.
    try:
        gdb.events.gdb_exiting.connect(lambda evt: _restore_all_impersonations())
    except AttributeError:
        pass

    # Wrap inferior function calls (e.g. `p foo()`, `call foo()`).  Without
    # this, calling a function via gdb while impersonating runs the call on
    # the parked user thread's mmap'd stack; non-trivial calls overflow it
    # and crash the host pthread via realm_freeze.
    try:
        gdb.events.inferior_call.connect(_on_inferior_call)
    except AttributeError:
        print(
            "realm-gdb: WARNING: gdb.events.inferior_call not available -- "
            "avoid 'p foo()' / 'call foo()' while impersonating (may crash "
            "the host pthread)."
        )


def _dump_parked_threads():
    parked = []
    for ut in _iter_user_threads():
        if _is_parked(ut):
            parked.append(ut)
    if not parked:
        return
    print()
    print("=== Parked Realm user threads ({}) ===".format(len(parked)))
    saved = _current_regs()
    if saved["thread_num"] is None:
        return
    # If we're called while the user is already inside a realm-thread
    # impersonation, the currently-loaded registers ARE the impersonated
    # state, not the real host pthread state.  An inferior call during the
    # parked-thread bt needs to swap back to the *actual* safe host
    # registers, which live at the bottom of the impersonate stack.
    if _impersonate_stack:
        safe_host = _impersonate_stack[0]["host_regs"]
    else:
        safe_host = saved
    # On a read-only target (a core file) the register writes fail, so
    # once one does, or if a view is already active, show the remaining
    # parked stacks through views instead.
    saved_view = _view
    use_view = saved_view is not None
    wrote_regs = False
    try:
        for ut in parked:
            try:
                regs = _saved_regs(ut)
            except gdb.error as e:
                print("\nUser thread {:#x}: <cannot read ctx: {}>".format(int(ut), e))
                continue
            print("\nUser thread {:#x}:".format(int(ut)))
            if not use_view:
                parked_tagged = dict(regs, thread_num=saved["thread_num"])
                # Push a synthetic impersonation entry so the inferior_call
                # event handler will swap to safe_host if a pretty-printer,
                # display expression, or conditional breakpoint expression
                # fires a function call while we're walking the parked stack.
                _impersonate_stack.append({
                    "host_regs": safe_host,
                    "parked_regs": parked_tagged,
                    "user_thread_ptr": int(ut),
                })
                try:
                    if _write_regs_or_restore(parked_tagged, saved):
                        wrote_regs = True
                        gdb.execute("bt", to_string=False)
                        continue
                    use_view = True
                except gdb.error as e:
                    print("  <bt failed: {}>".format(e))
                    continue
                finally:
                    _impersonate_stack.pop()
            try:
                view = _make_view(ut)
                start = _not_started(view)
                if start is not None:
                    print("  <not started yet; will begin in {}>".format(start))
                    continue
                _set_view(view)
                gdb.execute("bt", to_string=False)
            except gdb.error as e:
                print("  <bt failed: {}>".format(e))
    finally:
        if wrote_regs:
            try:
                _set_regs(saved)
            except gdb.error as e:
                print("realm-gdb: WARNING: failed to restore registers: {}".format(e))
        if use_view:
            _set_view(saved_view, select=True)


def _looks_like_bt(tokens):
    """Return True if `tokens` is a `thread apply` sub-command that runs bt."""
    # Skip leading -flags ("-ascending", "-c", "-q", "-s", etc.).
    i = 0
    while i < len(tokens) and tokens[i].startswith("-"):
        # `-c <cmd>` takes an argument that IS the command; treat as opaque
        if tokens[i] == "-c" and i + 1 < len(tokens):
            return False
        i += 1
    if i >= len(tokens):
        return False
    return tokens[i] in ("bt", "backtrace", "where")


class ThreadApply(gdb.Command):
    """thread apply <id-list> [-flags] command...

    Realm-aware override: on `thread apply all bt`/`backtrace`/`where`,
    parked Realm user-thread backtraces are appended after the kernel
    threads.  Other invocations behave like the built-in (minimal flag
    support; -ascending/-c are not honored)."""

    def __init__(self):
        super(ThreadApply, self).__init__("thread apply", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        tokens = gdb.string_to_argv(arg)
        if not tokens:
            print("Usage: thread apply <id-list> [-flags] command...")
            return

        spec = tokens[0]
        rest = tokens[1:]
        # strip -flag tokens for command extraction
        cmd_tokens = list(rest)
        while cmd_tokens and cmd_tokens[0].startswith("-"):
            cmd_tokens = cmd_tokens[1:]
        cmd_str = " ".join(cmd_tokens)
        if not cmd_str:
            print("Usage: thread apply <id-list> [-flags] command...")
            return

        inferior = gdb.selected_inferior()
        if spec == "all":
            targets = list(inferior.threads())
        else:
            targets = self._parse_spec(spec, inferior)

        cur = gdb.selected_thread()
        for t in targets:
            try:
                t.switch()
            except gdb.error:
                continue
            lwp = t.ptid[1] if isinstance(t.ptid, tuple) else "?"
            name = " \"{}\"".format(t.name) if t.name else ""
            print("\nThread {} (LWP {}){}:".format(t.num, lwp, name))
            try:
                gdb.execute(cmd_str, from_tty=False)
            except gdb.error as e:
                print("  <{}>".format(e))
        if cur is not None and cur.is_valid():
            cur.switch()

        if spec == "all" and _looks_like_bt(rest):
            _dump_parked_threads()

    def _parse_spec(self, spec, inferior):
        # Minimal: "N", "N-M", and comma-separated combos.
        by_num = {t.num: t for t in inferior.threads()}
        ids = []
        seen = set()
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                try:
                    a, b = part.split("-", 1)
                    rng = range(int(a), int(b) + 1)
                except ValueError:
                    continue
            else:
                try:
                    rng = [int(part)]
                except ValueError:
                    continue
            for n in rng:
                if n in seen:
                    continue
                seen.add(n)
                if n in by_num:
                    ids.append(by_num[n])
                else:
                    print("Thread {}: not found".format(n))
        return ids


class RealmThread(gdb.Command):
    """realm-thread [<ptr> | back]

    With no args, list parked Realm user threads.
    With a UserThread pointer, redirect the currently selected pthread's
    registers to that user thread's saved context so that `bt`, `frame N`,
    `info locals`, and `print VAR` operate on its stack.  On a core file,
    frame #0 becomes a placeholder and the user thread's stack starts at
    frame #1.
    `realm-thread back` restores the previous registers (stack of saves)."""

    def __init__(self):
        super(RealmThread, self).__init__("realm-thread", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        tokens = gdb.string_to_argv(arg)
        if not tokens:
            self._list()
            return
        if tokens[0] in ("back", "restore"):
            self._restore()
            return
        self._impersonate(tokens[0])

    def _list(self):
        found = False
        for ut in _iter_user_threads():
            running = "RUNNING" if bool(ut["running"]) else "parked"
            print("  {:#x}  {}".format(int(ut), running))
            found = True
        if not found:
            print("(no Realm user threads in registry)")

    def _impersonate(self, expr):
        try:
            val = gdb.parse_and_eval(expr)
        except gdb.error as e:
            print("realm-thread: cannot parse {!r}: {}".format(expr, e))
            return
        try:
            ut_type = gdb.lookup_type("Realm::UserThread").pointer()
        except gdb.error as e:
            print("realm-thread: cannot find Realm::UserThread type: {}".format(e))
            return
        ut = val.cast(ut_type)
        try:
            parked = _saved_regs(ut)
        except gdb.error as e:
            print("realm-thread: cannot read ctx: {}".format(e))
            return
        host = _current_regs()
        if host["thread_num"] is None:
            print("realm-thread: no selected thread to impersonate on.")
            return
        # Tag parked regs with the same thread_num so _apply_snapshot writes
        # them to the right thread on re-impersonation after an inferior call.
        parked_with_thread = dict(parked, thread_num=host["thread_num"])
        entry = {
            "host_regs": host,
            "parked_regs": parked_with_thread,
            "user_thread_ptr": int(ut),
        }
        if _view is None:
            _impersonate_stack.append(entry)
            if _write_regs_or_restore(parked_with_thread, host):
                print("realm-thread: now impersonating user thread {:#x}.".format(int(ut)))
                print("Use 'bt', 'frame N', 'info locals', 'print VAR' as normal.")
                print("Run 'realm-thread back' to restore.")
                return
            _impersonate_stack.pop()
        # Registers are read-only (e.g. a core file), or we're already in
        # a view: show the parked stack through a view instead.
        try:
            view = _make_view(ut)
        except gdb.error as e:
            print("realm-thread: cannot read ctx: {}".format(e))
            return
        start = _not_started(view)
        if start is not None:
            print("realm-thread: user thread {:#x} has not started yet; it will "
                  "begin in {}.".format(int(ut), start))
            return
        entry["view"] = view
        _impersonate_stack.append(entry)
        _set_view(view, select=True)
        print("realm-thread: now viewing user thread {:#x}.  Registers can't be "
              "written here,".format(int(ut)))
        print("so frame #0 is a placeholder and the user thread's stack starts "
              "at frame #1.")
        print("Use 'bt', 'frame N', 'info locals', 'print VAR' as normal.")
        print("Run 'realm-thread back' to restore.")

    def _restore(self):
        if not _impersonate_stack:
            print("realm-thread: nothing to restore.")
            return
        # _pop_impersonation restores to the originating thread by number,
        # not to whatever thread is currently selected.
        if not _pop_impersonation():
            return
        print("realm-thread: restored.")


_HELP_BANNER = """\
======================================================================
realm-gdb.py loaded.  Realm user-thread debugging commands:

  thread apply all bt       Built-in 'thread apply all bt' plus an
                            extra section listing backtraces for every
                            parked Realm user thread.  Also recognises
                            'backtrace' and 'where' in place of 'bt'.

  realm-thread              List parked Realm user threads with their
                            UserThread pointers.

  realm-thread <ptr>        Redirect the currently-selected pthread's
                            registers to the given user thread's saved
                            context.  Use 'bt', 'frame N', 'info locals',
                            'print VAR' etc. as usual.  Calls nest --
                            you can switch into another parked thread
                            without restoring first.

  realm-thread back         Pop one level of impersonation, restoring
                            the previous register snapshot.

  realm-help                Reprint this banner.

Safety net:
  * continue, run, signal, thread, detach, quit auto-restore every
    outstanding impersonation before running, so you can't resume the
    inferior (or detach gdb) with clobbered registers.
  * step, stepi, next, nexti, finish, until, advance, jump are REFUSED
    while impersonating -- they would step the host pthread, not the
    parked user thread, which is almost never what you want.  Run
    'realm-thread back' first, or use 'continue' if you just want to
    let the program proceed.
  * Inferior function calls (e.g. 'p foo()', 'call foo()') temporarily
    swap the host pthread's real registers back in for the call's
    duration, then re-impersonate.  Without this, the call would run on
    the parked user thread's small mmap'd stack and likely overflow it.

Core files:
  * Registers can't be written, so both commands use a virtual unwinder
    instead.  Frame #0 is a placeholder ('<parked Realm user thread
    ...>' in 'bt') and the user thread's stack starts at frame #1.

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
    ThreadApply()
    RealmThread()
    RealmHelp()
    _install_safety_hooks()
    _install_view_support()
    print(_HELP_BANNER)
else:
    # Other platforms / architectures: do nothing rather than provide a
    # half-working command.
    pass
