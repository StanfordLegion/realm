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
#   - `realm-thread` lists the user threads.
#   - `realm-thread apply <ptr>|all COMMAND` runs COMMAND (`bt full`,
#     `frame apply level 3 print VAR`, ...) on a parked user thread, or on
#     each of them.
#   - `realm-thread <ptr>` selects a parked user thread until `realm-thread
#     back`.  Meanwhile `bt`, `frame N`, `print VAR`, `info locals`, `set
#     var`, `break` and the like show it, and commands that would act on the
#     selected thread as it really is (`step`, `finish`, `return`, ...) are
#     refused; `continue` and `thread N` end the selection.  (In a live process, only in a console gdb session:
#     elsewhere `rt COMMAND` shows it.)
#
# How a parked user thread is shown depends on the target (see "Register
# windows" and "Core-file views" below):
#
#   - In a live process (all-stop mode), its saved context is written into a
#     stopped thread's registers for the duration of one command, and that
#     thread's own registers are put back before the command returns.  So gdb
#     unwinds, prints and evaluates on the parked stack natively, function
#     calls (a pretty-printer's included) work, and between commands the
#     program is exactly as it was.
#   - In a core file, which can't take register writes, a Python unwinder
#     makes the selected thread's innermost frame unwind into the parked
#     thread's saved context instead, and backtraces draw that frame as a
#     `<parked Realm user thread ...>` marker line.
#
# Linux/FreeBSD on x86-64 or aarch64.  On other targets the script loads
# as a no-op.

import gdb
import gdb.unwinder
import os
import platform
import re
import struct
import sys
import traceback
from gdb.FrameDecorator import FrameDecorator

# 64-bit unsigned mask used to coerce gdb's signed-int reads into the
# unsigned address space the kernel saved.
_U64 = (1 << 64) - 1


def _debug_exc():
    """Show an exception that realm-gdb swallowed (set REALM_GDB_DEBUG=1 in
    gdb's environment)."""
    if os.environ.get("REALM_GDB_DEBUG"):
        traceback.print_exc()


def _parameter(name, default=None):
    try:
        return gdb.parameter(name)
    except (gdb.error, RuntimeError):
        return default


def _has_parameter(name):
    try:
        gdb.parameter(name)
        return True
    except (gdb.error, RuntimeError):
        return False


# An MI stream record (console, target or log output) on a line of its own.
_MI_STREAM_RECORD = re.compile(r'[~@&]"((?:[^"\\]|\\.)*)"\n?$', re.S)
_C_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "a": "\a",
              "b": "\b", "e": "\x1b"}


# realm-gdb's own commands in progress (see _internal).
_internal_depth = 0


def _internal(command):
    """Run gdb command `command` for realm-gdb itself, where its hooks
    leave it alone (`return` to pop a dummy frame in a window, say)."""
    global _internal_depth
    _internal_depth += 1
    try:
        return gdb.execute(command, to_string=True)
    finally:
        _internal_depth -= 1


def _execute(command):
    """gdb.execute(command, to_string=True): the command's output, as the
    CLI would print it.  (Under MI, gdb 18 returns some of it, line by line,
    as MI stream records: ~"text\\n".)"""
    out = gdb.execute(command, to_string=True)
    if not re.search(r'(?:^|\n)[~@&]"', out):
        return out
    return "".join(_mi_unquote(line) for line in out.splitlines(True))


def _mi_unquote(line):
    """An MI stream record's text (any other line as it is)."""
    m = _MI_STREAM_RECORD.match(line)
    if m is None:
        return line
    text = bytearray()
    for piece in re.findall(r"\\[0-7]{1,3}|\\.|[^\\]+", m.group(1), re.S):
        if piece[0] != "\\":
            text += piece.encode("utf-8", "surrogateescape")
        elif piece[1] in "01234567":
            text.append(int(piece[1:], 8) & 0xff)
        else:
            text += _C_ESCAPES.get(piece[1], piece[1]).encode("utf-8")
    return text.decode("utf-8", "replace")


# --------------------------------------------------------------------------
# Architecture abstraction
# --------------------------------------------------------------------------
#
# Each supported architecture reads a UserThread::ctx (a ucontext_t, laid out
# differently per CPU and OS):
#
#   context_regs(ut) -> ({register: value}, [note]): what to write into a
#     stopped thread to make it look like the parked user thread `ut`: PC,
#     SP, every callee-saved register (floating-point ones included), the
#     other registers getcontext() saves, and the return-value register as
#     swapcontext() returns it (0).  The registers the context doesn't hold
#     are dead at that PC, which is just after a call.  A value is an int,
#     or for a vector register (lo, hi) 64-bit halves.  A note says what
#     couldn't be read, and so shows the host thread's value instead.
#   unwind_regs_from_ctx(ut) -> {register: int}: PC, SP and the callee-saved
#     general registers, for the core-file unwinder.
#
# fallback_regs are the registers to keep and put back around a register
# window with gdbs that can't list the "save" register group (gdb < 10):
# the group's members, those of them this gdb has.  That is those a window
# writes, those gdb itself writes meanwhile (a function call sets orig_rax
# on x86-64 Linux, which decides whether the kernel restarts an interrupted
# system call, until it returns), and any the user changes meanwhile (`set
# var $xmm1 = ...`), which would otherwise stay changed on the thread.

class _Arch(object):
    pc_reg = None
    sp_reg = None
    fallback_regs = ()
    # Registers that a parked user thread shares with whichever thread shows
    # it, which gdb may need to unwind its frames; a core-file view passes
    # them through from the host frame (those that exist).
    thread_regs = ()

    @staticmethod
    def matches():
        raise NotImplementedError

    @staticmethod
    def context_regs(ut):
        raise NotImplementedError

    @staticmethod
    def unwind_regs_from_ctx(ut):
        raise NotImplementedError


_X86_64_GP_REGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
                   "rip", "eflags")
# The rest of gdb's "save" group on x86-64 (raw registers only: ymm0h is
# ymm0's upper half, zmm0h zmm0's).
_X86_64_OTHER_REGS = (
    ("cs", "ss", "ds", "es", "fs", "gs")
    + tuple("st{}".format(i) for i in range(8))
    + ("fctrl", "fstat", "ftag", "fiseg", "fioff", "foseg", "fooff", "fop")
    + tuple("xmm{}".format(i) for i in range(32)) + ("mxcsr",)
    + tuple("ymm{}h".format(i) for i in range(32))
    + tuple("k{}".format(i) for i in range(8))
    + tuple("zmm{}h".format(i) for i in range(32)) + ("pkru", "fs_base", "gs_base"))


class _X86_64Linux(_Arch):
    pc_reg = "rip"
    sp_reg = "rsp"
    fallback_regs = _X86_64_GP_REGS + _X86_64_OTHER_REGS + ("orig_rax",)
    # glibc x86-64 ucontext gregs[] indices (see /usr/include/sys/ucontext.h).
    # getcontext()/swapcontext() save the callee-saved registers, the
    # argument registers, and RIP/RSP as of the return from the call.  (No
    # SSE register is callee-saved in the SysV ABI.)
    _CTX_GREGS = (("r8", 0), ("r9", 1), ("r12", 4), ("r13", 5), ("r14", 6),
                  ("r15", 7), ("rdi", 8), ("rsi", 9), ("rbp", 10), ("rbx", 11),
                  ("rdx", 12), ("rcx", 14), ("rsp", 15), ("rip", 16))
    _UNWIND_GREGS = (("r12", 4), ("r13", 5), ("r14", 6), ("r15", 7),
                     ("rbp", 10), ("rbx", 11), ("rsp", 15), ("rip", 16))

    @staticmethod
    def matches():
        return (platform.system() == "Linux"
                and platform.machine() in ("x86_64", "amd64"))

    @staticmethod
    def context_regs(ut):
        gregs = ut["ctx"]["uc_mcontext"]["gregs"]
        regs = {name: int(gregs[i]) & _U64 for name, i in _X86_64Linux._CTX_GREGS}
        regs["rax"] = 0
        return regs, []

    @staticmethod
    def unwind_regs_from_ctx(ut):
        gregs = ut["ctx"]["uc_mcontext"]["gregs"]
        return {name: int(gregs[i]) & _U64
                for name, i in _X86_64Linux._UNWIND_GREGS}


class _X86_64FreeBSD(_Arch):
    pc_reg = "rip"
    sp_reg = "rsp"
    fallback_regs = _X86_64_GP_REGS + _X86_64_OTHER_REGS
    # FreeBSD mcontext_t (sys/x86/include/ucontext.h) uses named fields
    # (mc_rip, mc_rsp, ...), not a gregs[] array.
    _CTX = ("r8", "r9", "r12", "r13", "r14", "r15", "rdi", "rsi", "rbp",
            "rbx", "rdx", "rcx", "rsp", "rip")

    @staticmethod
    def matches():
        return (platform.system() == "FreeBSD"
                and platform.machine() in ("x86_64", "amd64"))

    @staticmethod
    def context_regs(ut):
        mc = ut["ctx"]["uc_mcontext"]
        regs = {name: int(mc["mc_" + name]) & _U64 for name in _X86_64FreeBSD._CTX}
        regs["rax"] = 0
        return regs, []

    @staticmethod
    def unwind_regs_from_ctx(ut):
        mc = ut["ctx"]["uc_mcontext"]
        return {name: int(mc["mc_" + name]) & _U64
                for name in ("r12", "r13", "r14", "r15",
                             "rbp", "rbx", "rsp", "rip")}


# gdb needs the pointer-authentication masks to unwind return addresses
# signed with -mbranch-protection=pac-ret; tpidr, fpsr/fpcr and vg are
# per-thread too.  (Linux on aarch64 is tested in CI, FreeBSD on aarch64
# not yet.)
_AARCH64_THREAD_REGS = ("pauth_dmask", "pauth_cmask", "tpidr", "fpsr", "fpcr", "vg")
_AARCH64_GP_REGS = tuple("x{}".format(i) for i in range(31)) + ("sp", "pc", "cpsr")
# d8..d15 (the low halves of v8..v15) are callee-saved.  With SVE, v8..v15
# are the low 128 bits of z8..z15, and writing v8 clears the rest of z8, so
# keep z8..z15 too.
_AARCH64_FP_SAVED = tuple("v{}".format(i) for i in range(8, 16))
_AARCH64_FALLBACK = (_AARCH64_GP_REGS + tuple("v{}".format(i) for i in range(32))
                     + tuple("z{}".format(i) for i in range(32))
                     + tuple("p{}".format(i) for i in range(16)) + ("ffr", "fpsr", "fpcr"))


def _read_bytes(addr, size):
    return bytes(gdb.selected_inferior().read_memory(addr, size))


def _vregs_note(why):
    return ("its floating-point registers d8-d15 ({}), so a value kept in one "
            "of them shows the selected thread's".format(why))


class _AArch64Linux(_Arch):
    pc_reg = "pc"
    sp_reg = "sp"
    fallback_regs = _AARCH64_FALLBACK
    thread_regs = _AARCH64_THREAD_REGS
    # Linux/glibc aarch64 mcontext_t is `struct sigcontext` (see
    # linux/arch/arm64/include/uapi/asm/sigcontext.h):
    #     __u64 fault_address;
    #     __u64 regs[31];     // X0..X30
    #     __u64 sp;
    #     __u64 pc;
    #     __u64 pstate;
    #     __u8 __reserved[4096];
    # Saved by glibc's swapcontext(): sp holds the SP, and pc where the
    # context resumes: the return address (the LR swapcontext was called
    # with), or (glibc >= 2.35 at least) a stub in swapcontext that returns
    # to it from x1, where it is saved instead.  __reserved starts with a
    # struct fpsimd_context: magic FPSIMD_MAGIC, size, fpsr, fpcr, then
    # vregs[32] (128 bits each, at offset 16), of which swapcontext() saves
    # v8..v15.
    _FPSIMD_MAGIC = 0x46508001

    @staticmethod
    def matches():
        return (platform.system() == "Linux"
                and platform.machine() in ("aarch64", "arm64"))

    @staticmethod
    def context_regs(ut):
        regs = _AArch64Linux.unwind_regs_from_ctx(ut)
        regs["x0"] = 0
        notes = []
        try:
            buf = _read_bytes(int(ut["ctx"]["uc_mcontext"]["__reserved"].address),
                              16 + 16 * 16)
            magic, size = struct.unpack_from("<II", buf, 0)
            if magic != _AArch64Linux._FPSIMD_MAGIC or size < 16 + 16 * 32:
                notes.append(_vregs_note("its saved context has no FP/SIMD record"))
            else:
                for n in range(8, 16):
                    regs["v{}".format(n)] = struct.unpack_from("<QQ", buf, 16 + 16 * n)
        except (gdb.error, gdb.MemoryError, ValueError, struct.error) as e:
            notes.append(_vregs_note("they can't be read: {}".format(e)))
        return regs, notes

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
                words = struct.unpack("<4I", _read_bytes(pc, 16))
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
    fallback_regs = _AARCH64_FALLBACK
    thread_regs = _AARCH64_THREAD_REGS
    # FreeBSD aarch64 mcontext_t (sys/arm64/include/ucontext.h):
    #     struct gpregs {
    #         __register_t  gp_x[30];   // X0..X29
    #         __register_t  gp_lr;      // X30
    #         __register_t  gp_sp;
    #         __register_t  gp_elr;     // exception link register == PC
    #         ...
    #     };
    #     struct fpregs {
    #         __uint128_t   fp_q[32];
    #         ...
    #     };
    #     typedef struct __mcontext {
    #         struct gpregs  mc_gpregs;
    #         struct fpregs  mc_fpregs;
    #         int            mc_flags;  // _MC_FP_VALID (1) if mc_fpregs is
    #         ...
    #     } mcontext_t;
    _MC_FP_VALID = 0x1

    @staticmethod
    def matches():
        return (platform.system() == "FreeBSD"
                and platform.machine() in ("aarch64", "arm64"))

    @staticmethod
    def context_regs(ut):
        regs = _AArch64FreeBSD.unwind_regs_from_ctx(ut)
        regs["x0"] = 0
        notes = []
        try:
            mc = ut["ctx"]["uc_mcontext"]
            if not int(mc["mc_flags"]) & _AArch64FreeBSD._MC_FP_VALID:
                notes.append(_vregs_note("its saved context has none"))
            else:
                buf = _read_bytes(int(mc["mc_fpregs"]["fp_q"].address), 16 * 16)
                for n in range(8, 16):
                    regs["v{}".format(n)] = struct.unpack_from("<QQ", buf, 16 * n)
        except (gdb.error, gdb.MemoryError, ValueError, struct.error) as e:
            notes.append(_vregs_note("they can't be read: {}".format(e)))
        return regs, notes

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


def _stack_of(ut):
    """(base, end) of a user thread's stack.  Raises gdb.error."""
    base = int(ut["stack_base"]) & _U64
    return base, base + (int(ut["stack_size"]) & _U64)


def _host_sps():
    """{thread number: SP} for every stopped thread of the selected
    inferior, and whether any of its threads is running."""
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
        base, end = _stack_of(ut)
    except gdb.error:
        return None
    for num, sp in sorted(sps.items()):
        if base <= sp < end:
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


def _not_started(pc):
    """A user thread that has never run still holds makecontext()'s
    context, whose PC is its entry point rather than a return address:
    there is no stack to show.  Returns the entry function's name for such
    a thread, else None."""
    try:
        block = gdb.block_for_pc(pc)
    except RuntimeError:
        return None
    while block is not None and block.function is None:
        block = block.superblock
    if block is None or int(block.function.value().address) != pc:
        return None
    return block.function.print_name


# --------------------------------------------------------------------------
# Frames, threads and selection
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
    """Restore the selected thread and frame on exit (with its core-file
    view, if any, shown again).  Some gdbs (10, on core files at least)
    lose the selected frame when Inferior.threads() refreshes the thread
    list, and commands that switch threads without unwinding them leave a
    view unarmed on gdb < 11."""

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


# --------------------------------------------------------------------------
# Where parked user threads can be shown
# --------------------------------------------------------------------------
#
# _how_shown() decides, for the selected inferior: "window" (a live process
# in all-stop mode: register windows), "view" (a core file: core-file views),
# or a _NotShown saying why they can't be shown and how to see them.  Parked
# threads are never left out silently.

class _NotShown(object):
    def __init__(self, why, advice):
        self.why = why
        self.advice = advice

    def sentence(self):
        return "{}{}.  To see them, {}.".format(self.why[0].upper(), self.why[1:],
                                                self.advice)


def _target_stack():
    try:
        return _execute("maint print target-stack")
    except gdb.error:
        return ""


def _is_core():
    """Whether the selected inferior is a core file.  (It has a pid too.
    A probe write wouldn't tell: gdb skips writing an unchanged value.)"""
    return re.search(r"^\s*- core \(", _target_stack(), re.M) is not None


def _recording():
    try:
        if hasattr(gdb, "current_recording") and gdb.current_recording() is not None:
            return True
    except gdb.error:
        pass
    return re.search(r"^\s*- record", _target_stack(), re.M) is not None


def _how_shown():
    if not _has_target():
        return _NotShown("there is no process or core file", "start the program")
    if _is_core():
        return "view"
    if _parameter("non-stop", False):
        return _NotShown(
            "the program is in non-stop mode, where realm-gdb can't show them: it "
            "would have to put a parked user thread's registers into a stopped "
            "thread while other threads may run and resume the parked one",
            "write a core file ('gcore FILE') and load it in another gdb, or debug "
            "the program with 'set non-stop off'")
    if any(t.is_valid() and not t.is_stopped() for t in gdb.selected_inferior().threads()):
        return _NotShown("the program is running", "stop it ('interrupt') and rerun "
                         "this command")
    if _recording():
        return _NotShown(
            "the program is being recorded (record/replay), and showing them "
            "changes a thread's registers, which would spoil the execution log",
            "stop recording ('record stop') or write a core file ('gcore FILE') "
            "and load it in another gdb")
    return "window"


def _parked_problem(ut, sps):
    """Why parked user thread `ut` can't be shown (a string), or None.  Its
    saved stack pointer must be inside its own stack: a function call made
    while it is shown in a register window runs just below it."""
    try:
        if not _is_parked(ut):
            if not _state_readable(ut):
                bool(ut["running"])  # (raises)
            host = _host_thread_of(ut)
            return "it is running{}".format(
                " on thread {}".format(host.num) if host is not None else "")
        executing = _executing_on(ut, sps)
        if executing is not None:
            return "running on thread {}: it is switching contexts".format(executing)
        regs = _arch.unwind_regs_from_ctx(ut)
        base, end = _stack_of(ut)
    except gdb.error as e:
        return "cannot read it: {}".format(e)
    start = _not_started(regs[_arch.pc_reg])
    if start is not None:
        return "not started yet; will begin in {}".format(start)
    sp = regs[_arch.sp_reg]
    if not base < sp <= end:
        return "its saved stack pointer {:#x} is outside its stack ({:#x}-{:#x})".format(
            sp, base, end)
    return None


# --------------------------------------------------------------------------
# Register windows (live processes)
# --------------------------------------------------------------------------
#
# A register window shows a parked user thread on a stopped thread of a live
# process (the "host") for the duration of one command: the host's registers
# are kept (the "save" register group: everything gdb itself keeps around an
# inferior call, floating-point and vector registers included; before gdb 10,
# _Arch.fallback_regs), the parked thread's saved context is written in, the
# command runs, and every register that differs from what was kept is written
# back before the command returns.  gdb then unwinds, prints and evaluates on
# the parked stack exactly as it would on a running thread's: frame #0 is the
# parked thread's own, and pretty-printers that call functions work.  Its
# output is collected and printed once the host's registers are back.
#
# Nothing may let the program run, or let go of it, while a window is open,
# or the host would run the parked thread's code, on its stack, with Realm
# none the wiser:
#
#   - Every CLI command that would (`continue`, `step`, `detach`, `quit`,
#     `kill`, `run`, `interpreter-exec`, ...) is refused, by a hook-COMMAND
#     (see "Refusing commands").  Commands inside a window can't run windows
#     of their own either.
#   - Function calls (a pretty-printer's, or the user's) run on the host
#     with the parked thread's registers, so on the parked thread's stack,
#     just below its saved SP: gdb takes the call's SP from frame #0 before it
#     tells Python about the call.  That memory is unused while the thread is
#     parked.  But Realm can resume a parked thread on any thread
#     (user_switch() sets host_pthread on resuming), so for the whole window
#     every call holds the other threads (scheduler-locking), is unwound if it
#     gets a signal (unwindonsignal: gdb reads it only once the call is over,
#     so it must stay on until then; otherwise a faulting call would be left
#     on the parked stack, with its signal still to come), or throws past
#     std::terminate, or (gdb >= 15) runs longer than _CALL_TIMEOUT seconds.
#     User breakpoints are disabled while a call runs, so that it can't stop
#     at one.  The user's settings are put back afterwards.
#   - Output isn't paged, and queries are answered (`confirm off`), inside a
#     window: gdb's pager and queries wait for input in a nested event loop,
#     where another UI (a second MI `new-ui`, say) could run a command.
#   - SIGTERM and SIGHUP to gdb are held off until the window is closed (gdb
#     acts on them by killing or detaching from the program and exiting, in
#     the middle of a command if need be).  Ctrl-C is not: an interrupted
#     command closes its window on the way out.
#   - gdb.events.gdb_exiting (gdb >= 12) closes any window left open.
#
# What can't be covered: gdb itself dying (kill -9, a crash) with a window
# open leaves an attached process running the parked thread's context on the
# host, and so corrupts it.  A process gdb started dies with gdb anyway.
# Windows are short (well under a second each), so this needs gdb to die in
# the middle of one.  And a resume that bypasses CLI commands, from Python's
# gdb.execute_mi() inside a window, can't be refused: gdb.events.cont comes
# too late to put the registers back (gdb's proceed() holds on to the
# thread's registers across it, and writing one frees them all).  Then
# realm-gdb says the program may crash.

# The open window, if any; windows whose host's registers could not be put
# back (the program must not run until they are).  (Kept if the script is
# loaded again: a window a failed plain command left open is still open.)
_window = globals().get("_window")
_unrestored = globals().get("_unrestored", [])

# Inferior calls in progress (in or out of a window); the user breakpoints
# disabled for the one in a window.
_infcall_depth = 0
_disabled_breakpoints = []
# The numbers of the catchpoints disabled for it, on gdbs whose Python can't
# see them (see _hidden_catchpoints).
_disabled_catchpoints = []

# Seconds a call made in a window may run (gdb >= 15), after which gdb
# unwinds it.
_CALL_TIMEOUT = 10

# Free stack below a parked thread's saved SP under which function calls are
# refused in its window (gdb >= 9; older gdbs don't show it).  Realm's stacks
# have no guard page below them, so a call that overran the stack would
# write over whatever is mapped there.
_CALL_HEADROOM = 64 << 10

_can_refuse_calls = _has_parameter("may-call-functions")


def _snapshot_names(frame):
    try:
        names = tuple(r.name for r in frame.architecture().registers("save"))
        if names:
            return names
    except Exception:
        pass
    return _arch.fallback_regs


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


_SCALAR_REG_TYPES = (gdb.TYPE_CODE_INT, gdb.TYPE_CODE_PTR, gdb.TYPE_CODE_FLAGS,
                     getattr(gdb, "TYPE_CODE_BOOL", gdb.TYPE_CODE_INT))


def _raw(value):
    """A register's contents, as an int or bytes (None if unavailable)."""
    try:
        if value.type.code in _SCALAR_REG_TYPES and value.type.sizeof <= 8:
            return int(value) & _U64
        if hasattr(value, "bytes"):  # gdb >= 15
            return bytes(value.bytes)
        size = value.type.sizeof
        vec = value.cast(gdb.lookup_type("unsigned char").vector(size - 1))
        return bytes(int(vec[i]) & 0xff for i in range(size))
    except Exception:
        return None


def _int_writable(value):
    """Whether a register's value can be written back as an integer."""
    return value.type.code in _SCALAR_REG_TYPES and value.type.sizeof <= 8


def _keep_in_gdb(saved):
    """Without gdb.set_convenience_variable (gdb < 8.3): keep a copy of each
    register in `saved` that can't be written back as an integer (vector
    and x87 registers) in a gdb convenience variable, for _write_reg."""
    if hasattr(gdb, "set_convenience_variable"):
        return
    gdb.newest_frame().select()
    for name, value in saved.items():
        if not _int_writable(value):
            gdb.execute("set var $_realm_gdb_saved_{0} = ${0}".format(name),
                        to_string=True)


def _write_reg(name, value):
    """Set the selected thread's register `name` to `value`: a gdb.Value of
    its type (as read), an int, or (lo, hi) 64-bit halves of an aarch64
    vector register.  (A register is written in the selected frame: in an
    outer frame that would be the frame's saved copy, on the stack.)"""
    gdb.newest_frame().select()
    if isinstance(value, tuple):
        for i, half in enumerate(value):
            gdb.execute("set var ${}.d.u[{}] = {:#x}".format(name, i, half),
                        to_string=True)
    elif isinstance(value, gdb.Value) and hasattr(gdb, "set_convenience_variable"):
        gdb.set_convenience_variable("_realm_gdb_reg", value)
        gdb.execute("set var ${} = $_realm_gdb_reg".format(name), to_string=True)
    elif isinstance(value, gdb.Value) and not _int_writable(value):
        # (A window's own copy: see _keep_in_gdb.)
        gdb.execute("set var ${0} = $_realm_gdb_saved_{0}".format(name), to_string=True)
    else:
        gdb.execute("set var ${} = {:#x}".format(name, int(value) & _U64),
                    to_string=True)


def _raw_regs(frame):
    """{name: _raw(value)} of `frame`'s registers that a window keeps."""
    return {n: _raw(v) for n, v in _read_regs(frame, _snapshot_names(frame)).items()}


def _put_back(saved):
    """Write every register in `saved` that differs back into the selected
    thread, until they all match.  Returns what still differs ([] when
    done).  An interruption (Ctrl-C) is retried, so that a half-restored
    thread is never left."""
    left = list(saved)
    for _ in range(10):
        try:
            frame = gdb.newest_frame()
            left = [n for n, v in saved.items()
                    if _raw(frame.read_register(n)) != _raw(v)]
            if not left:
                return []
            for name in left:
                _write_reg(name, saved[name])
        except KeyboardInterrupt:
            continue
        except gdb.error:
            _debug_exc()
    return left


def _setting_words(value):
    if value is True:
        return "on"
    if value is False:
        return "off"
    if value is None:
        return "unlimited"
    return str(value)


def _set(name, value):
    """`set NAME VALUE`, retried if interrupted (Ctrl-C)."""
    for _ in range(10):
        try:
            gdb.execute("set {} {}".format(name, value), to_string=True)
            return
        except KeyboardInterrupt:
            continue


class _CallSettings(object):
    """The settings function calls made in a window need (see "Register
    windows"), and the user's, put back afterwards.  The user's
    may-call-functions is kept too, for set_calls.  With paging=False, gdb's
    pager is off too (in a window whose command prints as it goes)."""

    def __init__(self, paging=True):
        settings = [("scheduler-locking", "on"), ("unwindonsignal", "on"),
                    ("unwind-on-terminating-exception", "on"), ("confirm", "off")]
        if not paging:
            settings.append(("pagination", "off"))
        for name in ("direct-call-timeout", "indirect-call-timeout"):
            user = _parameter(name)
            if user is None or user > _CALL_TIMEOUT:  # (None: unlimited)
                settings.append((name, str(_CALL_TIMEOUT)))
        settings.append(("unwind-on-timeout", "on"))
        self.settings = settings
        self.user_calls = _parameter("may-call-functions")
        self.calls = self.user_calls

    def __enter__(self):
        self.saved = []
        for name, value in self.settings:
            if not _has_parameter(name):
                continue
            old = _parameter(name)
            if _setting_words(old) == value:
                continue
            _set(name, value)
            self.saved.append((name, old))
        return self

    def set_calls(self, allowed):
        """Allow function calls (as the user has it), or refuse them."""
        want = self.user_calls if allowed else False
        if self.user_calls is not None and want != self.calls:
            _set("may-call-functions", _setting_words(want))
            self.calls = want

    def for_call(self):
        """Just before a function call in the window: put back what the call
        needs, if a command in the window changed it since (`with
        scheduler-locking off -- print f()`, a user-defined command that
        sets it, Python).  Returns what was put back, as "NAME VALUE"."""
        put = []
        for name, value in self.settings:
            if (name in ("confirm", "pagination") or name.endswith("call-timeout")
                    or not _has_parameter(name)):
                continue
            if _setting_words(_parameter(name)) != value:
                _set(name, value)
                put.append("{} {}".format(name, value))
        for name in ("direct-call-timeout", "indirect-call-timeout"):
            if not _has_parameter(name):
                continue
            now = _parameter(name)
            if now is None or now > _CALL_TIMEOUT:  # (None: unlimited)
                _set(name, str(_CALL_TIMEOUT))
                put.append("{} {}".format(name, _CALL_TIMEOUT))
        return put

    def calls_refused(self):
        """Whether the window refuses function calls, but a command in it
        turned may-call-functions back on."""
        return self.calls is False and bool(_parameter("may-call-functions"))

    def __exit__(self, *exc):
        try:
            self.set_calls(True)
        finally:
            for name, old in reversed(self.saved):
                try:
                    _set(name, _setting_words(old))
                except gdb.error:
                    _debug_exc()
        return False


class _HeldSignals(object):
    """Hold off SIGTERM and SIGHUP to gdb (see "Register windows"), and
    deliver one that came meanwhile once the window is closed.  gdb's
    handler is swapped for one that just notes the signal (installed by
    Python, which runs it on gdb's main thread), and put back as it was
    (with libc's sigaction(), which Python's signal module can't do for a
    handler it didn't install).  Without ctypes, the signals are only
    blocked on gdb's main thread, which before gdb 10 is its only thread."""

    _SIGNALS = ("SIGTERM", "SIGHUP")

    def __enter__(self):
        import signal
        self.came = []
        self.swapped = []
        self.blocked = None
        sigs = [getattr(signal, s) for s in self._SIGNALS if hasattr(signal, s)]
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            self._sigaction = libc.sigaction
            self._sigaction.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
            for sig in sigs:
                # (struct sigaction is well under 256 bytes on every libc.)
                old = ctypes.create_string_buffer(256)
                if self._sigaction(sig, None, old) != 0:
                    continue
                signal.signal(sig, self._note)
                self.swapped.append((sig, old))
        except Exception:
            _debug_exc()
            self._put_back()
            try:
                self.blocked = signal.pthread_sigmask(signal.SIG_BLOCK, sigs)
            except Exception:
                self.blocked = None
        return self

    def _note(self, sig, frame):
        self.came.append(sig)

    def _put_back(self):
        while self.swapped:
            sig, old = self.swapped.pop()
            try:
                self._sigaction(sig, old, None)
            except Exception:
                _debug_exc()

    def __exit__(self, *exc):
        import signal
        self._put_back()
        if self.blocked is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, self.blocked)
        for sig in sorted(set(self.came)):
            os.kill(os.getpid(), sig)
        return False


class _Window(object):
    """A register window on the selected thread (see "Register windows"):
    show(ut, regs) puts a parked thread's context in place, and leaving the
    `with` block puts the thread's own registers back.  One window may show
    several parked threads in turn."""

    def __init__(self, via=None):
        # What opened it, if not `realm-thread apply` or `thread apply all
        # bt`: "rt", or the plain command it was opened for (see "Plain
        # commands on a selected parked thread").
        self.via = via
        self.resumed = False
        self.calls = 0
        self.breakpoints_skipped = False
        # Settings its function calls needed put back ("NAME VALUE"), and
        # whether a call was refused after all (see _on_inferior_call).
        self.settings_put_back = []
        self.call_refused = False
        self.refusals = []
        # Whether a call made in it was left stopped (None: not looked yet).
        self.abandoned = None
        self.showing = None

    def __enter__(self):
        global _window
        if _window is not None:
            raise gdb.GdbError("realm-gdb: can't show a parked Realm user thread "
                               "from inside 'realm-thread apply' or 'thread apply "
                               "all bt'.")
        self.thread = gdb.selected_thread()
        self.thread_num = self.thread.num
        self.selection = _KeepSelection().__enter__()
        frame = gdb.newest_frame()
        self.saved = _read_regs(frame, _snapshot_names(frame))
        self.signals = _HeldSignals().__enter__()
        try:
            self.settings = _CallSettings(paging=self.via in (None, "rt")).__enter__()
        except BaseException:
            self.signals.__exit__(None, None, None)
            raise
        _window = self
        try:
            _keep_in_gdb(self.saved)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def show(self, ut, regs, calls=True):
        """Put user thread `ut`'s context `regs` in place (see
        context_regs), with function calls refused unless `calls`."""
        self.showing = int(ut)
        self.settings.set_calls(calls)
        try:
            for name, value in sorted(regs.items()):
                _write_reg(name, value)
        except BaseException:
            self.showing = None
            raise
        gdb.newest_frame().select()

    def __exit__(self, *exc):
        global _window
        try:
            self.close()
        finally:
            _window = None
            try:
                self.settings.__exit__(None, None, None)
            finally:
                self.signals.__exit__(None, None, None)
        return False

    def close(self):
        """Put the host's own registers back, and select what was selected.
        (Ctrl-C can't stop it before they are back: it is held till then.)"""
        thread = self.thread
        try:
            if not thread.is_valid():
                return  # (the program has gone)
        except RuntimeError:
            return
        if self.resumed or thread.is_running():
            print("realm-gdb: WARNING: the program ran while thread {} had user thread "
                  "{:#x}'s registers (something resumed it without a gdb command, "
                  "e.g. Python's gdb.execute_mi()): it may crash.".format(
                      self.thread_num, self.showing or 0))
            return
        interrupted = False
        left = list(self.saved)
        for _ in range(10):
            try:
                if gdb.selected_thread() != thread:
                    thread.switch()
                if self.abandoned is None:
                    self._pop_abandoned_call()
                left = _put_back(self.saved)
                if left and self not in _unrestored:
                    _unrestored.append(self)
                break
            except KeyboardInterrupt:
                interrupted = True
        if left:
            print("realm-gdb: WARNING: could not put back thread {}'s own registers "
                  "({}).  Don't let the program run; run 'realm-thread back' to try "
                  "again.".format(self.thread_num, ", ".join("$" + n for n in left)))
        if hasattr(gdb, "set_convenience_variable"):
            gdb.set_convenience_variable("_realm_gdb_reg", None)
        self.selection.__exit__(None, None, None)
        if self.abandoned and not self.call_refused:  # (Those didn't run.)
            print("realm-gdb: WARNING: a function call made on thread {} while it "
                  "showed user thread {:#x} stopped (at a catchpoint?), and has been "
                  "abandoned: the program may be left inconsistent (a lock the "
                  "function held, say).".format(self.thread_num, self.showing or 0))
        if interrupted:
            raise KeyboardInterrupt

    def _pop_abandoned_call(self):
        """If a function call made in the window is stopped on the host (it
        shouldn't be: see "Register windows"), pop its dummy frame, which
        comes before the parked thread's frames: that puts back gdb's state
        of the host from before the call (otherwise, resumed, the host would
        stop again at once where it had stopped, or get a signal the call
        stopped with).  Sets self.abandoned."""
        self.abandoned = False
        if not self.calls:
            return
        try:
            frame = gdb.newest_frame()
            for _ in range(64):
                if frame is None or frame.type() == gdb.DUMMY_FRAME:
                    break
                frame = frame.older()
        except gdb.error:
            frame = None
        if frame is None or frame.type() != gdb.DUMMY_FRAME:
            return
        self.abandoned = True
        try:
            frame.select()
            _internal("return")
        except gdb.error:
            _debug_exc()


def _retry_unrestored():
    """Try again to put back the registers of hosts whose windows couldn't
    (see _Window.close).  Returns whether none are left."""
    for window in list(_unrestored):
        try:
            thread = window.thread
            if not thread.is_valid():
                _unrestored.remove(window)
                continue
            with _KeepSelection():
                thread.switch()
                if not _put_back(window.saved):
                    _unrestored.remove(window)
                    print("realm-gdb: put back thread {}'s own registers.".format(
                        window.thread_num))
        except Exception:
            _debug_exc()
    return not _unrestored


def _breakpoints():
    try:
        return gdb.breakpoints() or ()
    except Exception:
        return ()


def _on_inferior_call(event):
    """A call made in a window can't stop at a user breakpoint (see
    "Register windows"): disable them while it runs.  And it gets the
    settings it needs, whatever a command in the window changed."""
    global _infcall_depth
    if isinstance(event, getattr(gdb, "InferiorCallPreEvent", ())):
        _infcall_depth += 1
        if _window is not None and _infcall_depth == 1:
            _window.calls += 1
            for bp in _breakpoints():
                try:
                    if bp.is_valid() and bp.enabled:
                        bp.enabled = False
                        _disabled_breakpoints.append(bp)
                except Exception:
                    _debug_exc()
            for num in _hidden_catchpoints():
                try:
                    gdb.execute("disable " + num, to_string=True)
                    _disabled_catchpoints.append(num)
                except gdb.error:
                    _debug_exc()
            if _disabled_breakpoints or _disabled_catchpoints:
                _window.breakpoints_skipped = True
        if _window is not None:
            _for_call(_window, event)
    elif isinstance(event, getattr(gdb, "InferiorCallPostEvent", ())):
        _infcall_depth = max(0, _infcall_depth - 1)
        if _infcall_depth == 0:
            _enable_breakpoints()
            if _window is not None:
                _end_refusals(_window)


def _for_call(window, event):
    """Just before a function call in `window`: put back the settings the
    call needs (see _CallSettings.for_call).  A call the window refuses,
    which may-call-functions turned back on let through, can't be refused
    from here: it is stopped at its first instruction instead, before it
    can write past the parked thread's stack (see _CALL_HEADROOM), and
    abandoned when the window puts the host's registers back.  (Its stack
    pointer can't be spoiled instead: gdb has already worked it out.)"""
    try:
        for item in window.settings.for_call():
            if item not in window.settings_put_back:
                window.settings_put_back.append(item)
        if window.settings.calls_refused():
            window.call_refused = True
            bp = gdb.Breakpoint("*{:#x}".format(int(event.address) & _U64),
                                internal=True)
            window.refusals.append(bp)
            bp.silent = True
    except Exception:
        _debug_exc()


def _end_refusals(window):
    refusals = getattr(window, "refusals", [])  # (A window of an earlier load's.)
    for bp in refusals:
        try:
            if bp.is_valid():
                bp.delete()
        except Exception:
            _debug_exc()
    del refusals[:]


def _call_notes(window):
    """The notes on the function calls made in `window`, to print once it
    is closed."""
    notes = []
    if window.breakpoints_skipped:
        notes.append("realm-gdb: breakpoints were ignored in the function calls made "
                     "meanwhile (a call made while a parked Realm user thread is shown "
                     "can't stop).")
    put = getattr(window, "settings_put_back", None)  # (See _end_refusals.)
    if put:
        notes.append("realm-gdb: the function calls made meanwhile ran with {} all the "
                     "same, as a call made while a parked Realm user thread is shown "
                     "must.".format(", ".join(put[:-1]) + " and " + put[-1]
                                    if len(put) > 1 else put[0]))
    if getattr(window, "call_refused", False):
        notes.append("realm-gdb: function calls were refused all the same, with "
                     "may-call-functions turned on meanwhile: stopped at their first "
                     "instruction, and abandoned, so that nothing of them ran.")
    return notes


def _hidden_catchpoints():
    """The numbers of the enabled catchpoints, on gdbs whose gdb.breakpoints()
    leaves them out (before gdb 11).  A call that stopped at one would be
    abandoned, perhaps in a system call, which putting the registers back
    doesn't undo (and on aarch64, where gdb has no register for the system
    call's number, the host would make that call when resumed)."""
    if hasattr(gdb, "BP_CATCHPOINT"):
        return []
    try:
        out = _execute("info breakpoints")
    except gdb.error:
        return []
    return re.findall(r"(?m)^(\d+)\s+catchpoint\s+\S+\s+y\b", out)


def _enable_breakpoints():
    while _disabled_breakpoints:
        bp = _disabled_breakpoints.pop()
        try:
            if bp.is_valid():
                bp.enabled = True
        except Exception:
            _debug_exc()
    while _disabled_catchpoints:
        try:
            gdb.execute("enable " + _disabled_catchpoints.pop(), to_string=True)
        except gdb.error:
            _debug_exc()


def _on_cont(event):
    """Something is resuming the program.  Inside a window, that is only
    expected of function calls; anything else got past the hooks."""
    if _window is not None and _infcall_depth == 0:
        _window.resumed = True
        print("realm-gdb: WARNING: the program is resuming while thread {} has user "
              "thread {:#x}'s registers, and realm-gdb can't put its own back now: "
              "the program may crash.".format(_window.thread_num, _window.showing or 0))
    if _infcall_depth == 0 and _views:
        # Views are for core files (see "Core-file views"): whatever runs
        # isn't one.
        _drop_views(lambda v: _thread_running(v["thread_object"]))
    if _infcall_depth == 0 and _window is None and _selections:
        _drop_running_selections(event)


def _thread_running(thread):
    """Whether a thread is running (or gone)."""
    try:
        return not thread.is_valid() or thread.is_running()
    except RuntimeError:
        return True


def _on_exited(event):
    global _infcall_depth
    inferior = getattr(event, "inferior", None)
    _drop_views(lambda v: inferior is None or v["inferior"] == inferior.num)
    _drop_selections(lambda sel: inferior is None or sel["inferior"] == inferior.num)
    for window in list(_unrestored):
        try:
            if not window.thread.is_valid():
                _unrestored.remove(window)
        except Exception:
            _unrestored.remove(window)
    if _infcall_depth:
        _infcall_depth = 0
        _enable_breakpoints()


def _on_gdb_exiting(event):
    """gdb >= 12: put back any host's registers before gdb lets go of the
    program, however it exits."""
    global _plain
    if _plain is not None and _plain["input"] is not None:
        _plain["input"].__exit__(None, None, None)
    _plain = None
    if _window is not None:
        try:
            _window.close()
        except BaseException:
            _debug_exc()
    _retry_unrestored()


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------
#
# For the commands below, realm-gdb defines a hook-COMMAND that runs
# `realm-thread-guard COMMAND`, and for some a hookpost-COMMAND that runs
# `realm-thread-guard --after COMMAND`.  gdb runs a command's hooks however
# the command comes: by an alias or an abbreviation, from a script, a
# user-defined command, Python's gdb.execute or MI's `-interpreter-exec
# console`.  It doesn't run them for MI's own commands.  A hook gets no
# arguments, and a hook that fails stops its command.  If the user already
# has such a hook it is left alone, and realm-gdb says what that means.
#
# Inside a window, CLI commands that would let the program run, or let go of
# it, are refused (_GUARDED_COMMANDS), and so are those that read the
# terminal themselves (_TERMINAL_COMMANDS).  Outside windows they aren't,
# unless a host's registers couldn't be put back.  MI commands bypass CLI
# hooks, but inside a window they can only come from the CLI's
# `interpreter-exec` (refused) or Python's gdb.execute_mi.
#
# While a parked thread is selected, commands that look at the selected
# thread or frame show it, and commands that would act on the thread as it
# really is are refused (see "Plain commands on a selected parked thread").

# (Each that this gdb has; and every `target` subcommand, since connecting
# to a new target kills the program, which a window's `confirm off` doesn't
# ask about.)
_GUARDED_COMMANDS = (
    "continue", "signal", "step", "stepi", "next", "nexti", "finish", "until",
    "advance", "jump", "return", "reverse-step", "reverse-stepi", "reverse-next",
    "reverse-nexti", "reverse-finish", "reverse-continue", "detach",
    "detach inferiors", "disconnect", "kill", "kill inferiors", "quit", "run",
    "start", "starti", "attach", "core-file", "generate-core-file", "load",
    "checkpoint", "restart", "record", "record full", "record btrace",
    "interpreter-exec")
# Commands that read the terminal themselves (Python's REPL), refused in a
# window too: in `rt`, the output is collected until the command is over,
# so what they ask wouldn't show; and the window would stay open while they
# wait.  (Python's input() and sys.stdin are _NoTerminalInput's.)
_TERMINAL_COMMANDS = ("python-interactive",)

# Commands that work with the selected thread or frame, shown a selected
# parked thread ("PREFIX *" is every subcommand of PREFIX, theirs included,
# but those refused below): those that look at it, those that evaluate an
# expression in its scope, and those that take a location relative to it.
# `set var X = V` and `print X = V` change X on the parked thread, though
# not lastingly if X is in a register (the window puts the host's
# back; realm-gdb says so); a breakpoint command without a location uses the
# parked thread's selected frame.
_SHOWING_COMMANDS = (
    "backtrace", "info stack", "frame", "frame *", "select-frame", "select-frame *",
    "up", "down", "up-silently", "down-silently", "info frame", "info frame *",
    "info locals", "info args", "info registers", "info all-registers",
    "info vector", "info float", "info line", "info source", "info symbol",
    "info address", "info scope", "list", "print", "call", "output", "printf",
    "x", "ptype", "whatis", "display", "disassemble", "find", "set", "set variable",
    "eval", "init-if-undefined", "print-object", "dump *", "append *", "restore",
    "mem", "memory-tag *", "info vtbl", "info common", "info macro", "info macros",
    "macro expand", "macro expand-once", "break", "tbreak", "hbreak", "thbreak",
    "dprintf", "clear", "break-range", "trace", "ftrace", "strace", "skip",
    "skip function", "skip file")
# Commands refused while a parked thread is selected: they would act on the
# selected thread as it really is.  `rt` runs those of _RT_COMMANDS on the
# parked thread.  (`thread apply` and `frame apply` run other commands: see
# "Plain commands on a selected parked thread".)
_ACTING_COMMANDS = (
    "signal", "queue-signal", "step", "stepi", "next", "nexti",
    "finish", "until", "advance", "jump", "return", "reverse-step",
    "reverse-stepi", "reverse-next", "reverse-nexti", "reverse-finish",
    "reverse-continue", "watch", "rwatch", "awatch", "thread apply",
    "interpreter-exec", "python-interactive", "guile-repl", "explore", "explore *",
    "compile", "compile *")
_RT_COMMANDS = ("python", "guile", "frame apply", "frame apply *", "faas")
# Commands that end the selection (all of the inferior's) before they run,
# rather than being refused: the user means to leave it.
_LEAVING_COMMANDS = ("continue",)
# Commands that end it once they have switched to another thread (`thread
# N`; `thread` alone, or naming the thread already selected, keeps it).
_SWITCHING_COMMANDS = ("thread",)
# Commands after which plain commands no longer show a selected parked
# thread.
_SESSION_COMMANDS = ("new-ui",)
# Commands that start another program (see "Commands that start another
# program"), refused inside a window: `rt` runs them without one, but in a
# command that has one (a user-defined command run with `rt`, `eval` shown a
# selected parked thread), the program would run with the host holding the
# parked thread's registers.
_EXTERNAL_COMMANDS = ("shell", "make", "pipe", "edit")

# The commands hooked, by what their hooks do: "guarded" (_GUARDED_COMMANDS
# and _TERMINAL_COMMANDS), "acting", "rt" (a subset of "acting"), "leaving",
# "switching", "showing", "session" and "external".
_hooked = {}

# Where a switching command was run from, while it runs: (thread, inferior
# number) of the selected thread then.
_switch_from = None


def _existing_commands(names):
    """Those of `names` that this gdb has, found without running a command
    that fails (see _install_hooks): `complete NAME` lists NAME itself."""
    found = []
    for name in names:
        try:
            out = _execute("complete " + name)
        except gdb.error:
            continue
        if name in [line.strip() for line in out.splitlines()]:
            found.append(name)
    return found


def _subcommands(prefix):
    """The subcommands of `prefix` (none if it isn't a prefix command)."""
    try:
        out = _execute("help " + prefix)
    except gdb.error:
        return []
    return sorted(set(prefix + " " + word for word in re.findall(
        r"^{} ([\w-]+)(?: --|,)".format(re.escape(prefix)), out, re.M)))


def _expand(names):
    """`names`, with each "PREFIX *" replaced by PREFIX's subcommands, and
    theirs (`dump binary memory`).  (`help PREFIX` only for a PREFIX this
    gdb has: see _install_hooks.)"""
    out = []
    for name in names:
        if not name.endswith(" *"):
            out.append(name)
            continue
        prefixes = _existing_commands([name[:-2]])
        while prefixes:
            subs = _subcommands(prefixes.pop())
            out += subs
            prefixes += subs
    return out


def _target_subcommands():
    return _subcommands("target")


def _window_words(window):
    """What opened `window`, for messages."""
    if window.via is None:
        return "'realm-thread apply' or 'thread apply all bt'"
    if window.via == "rt":
        return "'rt'"
    return "'{}', which shows the selected one".format(window.via)


def _guard(cmd):
    if _window is not None and cmd in _TERMINAL_COMMANDS:
        raise gdb.GdbError(
            "realm-gdb: refusing '{}' while thread {} shows a parked Realm user "
            "thread (in {}): it reads the terminal, and {}.".format(
                cmd, _window.thread_num, _window_words(_window),
                "its output is only shown once the command is over"
                if _window.via in (None, "rt") else "the thread would keep the "
                "parked thread's registers while it waits"))
    if cmd in _TERMINAL_COMMANDS:
        return
    if _window is not None:
        num = _window.thread_num
        hint = ""
        if _window.via == "rt":
            # (Plain commands that would act on the thread as it really is
            # are refused while they show the selection.)
            hint = ("  To run it on thread {} as it really is, run 'realm-thread back' "
                    "first.".format(num) if _plain_off_why is None and
                    cmd in _hooked.get("acting", ()) else
                    "  To run it on thread {} as it really is, leave out 'rt'.".format(num))
        raise gdb.GdbError(
            "realm-gdb: refusing '{}' while thread {} shows a parked Realm user "
            "thread (in {}): that thread has the parked thread's registers until "
            "the command is over, so letting the program run, or letting go of "
            "it, would corrupt it.{}".format(cmd, num, _window_words(_window), hint))
    # (Killing the program is the way out, as the message says.)
    if _unrestored and cmd.split()[0] != "kill" and not _retry_unrestored():
        raise gdb.GdbError(
            "realm-gdb: refusing '{}': thread {} still has a parked Realm user "
            "thread's registers, which realm-gdb could not put back, so letting "
            "the program run, or letting go of it, would corrupt it.  Run "
            "'realm-thread back' to try again, or 'kill' the program.".format(
                cmd, ", ".join(str(w.thread_num) for w in _unrestored)))


def _outer_frames(frame):
    """The Python frames, innermost first, from `frame` out: those of
    whatever ran the gdb command that called realm-gdb's invoke(), whose
    caller `frame` is (None if no Python ran it)."""
    frames = []
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    return tuple(frames)


def _hook(cmd, after, outer):
    """realm-thread-guard: hook-CMD's work (hookpost-CMD's, `after`), run
    from Python frames `outer` (see _outer_frames)."""
    global _switch_from
    if _internal_depth:
        return
    if after:
        if cmd in _hooked.get("switching", ()):
            _switched(cmd)
        _plain_after(cmd, outer)
        return
    _close_stale(outer)
    if cmd in _hooked.get("guarded", ()):
        _guard(cmd)
    if cmd in _hooked.get("external", ()) and _window is not None:
        raise gdb.GdbError(
            "realm-gdb: refusing '{}' while thread {} shows a parked Realm user "
            "thread (in {}): the program it starts would run with that thread "
            "holding the parked thread's registers.  Run it on its own.".format(
                cmd, _window.thread_num, _window_words(_window)))
    if cmd in _hooked.get("session", ()):
        _plain_off("a second UI was opened ('{}')".format(cmd))
    if cmd in _hooked.get("acting", ()):
        _refuse_acting(cmd)
    if cmd in _hooked.get("leaving", ()) and _plain_shows() is not None:
        inferior = gdb.selected_inferior().num
        _drop_selections(lambda sel: sel["inferior"] == inferior, "continuing")
    if cmd in _hooked.get("switching", ()):
        _switch_from = None
        if _plain_shows() is not None:
            _switch_from = (gdb.selected_thread(), gdb.selected_inferior().num)
    if cmd in _hooked.get("showing", ()):
        _plain_before(cmd, outer)


class RealmThreadGuard(gdb.Command):
    """realm-thread-guard [--after] COMMAND

    Used by realm-gdb.py's hook-COMMAND and hookpost-COMMAND definitions.
    Fails, which stops COMMAND from running, if COMMAND would let the program
    run or let go of it while a thread has a parked Realm user thread's
    registers (inside `rt`, `realm-thread apply` or `thread apply all bt`);
    or, while a parked thread is selected (by `realm-thread <ptr>`), if it
    would act on the selected thread as it really is.  If COMMAND looks at
    the selected thread or frame, a selected parked thread is shown for its
    duration."""

    def __init__(self):
        super(RealmThreadGuard, self).__init__("realm-thread-guard",
                                               gdb.COMMAND_MAINTENANCE)

    def invoke(self, arg, from_tty):
        after = arg.startswith("--after ")
        cmd = (arg[len("--after "):] if after else arg).strip()
        try:
            _hook(cmd, after, _outer_frames(sys._getframe().f_back))
        except gdb.GdbError:
            raise
        except Exception:
            _debug_exc()


def _switched(cmd):
    """hookpost-CMD of a switching command: if it switched to another
    thread while plain commands showed a selection, end the selection (all
    of the inferior's), saying so."""
    global _switch_from
    if _switch_from is None:
        return
    (old, inferior), _switch_from = _switch_from, None
    new = gdb.selected_thread()
    if new is None or new == old:
        return
    where = (str(new.num) if new.inferior.num == inferior else
             "{}.{}".format(new.inferior.num, new.num))
    _drop_selections(lambda sel: sel["inferior"] == inferior,
                     "switched to thread {}".format(where))


def _hook_name(command, kind):
    """`define`'s name for COMMAND's hook (kind "hook") or hookpost: `detach
    inferiors` is hooked by `detach hook-inferiors`."""
    prefix, _, word = command.rpartition(" ")
    return (prefix + " " if prefix else "") + kind + "-" + word


def _install_hooks():
    # This runs while librealm is being loaded, often in the middle of
    # `run`; a command that fails here makes gdb 10 and 12 (at least)
    # return from `run` without waiting for the program to stop.  So find
    # the commands and hooks that exist without asking for ones that don't
    # (`help memory-tag` fails before gdb 11).
    try:
        defined = _execute("help user-defined")
    except gdb.error:
        defined = ""
    kinds = {
        "guarded": set(_existing_commands(_GUARDED_COMMANDS + _TERMINAL_COMMANDS) +
                       _target_subcommands()),
        "rt": set(_existing_commands(_expand(_RT_COMMANDS))),
        "acting": set(_existing_commands(_expand(_ACTING_COMMANDS))),
        "leaving": set(_existing_commands(_LEAVING_COMMANDS)),
        "switching": set(_existing_commands(_SWITCHING_COMMANDS)),
        "session": set(_existing_commands(_SESSION_COMMANDS)),
        "external": set(_existing_commands(_EXTERNAL_COMMANDS)),
    }
    kinds["acting"] |= kinds["rt"]
    kinds["showing"] = set(_existing_commands(_expand(_SHOWING_COMMANDS))) - kinds["acting"]
    _hooked.clear()
    for c in sorted(set().union(*kinds.values())):
        hooks = [_hook_name(c, "hook")]
        if c in kinds["showing"] or c in kinds["switching"]:
            hooks.append(_hook_name(c, "hookpost"))
        # Leave the user's own hooks alone.
        theirs = []
        for hook in hooks:
            if re.search(r"^{} -- ".format(re.escape(hook)), defined, re.M):
                try:
                    existing = _execute("show user " + hook)
                except gdb.error:
                    existing = ""
                if "realm-thread-guard" not in existing:
                    theirs.append(hook)
        if theirs:
            print(_their_hook_note(c, theirs[0], kinds))
            if c in kinds["session"]:
                _plain_off("you defined {}, so realm-gdb can't tell when a second "
                           "UI is opened".format(theirs[0]))
            continue
        try:
            for hook in hooks:
                gdb.execute("define {}\nrealm-thread-guard {}{}\nend\n".format(
                    hook, "--after " if "hookpost-" in hook else "", c), to_string=True)
        except gdb.error as e:
            print("realm-gdb: could not define {}: {}".format(hook, e))
            continue
        for kind, commands in kinds.items():
            if c in commands:
                _hooked.setdefault(kind, set()).add(c)


def _their_hook_note(c, hook, kinds):
    """What it means that the user's own `hook` for command `c` was left."""
    if c in kinds["guarded"]:
        return ("realm-gdb: you defined {}, so '{}' is not refused in 'rt', "
                "'realm-thread apply' or 'thread apply all bt' commands, where it "
                "would act on a thread holding a parked Realm user thread's "
                "registers and may corrupt or crash the program.".format(hook, c))
    if c in kinds["acting"]:
        return ("realm-gdb: you defined {}, so '{}' is not refused while a parked "
                "Realm user thread is selected, though it acts on the selected "
                "thread as it really is.".format(hook, c))
    if c in kinds["leaving"]:
        return ("realm-gdb: you defined {}, so a selected parked Realm user thread "
                "is only dropped once '{}' lets the program run.".format(hook, c))
    if c in kinds["switching"]:
        return ("realm-gdb: you defined {}, so '{} N' doesn't end the selection of a "
                "parked Realm user thread, which plain commands then show on thread "
                "N.".format(hook, c))
    if c in kinds["showing"]:
        return ("realm-gdb: you defined {}, so '{}' shows a selected parked Realm "
                "user thread only as 'rt {}'.".format(hook, c, c))
    if c in kinds["external"]:
        return ("realm-gdb: you defined {}, so '{}' is not refused inside a command "
                "that shows a parked Realm user thread, where the program it starts "
                "would run with a thread holding the parked thread's registers."
                .format(hook, c))
    return ("realm-gdb: you defined {}, so plain commands show a selected parked "
            "Realm user thread only with 'rt'.".format(hook))


# --------------------------------------------------------------------------
# Core-file views
# --------------------------------------------------------------------------
#
# A core file can't take register writes, so there a view shows a parked
# user thread's stack on a gdb thread (the "host"): the unwinder below claims
# the host thread's innermost real frame (the "host frame") and reports the
# parked user thread's saved context as its caller's registers, so frame #0
# is the host frame and frames #1 and up are the parked stack.  A frame
# filter prints the host frame as a marker line (see "the marker line").
# Nothing in the core changes, so dropping a view only needs the frame cache
# rebuilt.  (A core file can't run, so its views never go stale; they are
# dropped if its inferior turns into a live process.)
#
# The saved context holds PC, SP and the callee-saved general registers.
# gdb reports any register an unwinder doesn't supply as unavailable, so a
# parked frame's value kept only in another (aarch64's d8-d15 among them)
# shows as <optimized out>.  The host frame's ID is at the parked thread's
# SP, since gdb stops unwinding at a caller whose frame ID is inner to its
# callee's, and the parked stack can be anywhere; so frame #0's own values,
# read relative to it, are wrong (the marker line doesn't print them).
#
# Finding the host frame:
#
#   gdb >= 11 tells an unwinder the pending frame's level.  The unwinder
#   claims the frame at the host frame's level if its PC and SP are still
#   the host frame's.
#
#   Older gdbs don't, and reading a register of the pending frame hits an
#   internal error in gdb 9 ("get_frame_id: Assertion `fi->level == 0'
#   failed") when the frame just inside it is an inline frame.  So there a
#   view is "armed" when the host thread's frame cache is about to be
#   rebuilt from scratch, and the unwinder claims the first frame it is then
#   asked about, which is the innermost one.  gdb rebuilds the cache on
#   every thread switch (the unwinder notices those).  Wherever the host
#   thread might not be at the host frame, the view is "hidden" until
#   _rearm() has checked that it is.
#
# A view is a dict:
#   thread:           global number of the host thread
#   thread_num:       its number within its inferior, for messages
#   thread_object:    its gdb.InferiorThread
#   inferior, pid:    its inferior and process, so that a view never
#                     outlives it
#   ut:               the UserThread*, as a gdb.Value
#   user_thread:      the same, as an int
#   host_pc, host_sp: the host frame's PC and SP
#   host_level:       the host frame's level (> 0 below inline frames)
#   regs:             unwind_regs_from_ctx() of the parked user thread
#   values:           regs (plus _Arch.thread_regs) as gdb.Values
#   frame_id:         the host frame's ID while the view is shown
#   temporary:        True for the views shown for one command
#   armed, hidden:    see above (gdb < 11 only)
#   level:            the frame the user last had selected while the view
#                     was shown, to go back to if gdb loses it (gdb < 11)
#
# Views are kept per host thread, in a stack (realm-thread nests); only the
# top one is shown.
#
# (This script runs in gdb's __main__ namespace, and may be loaded again, by
# another copy of librealm or by hand: keep the state of the earlier load.)
_views = globals().get("_views", {})

# gdb thread the unwinder last ran for (gdb < 11).
_last_unwound_thread = globals().get("_last_unwound_thread")

# Whether PendingFrame.level() exists (gdb >= 11).
_have_level = hasattr(getattr(gdb, "PendingFrame", None), "level")


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
    stack = _views.get(view["thread"], [])
    for i, v in enumerate(stack):
        if v is view:
            del stack[i]
            break
    if not stack:
        _views.pop(view["thread"], None)
    _frames_changed()


def _drop_views(which):
    dropped = False
    for stack in list(_views.values()):
        for view in list(stack):
            if which(view):
                _remove_view(view)
                dropped = True
    return dropped


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
        "host_pc": frame.pc(),
        "host_sp": _frame_sp(frame),
        "host_level": level,
        "regs": regs,
        "values": values,
        "frame_id": frame_id,
        "temporary": temporary,
        "armed": False,
        "hidden": False,
        "level": None,
    }


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
    if select == "parked" and view is not None:
        frame = _first_parked_frame(view)
        if frame is not None:
            frame.select()
            return
    _select_level(select if isinstance(select, int) else 0)


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
                if pending_frame.level() != view["host_level"]:
                    return None
                pc = int(pending_frame.read_register(_arch.pc_reg)) & _U64
                sp = int(pending_frame.read_register(_arch.sp_reg)) & _U64
                if pc != view["host_pc"] or sp != view["host_sp"]:
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
            info = pending_frame.create_unwind_info(view["frame_id"])
            for name, value in view["values"].items():
                info.add_saved_register(name, value)
            return info
        except Exception:
            return None


# ---- the marker line --------------------------------------------------------
#
# Any enabled frame filter makes gdb print the whole backtrace, every parked
# frame included, with its Python frame printer instead of its own.  Mostly
# that changes only how it looks, but in two ways it matters.  That printer
# gets each frame's locals from its decorator, and gdb's FrameDecorator
# leaves out some that gdb's printer shows (artificial ones such as __func__,
# and ones that are optimized out or constant) and, before gdb 13, adds those
# of the function an inlined one is in; so the filter below lists them as
# gdb's printer does (_frame_locals).  And from gdb 10 it hands
# pretty-printers values that are entirely optimized out, which most can't
# print (libstdc++'s raise an error), where gdb's own printer prints
# `<optimized out>` without asking them; so while it prints the filter's
# frames' values, a lookup function that gdb consults before all others does
# the same (_printing_lookup).  (Function calls, which that printer also
# mishandles, can't be made in a core file.)  gdb's DAP doesn't use the
# printer, so none of this applies to the frames it gets (see
# _for_frame_printer).

# How many _printing iterations are under way, and the printer list
# _printing_lookup is in meanwhile.
_printing_depth = 0
_printing_list = None


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
    printer, which prints each as it gets it: with _printing_lookup in
    place meanwhile."""
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
    global _printing_depth, _printing_list
    _printing_depth += 1
    if _printing_list is not None:
        return
    try:
        objfiles = gdb.objfiles()
        if objfiles:
            _printing_list = objfiles[0].pretty_printers
        else:
            _printing_list = gdb.current_progspace().pretty_printers
        _printing_list.insert(0, _printing_lookup)
    except Exception:
        _printing_list = None


def _end_printing():
    global _printing_depth, _printing_list
    _printing_depth -= 1
    if _printing_depth > 0:
        return
    _printing_depth = 0
    if _printing_list is not None:
        while _printing_lookup in _printing_list:
            _printing_list.remove(_printing_lookup)
        _printing_list = None


def _printing_over():
    """Once gdb has finished a command, make sure _printing_lookup isn't
    left in place (were an iteration of the filter's frames not closed)."""
    global _printing_depth
    if _printing_depth > 0:
        _printing_depth = 1
        _end_printing()


def _printing_lookup(val):
    """A pretty-printer lookup function that gdb consults before all others
    while its frame printer prints values from the frame filter's frames
    (see "the marker line")."""
    try:
        if _entirely_optimized_out(val, [_OPTIMIZED_OUT_BUDGET]):
            return _OptimizedOut(val)
        t = val.type.strip_typedefs()
        if (t.code == gdb.TYPE_CODE_ARRAY and t.target().sizeof == 1
                and t.sizeof > len(_UNREADABLE)
                and val.string(length=len(_UNREADABLE)) == _UNREADABLE):
            return _Text(val.string(length=t.sizeof - 1)[len(_UNREADABLE):])
    except Exception:
        pass  # (Can't tell: leave it to the others.)
    return None


class _Text(object):
    def __init__(self, text):
        self._text = text

    def to_string(self):
        return self._text


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
    printer does).  One that can't be read is shown as gdb's own printer
    shows it: gdb's frame printer would raise an error instead, which ends
    the frame's locals."""

    def __init__(self, sym, frame):
        self._sym = sym
        self._frame = frame

    def symbol(self):
        return self._sym

    def value(self):
        try:
            self._sym.value(self._frame)
        except gdb.error as e:
            return gdb.Value(_UNREADABLE + "<error reading variable {} ({})>".format(
                self._sym.print_name, e))
        except Exception:
            pass
        return None


# What starts the text _Local gives for a local it can't read, which
# _printing_lookup prints as it is.
_UNREADABLE = "\x01realm-gdb unreadable\x01"


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
                found.append(_Local(sym, frame))
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
    # frame printer, so do that only while a view is shown.
    @property
    def enabled(self):
        try:
            view = _current_view()
            return view is not None and self._user_enabled and _shown(view)
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


# (Keep the user's `disable frame-filter` across a reload.)
_frame_filter = _ParkedThreadFrameFilter()
_frame_filter._user_enabled = getattr(gdb.frame_filters.get("realm-parked-thread"),
                                      "_user_enabled", True)


def _marker_drawn():
    """Whether a shown view's frame #0 is drawn as the marker line."""
    return _frame_filter._user_enabled


def _frame0_words(thread_num):
    return "{}thread {}'s own frame".format(
        "a marker for " if _marker_drawn() else "", thread_num)


def _on_before_prompt():
    """Tidy up after a command (see "the marker line"), and on gdb < 11 show
    the selected thread's view again if gdb rebuilt its frames without the
    unwinder knowing (e.g. `thread apply 1 2 3 print 1`), going back to the
    frame the user had selected."""
    global _infcall_depth
    if _plain is not None:
        _close_plain(_plain)  # (its command failed: see "Plain commands ...")
    _printing_over()
    if _window is None and _infcall_depth:
        _infcall_depth = 0
        _enable_breakpoints()
    view = _current_view()
    if view is None or _have_level:
        return
    if _shown(view):
        view["level"] = _selected_level()
        return
    if not _at_host(view):
        if not view["hidden"]:
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
            _debug_exc()
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


def _install_event_handlers():
    gdb.unwinder.register_unwinder(None, _ParkedThreadUnwinder(), replace=True)
    gdb.frame_filters["realm-parked-thread"] = _frame_filter
    _connect("cont", "_on_cont")
    _connect("exited", "_on_exited")
    _connect("inferior_call", "_on_inferior_call")
    _connect("before_prompt", "_on_before_prompt")
    _connect("gdb_exiting", "_on_gdb_exiting")


# --------------------------------------------------------------------------
# Running a command on parked user threads
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


def _view_command(command, view):
    """`command` to run with a core-file view shown: a backtrace counts its
    frames from #0, and before the parked stack come the host frame and any
    inline frames inside it (one marker line, if that is drawn), so `bt N`
    shows N of the parked thread's frames."""
    bt_args = _bt_args(command)
    if bt_args is None:
        return command
    options, count = _split_bt_args(bt_args)
    if not count:
        return "bt " + " ".join(options)
    n = int(gdb.parse_and_eval(count))
    if n > 0:
        filtered = (_marker_drawn() and "no-filters" not in options
                    and not any(len(o) > 2 and "-no-filters".startswith(o)
                                for o in options))
        n += 1 if filtered else view["host_level"] + 1
    return "bt " + " ".join(options + [str(n)])


def _run_in_view(ut, command):
    """Run `command` with user thread `ut` shown in a temporary core-file
    view on the selected thread, returning its output."""
    view = _make_view(ut, temporary=True)
    _views.setdefault(view["thread"], []).append(view)
    # `set backtrace limit N` should show N parked frames, as `bt N` does:
    # allow for the frames before them.
    limit = (_parameter("backtrace limit")
             if not callable(command) and _bt_args(command) is not None else None)
    try:
        if limit:
            gdb.execute("set backtrace limit {}".format(
                limit + view["host_level"] + 1), to_string=True)
        _refresh_frames("parked")
        if not _shown(view):
            raise gdb.GdbError("cannot unwind it{}".format(
                " within 'set backtrace limit {}'".format(limit)
                if limit and limit <= view["host_level"] + 1 else ""))
        if callable(command):  # (see _apply_external)
            return command()
        return _execute(_view_command(command, view))
    finally:
        _remove_view(view)
        if limit:
            gdb.execute("set backtrace limit {}".format(limit), to_string=True)
        _refresh_frames()


def _call_room(ut):
    """Whether a function call made while `ut` is shown in a window has
    enough of its stack free (see _CALL_HEADROOM)."""
    try:
        base, _ = _stack_of(ut)
        sp = _arch.unwind_regs_from_ctx(ut)[_arch.sp_reg]
    except gdb.error:
        return False
    return sp - base >= _CALL_HEADROOM


# The words for gdb's `frame`, `up` and `down` (aliases and abbreviations).
_FRAME_WORDS = ("f", "fr", "fra", "fram", "frame")
_UP_WORDS = ("up",)
_DOWN_WORDS = ("do", "dow", "down")


def _frame_motion(command):
    """`frame N` (or `frame level N`), `up [N]` or `down [N]`, as gdb does
    them (same frame, same errors), but selecting the frame from Python, and
    printing it with a plain `frame`, returning that output; None for any
    other command.  gdb itself prints the frame these select from its
    notification that the selected frame changed, which goes straight to
    the console (on every gdb from 8.2 to 17), where it can't be collected
    to be printed once the window is closed (see _apply), and to an MI
    front end as `=thread-selected` with a parked frame in it."""
    words = command.split()
    if not words:
        return None
    word, args = words[0], words[1:]
    if word in _FRAME_WORDS:
        if args[:1] == ["level"]:
            args = args[1:]
        if len(args) != 1 or not re.match(r"\d+$", args[0]):
            return None
        frame = gdb.newest_frame()
        for _ in range(int(args[0])):
            frame = frame.older()
            if frame is None:
                raise gdb.GdbError("No frame at level {}.".format(args[0]))
    elif word in _UP_WORDS + _DOWN_WORDS:
        if len(args) > 1:
            return None
        up = word in _UP_WORDS
        count = int(gdb.parse_and_eval(args[0])) if args else 1
        if not up:
            count = -count
        frame = gdb.selected_frame()
        while count > 0 and frame.older() is not None:
            frame, count = frame.older(), count - 1
        while count < 0 and frame.newer() is not None:
            frame, count = frame.newer(), count + 1
        if count != 0 and not args:
            raise gdb.GdbError("Initial frame selected; you cannot go up." if up else
                               "Bottom (innermost) frame selected; you cannot go down.")
    else:
        return None
    frame.select()
    return _execute("frame")


def _show_in_window(window, ut, selection=None):
    """Show user thread `ut` in register window `window` (see "Register
    windows"), with `selection`'s frame selected (see "Selecting a parked
    thread").  Returns what _window_done needs."""
    room = _call_room(ut)
    if not room and not _can_refuse_calls:
        raise _Unshowable("too little of its stack is free below its stack pointer "
                          "for function calls, which this gdb can't refuse")
    regs, ut_notes = _arch.context_regs(ut)
    window.show(ut, regs, calls=room)
    if selection is not None:
        _select_level(selection["level"])
    return {"ut": ut, "selection": selection, "room": room, "notes": ut_notes,
            "regs": _raw_regs(gdb.newest_frame())}


def _window_done(window, shown, what):
    """After command `what` (its first word) has run with _show_in_window's
    `shown` in `window`: remember the frame it left selected, and return the
    notes to print once the window is closed."""
    ut = shown["ut"]
    notes = []
    if gdb.selected_thread() == window.thread:
        changed = sorted(n for n, v in _raw_regs(gdb.newest_frame()).items()
                         if n in shown["regs"] and v != shown["regs"][n])
        if changed:
            # (The window puts the thread's own registers back.)
            notes.append("realm-gdb: user thread {:#x}: '{}' changed {}, which doesn't "
                         "last: a parked thread's saved registers aren't changed, and "
                         "thread {}'s own are put back.".format(
                             int(ut), what, ", ".join("$" + n for n in changed),
                             window.thread_num))
        if shown["selection"] is not None:
            shown["selection"]["level"] = _selected_level()
    notes += ["realm-gdb: user thread {:#x}: {}.".format(int(ut), n) for n in shown["notes"]]
    if not shown["room"]:
        notes.append("realm-gdb: user thread {:#x} has under {} KiB of its stack free "
                     "below its stack pointer, so function calls were refused while "
                     "it was shown.".format(int(ut), _CALL_HEADROOM >> 10))
    return notes


def _run_in_window(window, ut, command, notes, selection=None):
    """Run `command` with user thread `ut` shown in register window
    `window`, returning its output (see "Register windows").  With a
    `selection` (see "Selecting a parked thread"), its frame is selected
    first, and the frame `command` leaves selected is remembered."""
    shown = _show_in_window(window, ut, selection)
    if callable(command):  # (see _apply_external)
        out = command()
    else:
        out = _frame_motion(command)
        if out is None:
            out = _execute(command)
    notes += _window_done(window, shown, "edit" if callable(command) else command.split()[0])
    return out


class _Unshowable(Exception):
    """A parked user thread that turns out not to be showable."""


def _apply(targets, command, header, selection=None, collect=None):
    """Run `command` on each parked user thread in `targets` ([(ut,
    problem)] of the selected inferior), shown on the selected thread, and
    print its output (or, with a list `collect`, append it there).  With
    `header`, each output gets a "User thread PTR:" line first.  A
    `selection` is for `rt` (see _run_in_window).  Returns the first error,
    if any (gdb.error).  (The output is printed once the window is closed:
    gdb's pager may wait for input, and so let another UI run a command; see
    "Register windows".)  A command that starts another program is
    _apply_external's."""
    external = _external(command) if collect is None and not callable(command) else None
    if external is not None:
        return _apply_external(targets, command, header, selection, external)
    how = _how_shown()
    if isinstance(how, _NotShown):
        raise gdb.GdbError("realm-thread: can't show parked user threads: " +
                           how.sentence())
    first_error = []
    what = "edit" if callable(command) else (command.split() or [command])[0]
    notes = []
    printed = []
    unshowable = None
    window = None
    if how == "window":
        window = _Window(via="rt" if selection is not None else None)
    with (window if window is not None else _NoWindow()), _NoTerminalInput(what, window):
        for ut, problem in targets:
            out = None
            if problem is None:
                try:
                    if window is not None:
                        out = _run_in_window(window, ut, command, notes, selection)
                    else:
                        out = _run_in_view(ut, command)
                except _Unshowable as e:
                    problem = str(e)
                except gdb.error as e:
                    out = e
            if problem is not None and not header:
                unshowable = gdb.GdbError("realm-thread: can't show user thread "
                                          "{:#x}: {}.".format(int(ut), problem))
                break
            if header:
                printed.append("\nUser thread {:#x}:\n".format(int(ut)))
            if problem is not None:
                printed.append("  <{}>\n".format(_problem_words(problem)))
            elif isinstance(out, gdb.error):
                if not first_error:
                    first_error.append(out)
                if header:  # (A single thread's error is the command's.)
                    printed.append("  <{} failed: {}>\n".format(what, out))
            else:
                printed.append(out)
    if collect is not None:
        collect.append("".join(printed))
    else:
        gdb.write("".join(printed))
        # (Output that doesn't end a line, such as `output`'s, would
        # otherwise wait until after the prompt, or the next command's.)
        gdb.flush()
    if unshowable is not None:
        raise unshowable
    if window is not None:
        notes += _call_notes(window)
    for note in notes:
        print(note)
    return first_error[0] if first_error else None


# --------------------------------------------------------------------------
# Commands that start another program
# --------------------------------------------------------------------------
#
# `shell` (`!`), `make` and `edit` start another program (a shell command,
# an editor), and look at nothing of the parked thread's: `rt` and
# `realm-thread apply` run them without showing it.  In a window, the host
# would hold the parked thread's registers for as long as that program ran
# (with SIGTERM and SIGHUP held), and gdb dying meanwhile would leave an
# attached program corrupted (see "Register windows").  `pipe COMMAND |
# SHELL` runs COMMAND with the parked thread shown, its output collected,
# then SHELL on that output; `edit *EXPR` works out where EXPR is with the
# parked thread shown, then edits there.  (Bare `edit`, and `edit` of a
# function or line, use gdb's current source position, which `rt frame` and
# the like set: not the registers.)

_COMMAND_WORD = re.compile(r"\s*([A-Za-z0-9_.-]+|!|\|)")
_has_pipe_command = globals().get("_has_pipe_command")


def _external(command):
    """How `command` starts another program: ("direct",), ("pipe", COMMAND,
    SHELL) or ("edit", LOCATION); or None, if it doesn't."""
    global _has_pipe_command
    m = _COMMAND_WORD.match(command)
    if not m:
        return None
    word, rest = m.group(1), command[m.end():].strip()

    def abbreviates(name, least):
        return name.startswith(word) and word.startswith(least)
    if word == "!" or abbreviates("shell", "she") or abbreviates("make", "mak"):
        return ("direct",)
    if abbreviates("edit", "ed"):
        return ("edit", rest) if rest.startswith("*") else ("direct",)
    if word == "|" or abbreviates("pipe", "pip"):
        if _has_pipe_command is None:
            _has_pipe_command = bool(_existing_commands(["pipe"]))
        if not _has_pipe_command:
            return None
        # (As gdb's pipe: -d DELIM, then COMMAND to the first DELIM.)
        delim = "|"
        m = re.match(r"-d\s+(\S+)\s*(.*)$", rest, re.S)
        if m:
            delim, rest = m.group(1), m.group(2)
        elif rest.startswith("--"):
            rest = rest[2:].lstrip()
        i = rest.find(delim)
        if i < 0 or not rest[i + len(delim):].strip():
            return ("direct",)  # (gdb's own error)
        if not rest[:i].strip():
            raise gdb.GdbError("realm-thread: 'pipe' needs a COMMAND here (gdb's would "
                               "repeat the last command).")
        if _external(rest[:i]) is not None:
            return ("direct",)  # (`pipe shell ... | ...`: nothing shown either)
        return ("pipe", rest[:i].strip(), rest[i + len(delim):].strip())
    return None


def _apply_external(targets, command, header, selection, external):
    """_apply for a command that starts another program (see above)."""
    first_error = None
    for ut, problem in targets:
        if header:
            gdb.write("\nUser thread {:#x}:\n".format(int(ut)))
            gdb.flush()
        if problem is not None and external[0] != "direct":
            if not header:
                raise gdb.GdbError("realm-thread: can't show user thread {:#x}: {}.".format(
                    int(ut), problem))
            gdb.write("  <{}>\n".format(_problem_words(problem)))
            continue
        try:
            if external[0] == "direct":
                gdb.execute(command, to_string=False)
            elif external[0] == "pipe":
                out = []
                error = _apply([(ut, None)], external[1], False, selection, collect=out)
                if error is not None:
                    raise error
                _run_shell(external[2], "".join(out))
            else:
                where = []
                error = _apply([(ut, None)], lambda: _locate(external[1], where), False,
                               selection)
                if error is not None:
                    raise error
                _edit_at(*where[0])
        except gdb.error as e:
            if first_error is None:
                first_error = e
            if header:
                gdb.write("  <{} failed: {}>\n".format(command.split()[0], e))
    return first_error


def _run_shell(shell, text):
    """pipe's SHELL, given `text`, as gdb's pipe runs it (with /bin/sh),
    setting $_shell_exitcode or $_shell_exitsignal."""
    import subprocess
    gdb.flush()
    try:
        proc = subprocess.Popen(shell, shell=True, stdin=subprocess.PIPE)
    except OSError as e:
        raise gdb.GdbError("Error launching \"{}\": {}".format(shell, e))
    try:
        proc.communicate(text.encode("utf-8", "surrogateescape"))
    except KeyboardInterrupt:
        # (Ctrl-C: SHELL had the terminal's SIGINT too, as with gdb's pipe.)
        try:
            proc.wait(2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _shell_status(proc.returncode)
        raise
    _shell_status(proc.returncode)


def _shell_status(status):
    if hasattr(gdb, "set_convenience_variable"):
        gdb.set_convenience_variable("_shell_exitcode", status if status >= 0 else None)
        gdb.set_convenience_variable("_shell_exitsignal", -status if status < 0 else None)


def _locate(location, where):
    """With the parked thread shown: where `edit LOCATION` would edit."""
    sals = gdb.decode_line(location)[1]
    if not sals:
        raise gdb.GdbError("No line number known for {}.".format(location))
    sal = sals[0]
    block = gdb.block_for_pc(sal.pc) if sal.pc else None
    while block is not None and block.function is None:
        block = block.superblock
    where.append((sal.pc, sal.symtab, sal.line,
                  block.function.print_name if block is not None else None))
    return ""


def _edit_at(pc, symtab, line, function):
    """`edit *ADDR`'s editing, once ADDR is known (as gdb's edit_command)."""
    if symtab is None:
        raise gdb.GdbError("No source file for address {:#x}.".format(pc))
    if function:
        print("{:#x} is in {} ({}:{}).".format(pc, function, symtab.filename, line))
    else:
        print("{:#x} is at {}:{}.".format(pc, symtab.filename, line))
    if not line:
        raise gdb.GdbError("No line number known for {:#x}.".format(pc))
    gdb.execute('shell {} +{} "{}"'.format(os.environ.get("EDITOR") or "/bin/ex", line,
                                          symtab.fullname()))


class _NoTerminalInput(object):
    """While `rt` or `realm-thread apply` runs a command, or a plain command
    shows a selected parked thread: Python's input() and sys.stdin refuse,
    rather than wait (`explore` asks its questions so).  In `rt`, the
    command's output is collected until it is over, so the question wouldn't
    show; and in a window the thread would hold the parked thread's
    registers meanwhile, with SIGTERM and SIGHUP held."""

    def __init__(self, what, window):
        self.what, self.window = what, window

    def refuse(self, *args, **kwargs):
        if self.window is not None and self.window.via not in (None, "rt"):
            raise gdb.GdbError(
                "realm-gdb: '{}' reads the terminal, which can't be done while "
                "thread {} shows a parked Realm user thread: it would keep the "
                "parked thread's registers while it waits.".format(
                    self.what, self.window.thread_num))
        raise gdb.GdbError(
            "realm-gdb: '{}' reads the terminal, which can't be done inside 'rt' or "
            "'realm-thread apply': its output, and so its question, only shows once "
            "it is over{}.".format(
                self.what, ", and thread {} has a parked thread's registers "
                "meanwhile".format(self.window.thread_num) if self.window else ""))

    def __enter__(self):
        import builtins
        self.input, self.stdin = builtins.input, sys.stdin
        builtins.input = self.refuse
        sys.stdin = _RefusedInput(self.refuse, self.stdin)
        return self

    def __exit__(self, *exc):
        import builtins
        builtins.input, sys.stdin = self.input, self.stdin
        return False


class _RefusedInput(object):
    """sys.stdin while _NoTerminalInput is in force."""

    def __init__(self, refuse, real):
        self._refuse, self._real = refuse, real
        self.encoding = getattr(real, "encoding", None)

    def read(self, *args):
        self._refuse()

    readline = readlines = read

    def __iter__(self):
        self._refuse()

    def isatty(self):
        return False


class _NoWindow(object):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _problem_words(problem):
    """How a problem is shown in place of a parked thread's output."""
    if problem.startswith(("running on thread", "not started yet")):
        return problem
    return "not shown: " + problem


# --------------------------------------------------------------------------
# thread apply all
# --------------------------------------------------------------------------

def _dump_parked_threads(command):
    """Run backtrace `command` on every parked user thread of each inferior,
    in a section that begins and ends with a line of its own.  (The caller
    restores the selection.)"""
    for inferior in sorted(gdb.inferiors(), key=lambda inf: inf.num):
        if _select_inferior(inferior):
            _dump_inferior_parked_threads(command)


def _dump_inferior_parked_threads(command):
    """_dump_parked_threads for the selected inferior, on the selected
    thread."""
    where = ""
    if len(gdb.inferiors()) > 1:
        where = " of inferior {}".format(gdb.selected_inferior().num)
    end = "=== End of parked Realm user threads{} ===".format(where)
    how = _how_shown()
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
        if isinstance(how, _NotShown):
            # (gdb 9 can't read memory through a running thread.)
            _print_not_shown(where, None, how)
        else:
            print("\n=== Parked Realm user threads{} ===".format(where))
            print("realm-gdb: cannot read the Realm user thread registry: {}".format(e))
        print(end)
        return
    if not parked and loop is None:
        return
    if isinstance(how, _NotShown):
        _print_not_shown(where, len(parked), how)
        if loop is not None:
            print("realm-gdb: {}; there may be more parked threads.".format(loop))
        print(end)
        return
    host = gdb.selected_thread()
    if how == "window":
        print("\n=== Parked Realm user threads{} ({}), each shown on thread {} (whose "
              "thread-local variables they show) ===".format(where, len(parked),
                                                            host.num))
    else:
        print("\n=== Parked Realm user threads{} ({}), shown on thread {}: each stack "
              "starts at frame #1 (#0 is {}) ===".format(
                  where, len(parked), host.num, _frame0_words(host.num)))
    try:
        _apply([(ut, _parked_problem(ut, sps)) for ut in parked], command, header=True)
    except gdb.error as e:
        print("realm-gdb: {}".format(e))
    if loop is not None:
        print("\nrealm-gdb: {}; there may be more parked threads.".format(loop))
    print("\n" + end)


def _print_not_shown(where, count, how):
    print("\n=== Parked Realm user threads{}{}: not shown ===".format(
        where, "" if count is None else " ({})".format(count)))
    print(how.sentence())


class ThreadApplyAll(gdb.Command):
    """Apply a command to all threads, plus parked Realm user threads for bt.

    Usage: thread apply all [-ascending] [-q] [-c] [-s] [--] COMMAND

    Like gdb's own `thread apply all` (see `help thread apply`), which this
    replaces: the flags mean the same and threads are visited in descending
    order unless -ascending is given.  When COMMAND is `bt` (or `backtrace`,
    `where` or `info stack`, with any options), it is then run on every
    parked Realm user thread of each inferior, between a "=== Parked Realm
    user threads ... ===" line and a "=== End of parked Realm user threads
    ===" line.  In a live process, each parked thread is shown on a stopped
    thread with its saved registers in place, and that thread's own put
    back before the command returns (see `help realm-thread`).  Where they
    can't be shown (non-stop mode, a running or recorded program), it says
    so and how to see them.  If COMMAND fails on a thread, gdb stops there
    as usual (unless -c is given), and the command fails with gdb's error,
    but only after the parked threads."""

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
        _close_stale(_outer_frames(sys._getframe().f_back))
        _printing_over()
        flags, rest = self._split_flags(arg.strip())
        if not rest:
            raise gdb.GdbError("Please specify a command at the end of "
                               "'thread apply all'")
        bt = _bt_args(rest) is not None
        if bt and _window is not None:
            raise gdb.GdbError("realm-gdb: 'thread apply all {}' can't show parked "
                               "Realm user threads from inside 'realm-thread "
                               "apply'.".format(rest))
        with _KeepSelection():
            self._invoke(flags, rest, bt, from_tty)

    def _invoke(self, flags, rest, bt, from_tty):
        global _plain_suppressed
        ascending = any(len(f) > 1 and "-ascending".startswith(f) for f in flags)
        flags = [f for f in flags if not (len(f) > 1 and "-ascending".startswith(f))]
        threads = []
        for inf in gdb.inferiors():
            threads += [(inf.num, t.num) for t in inf.threads()]
        threads.sort(reverse=not ascending)
        error = None
        try:
            if threads:
                ids = " ".join("{}.{}".format(i, t) for i, t in threads)
                # (Threads as they are, even with a parked thread selected.)
                _plain_suppressed += 1
                try:
                    gdb.execute("thread apply {} {}{}".format(
                        ids, "".join(f + " " for f in flags), rest), from_tty=from_tty)
                finally:
                    _plain_suppressed -= 1
        except gdb.error as e:
            # Show the parked threads even so (e.g. after a running thread in
            # non-stop mode), but not after a quit (e.g. 'q' at the pager),
            # which arrives as KeyboardInterrupt and so isn't caught here.
            error = e
        if bt:
            _dump_parked_threads(rest)
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
        return _complete_command(rest, word)


def _complete_command(rest, word):
    """Completions of command line `rest` (whose last word is `word`), as
    gdb's `thread apply 1 COMMAND` completes it."""
    prefix = "thread apply 1 "
    try:
        out = _execute("complete " + prefix + rest)
    except gdb.error:
        return []
    start = len(rest) - len(word)
    return [line[len(prefix):][start:] for line in out.splitlines()
            if line.startswith(prefix)]


# --------------------------------------------------------------------------
# realm-thread
# --------------------------------------------------------------------------

def _split_target(arg):
    """(user thread expression or "all", command) of `realm-thread apply`'s
    arguments: the expression is one word, or a parenthesized one."""
    arg = arg.lstrip()
    if arg.startswith("("):
        depth = 0
        for i, c in enumerate(arg):
            depth += {"(": 1, ")": -1}.get(c, 0)
            if depth == 0:
                return arg[:i + 1], arg[i + 1:].lstrip()
        return arg, ""
    parts = arg.split(None, 1)
    return parts[0] if parts else "", parts[1] if len(parts) > 1 else ""


def _lookup_user_thread(expr):
    """The parked-or-not Realm::UserThread* that `expr` evaluates to, in the
    selected inferior.  Raises gdb.GdbError saying what is wrong."""
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
        for num, addrs in (_user_threads_elsewhere() if len(gdb.inferiors()) > 1 else []):
            if addr in addrs:
                raise gdb.GdbError(
                    "realm-thread: {:#x} is a user thread of inferior {}: select it "
                    "('inferior {}') and rerun this command.".format(addr, num, num))
        raise gdb.GdbError("realm-thread: {:#x} is not a Realm user thread (run "
                           "'realm-thread' to list them).".format(addr))
    return ut


# --------------------------------------------------------------------------
# Selecting a parked thread (live processes)
# --------------------------------------------------------------------------
#
# In a live process, `realm-thread <ptr>` selects a parked user thread without
# changing anything in the program: realm-gdb only remembers it, and the frame
# selected on it.  `rt COMMAND` then runs COMMAND on it in a register window
# (see "Register windows"), with that frame selected, and remembers the frame
# COMMAND left selected; so `rt frame 3` then `rt print VAR` work as `frame 3`
# then `print VAR` would on a running thread.  Selections nest: `realm-thread
# back` goes back to the one before.  The prompt shows the latest one.
#
# A selection is dropped when its program runs (it can't be checked from
# gdb.events.cont whether the parked thread moved meanwhile), and checked
# before each `rt` against the registry: the user thread must still be
# parked, with the saved context it had.  (One that ran and parked again with
# an identical context isn't noticed; nor was it on the other approach.)
#
# A selection is a dict:
#   inferior, pid:  its inferior and process
#   user_thread:    the UserThread*, as an int
#   regs:           unwind_regs_from_ctx() of it as it was selected
#   level:          the frame selected on it

_selections = globals().get("_selections", [])

# The prompt hook realm-gdb installed while a thread is selected, and the one
# it replaced (called first).
_prompt_hooks = globals().get("_prompt_hooks", {"ours": None, "theirs": None})

# The reminder that realm-gdb's prompt hook puts in front of the prompt.  gdb
# makes the prompt a hook returns its `prompt` setting, so the prompt the
# hook gets next time has the reminder already, as `show prompt` does
# meanwhile; once nothing is selected, the setting is put back without it.
_PROMPT_REMINDER = re.compile(r"^(?:\[realm-thread 0x[0-9a-f]+\] )+")


def _selection():
    """The latest selection in the selected inferior, if any."""
    try:
        inferior = gdb.selected_inferior()
    except gdb.error:
        return None
    for sel in reversed(_selections):
        if sel["inferior"] == inferior.num and sel["pid"] == inferior.pid:
            return sel
    return None


def _drop_selections(which, why=None):
    dropped = [sel for sel in _selections if which(sel)]
    _selections[:] = [sel for sel in _selections if not which(sel)]
    if dropped and why:
        print("realm-thread: {}; no longer selecting user thread{} {}.".format(
            why, "s" if len(dropped) > 1 else "",
            ", ".join("{:#x}".format(sel["user_thread"]) for sel in dropped)))
    if not _selections:
        _remove_prompt_hook()


def _drop_running_selections(event):
    """From gdb.events.cont: the selected inferior's program runs (or every
    inferior's, with schedule-multiple), so Realm may resume a selected
    parked thread.  (Not Inferior.threads(), which gdb 12 can't refresh in
    the middle of resuming.)"""
    thread = getattr(event, "inferior_thread", None)
    if _parameter("schedule-multiple", False):
        which = lambda sel: True
    else:
        num = thread.inferior.num if thread is not None else gdb.selected_inferior().num
        which = lambda sel: sel["inferior"] == num
    _drop_selections(which, "the program is running")


def _check_selection(sel, sps):
    """The selected user thread, if it is still parked as it was selected;
    else drop the selection and raise gdb.GdbError saying why."""
    try:
        ut = _find_user_thread(sel["user_thread"])
    except gdb.error as e:
        raise gdb.GdbError("realm-thread: cannot read the Realm user thread "
                           "registry: {}".format(e))
    why = None
    if ut is None:
        why = "is no longer a Realm user thread"
    else:
        problem = _parked_problem(ut, sps)
        if problem is not None:
            why = "can't be shown: " + problem
        else:
            try:
                if _arch.unwind_regs_from_ctx(ut) != sel["regs"]:
                    why = "has run since it was selected"
            except gdb.error as e:
                why = "can't be read: {}".format(e)
    if why is not None:
        _drop_selections(lambda s: s is sel)
        raise gdb.GdbError("realm-thread: user thread {:#x} {}; no longer selecting "
                           "it.".format(sel["user_thread"], why))
    return ut


def _realm_prompt(current):
    """The prompt (gdb.prompt_hook) while a parked thread is selected: the
    user's, after a reminder of the selection."""
    prompt = current = _PROMPT_REMINDER.sub("", current)
    theirs = _prompt_hooks["theirs"]
    try:
        if theirs is not None:
            got = theirs(current)
            if got is not None:
                prompt = got
        sel = _selection()
        if sel is not None:
            prompt = "[realm-thread {:#x}] {}".format(sel["user_thread"], prompt)
    except Exception:
        _debug_exc()
    return prompt


def _install_prompt_hook():
    if _prompt_hooks["ours"] is not None and gdb.prompt_hook is _prompt_hooks["ours"]:
        return
    namespace = globals()
    _prompt_hooks["theirs"] = gdb.prompt_hook
    _prompt_hooks["ours"] = lambda current: namespace["_realm_prompt"](current)
    gdb.prompt_hook = _prompt_hooks["ours"]


def _remove_prompt_hook():
    """Put the prompt hook back as it was, unless something else has
    replaced realm-gdb's meanwhile, and the prompt without the reminder."""
    if _prompt_hooks["ours"] is not None and gdb.prompt_hook is _prompt_hooks["ours"]:
        gdb.prompt_hook = _prompt_hooks["theirs"]
    _prompt_hooks["ours"] = None
    _prompt_hooks["theirs"] = None
    current = _parameter("prompt")
    if current and _PROMPT_REMINDER.match(current):
        # (Escaped: `set prompt` would drop spaces at either end, and takes
        # backslashes as escapes.)
        prompt = "".join("\\{:03o}".format(ord(c)) if c in " \t\\" or c < " " or c == "\x7f"
                         else c for c in _PROMPT_REMINDER.sub("", current))
        try:
            gdb.execute("set prompt " + prompt, to_string=True)
        except gdb.error:
            _debug_exc()


def _live_checks(what):
    """The checks before showing a parked thread in a live process: a
    target, a showable state, a stopped selected thread."""
    if _window is not None:
        raise gdb.GdbError("realm-thread: can't {} from inside 'rt', 'realm-thread "
                           "apply' or 'thread apply all bt'.".format(what))
    if not _has_target():
        raise gdb.GdbError("realm-thread: no process or core file.")
    how = _how_shown()
    if isinstance(how, _NotShown):
        raise gdb.GdbError("realm-thread: can't show parked user threads: " +
                           how.sentence())
    thread = gdb.selected_thread()
    if thread is None or not thread.is_stopped():
        raise gdb.GdbError("realm-thread: the selected thread must be stopped.")
    return how


# --------------------------------------------------------------------------
# Plain commands on a selected parked thread (live processes)
# --------------------------------------------------------------------------
#
# While a parked thread is selected (in a console gdb session), a command
# that works with the selected thread or frame (_SHOWING_COMMANDS: `bt`,
# `frame N`, `up`, `print VAR`, `info locals`, `x`, `list`, `set var`,
# `break`, ...) shows it,
# as `rt COMMAND` would: its hook-COMMAND opens a register window showing
# it, with its selected frame selected, and its hookpost-COMMAND remembers
# the frame left selected and closes the window.  So gdb runs its own
# command, as it was typed, on the parked thread's registers, and prints as
# it goes (not collected, as `rt` collects it; so gdb's pager is off in such
# a window).  A command that would act on the selected thread as it really
# is (_ACTING_COMMANDS: `step`, `next`, `finish`, `return`, `watch`,
# `python`, `thread apply`, ...) is refused, saying to run `realm-thread
# back` first, or `rt COMMAND` where that works.  Two end the
# selection instead (all of the inferior's), since the user means to leave
# it: `continue`, before it runs (the program running would end it anyway:
# see "Selecting a parked thread"); and `thread N`, once it has switched to
# another thread.  realm-gdb's own `thread apply all` shows threads as they
# are.
#
# gdb doesn't run a hookpost when its command fails (an error, Ctrl-C), so a
# window can be left open.  The next top-level prompt closes it
# (gdb.events.before_prompt), and so does gdb exiting (gdb.events.
# gdb_exiting, gdb >= 12), and so does the next hooked command, or realm-gdb
# command, that isn't part of the failed one.  Python's stack tells which:
# a command run as part of another (by a pretty-printer's gdb.execute, say)
# is called from newer Python frames than the other was, the other's still
# where they were; one run after it isn't.  A command run as part of another
# without Python, as `frame apply` and `thread apply` run theirs, couldn't
# be told from one run after it: so those two are refused (realm-gdb's own
# `thread apply all` runs gdb's `thread apply` with plain commands showing
# nothing).  Until a window left open is closed, its guards refuse what
# would let the program run or let go of it.
#
# Where a window left open might not be closed in time, plain commands
# don't show a selected parked thread, nor are any refused, and realm-gdb
# says so on selecting it:
#   - MI and DAP sessions: MI's own commands run no hooks, so an IDE's
#     -exec-continue would resume the program with a window left open;
#   - batch mode without gdb.events.gdb_exiting (gdb < 12): gdb would let
#     go of the program at the end with a window left open;
#   - once a second UI is open (`new-ui`), whose MI commands could come
#     with a window left open.
# They are found from gdb's own command line (/proc/self/cmdline, or ps)
# and, on Linux, the terminals gdb has open, and from hook-new-ui.
#
# Python's gdb.execute_mi runs no hooks either: realm-gdb has it close a
# failed command's window first (_install_execute_mi).  Inside a plain
# command (a pretty-printer's), it can resume the program with the window
# open, as inside `rt` (see "Register windows").

# The window a plain command opened, if any: a dict
#   window:   the _Window
#   command:  the command
#   outer:    the Python frames it was run from (_outer_frames), and
#   lasti:    the innermost one's f_lasti then
#   shown:    _show_in_window's
#   input:    its _NoTerminalInput
_plain = globals().get("_plain")

# Plain commands on realm-gdb's behalf that show threads as they are (in
# `thread apply all`).
_plain_suppressed = 0

# Why plain commands don't show a selected parked thread in this session,
# if they don't (see _plain_off).
_plain_off_why = globals().get("_plain_off_why")


def _gdb_args():
    """gdb's own command line (Linux: /proc; else ps), or None."""
    try:
        with open("/proc/self/cmdline", "rb") as f:
            return [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
    except OSError:
        pass
    try:
        import subprocess
        out = subprocess.run(["ps", "-ww", "-o", "args=", "-p", str(os.getpid())],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=10).stdout
        # (Arguments with spaces come apart, which can only find too much.)
        return out.decode("utf-8", "replace").split() or None
    except Exception:
        return None


# gdb's options that take a value.
_GDB_VALUE_OPTIONS = ("ex", "eval-command", "iex", "init-eval-command", "x", "command",
                      "ix", "init-command", "p", "pid", "c", "core", "cd", "d",
                      "directory", "s", "symbols", "e", "exec", "se", "data-directory",
                      "D", "tty", "t", "b", "l")


def _session_problem():
    """Why plain commands can't show a selected parked thread in this gdb
    session, judged from gdb's command line; or None."""
    args = _gdb_args()
    if args is None:
        return "realm-gdb can't tell whether this is an MI session"
    interpreter, batch = None, False
    i = 1
    while i < len(args):
        a = args[i]
        if a in ("--args", "-args", "--") or not a.startswith("-"):
            if a.startswith("new-ui"):  # (a command, for -ex)
                return "a second UI is open ('new-ui')"
            if a in ("--args", "-args", "--"):
                break
            i += 1
            continue
        name, eq, value = a.lstrip("-").partition("=")
        if name in ("i", "interpreter"):
            interpreter = value if eq else (args[i + 1] if i + 1 < len(args) else "")
        elif name in ("batch", "batch-silent"):
            batch = True
        elif name in ("ex", "eval-command", "iex", "init-eval-command") and eq:
            if value.lstrip().startswith("new-ui"):
                return "a second UI is open ('new-ui')"
        if name in _GDB_VALUE_OPTIONS and not eq:
            i += 1
            if name in ("ex", "eval-command", "iex", "init-eval-command") and \
                    i < len(args) and args[i].lstrip().startswith("new-ui"):
                return "a second UI is open ('new-ui')"
        i += 1
    if interpreter and not interpreter.startswith(("console", "tui")):
        return "this is {} {} session".format(
            "a" if interpreter.startswith("dap") else "an",
            re.match(r"[A-Za-z]*", interpreter).group(0).upper() or interpreter)
    if batch and not hasattr(gdb.events, "gdb_exiting"):
        return ("in batch mode, this gdb (< 12) could let go of the program at the "
                "end with a parked thread's registers in place")
    return None


def _other_terminal():
    """Whether gdb has a terminal open besides its standard input, output
    and error: a second UI ('new-ui'), most likely.  (Linux only.  Not a
    pseudo-terminal's master side, which a UI isn't on.)"""
    try:
        fds = sorted(int(fd) for fd in os.listdir("/proc/self/fd"))
    except (OSError, ValueError):
        return False
    paths = {}
    for fd in fds:
        try:
            paths[fd] = os.readlink("/proc/self/fd/{}".format(fd))
        except OSError:
            pass
    own = set(paths.get(fd) for fd in (0, 1, 2))
    for fd, path in paths.items():
        try:
            if (fd > 2 and path.startswith("/dev/") and path not in own and
                    path != "/dev/ptmx" and os.isatty(fd)):
                return True
        except OSError:
            pass
    return False


def _plain_off(why):
    """From now on, plain commands don't show a selected parked thread."""
    global _plain_off_why
    if _plain_off_why is not None:
        return
    _plain_off_why = why
    if _selections:
        print("realm-gdb: {}, so from now on plain commands show threads as they "
              "really are: use 'rt COMMAND' for a selected parked Realm user "
              "thread.".format(why[0].upper() + why[1:]))


def _plain_shows():
    """The selection that plain commands show now, if any."""
    if _plain_off_why is not None or _plain_suppressed or _window is not None:
        return None
    if not _has_target():
        return None
    return _selection()


def _refuse_acting(cmd):
    """hook-CMD of a command that would act on the selected thread as it
    really is: refuse it while plain commands show a selected parked
    thread."""
    sel = _plain_shows()
    if sel is None:
        return
    if cmd in ("thread apply", "frame apply") or cmd.startswith("frame apply ") or \
            cmd == "faas":
        why = "realm-gdb can't show the user thread to the commands it runs"
    else:
        why = "it would act on thread {} as it really is".format(
            gdb.selected_thread().num)
    raise gdb.GdbError(
        "realm-thread: refusing '{}' while user thread {:#x} is selected: {}.  Run "
        "'realm-thread back' first{}.".format(
            cmd, sel["user_thread"], why,
            ", or 'rt {} ...' to run it on the user thread".format(cmd)
            if cmd in _hooked.get("rt", ()) else
            ", or use 'thread apply all'" if cmd == "thread apply" else ""))


def _plain_before(cmd, outer):
    """hook-CMD of a command that looks at the selected thread or frame
    (run from Python frames `outer`): open a window showing a selected
    parked thread for it."""
    global _plain
    if _plain is not None:
        return  # (CMD is part of the command that opened it)
    sel = _plain_shows()
    if sel is None:
        return
    try:
        _live_checks("show a parked user thread")
        sps, _ = _host_sps()
        ut = _check_selection(sel, sps)
        window = _Window(via=cmd)
        window.__enter__()
    except (gdb.error, gdb.GdbError) as e:
        raise gdb.GdbError("{}  ('{}' was not run.)".format(e, cmd))
    except Exception as e:  # (shouldn't be: but CMD mustn't run on the host)
        _debug_exc()
        raise gdb.GdbError("realm-thread: can't show user thread {:#x}: {}  ('{}' "
                           "was not run.)".format(sel["user_thread"], e, cmd))
    try:
        entry = {"window": window, "command": cmd, "outer": outer,
                 "lasti": outer[0].f_lasti if outer else None, "input": None}
        entry["shown"] = _show_in_window(window, ut, sel)
        entry["input"] = _NoTerminalInput(cmd, window).__enter__()
        _plain = entry
    except BaseException as e:
        if entry["input"] is not None:
            entry["input"].__exit__(None, None, None)
        window.__exit__(None, None, None)
        if isinstance(e, (_Unshowable, gdb.error)):
            raise gdb.GdbError("realm-thread: can't show user thread {:#x}: {}.  ('{}' "
                               "was not run.)".format(int(ut), e, cmd))
        raise


def _same_frames(entry, outer):
    mine = entry["outer"]
    return (len(outer) == len(mine) and all(a is b for a, b in zip(outer, mine)) and
            (not mine or mine[0].f_lasti == entry["lasti"]))


def _inside(entry, outer):
    """Whether code running from Python frames `outer` is part of the command
    that opened `entry`'s window (see above)."""
    mine = entry["outer"]
    if len(outer) <= len(mine):
        return False
    if not all(a is b for a, b in zip(outer[len(outer) - len(mine):], mine)):
        return False
    return not mine or mine[0].f_lasti == entry["lasti"]


def _plain_after(cmd, outer):
    """hookpost-CMD: close the window CMD opened."""
    entry = _plain
    if entry is not None and entry["command"] == cmd and _same_frames(entry, outer):
        _close_plain(entry)


def _close_stale(outer):
    """Close a plain command's window if the code now running (from Python
    frames `outer`) isn't part of that command, which then failed.  (Not in
    the middle of a function call, which only a command can be making.)"""
    entry = _plain
    if entry is not None and not _infcall_depth and not _inside(entry, outer):
        _close_plain(entry)


def _install_execute_mi():
    """Python's gdb.execute_mi (gdb >= 14) runs MI commands, which run no
    hooks: have it close a failed plain command's window first, as a hooked
    command would (a script that catches the failure, then resumes the
    program with -exec-continue, would otherwise resume it with the window
    open)."""
    real = getattr(gdb, "execute_mi", None)
    if real is None or getattr(real, "_realm_gdb", False):
        return
    namespace = globals()

    def execute_mi(*args, **kwargs):
        try:
            namespace["_close_stale"](namespace["_outer_frames"](sys._getframe().f_back))
        except Exception:
            _debug_exc()
        return real(*args, **kwargs)
    execute_mi._realm_gdb = True
    execute_mi.__doc__ = real.__doc__
    gdb.execute_mi = execute_mi


def _close_plain(entry):
    """Close the window a plain command opened, and print its notes."""
    global _plain
    if _plain is entry:
        _plain = None
    window, notes = entry["window"], []
    try:
        if entry["input"] is not None:
            entry["input"].__exit__(None, None, None)
        if not window.resumed:
            try:
                notes = _window_done(window, entry["shown"], entry["command"])
            except gdb.error:
                _debug_exc()
    finally:
        window.__exit__(None, None, None)
    for note in notes + _call_notes(window):
        print(note)


class RealmThreadRun(gdb.Command):
    """Run a command on the selected parked Realm user thread.

    Usage: rt COMMAND

    Runs COMMAND (`bt`, `frame 3`, `up`, `info locals`, `print VAR`, ...) on
    the parked user thread that `realm-thread <ptr>` selected, as `realm-thread
    apply <ptr> COMMAND` would, but with the frame last selected on it (by an
    earlier `rt frame N`, `rt up`, ...) selected first.  In a console gdb
    session, plain commands that work with the selected frame do the same
    without `rt` (see `help realm-thread`); `rt` also runs those refused
    there that work on a parked thread (`rt python ...`, `rt frame apply
    ...`), and works in MI and DAP sessions too.  Commands
    that would let the program run or let go of it are refused: run them
    without `rt` (after `realm-thread back`, for those refused there too),
    which then no longer selects the parked thread, since Realm may resume
    it.
    Commands that start another program and look at nothing of the parked
    thread's (`shell`, `!`, `make`, `edit`) run as they would without `rt`;
    `pipe COMMAND | SHELL` runs COMMAND on it, then SHELL, and `edit *EXPR`
    works out EXPR on it, then edits (as `realm-thread apply` does too).
    In a core file, where `realm-thread <ptr>` keeps the parked thread shown,
    `rt COMMAND` is just COMMAND."""

    def __init__(self):
        super(RealmThreadRun, self).__init__("rt", gdb.COMMAND_STACK)

    def invoke(self, arg, from_tty):
        _close_stale(_outer_frames(sys._getframe().f_back))
        _printing_over()
        command = arg.strip()
        if not command:
            raise gdb.GdbError("usage: rt COMMAND (after 'realm-thread <ptr>')")
        if _has_target() and _window is None and _how_shown() == "view":
            view = _current_view()
            if view is None or view["temporary"]:
                raise gdb.GdbError("realm-thread: no parked user thread is shown on "
                                   "this thread: run 'realm-thread <ptr>' first.")
            gdb.execute(command, from_tty=from_tty)
            return
        _live_checks("run 'rt'")
        sel = _selection()
        if sel is None:
            elsewhere = sorted(set(s["inferior"] for s in _selections))
            raise gdb.GdbError(
                "realm-thread: no parked user thread is selected{}: run "
                "'realm-thread <ptr>' first, or 'realm-thread apply <ptr> COMMAND'"
                ".".format(" in this inferior (inferior {} has one)".format(
                    elsewhere[-1]) if elsewhere else ""))
        with _KeepSelection():
            sps, _ = _host_sps()
            ut = _check_selection(sel, sps)
            error = _apply([(ut, None)], command, header=False, selection=sel)
        if error is not None:
            raise gdb.GdbError(str(error))

    def complete(self, text, word):
        if word is None:
            word = re.split(r"[ \t]", text)[-1]
        return _complete_command(text, word)


class RealmThread(gdb.Command):
    """realm-thread [apply <ptr>|all COMMAND | <ptr> | back]

    With no argument, list the Realm user threads of the selected inferior.

    `realm-thread apply <ptr> COMMAND` runs COMMAND on the parked user thread
    at <ptr> (a UserThread pointer: one word, or an expression in
    parentheses), for example `realm-thread apply <ptr> bt full`, or (gdb >=
    8.3) `realm-thread apply <ptr> frame apply level 3 print VAR`.
    `realm-thread apply all COMMAND` runs it on each parked user thread.

    In a live process, the parked thread's saved registers are put in place
    on the selected thread for the duration of COMMAND, and the thread's own
    put back before it returns: so its stack starts at frame #0, and gdb
    unwinds, prints and evaluates it as usual, function calls included
    (pretty-printers' too).  Thread-local variables are the selected
    thread's.  Meanwhile, commands that would let the program run or let go
    of it (`continue`, `step`, `finish`, `detach`, `quit`, ...) are refused,
    function calls hold the other threads and can't stop at breakpoints, and
    a call that gets a signal (or on gdb >= 15 runs for 10 seconds) is
    unwound.  Not in non-stop mode, or while the program runs or is recorded
    (realm-thread says how to see them then).

    `realm-thread <ptr>` selects the parked user thread at <ptr>, until
    `realm-thread back` (or `restore`); selections nest.  In a live process
    this changes nothing in the program: `rt COMMAND` then runs COMMAND on
    it, as `realm-thread apply <ptr> COMMAND` does, with the frame selected
    by an earlier `rt frame N` (or `rt up`, ...) selected first.  The prompt
    shows the selection.  It is dropped once the program runs.

    In a console gdb session, plain commands that work with the selected
    frame do the same as `rt COMMAND` meanwhile: `bt`, `frame N`, `up`,
    `down`, `print VAR`, `call`, `info locals`, `info frame`, `info
    registers`, `list`, `x`, `ptype`, `display`, `disassemble`, `set var`,
    `break`, `tbreak`, `clear`, ...  (gdb's pager is off in them, and its
    queries take their default answers: so `break` makes no pending
    breakpoint unless `set breakpoint pending on`.)  `set var` and `print X =
    V` change a variable that lives in a register only until the command is
    over (realm-gdb says so), and a thread-local variable is the selected
    thread's own.  Commands that would act on the selected thread as it
    really is are refused meanwhile, saying to run `realm-thread back` first:
    `step`, `next`, `finish`, `return`, `until`, `jump`, `signal`, `watch`,
    `python`, `thread apply` and `frame apply` among them.  `rt` runs some
    of those on the parked thread (`rt python`, `rt frame apply`).  `continue` ends the selection,
    then runs; `thread N` switches to thread N, then ends it (unless N is
    the thread already selected).  `thread apply all bt` shows threads as
    they are.
    Commands written in Python (other than these) see threads as they are.
    Not in MI or DAP sessions (an IDE's own MI commands can't be followed),
    nor after a second UI is opened (`new-ui`), nor in batch mode with gdb <
    12: there `rt` is needed, and nothing is refused.

    In a core file, a Python unwinder shows the parked thread's stack on the
    selected thread instead: frame #0 stays the thread's own frame (which
    backtraces show as a `<parked Realm user thread ...>` line), and the
    user thread's stack starts at frame #1.  There `realm-thread <ptr>` keeps
    it shown, so that `bt`, `frame N`, `info locals` and `print VAR` (and
    `rt COMMAND`, which is the same there) operate on it."""

    def __init__(self):
        super(RealmThread, self).__init__("realm-thread", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        _close_stale(_outer_frames(sys._getframe().f_back))
        _printing_over()
        arg = arg.strip()
        if arg and _window is not None:
            raise gdb.GdbError("realm-thread: can't change what is shown from inside "
                               "'rt', 'realm-thread apply' or 'thread apply all bt'.")
        if not arg:
            self._list()
        elif arg in ("back", "restore"):
            self.dont_repeat()  # (an empty line mustn't go back again)
            self._back()
        elif arg.split()[0] == "apply":
            self._apply(arg[len("apply"):])
        else:
            self.dont_repeat()  # (nor select it again)
            self._show_thread(arg)

    def complete(self, text, word):
        # (gdb >= 13 first asks with word=None, to find the word breaks.)
        if word is None:
            word = re.split(r"[ \t]", text)[-1]
        words = text.split()
        if len(words) <= 1 and not text[-1:].isspace():
            return [w for w in ("apply", "back", "restore") if w.startswith(word)]
        if words[0] != "apply":
            return []
        after = text.lstrip()[len("apply"):].lstrip()
        target, rest = _split_target(after)
        if not rest and not after[len(target):][:1].isspace():
            return [w for w in ("all",) if w.startswith(word)]
        return _complete_command(rest, word)

    def _list(self):
        sps = {}
        others_running = False
        if _has_target():
            try:
                sps, others_running = _host_sps()
            except gdb.error:
                pass
        inferior = gdb.selected_inferior().num
        shown = {}
        for stack in _views.values():
            for view in stack:
                if view["inferior"] == inferior and not view["temporary"]:
                    shown.setdefault(view["user_thread"], []).append(view["thread_num"])
        selected = set(sel["user_thread"] for sel in _selections
                       if sel["inferior"] == inferior
                       and sel["pid"] == gdb.selected_inferior().pid)
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
                    elif int(ut) in selected:
                        current = _selection()
                        state = "parked  (selected{})".format(
                            "" if current is not None and current["user_thread"] == int(ut)
                            else " earlier")
                    else:
                        state = "parked"
                    print("  {:#x}  {}".format(int(ut), state))
        except _RegistryLoop as e:
            print("realm-thread: {}; the list above may be incomplete.".format(e))
        except gdb.error as e:
            if not others_running:
                raise gdb.GdbError("realm-thread: cannot read the Realm user "
                                   "thread registry: {}".format(e))
            print("realm-thread: cannot read the Realm user thread registry while "
                  "the program is running.  To list them, stop it ('interrupt') "
                  "and rerun this command.")
        else:
            if not found:
                print("realm-thread: no Realm user threads in the registry.")
            elif _has_target():
                how = _how_shown()
                if isinstance(how, _NotShown):
                    print("realm-thread: parked ones can't be shown now.  " +
                          how.sentence())
        if len(gdb.inferiors()) > 1:
            for num, addrs in _user_threads_elsewhere():
                print("realm-thread: inferior {} has {} Realm user thread{} too: "
                      "select it ('inferior {}') to list {}.".format(
                          num, len(addrs), "s" if len(addrs) > 1 else "", num,
                          "them" if len(addrs) > 1 else "it"))

    def _apply(self, arg):
        target, command = _split_target(arg)
        command = command.strip()
        if command.startswith("-- ") or command == "--":
            command = command[2:].strip()
        if not target or not command:
            raise gdb.GdbError("usage: realm-thread apply <ptr>|all COMMAND")
        if _window is not None:
            raise gdb.GdbError("realm-thread: can't nest 'realm-thread apply'.")
        if not _has_target():
            raise gdb.GdbError("realm-thread: no process or core file.")
        how = _how_shown()
        if isinstance(how, _NotShown):
            raise gdb.GdbError("realm-thread: can't show parked user threads: " +
                               how.sentence())
        thread = gdb.selected_thread()
        if thread is None or not thread.is_stopped():
            raise gdb.GdbError("realm-thread: the selected thread must be stopped.")
        with _KeepSelection():
            sps, _ = _host_sps()
            if target == "all":
                try:
                    parked = [ut for ut in _iter_user_threads()
                              if _is_parked(ut) or not _state_readable(ut)]
                except gdb.error as e:
                    raise gdb.GdbError("realm-thread: cannot read the Realm user "
                                       "thread registry: {}".format(e))
                if not parked:
                    print("realm-thread: no parked Realm user threads.")
                    return
                error = _apply([(ut, _parked_problem(ut, sps)) for ut in parked],
                               command, header=True)
            else:
                ut = _lookup_user_thread(target)
                problem = _parked_problem(ut, sps)
                if problem is not None:
                    raise gdb.GdbError("realm-thread: can't show user thread {:#x}: "
                                       "{}.".format(int(ut), problem))
                error = _apply([(ut, None)], command, header=False)
        if error is not None:
            raise gdb.GdbError(str(error))

    def _show_thread(self, expr):
        if not _has_target():
            raise gdb.GdbError("realm-thread: no process or core file.")
        how = _how_shown()
        if how == "window":
            return self._select(expr)
        if isinstance(how, _NotShown):
            raise gdb.GdbError("realm-thread: can't show a parked user thread: " +
                               how.sentence())
        thread = gdb.selected_thread()
        if thread is None or not thread.is_stopped():
            raise gdb.GdbError("realm-thread: the selected thread must be stopped.")
        ut = _lookup_user_thread(expr)
        sps, _ = _host_sps()
        problem = _parked_problem(ut, sps)
        if problem is not None:
            raise gdb.GdbError("realm-thread: can't show user thread {:#x}: {}.".format(
                int(ut), problem))
        try:
            view = _make_view(ut)
        except gdb.error as e:
            raise gdb.GdbError("realm-thread: cannot read the context of user "
                               "thread {:#x}: {}".format(int(ut), e))
        _views.setdefault(view["thread"], []).append(view)
        _refresh_frames("parked")
        if not _shown(view):
            _remove_view(view)
            _refresh_frames("parked")
            limit = _parameter("backtrace limit")
            raise gdb.GdbError("realm-thread: cannot unwind user thread {:#x}{}.".format(
                int(ut), "" if not limit or limit > view["host_level"] + 1 else
                " within 'set backtrace limit {}'".format(limit)))
        print("realm-thread: showing user thread {:#x} on thread {}: its stack "
              "starts at frame #1 (#0 is {}).".format(
                  int(ut), thread.num, _frame0_words(thread.num)))
        print("Use 'bt', 'frame N', 'info locals', 'print VAR' as normal.  "
              "Run 'realm-thread back' to undo.")

    def _select(self, expr):
        """`realm-thread <ptr>` in a live process (see "Selecting a parked
        thread")."""
        _live_checks("select a parked user thread")
        with _KeepSelection():
            ut = _lookup_user_thread(expr)
            sps, _ = _host_sps()
            problem = _parked_problem(ut, sps)
            if problem is not None:
                raise gdb.GdbError("realm-thread: can't show user thread {:#x}: "
                                   "{}.".format(int(ut), problem))
            inferior = gdb.selected_inferior()
            sel = {"inferior": inferior.num, "pid": inferior.pid,
                   "user_thread": int(ut), "regs": _arch.unwind_regs_from_ctx(ut),
                   "level": 0}
        if _plain_off_why is None and _other_terminal():
            _plain_off("a second UI may be open (gdb has another terminal open)")
        _selections.append(sel)
        _install_prompt_hook()
        host = gdb.selected_thread().num
        if _plain_off_why is None:
            print("realm-thread: selected user thread {:#x}.  Until 'realm-thread "
                  "back', commands that work with the selected frame ('bt', 'frame N', "
                  "'up', 'print VAR', 'info locals', 'x', 'set var', 'break', ...) show "
                  "it, with its saved registers in place on thread {} for each command "
                  "only; nothing in the program changes in between.  Commands that "
                  "would act on thread {} as it really is ('step', 'finish', 'return', "
                  "'watch', 'python', ...) are refused meanwhile; 'continue', or "
                  "'thread N' to another thread, ends the selection.".format(
                      int(ut), host, host))
        else:
            print("realm-thread: selected user thread {:#x}.  'rt COMMAND' runs "
                  "COMMAND on it (e.g. 'rt bt', 'rt frame 2', 'rt print VAR'), with "
                  "its saved registers in place on the selected thread for that "
                  "command only; nothing in the program changes in between.  Plain "
                  "commands show thread {} as it really is: {}.  Run 'realm-thread "
                  "back' to undo.".format(int(ut), host, _plain_off_why))
        with _KeepSelection():
            error = _apply([(ut, None)], "frame", header=False, selection=sel)
        if error is not None:
            print("realm-thread: {}".format(error))

    def _back(self):
        if _unrestored:
            _retry_unrestored()
            return
        view = _current_view()
        if view is None:
            sel = _selection()
            if sel is None:
                print("realm-thread: nothing to restore.")
                return
            _drop_selections(lambda s: s is sel)
            sel = _selection()
            if sel is not None:
                print("realm-thread: back to user thread {:#x}.".format(sel["user_thread"]))
            else:
                print("realm-thread: no parked user thread is selected now.")
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

  realm-thread apply <ptr>|all COMMAND
                            Run COMMAND on a parked user thread (or each
                            of them): e.g. 'bt full', or 'frame apply
                            level 3 print VAR'.

  realm-thread <ptr>        Select a parked user thread: then plain 'bt',
  realm-thread back         'frame N', 'up', 'print VAR', 'info locals', ...
                            show it, keeping the frame you select, and
                            'step', 'finish', 'return', 'watch', ... are
                            refused, until 'back' undoes the selection
                            ('continue' and 'thread N' undo it too).
  rt COMMAND                Run COMMAND on the selected parked thread
                            (also in MI and DAP sessions, where plain
                            commands show threads as they really are).

  realm-help                Reprint this banner.

In a live process, a parked user thread is shown by putting its saved
registers into the selected thread for the duration of one command, then
putting that thread's own back.  Between commands nothing in the program
is changed.  During such a command, commands that would let the program
run or let go of it (continue, step, detach, quit, ...) are refused,
function calls hold the other threads, can't stop at breakpoints, and
are unwound if they get a signal.  (So if gdb itself were killed in the
middle of one, an attached program would be left corrupted.)  Parked
user threads can't be shown in non-stop mode or while the program runs.

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
    RealmThreadRun()
    RealmThreadGuard()
    RealmHelp()
    _install_hooks()
    _install_execute_mi()
    if _plain_off_why is None:
        _plain_off_why = _session_problem()
    # Registering the unwinder rebuilds the frames; if this is a reload
    # while a parked thread is shown, keep the user's frame selected.
    _reload_level = _selected_level() if _current_view() is not None else None
    _install_event_handlers()
    if _reload_level is not None:
        _refresh_frames(_reload_level)
    # (Loaded again: an installed copy sourced on top of librealm's embedded
    # one, say, or the other way round; the banner once is enough.)
    if globals().get("_banner_printed"):
        objfile = gdb.current_objfile()
        print("realm-gdb.py loaded again{}.".format(
            " (the copy in {})".format(objfile.filename) if objfile is not None else ""))
    else:
        print(_HELP_BANNER)
    _banner_printed = True
else:
    # Other platforms / architectures: do nothing rather than provide a
    # half-working command.
    pass
