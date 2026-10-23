# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Auto-loaded by gdb (via the .debug_gdb_scripts section in librealm) to make
# Realm's user-level threads visible to the debugger.
#
# REALM_GDB.md, alongside this script, describes the design: how parked user
# threads are shown, why, how it is tested, and the script's known limits.
# IMPORTANT: keep the design doc up to date if you modify this script!
#
# Provides `thread apply all bt` with the stack of every PARKED user thread
# after gdb's own (a RUNNING one is its host pthread's stack already), and
# `realm-thread` (to list, select, and `apply` commands to, parked user
# threads), `rt` and `realm-help`, which summarizes them (see also `help
# realm-thread`).
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


# (A setting this gdb doesn't have.  Kept if the script is loaded again: a
# window's _Temporarily settings, made before, compare with it.)
_NO_SETTING = globals().get("_NO_SETTING", object())


def _parameter(name, default=None):
    try:
        return _gdb_parameter()(name)
    except (gdb.error, RuntimeError):
        return default


def _has_parameter(name):
    return _parameter(name, _NO_SETTING) is not _NO_SETTING


def _gdb_parameter():
    """gdb.parameter itself, not realm-gdb's wrapper (_install_parameter)."""
    return getattr(gdb.parameter, "_realm_gdb_real", gdb.parameter)


# An MI stream record (console, target or log output) on a line of its own.
_MI_STREAM_RECORD = re.compile(r'[~@&]"((?:[^"\\]|\\.)*)"\n?$', re.S)
_C_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "a": "\a",
              "b": "\b", "e": "\x1b"}


# realm-gdb's own commands in progress (see _internal).  (Kept if the script
# is loaded again, as the state below is: in the middle of one, perhaps.)
_internal_depth = globals().get("_internal_depth", 0)
_internal_commands = globals().get("_internal_commands", [])


def _internal(command):
    """_execute(command) for realm-gdb itself, where its hooks leave it
    alone (`return` to pop a dummy frame in a window, say)."""
    global _internal_depth
    _internal_depth += 1
    _internal_commands.append(command)
    try:
        return _execute(command)
    finally:
        _internal_commands.pop()
        _internal_depth -= 1


def _execute(command, from_tty=False):
    """gdb.execute(command, from_tty, to_string=True): the command's output,
    as the CLI would print it.  (Under MI, gdb 18 returns some of it, line
    by line, as MI stream records: ~"text\\n".)  Without the echo of the
    command itself that `set trace-commands on` puts first (see
    _untraced)."""
    return _cli_output(command, gdb.execute(command, from_tty, to_string=True))


def _cli_output(command, out):
    if re.search(r'(?:^|\n)[~@&]"', out):
        out = "".join(_mi_unquote(line) for line in out.splitlines(True))
    return _untraced(command, out) if out.startswith("+") else out


def _say(text):
    """Print one of realm-gdb's messages: a WARNING to gdb's stderr, which
    `-batch-silent` leaves (it silences only stdout), and the others as
    print() does."""
    if "WARNING:" in text:
        gdb.write(text + "\n", gdb.STDERR)
    else:
        print(text)


def _untraced(command, out):
    """`out` without the echo of `command` that gdb puts first with `set
    trace-commands on` (or in `source -v`): "+COMMAND", a "+" more for each
    command it runs in.  Else realm-gdb would read it as the output's first
    line: a user's hook kept with its header for a line ("User command
    ..."), which then failed, `quit` and `continue` with it."""
    lines = out.splitlines(True)
    mine = set(line.strip() for line in command.splitlines())
    while lines:
        m = re.match(r"\++(.*)$", lines[0].rstrip("\r\n"))
        if m is None or m.group(1).strip() not in mine:
            break
        lines.pop(0)
    return "".join(lines)


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
    # An extended register (x87, SSE, AVX state) that a window writes once,
    # and back, to see whether gdb can (see _calls_unwritable); None: not
    # checked.
    probe_reg = None
    # Registers that a parked user thread shares with whichever thread shows
    # it, which gdb may need to unwind its frames; a core-file view passes
    # them through from the host frame (those that exist).
    thread_regs = ()
    # Bits of registers that the kernel keeps itself, which can't be written
    # back as they were read: {name: mask}.
    kernel_bits = {}

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


class _X86_64(_Arch):
    pc_reg = "rip"
    sp_reg = "rsp"
    probe_reg = "xmm15.v2_int64[0]"
    # getcontext()/swapcontext() save the callee-saved registers, the
    # argument registers, and RIP/RSP as of the return from the call.  (No
    # SSE register is callee-saved in the SysV ABI.)  The core-file unwinder
    # gets the callee-saved ones, RSP and RIP.
    _CTX = ("r8", "r9", "r12", "r13", "r14", "r15", "rdi", "rsi", "rbp",
            "rbx", "rdx", "rcx", "rsp", "rip")
    _UNWIND = ("r12", "r13", "r14", "r15", "rbp", "rbx", "rsp", "rip")

    @classmethod
    def context_regs(cls, ut):
        regs = cls._saved(ut, cls._CTX)
        regs["rax"] = 0
        return regs, []

    @classmethod
    def unwind_regs_from_ctx(cls, ut):
        return cls._saved(ut, cls._UNWIND)


class _X86_64Linux(_X86_64):
    fallback_regs = _X86_64_GP_REGS + _X86_64_OTHER_REGS + ("orig_rax",)
    # glibc x86-64 ucontext gregs[] indices (see /usr/include/sys/ucontext.h).
    _GREGS = {"r8": 0, "r9": 1, "r12": 4, "r13": 5, "r14": 6, "r15": 7, "rdi": 8,
              "rsi": 9, "rbp": 10, "rbx": 11, "rdx": 12, "rcx": 14, "rsp": 15, "rip": 16}

    @staticmethod
    def matches():
        return (platform.system() == "Linux"
                and platform.machine() in ("x86_64", "amd64"))

    @staticmethod
    def _saved(ut, names):
        gregs = ut["ctx"]["uc_mcontext"]["gregs"]
        return {name: int(gregs[_X86_64Linux._GREGS[name]]) & _U64 for name in names}


class _X86_64FreeBSD(_X86_64):
    fallback_regs = _X86_64_GP_REGS + _X86_64_OTHER_REGS

    @staticmethod
    def matches():
        return (platform.system() == "FreeBSD"
                and platform.machine() in ("x86_64", "amd64"))

    @staticmethod
    def _saved(ut, names):
        # (FreeBSD's mcontext_t, sys/x86/include/ucontext.h, has named
        # fields: mc_rip, mc_rsp, ...)
        mc = ut["ctx"]["uc_mcontext"]
        return {name: int(mc["mc_" + name]) & _U64 for name in names}


# gdb needs the pointer-authentication masks to unwind return addresses
# signed with -mbranch-protection=pac-ret; tpidr, fpsr/fpcr and vg are
# per-thread too.  (Linux on aarch64 is tested in CI, FreeBSD on aarch64
# not yet.)
_AARCH64_THREAD_REGS = ("pauth_dmask", "pauth_cmask", "tpidr", "fpsr", "fpcr", "vg")
_AARCH64_GP_REGS = tuple("x{}".format(i) for i in range(31)) + ("sp", "pc", "cpsr")
# d8..d15 (the low halves of v8..v15) are callee-saved.  With SVE, v8..v15
# are the low 128 bits of z8..z15, and writing v8 clears the rest of z8, so
# keep z8..z15 too.
_AARCH64_FALLBACK = (_AARCH64_GP_REGS + tuple("v{}".format(i) for i in range(32))
                     + tuple("z{}".format(i) for i in range(32))
                     + tuple("p{}".format(i) for i in range(16)) + ("ffr", "fpsr", "fpcr"))


def _read_bytes(addr, size):
    return bytes(gdb.selected_inferior().read_memory(addr, size))


def _vregs_note(why):
    return ("its saved d8-d15 can't be shown ({}): a variable kept in one of them shows "
            "the selected thread's value instead".format(why))


class _AArch64Linux(_Arch):
    pc_reg = "pc"
    sp_reg = "sp"
    # (PSTATE.SS, of a single step that a signal cut short, as when a host is
    # stopped before it runs: see _stop_before_run.)
    kernel_bits = {"cpsr": 1 << 21}
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
    kernel_bits = _AArch64Linux.kernel_bits
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


# gdb's error reading the registry's head, when it last couldn't.
_registry_error = None
# The programs whose registry gdb has read (see _program_key).
_registry_read = globals().get("_registry_read", set())


def _program_key():
    """The selected inferior's process and program: (number, pid, file)."""
    inf = gdb.selected_inferior()  # (Whose program space is the current one.)
    return (inf.num, inf.pid, gdb.current_progspace().filename)


def _registry_head():
    global _registry_error
    try:
        with _CLanguage():
            head = gdb.parse_and_eval("Realm::user_thread_registry_head")
    except gdb.error as e:
        _registry_error = str(e)
        return None
    _registry_error = None
    try:
        if gdb.selected_inferior().pid:  # (Not a program's file, with no process.)
            _registry_read.add(_program_key())
    except Exception:
        _debug_exc()
    return head


def _registry_dropped():
    """Whether the registry's symbol is gone from a process whose registry
    gdb has read: `nosharedlibrary` dropped librealm's symbols, say."""
    try:
        return _registry_error is not None and \
            _registry_error.startswith('No symbol "') and _program_key() in _registry_read
    except Exception:
        _debug_exc()
        return False


# What _check_selection and _no_registry_words say of that.
_REGISTRY_DROPPED = ("gdb no longer has librealm's symbols, which 'nosharedlibrary' "
                     "drops ('sharedlibrary' reads them again)")


def _no_registry_words(no_realm=True):
    """Why no user thread is listed, if the registry couldn't be read:
    librealm's symbols dropped (see _registry_dropped), none of Realm's
    (another program, or librealm not loaded yet; None then unless
    `no_realm`), or its type unknown, say, or no symbols read at all."""
    if _registry_dropped():
        return "realm-thread: no user thread can be found: {}.".format(_REGISTRY_DROPPED)
    if _registry_error is None:
        return None
    if _registry_error.startswith('No symbol "') or not _registry_symbol():
        if not no_realm:
            return None
        return ("realm-thread: inferior {}'s program has no Realm user thread registry: "
                "it doesn't use Realm, or librealm isn't loaded yet.".format(
                    gdb.selected_inferior().num))
    return ("realm-thread: can't read the Realm user thread registry "
            "(Realm::user_thread_registry_head: {}), so no user thread can be found: "
            "librealm's debug information may be missing (gdb's --readnever, or a "
            "librealm built without it).".format(
                re.sub(r'\.?\s+Use the "file" command\.?$', "", _registry_error.rstrip("."))))


def _registry_symbol():
    """Whether gdb knows of the registry's head at all, without its debug
    information ("in a file compiled without debugging", with --readnever,
    say)."""
    try:
        _internal("info address Realm::user_thread_registry_head")  # (Hooked.)
        return True
    except gdb.error:
        return False


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
            raise _RegistryLoop("the Realm user thread registry is corrupt: it loops "
                                "back to {:#x}".format(int(cur)))
        seen.add(int(cur))
        yield cur
        cur = cur["registry_next"]


def _listed_parked(ut):
    """Whether user thread `ut` goes among the parked ones: no host pthread
    is executing it (a running one is already in its host's backtrace), or
    its state can't be read (listed with the error, rather than left out:
    see _parked_problem)."""
    try:
        return not bool(ut["running"])
    except gdb.error:
        return True


# Numbers for user threads, as gdb numbers its threads, in each process:
# (inferior number, pid) -> {UserThread*: N}, given in the order they were
# made (the registry's, newest first, reversed) as `realm-thread` first
# lists or looks them up, and not given again.  (Kept if the script is
# loaded again.)
_user_thread_numbers = globals().get("_user_thread_numbers", {})


def _numbered(addrs):
    """{UserThread*: number} of the selected inferior's, numbering those of
    `addrs` (the registry's, newest first) not seen before."""
    inferior = gdb.selected_inferior()
    numbers = _user_thread_numbers.setdefault((inferior.num, inferior.pid), {})
    for addr in reversed(addrs):
        if addr not in numbers:
            numbers[addr] = len(numbers) + 1
    return numbers


def _ut_number(addr):
    """The number `realm-thread` lists user thread `addr` by in the selected
    inferior, once it has one (see _numbered), else None."""
    try:
        inferior = gdb.selected_inferior()
        return _user_thread_numbers.get((inferior.num, inferior.pid), {}).get(addr)
    except Exception:
        return None


def _ut_words(addr, capital=False):
    """"user thread N (0x...)", as gdb names a thread "thread N (Thread
    0x... (LWP n))" (or "user thread 0x...", with no number yet)."""
    n = _ut_number(addr)
    words = ("user thread {} ({:#x})".format(n, addr) if n is not None else
             "user thread {:#x}".format(addr))
    return words[0].upper() + words[1:] if capital else words


def _parked_threads():
    """The selected inferior's parked user threads (see _listed_parked:
    one whose state can't be read is among them, rather than left out), and
    why the registry couldn't be read past them, or None.  Raises gdb.error
    if none could be read.  (Numbers them all: see _ut_words.)"""
    parked, addrs = [], []
    try:
        for ut in _iter_user_threads():
            addrs.append(int(ut))
            if _listed_parked(ut):
                parked.append(ut)
    except gdb.error as e:
        if not parked:
            raise
        return parked, (str(e) if isinstance(e, _RegistryLoop) else
                        "cannot read the rest of the Realm user thread registry: {}".format(e))
    finally:
        _numbered(addrs)
    return parked, None


def _registry_addrs():
    """The addresses of the selected inferior's user threads, the registry's
    (newest first), as far as it can be read."""
    addrs = []
    try:
        for ut in _iter_user_threads():
            addrs.append(int(ut))
    except (gdb.error, _RegistryLoop):
        pass
    return addrs


def _uts_words(addrs):
    """"user thread N (0x...)", or "user threads N (0x...) and M (0x...)"."""
    words = [_ut_words(addr)[len("user thread "):] for addr in addrs]
    if len(words) == 1:
        return "user thread " + words[0]
    return "user threads {} and {}".format(", ".join(words[:-1]), words[-1])


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
    selected = _selected_thread()
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

def _selected_thread():
    """gdb.selected_thread(), or None: also where gdb has lost its selected
    thread, after the program died in a function call (SystemError)."""
    try:
        return gdb.selected_thread()
    except (gdb.error, SystemError):
        return None


def _stopped_thread():
    """The selected thread, which must be stopped (else gdb.GdbError)."""
    thread = _selected_thread()
    if thread is None:
        raise gdb.GdbError("realm-thread: no thread is selected (see 'info threads').")
    if not thread.is_stopped():
        raise gdb.GdbError("realm-thread: the selected thread must be stopped.")
    return thread


def _selected_thread_num():
    t = _selected_thread()
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
    view, if any, shown again), or the selected inferior, if it has no
    thread.  Some gdbs (10, on core files at least) lose the selected frame
    when Inferior.threads() refreshes the thread list, and commands that
    switch threads without unwinding them leave a view unarmed on gdb <
    11."""

    def __enter__(self):
        self.inferior = gdb.selected_inferior().num
        self.skipped = ()
        try:
            self.thread = _selected_thread()
            self.level = _selected_level() if self.thread is not None else 0
        except gdb.error:
            self.thread = None
        if self.thread is not None and self.thread.is_stopped():
            try:
                self.skipped = _skipped_inline(gdb.newest_frame())
            except gdb.error:
                pass
        if self.thread is None and gdb.selected_inferior().threads():
            # (gdb has lost its selected thread: see _selected_thread.  A
            # thread switched to now couldn't be switched back from, and if
            # its registers can't be read, every command would then fail.)
            raise gdb.GdbError("realm-gdb: no thread is selected (see 'info threads').")
        return self

    def __exit__(self, *exc):
        try:
            if self.thread is not None and self.thread.is_valid():
                if _selected_thread() != self.thread:
                    self.thread.switch()
                if self.thread.is_stopped():
                    # (The same frame, if gdb now shows the inlined functions
                    # the thread stopped at the start of: see _skipped_inline.)
                    entered = self.skipped and _inline_entered(self.skipped)
                    _refresh_frames(self.level + (len(self.skipped) if entered else 0))
            elif gdb.selected_inferior().num != self.inferior and \
                    self.inferior in [inf.num for inf in gdb.inferiors()]:
                # (Selecting another inferior's thread, to read memory
                # through, left it: `inferior N` is the only way back to one
                # with no thread.)
                _internal("inferior {}".format(self.inferior))
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

    def sentence(self, capital=False):
        """Why and what to do: after a prefix's colon, or (`capital`) on its
        own."""
        cap = capital and not re.match(r"(realm-)?gdb\b", self.why)  # (Names keep their case.)
        return "{}.  To see them, {}.".format(
            self.why[0].upper() + self.why[1:] if cap else self.why, self.advice)

    def error(self):
        return gdb.GdbError("realm-thread: can't show parked user threads: " +
                            self.sentence())


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
    if _unguarded:
        return _NotShown(
            "realm-gdb couldn't install its guards on commands that would let the "
            "program run or let go of it ({}), which must be refused while one is "
            "shown".format(", ".join("'{}'".format(c) for c in _unguarded)),
            "load realm-gdb.py again ('source' it, from src/realm in Realm's sources), "
            "or write a core file ('gcore FILE') and load it in another gdb")
    if _parameter("non-stop", False):
        return _NotShown(
            "the program is in non-stop mode, where other threads could run, and resume "
            "a parked one, while it is shown on a stopped thread",
            "debug the program with 'set non-stop off', or write a core file ('gcore "
            "FILE') and load it in another gdb")
    running = [inf.num for inf in gdb.inferiors()
               if any(t.is_valid() and not t.is_stopped() for t in inf.threads())]
    if gdb.selected_inferior().num in running:
        return _NotShown("the program is running", "stop it ('interrupt') and rerun "
                         "this command")
    if running:
        # (A window's function calls disable breakpoints, which are every
        # inferior's: that program would run past them.)
        return _NotShown("inferior {}'s program is running".format(running[0]),
                         "stop it ('interrupt -a') and rerun this command")
    if _recording():
        return _NotShown(
            "the program is being recorded (record/replay), and showing them "
            "changes a thread's registers, which would spoil the execution log",
            "stop recording ('record stop') or write a core file ('gcore FILE') "
            "and load it in another gdb")
    if _parameter("may-write-registers") is False:
        return _NotShown(
            "showing one writes a thread's registers (and puts them back), and "
            "'may-write-registers' is off",
            "run 'set may-write-registers on', or write a core file ('gcore FILE') and "
            "load it in another gdb")
    takers = _sigint_takers()
    if takers:
        # (gdb 17 and later die of one taken by a thread other than gdb's
        # main one, 8.2 and 9.2 of one that lands as Python runs.)
        harm = ("would kill gdb" if _SIGINT_MAIN_THREAD_ONLY or not _SIGINT_IN_PYTHON_SAFE
                else "would quit gdb in the middle, which can leave the program unable to "
                "run on")
        many = len(takers) > 1
        return _NotShown(
            "{} that Python started in gdb (LWP{} {}) {} block SIGINT, and a Ctrl-C that "
            "{} took while a parked thread was shown {}".format(
                "{} threads".format(len(takers)) if many else "a thread",
                "s" if many else "", ", ".join(takers), "don't" if many else "doesn't",
                "one of them" if many else "it", harm),
            "have {} block SIGINT (signal.pthread_sigmask){}, or write a core file ('gcore "
            "FILE') and load it in another gdb".format(
                "those threads" if many else "that thread",
                " or start {} as {}".format("them" if many else "it",
                                            "gdb.Threads" if many else "a gdb.Thread")
                if hasattr(gdb, "Thread") else ""))
    return "window"


def _parked_problem(ut, sps):
    """Why parked user thread `ut` can't be shown (a string), or None.  Its
    saved stack pointer must be inside its own stack: a function call made
    while it is shown in a register window runs just below it."""
    try:
        if bool(ut["running"]):
            host = _host_thread_of(ut)
            return "it is running{}".format(
                " on thread {}".format(host.num) if host is not None else "")
        executing = _executing_on(ut, sps)
        if executing is not None:
            return "it is switching contexts on thread {}".format(executing)
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
#     User breakpoints of every kind are disabled while a call runs, so that
#     it can't stop at one.  The user's settings are put back afterwards.  A
#     call left stopped nonetheless is popped before the next parked thread
#     is shown (see _Window.pop_calls).
#   - Output isn't paged, and queries are answered (`confirm off`), inside a
#     window: gdb's pager and queries wait for input in a nested event loop,
#     where another UI (a second MI `new-ui`, say) could run a command.
#   - SIGTERM and SIGHUP to gdb are held off until the window is closed (gdb
#     acts on them by killing or detaching from the program and exiting, in
#     the middle of a command if need be), and so are those whose default
#     action, which gdb leaves them, ends gdb (SIGUSR1, SIGALRM, SIGXCPU,
#     ...).  So is Ctrl-C (SIGINT), but for the window's function calls,
#     which it must be able to interrupt, and a plain command typed at a
#     terminal, which gdb runs itself (see _block_sigint).
#   - gdb.events.gdb_exiting (gdb >= 12) closes any window left open, and so
#     does Python's exit (a SystemExit, which ends gdb 14 and older at once).
#   - A command in the window that selects another thread (`thread N`) or
#     inferior leaves the window on the host: the next parked thread is
#     shown there again, and the host's own registers put back there.
#   - Function calls are refused in windows if gdb can't write the extended
#     registers (x87, SSE, AVX state), which a call changes: gdb 13 and older
#     can't where the kernel's XSAVE area is bigger than their buffer (with
#     AMX on).  Then gdb itself can't put back what a call changed (it gives
#     up at the first of them, leaving $orig_rax changed too), nor can the
#     window, and the program mustn't run until it is (see _unrestored).
#     The first window that would allow calls checks it, once per process
#     (_calls_unwritable): once gdb has started a call it is too late, as
#     stopping one cleanly needs those writes too.
#
# What can't be covered: gdb itself dying (kill -9, a crash) with a window
# open would leave an attached process running the parked thread's context
# on the host, and so corrupt it: on Linux the kernel kills it instead (see
# _ExitKill), as it does a process gdb started; elsewhere (a remote target)
# it runs on.  Windows are short (milliseconds, but up to _CALL_TIMEOUT
# seconds for one making a function call), so this needs gdb to die in the
# middle of one.  And a resume that bypasses the CLI commands' hooks and the
# MI commands that realm-gdb refuses (see _install_execute_mi) can't be
# refused: gdb.events.cont comes too late to put the registers back (gdb's
# proceed() holds on to the thread's registers across it, and writing one
# frees them all).  The host is stopped before it runs instead, with a
# SIGINT of its own, where it can be (see _stop_before_run).

# The open window, if any; windows whose host's registers could not be put
# back (the program must not run until they are).  (Kept if the script is
# loaded again: a window a failed plain command left open is still open.)
_window = globals().get("_window")
_unrestored = globals().get("_unrestored", [])

# Per process ((inferior number, pid)): gdb's error writing an extended
# register, or None if it can (see _calls_unwritable).
_extended_unwritable = globals().get("_extended_unwritable", {})

# Inferior calls in progress (in or out of a window); the user breakpoints
# disabled for the one in a window: (number, gdb.Breakpoint or None if
# Python can't see it, enable count) (see _disable_breakpoints).  (Kept if
# the script is loaded again, as by a library a call in a window loads.)
_infcall_depth = globals().get("_infcall_depth", 0)
_disabled_breakpoints = globals().get("_disabled_breakpoints", [])
# For each call counted in _infcall_depth, oldest first: (ptid, the thread's
# newest frame as it began, or None for one that can't be in progress), to
# tell its dummy frame by (see _call_running).
_call_starts = globals().get("_call_starts", [])

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


def _same_reg(name, a, b):
    """Whether register `name`'s contents `a` and `b` (see _raw) are the
    same, but for the bits the kernel keeps itself (see _Arch.kernel_bits)."""
    mask = _arch.kernel_bits.get(name)
    if mask and a is not None and b is not None:
        a, b = [int.from_bytes(x, "little") if isinstance(x, bytes) else x for x in (a, b)]
        return a & ~mask == b & ~mask
    return a == b


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


class _Temporarily(object):
    """Set gdb's setting `name` to `value` while realm-gdb does something of
    its own, if `when` (given the setting's value) says it must, and put the
    old value back afterwards, unless something has changed it since."""

    def __init__(self, name, value, when=None):
        self.name, self.value, self.when = name, value, when
        self.saved = _NO_SETTING

    def __enter__(self):
        old = _get_setting(self.name)
        if old is not _NO_SETTING and _setting_words(old) != self.value and \
                (self.when is None or self.when(old)):
            _set(self.name, self.value)
            self.saved, self.mine = old, _get_setting(self.name)
        return self

    def __exit__(self, *exc):
        if self.saved is not _NO_SETTING and _get_setting(self.name) == self.mine:
            _set(self.name, _setting_words(self.saved))
        return False


def _current_language():
    try:
        out = _execute("show language")
    except gdb.error:
        return None
    # ('The current source language is "auto; currently c++".')
    m = re.search(r'"(?:auto; currently )?([^"]*)"', out)
    return m.group(1) if m is not None else None


def _CLanguage():
    """Parse realm-gdb's own expressions and `set var`s as C++ (`0x...`,
    `Realm::...`, `=`), whatever the current language: gdb parses them in
    it, and some read them otherwise without an error (Pascal and Ada take
    `set var $r12 = 0x10` for a comparison, and write $r12 some other
    value), or fail on `Realm::`.  Switches to C++ only when it isn't the
    language already (`set language pascal`, say, perhaps by a command in a
    window, before the window puts the host's registers back)."""
    return _Temporarily("language", "c++",
                        lambda old: _current_language() not in (None, "c++"))


def _BigValues():
    """Let gdb read values as big as any register while realm-gdb reads or
    writes them, whatever `max-value-size` (16 bytes, say, set by a command
    in a window, which a 64-byte AVX-512 register is over): else no
    register could be put back.  Changes it only when it is below gdb's
    default (64 KiB)."""
    return _Temporarily("max-value-size", "unlimited",
                        lambda old: isinstance(old, int) and 0 <= old < 65536)


def _AllFrames():
    """Lift `set backtrace limit` while realm-gdb walks a stack itself,
    which Frame.older() stops at, as `bt` does: a limit of 1 would hide the
    dummy frame (frame #1) of a call that a window must pop, say.  (Not for
    the frames a user's command shows.)"""
    return _Temporarily("backtrace limit", "unlimited",
                        lambda old: isinstance(old, int) and old > 0)


def _keep_in_gdb(saved, tag):
    """Without gdb.set_convenience_variable (gdb < 8.3): keep a copy of each
    register in `saved` that can't be written back as an integer (vector
    and x87 registers) in a gdb convenience variable, for _write_reg; `tag`
    tells the window's copies from those of windows that couldn't put them
    back (see _unrestored)."""
    if hasattr(gdb, "set_convenience_variable"):
        return
    gdb.newest_frame().select()
    with _CLanguage():
        for name, value in saved.items():
            if not _int_writable(value):
                _internal("set var $_realm_gdb_saved_{1}_{0} = ${0}".format(name, tag))


def _write_reg(name, value, tag=0):
    """Set the selected thread's register `name` to `value`: a gdb.Value of
    its type (as read), an int, or (lo, hi) 64-bit halves of an aarch64
    vector register.  (A register is written in the selected frame: in an
    outer frame that would be the frame's saved copy, on the stack.  And as
    realm-gdb's own: a `set var` of the user's shows a selected parked
    thread.)"""
    gdb.newest_frame().select()
    with _CLanguage():
        if isinstance(value, tuple):
            for i, half in enumerate(value):
                _internal("set var ${}.d.u[{}] = {:#x}".format(name, i, half))
        elif isinstance(value, gdb.Value) and hasattr(gdb, "set_convenience_variable"):
            gdb.set_convenience_variable("_realm_gdb_reg", value)
            _internal("set var ${} = $_realm_gdb_reg".format(name))
        elif isinstance(value, gdb.Value) and not _int_writable(value):
            # (A window's own copy: see _keep_in_gdb.)
            _internal("set var ${0} = $_realm_gdb_saved_{1}_{0}".format(name, tag))
        else:
            _internal("set var ${} = {:#x}".format(name, int(value) & _U64))


def _raw_regs(frame):
    """{name: _raw(value)} of `frame`'s registers that a window keeps."""
    return {n: _raw(v) for n, v in _read_regs(frame, _snapshot_names(frame)).items()}


def _put_back(saved, errors=None, tag=0):
    """Write every register in `saved` (kept with `tag`: see _keep_in_gdb)
    that differs back into the selected thread, until they all match.
    Returns what still differs ([] when done), with gdb's error writing each
    in `errors`, a dict, if given.  An interruption (Ctrl-C) is retried, so
    that a half-restored thread is never left."""
    with _BigValues():
        return _put_back_values(saved, errors, tag)


def _put_back_values(saved, errors, tag):
    """_put_back's work."""
    left = list(saved)
    for _ in range(10):
        try:
            frame = gdb.newest_frame()
            left = []
            for n, v in saved.items():
                # (One that can't be read, or written, mustn't keep the rest
                # from being put back: a register the target's description
                # has lost, say.)
                try:
                    if not _same_reg(n, _raw(frame.read_register(n)), _raw(v)):
                        left.append(n)
                except Exception as e:
                    left.append(n)
                    if errors is not None:
                        errors[n] = str(e) or type(e).__name__
            if not left:
                return []
            if _parameter("may-write-registers") is False:
                # (Turned off by a command in the window, after the window
                # wrote its registers: allowed again until they are back.)
                with _Temporarily("may-write-registers", "on"):
                    return _put_back_values(saved, errors, tag)
            for name in left:
                # (One that can't be written mustn't keep the rest from
                # being: e.g. gdb 13 and older can't write the SSE registers
                # where the kernel's XSAVE area is bigger than gdb's own
                # buffer (with AMX), and $orig_rax must still be put back.)
                try:
                    _write_reg(name, saved[name], tag)
                except Exception as e:
                    _debug_exc()
                    if errors is not None:
                        errors[name] = str(e) or type(e).__name__
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
    """`set NAME VALUE` (`maint set ...` for "maint NAME"), retried if
    interrupted (Ctrl-C)."""
    command = ("maint set " + name[len("maint "):] if name.startswith("maint ")
               else "set " + name)
    number = re.match(r"-?\d+$", value) is not None
    if number:
        # (In hex, as C++: read the same in any input radix and language.)
        value = "{:#x}".format(int(value))
    for attempt in range(10):
        try:
            if number:
                with _CLanguage():
                    _internal("{} {}".format(command, value))
            else:
                _internal("{} {}".format(command, value))
            return
        except KeyboardInterrupt:
            if attempt == 9:
                raise


def _number_words(n):
    """Integer `n` as gdb reads a number now, in its input radix (`set
    input-radix 16`): for realm-gdb's own numbers in a user's command, which
    runs in the user's language.  (Digits that would start with a letter
    get C's `0x`.)"""
    radix = _parameter("input-radix") or 10
    if radix == 10:
        return str(n)
    digits, m = "", abs(n)
    while True:
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"[m % radix] + digits
        m //= radix
        if not m:
            break
    if not digits[0].isdigit():
        digits = "{:#x}".format(abs(n))  # (Not a name; nor `0a`, octal in C.)
    return ("-" if n < 0 else "") + digits


def _get_setting(name):
    """Setting `name`'s value, as gdb.parameter has it; for "maint NAME",
    which gdb.parameter can't read, the word `maint show NAME` gives.
    _NO_SETTING if this gdb has no such setting.  Retried if interrupted, as
    _set is."""
    for attempt in range(10):
        try:
            return _read_setting(name)
        except KeyboardInterrupt:
            if attempt == 9:
                raise


def _read_setting(name):
    if not name.startswith("maint "):
        return _parameter(name, _NO_SETTING)
    try:
        out = _execute("maint show " + name[len("maint "):])
    except gdb.error:
        return _NO_SETTING
    m = re.search(r' is "?([\w-]+)"?\.\s*$', out)
    return m.group(1) if m else _NO_SETTING


class _CallSettings(object):
    """The settings function calls made in a window need (see "Register
    windows"), and the user's, put back afterwards.  The user's
    may-call-functions is kept too, for set_calls.  With paging=False, gdb's
    pager is off too (in a window whose command prints as it goes)."""

    # Not put back before each call (for_call): they aren't about calls.
    _NOT_FOR_CALL = ("confirm", "pagination", "multiple-symbols")

    def __init__(self, paging=True):
        settings = [("unwindonsignal", "on"), ("unwind-on-terminating-exception", "on"),
                    ("confirm", "off"), ("multiple-symbols", "all")]
        if not paging:
            settings.append(("pagination", "off"))
        self.window = [_Temporarily(name, value) for name, value in settings]
        # (Kept lower if the user has them lower; None is unlimited.)
        self.window += [_Temporarily(name, str(_CALL_TIMEOUT),
                                     lambda old: old is None or old > _CALL_TIMEOUT)
                        for name in ("direct-call-timeout", "indirect-call-timeout")]
        self.window.append(_Temporarily("unwind-on-timeout", "on"))
        # gdb's internal problems end gdb with `confirm off`, whatever the
        # problem: in a window they end the command instead.  (And gdb 12 to
        # 16 die of the backtrace they print of one in a command whose output
        # is collected, as rt's is: "Recursive internal problem".  Before gdb
        # 12 there is no backtrace, nor setting.)
        self.window += [_Temporarily("maint " + problem, "no") for problem in (
            "internal-error quit", "internal-error corefile", "internal-warning quit",
            "internal-warning corefile", "demangler-warning quit")]
        self.window += [_Temporarily("maint " + problem, "off") for problem in (
            "internal-error backtrace", "internal-warning backtrace")]
        # Those that matter only while a call runs: set as one begins, and put
        # back as it ends (for_call, end_call), so that a command in the
        # window sees the user's (the gdb manual's idiom saves
        # `$_gdb_setting("scheduler-locking")` and sets it back afterwards).
        # (Not unwindonsignal and unwind-on-timeout, which gdb reads once
        # the call is over.)
        self.call_settings = [("scheduler-locking", "on"), ("stop-on-solib-events", "0"),
                              ("may-insert-breakpoints", "on"),
                              ("follow-fork-mode", "parent"), ("detach-on-fork", "on")]
        self.in_call = []
        self.at_open = {}
        # Those a `with` in progress around the command that opened the window
        # changed (see _with_hook), which it then overrides for its calls.
        self.with_changed = set()
        self.user_calls = _parameter("may-call-functions")
        self.calls = self.user_calls
        self.refusing = False

    def __enter__(self):
        try:
            if _with_settings is not None:
                for name in _WINDOW_SETTINGS:
                    was = _with_settings.get("call: " + name)
                    if was is not None and was != _setting_words(_get_setting(name)):
                        self.with_changed.add(name)
            for name, value in self.call_settings:
                self.at_open[name] = _setting_words(_get_setting(name))
            for setting in self.window:
                setting.__enter__()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def set_calls(self, allowed):
        """Allow function calls (as the user has it), or refuse them."""
        self.refusing = not allowed
        want = self.user_calls if allowed else False
        if self.user_calls is not None and want != self.calls:
            _set("may-call-functions", _setting_words(want))
            self.calls = want

    def for_call(self):
        """Just before a function call in the window: set what the call
        needs for while it runs (see end_call), and put back what it needs
        otherwise, if a command in the window changed it since (`with
        scheduler-locking off -- print f()`, a user-defined command that
        sets it, Python).  Returns what was put back, as "NAME VALUE", of
        what the window found otherwise."""
        put = []
        for name, value in self.call_settings:
            setting = _Temporarily(name, value).__enter__()
            self.in_call.append(setting)
            if setting.saved is not _NO_SETTING and (
                    _setting_words(setting.saved) != self.at_open.get(name) or
                    name in self.with_changed):
                put.append("{} {}".format(name, value))
        for setting in self.window:
            if setting.name in self._NOT_FOR_CALL or setting.name.startswith("maint "):
                continue
            now = _parameter(setting.name, _NO_SETTING)
            if now is not _NO_SETTING and (setting.when(now) if setting.when else
                                           _setting_words(now) != setting.value):
                _set(setting.name, setting.value)
                put.append("{} {}".format(setting.name, setting.value))
            elif setting.name in self.with_changed and setting.saved is not _NO_SETTING:
                put.append("{} {}".format(setting.name, setting.value))  # (At open.)
        return put

    def end_call(self):
        """As the window's function call ends (or a command whose call
        failed before it ran): put back what for_call set for it."""
        self._put_back(self.in_call)

    @staticmethod
    def _put_back(settings):
        """Put back each _Temporarily in list `settings`, newest first, that
        still has its value (else something changed it since, such as `with
        SETTING V -- COMMAND` putting it back as COMMAND, which opened the
        window, failed), dropping it from `settings` once it is."""
        while settings:
            try:
                settings[-1].__exit__(None, None, None)
            except gdb.error:
                _debug_exc()
            settings.pop()  # (Once it is back.)

    def calls_refused(self):
        """Whether the window refuses function calls, but a command in it
        turned may-call-functions back on.  (Not when it is the user who has
        them off, and the window allows them.)"""
        return self.refusing and bool(_parameter("may-call-functions"))

    def __exit__(self, *exc):
        try:
            try:
                self.end_call()
            finally:
                self.set_calls(True)
        finally:
            self._put_back(self.window)
        return False


class _HeldSignals(object):
    """Hold off the signals that would end gdb (see "Register windows"):
    SIGTERM and SIGHUP, which gdb handles by killing or detaching from the
    program and exiting, and those that gdb leaves to kill it (SIGUSR1,
    SIGALRM, SIGXCPU, ...), or to stop it (Ctrl-Z's SIGTSTP: gdb handles
    that only as it reads a line); and deliver one that came meanwhile once
    the window is closed.  The handler is swapped for one that just notes the
    signal (installed by Python, which runs it on gdb's main thread), and
    put back as it was (with libc's sigaction(), which Python's signal
    module can't do for a handler it didn't install).  Without ctypes, the
    signals are only blocked on gdb's main thread, which before gdb 10 is
    its only thread."""

    _SIGNALS = ("SIGTERM", "SIGHUP")
    # (Held only while their action is the default, which ends gdb; and the
    # real-time signals.  Not SIGILL, SIGTRAP or SIGSYS, which a fault in
    # gdb itself raises: a handler that only notes them would return to the
    # fault, again and again.)
    _FATAL_BY_DEFAULT = ("SIGUSR1", "SIGUSR2", "SIGALRM", "SIGVTALRM", "SIGPROF",
                         "SIGXCPU", "SIGXFSZ", "SIGPIPE", "SIGIO", "SIGPWR",
                         "SIGSTKFLT", "SIGLOST", "SIGEMT", "SIGTSTP")

    def __init__(self):
        self.came = []
        self.swapped = []
        self.blocked = None
        self.done = False
        # Whether one that came is delivered only by deliver(), once what the
        # window's command left is printed (see _Window.hold_sigint), rather
        # than as it closes; and whether it has been.
        self.deferred = self.delivered = False

    def __enter__(self):
        import signal
        sigs = [getattr(signal, s) for s in self._SIGNALS if hasattr(signal, s)]
        defaults = [getattr(signal, s) for s in self._FATAL_BY_DEFAULT
                    if hasattr(signal, s)]
        if sys.platform.startswith("linux") and not hasattr(signal, "SIGSTKFLT"):
            defaults.append(16)  # (SIGSTKFLT, which Python names from 3.11.)
        if hasattr(signal, "SIGRTMIN") and hasattr(signal, "SIGRTMAX"):
            defaults += range(int(signal.SIGRTMIN), int(signal.SIGRTMAX) + 1)
        try:
            try:
                import ctypes
                libc = ctypes.CDLL(None, use_errno=True)
                self._sigaction = libc.sigaction
                self._sigaction.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_void_p)
                for sig in sorted(set(sigs + defaults)):
                    # (struct sigaction is well under 256 bytes on every
                    # libc, and begins with the handler: SIG_DFL is 0.)
                    old = ctypes.create_string_buffer(256)
                    if self._sigaction(sig, None, old) != 0:
                        continue
                    if sig not in sigs and ctypes.c_void_p.from_buffer(old).value:
                        continue
                    self.swapped.append((sig, old))
                    signal.signal(sig, self._note)
            except Exception:
                _debug_exc()
                self._put_back()
                try:
                    self.blocked = signal.pthread_sigmask(
                        signal.SIG_BLOCK, sigs + [s for s in defaults
                                                  if signal.getsignal(s) == signal.SIG_DFL])
                except Exception:
                    self.blocked = None
        except BaseException:  # (Ctrl-C: don't leave any held.)
            self.__exit__(None, None, None)
            raise
        return self

    def _note(self, sig, frame):
        if self.done and (self.delivered or not self.deferred):
            # (Python runs a handler some time after the signal: this one
            # came as the window closed.  gdb's handler is back by now.)
            os.kill(os.getpid(), sig)
        else:
            self.came.append(sig)

    def _put_back(self):
        while self.swapped:
            sig, old = self.swapped[-1]
            try:
                self._sigaction(sig, old, None)
            except Exception:
                _debug_exc()
            self.swapped.pop()  # (Once it is back.)

    def __exit__(self, *exc):
        import signal
        try:
            self._put_back()
            if self.blocked is not None:
                blocked, self.blocked = self.blocked, None
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        finally:
            self.done = True
        if not self.deferred:
            self.deliver()
        return False

    def deliver(self):
        """Deliver the signals that came, gdb's handlers being back."""
        self.delivered = True
        came, self.came = sorted(set(self.came)), []
        for sig in came:
            os.kill(os.getpid(), sig)


def _sigint_is_blocked():
    """Whether Ctrl-C (SIGINT) is blocked on gdb's main thread, where Python
    runs (see _block_sigint)."""
    try:
        import signal
        return signal.SIGINT in signal.pthread_sigmask(signal.SIG_BLOCK, [])
    except Exception:
        return False


def _block_sigint(block):
    """Block Ctrl-C (SIGINT) on gdb's main thread, or let it through again
    (a pending one then arrives at once, and quits what runs next).  A
    window blocks it, but for the function calls made in it, which it must
    be able to interrupt, and a plain command typed at a terminal, which
    gdb runs itself (see _plain_before): else it can land in the middle of
    putting the host's registers back, or leave a window half closed; and
    on gdb 8.2 and 9.2 it can kill gdb whenever Python is running a
    command, as realm-gdb's own commands are run (see "Register windows").
    gdb's worker threads block it themselves.  Another thread that doesn't,
    one started from Python, would take it instead: where that kills gdb,
    no window is opened then (see _sigint_takers)."""
    if block == _sigint_is_blocked():
        return
    try:
        import signal
        signal.pthread_sigmask(signal.SIG_BLOCK if block else signal.SIG_UNBLOCK,
                               [signal.SIGINT])
    except Exception:
        _debug_exc()


# Whether Ctrl-C can land while Python runs a gdb command, or a function call
# made from Python, without killing gdb: gdb 8.2 and 9.2 (with Python 3.8)
# die of it, in _PyOS_InterruptOccurred, reached from gdb's own quit check
# without Python's lock; gdb 10 fixed that, and brought
# Architecture.registers with it.  Where it can't, Ctrl-C is held in windows
# but for calls a gdb command makes itself (see _block_sigint).
_SIGINT_IN_PYTHON_SAFE = hasattr(gdb.Architecture, "registers")

# Whether gdb takes a KeyboardInterrupt out of a Python command for a quit, as
# it does from gdb 9, which brought Value.format_string with it: gdb 8 makes
# it an error, saying "Python Exception <class 'KeyboardInterrupt'> Quit:".
_COMMANDS_QUIT = hasattr(gdb.Value, "format_string")

# Whether gdb dies of a Ctrl-C that a thread other than its main one takes:
# gdb 17 asserts, as gdb's handler sets the quit flag, that it runs on the
# main thread, and brought gdb.warning with it.
_SIGINT_MAIN_THREAD_ONLY = hasattr(gdb, "warning")


def _sigint_takers():
    """The threads of gdb's, but its main one, that don't block SIGINT
    (LWPs): while a window has Ctrl-C blocked on gdb's main thread (see
    _block_sigint), the kernel gives one to such a thread, and gdb's handler
    runs there.  gdb 8.2 and 9.2 then die of it as it lands while Python
    runs (see _SIGINT_IN_PYTHON_SAFE), gdb 17 and later wherever it lands
    (see _SIGINT_MAIN_THREAD_ONLY); and gdb 10 to 16 take it to quit what
    they are doing then, popping a function call's frame, say, which leaves
    the call's breakpoint in the parked thread's stack (see
    _Window._check_records).  gdb puts its handler back each time it runs
    Python or a command, so it can't be ignored instead.  gdb's own threads
    block it, as do Python's started once realm-gdb has loaded (see
    _install_thread_sigmask): one that doesn't was started from Python
    before that (or with _thread, or by a C extension).
    (Linux, which shows each thread's blocked signals in /proc.  Elsewhere a
    window opens all the same.)"""
    import signal
    bit = 1 << (int(signal.SIGINT) - 1)
    try:
        tasks = sorted((t for t in os.listdir("/proc/self/task") if t != str(os.getpid())),
                       key=int)
    except (OSError, ValueError):
        return []
    takers = []
    for task in tasks:
        try:
            with open("/proc/self/task/{}/status".format(task)) as f:
                for line in f:
                    if line.startswith("SigBlk:"):
                        if not int(line.split()[1], 16) & bit:
                            takers.append(task)
                        break
        except (OSError, ValueError, IndexError):
            pass  # (The thread ended.)
    return takers


# The held signals of windows closed with Ctrl-C held (see
# _Window.hold_sigint), to deliver with it (see _let_sigint).
_held_signals_after = globals().get("_held_signals_after", [])


def _let_sigint():
    """Let Ctrl-C through again once what a window's command left is
    printed (see _Window.hold_sigint): one pressed meanwhile lands then.
    (Not inside another window, which keeps it blocked.)"""
    if _window is None:
        _block_sigint(False)


def _let_held(idle=False):
    """Once what a window's command left is printed, and its notes: deliver
    the signals it held (see _HeldSignals), and let Ctrl-C through again
    (see _let_sigint).  When gdb is `idle`, about to prompt, a Ctrl-C held
    has nothing left to quit: it says "Quit", as gdb does for one at its
    prompt, rather than land in gdb's prompt event as a Python exception."""
    if _window is None:
        try:
            while _held_signals_after:
                _held_signals_after.pop().deliver()
        finally:
            if idle and _sigint_pending() and _take_sigint():
                gdb.write("Quit\n", gdb.STDERR)
            _block_sigint(False)


def _take_sigint():
    """Take a Ctrl-C pending (blocked) from gdb's main thread.  Returns
    whether it could (not without sigtimedwait, on macOS say)."""
    try:
        import signal
        return signal.sigtimedwait([signal.SIGINT], 0) is not None
    except Exception:
        _debug_exc()
        return False


def _sigint_pending():
    """Whether a Ctrl-C came while it was blocked."""
    try:
        import signal
        return signal.SIGINT in signal.sigpending()
    except Exception:
        return False


def _code_tree(code, into):
    """Code object `code` and those it defines, at any depth, into dict
    `into` by id (holding them, so that no id is used again)."""
    into[id(code)] = code
    for const in code.co_consts:
        if isinstance(const, type(code)):
            _code_tree(const, into)
    return into


# The code of realm-gdb's own functions (and of each earlier load's), told
# apart from other Python by that (see _on_python_sigint): they all run in
# gdb's __main__, from a script named "<string>", as the user's do.
_my_codes = _code_tree(sys._getframe().f_code, globals().get("_my_codes", {}))

# The code of the functions that gdb calls into realm-gdb by: _connect's
# event handlers, and realm-gdb-guard's invoke(), which runs its hooks
# (see _hand_sigint_to_gdb).  (Kept if the script is loaded again.)
_entry_codes = globals().get("_entry_codes", set())

# The code of realm-gdb's prompt hook, which gdb calls at its prompt (see
# _install_prompt_hook), and of what realm-gdb posts, which gdb runs in its
# event loop (see _post): between commands, unless a function call or the
# program runs (see _nothing_runs).  A Ctrl-C landing as one begins, then,
# has nothing left to quit (see _on_python_sigint).  (Kept if the script is
# loaded again.)
_idle_codes = globals().get("_idle_codes", set())
_posted_codes = globals().get("_posted_codes", set())

# The frame of the code that has let Ctrl-C through for what gdb does next,
# while it runs, and what for: "command" (see _hand_sigint_to_gdb) or "call"
# (see _let_sigint_for_call).
_sigint_letting = None

# Whether a function call made in a window runs, with Ctrl-C let through for
# it (see _let_sigint_for_call).
_sigint_call = False


def _on_python_sigint(signum, frame):
    """Python's handler of Ctrl-C (see _install_python_sigint).  It raises
    KeyboardInterrupt in whatever Python runs, as Python's own does, but
    not in realm-gdb's own code while a window is open: there that would
    leave the window half done, a function call made without what it needs
    (its settings, breakpoints disabled), a setting left changed, a call's
    breakpoint left in the parked thread's stack.  gdb hands Python a
    Ctrl-C that came while gdb ran as it next runs Python, which then
    raises it before the first line of what runs: realm-gdb's own handler
    of what gdb does next (inferior_call, before_prompt), whose `try`
    can't catch it.  Holding Ctrl-C blocked doesn't help with that one,
    which came while it was let through, in a plain command typed at a
    terminal (see _plain_before) or a function call.  So there it is held:
    blocked on gdb's main thread, and raised there again, to land once the
    window is closed (see _block_sigint).  Always in realm-gdb's handler of
    inferior_call: gdb pops a call's frame once its inferior_call_post is
    over, and a Ctrl-C landing in the middle of that would leave the call's
    breakpoint in the parked thread's stack (see _Window._check_records).
    But in the entry that lets it through for what gdb does next (see
    _hand_sigint_to_gdb), it is raised as Python would: for a plain
    command, its window then closes, and gdb takes it to quit the command.
    And where it is let through for a function call, from the end of the
    call's inferior_call_pre (see _let_sigint_for_call), and in realm-gdb's
    code that gdb runs while the call runs (a posted closer, see
    _post_close), it is passed to the call, as gdb would pass it (see
    _pass_sigint_to_call): held, it would keep the call from being
    interrupted.  (It can't be left for gdb, which takes it as Python
    returns: Python would raise it again before then, in here.)  At gdb's
    prompt, with no window open, it says "Quit", as gdb does there."""
    letting = None
    calls = mine = idle = posted = False
    f = frame
    # (Between the code that lets it through for a window's command and the
    # code it lands in: see _interruptibly.)
    passing = (_interruptibly.__code__, _apply_execute.__code__, _execute.__code__,
               _cli_output.__code__)
    while f is not None:
        if _sigint_letting is not None and f is _sigint_letting[0]:
            letting = _sigint_letting[1]
            mine = mine or f is frame
            break
        if f.f_code in _dispatch_codes:
            calls = calls or _dispatching(f, "_on_inferior_call")
            idle = idle or _dispatching(f, "_on_before_prompt")
        # (Or in realm-gdb's prompt hook, which gdb runs at its prompt too,
        # or what it posted: see _idle_codes.)
        idle = idle or f.f_code is _realm_prompt.__code__ or f.f_code in _idle_codes
        posted = posted or f.f_code in _posted_codes
        if id(f.f_code) in _my_codes and not (
                _sigint_letting is not None and _sigint_letting[1] == "rt" and
                f.f_code in passing):
            mine = True
        f = f.f_back
    if letting == "call" or letting is None and mine and _sigint_call and not calls:
        if _pass_sigint_to_call() or _window is not None and _hold_sigint():
            return
    elif letting in (None, "rt") and mine and _window is not None and _hold_sigint():
        return
    elif idle or posted and _nothing_runs():
        # (At gdb's prompt, with nothing left to quit: one pressed as a line
        # was typed, or as a command ended, say, which gdb hands realm-gdb's
        # prompt handler, prompt hook, or what it posted.  "Quit", as gdb
        # says of one at its prompt, not a Python exception.)
        gdb.write("Quit\n", gdb.STDERR)
        return
    prev = _python_sigint_prev
    if callable(prev):
        return prev(signum, frame)
    # (Python ignores Ctrl-C with SIG_IGN or SIG_DFL as its handler.)


def _nothing_runs():
    """Whether gdb runs no function call (one counted, see _call_running) and
    no thread of the program, as it runs its event loop: between commands,
    rather than waiting for a call or a resume to stop."""
    try:
        return not _infcall_depth and not any(
            t.is_valid() and t.is_running() for t in gdb.selected_inferior().threads())
    except Exception:
        return False


def _dispatching(frame, name):
    """Whether `frame`, one of _connect's dispatchers, runs handler `name`."""
    try:
        return frame.f_locals.get("handler_name") == name
    except Exception:
        return False


def _pass_sigint_to_call():
    """Pass a Ctrl-C to the function call made in the open window, running
    or about to run, as gdb passes one to the program it runs: SIGINT to the
    program, which the call's thread takes as it runs (the others are held,
    by scheduler-locking), stopping the call, which gdb then unwinds
    (unwindonsignal is on in a window).  Only where gdb runs the program
    itself, natively, so that its process ID is the program's (see
    _native_pid); and on Linux, which gives a process's signal only to a
    thread that runs, the call's.  (FreeBSD gave it to a thread gdb holds,
    which took it once the program ran, and the call ran on.)  Not before
    the call runs: one that then failed before it ran (writing its
    arguments, say) would leave the signal for the program.  Returns whether
    it could."""
    pid = getattr(_window, "native_pid", None)
    if not pid or not sys.platform.startswith("linux") or not _call_executing():
        return False
    try:
        import signal
        os.kill(pid, signal.SIGINT)
        return True
    except Exception:
        _debug_exc()
        return False


def _call_executing():
    """Whether the selected thread is running a function call now: gdb
    pretends it isn't running (is_running() is false), but won't read its
    registers."""
    try:
        gdb.newest_frame()
    except gdb.error as e:
        return "running" in str(e)
    except Exception:
        _debug_exc()
    return False


def _native_pid(inferior):
    """`inferior`'s process ID, if gdb debugs it natively (not a remote
    target's, whose ID is of another machine's process, or a core file's),
    else None."""
    # (From the target stack: not Inferior.connection, a Python object that
    # gdb 12 to 14 keep, and die of as they exit: "connpy_connection_dealloc:
    # Assertion `conn_obj->target == nullptr' failed".)
    try:
        pid = inferior.pid
        if not pid or inferior != gdb.selected_inferior():
            return None
        return pid if re.search(r"^\s*- native\b", _target_stack(), re.M) else None
    except Exception:
        _debug_exc()
        return None


_on_python_sigint._realm_gdb = True

# The Python handler of SIGINT that _on_python_sigint stands in front of
# (Python's own default_int_handler, as gdb has it).
_python_sigint_prev = globals().get("_python_sigint_prev")


def _hold_sigint():
    """Hold a Ctrl-C that Python is handling (see _on_python_sigint):
    block it on gdb's main thread, and raise it there again, as if it had
    come while it was blocked.  Returns whether it could."""
    try:
        import signal
        import _thread
        signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
        if not _sigint_is_blocked():
            return False  # (Raised again, it would come straight back.)
        signal.pthread_kill(_thread.get_ident(), signal.SIGINT)
        return True
    except Exception:
        _debug_exc()
        return False


def _hand_sigint_to_gdb():
    """Let Ctrl-C through, as the entry into realm-gdb (see _entry_codes)
    that calls this ends, for a plain command typed at a terminal, which gdb
    runs itself (see _plain_before).  One pending, or landing before that
    entry has returned, is raised then, rather than held (see
    _on_python_sigint)."""
    global _sigint_letting
    _sigint_letting = (_entry_frame(sys._getframe(1)), "command")
    _block_sigint(False)


def _entry_frame(frame):
    """The frame, `frame` or an older one, of the entry into realm-gdb (see
    _entry_codes) that runs it: `frame` itself if none is found."""
    entry = frame
    while entry is not None and entry.f_code not in _entry_codes:
        entry = entry.f_back
    return entry if entry is not None else frame


def _let_sigint_for_call():
    """Let Ctrl-C through for a function call made in a window, as its
    inferior_call_pre ends: one that lands before that handler has
    returned is passed to the call, as is one that lands in realm-gdb's code
    while the call runs (see _on_python_sigint), until the call's
    inferior_call_post, or the window finds the call failed before it ran
    (see _calls_over)."""
    global _sigint_call, _sigint_letting
    _sigint_call = True
    _sigint_letting = (_entry_frame(sys._getframe(1)), "call")
    _block_sigint(False)


def _install_python_sigint():
    """Have _on_python_sigint handle Ctrl-C in Python, if something has
    taken its place (or the script has been loaded again).  Python's
    signal.signal installs Python's own handler for the process too, in
    place of gdb's, which gdb puts back only as it next leaves Python with
    the terminal its own: so gdb's is put back at once, with libc's
    sigaction(), as _HeldSignals does.  (Not without ctypes; nor off gdb's
    main thread, which Python's signal module refuses.)"""
    global _python_sigint_prev
    try:
        import signal
        current = signal.getsignal(signal.SIGINT)
        if current is _on_python_sigint:
            return
        import ctypes
        sigaction = ctypes.CDLL(None, use_errno=True).sigaction
        sigaction.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        old = ctypes.create_string_buffer(256)  # (See _HeldSignals.)
        if sigaction(int(signal.SIGINT), None, old) != 0:
            return
        if not getattr(current, "_realm_gdb", False):
            _python_sigint_prev = current
        signal.signal(signal.SIGINT, _on_python_sigint)
        sigaction(int(signal.SIGINT), old, None)
    except Exception:
        _debug_exc()


def _stack():
    """The selected thread's frames, newest first, past any `set backtrace
    limit` while the caller holds _AllFrames.  Raises gdb.error as gdb does
    ("Selected thread is running", for one running a function call), and
    should there seem to be no end of them."""
    frame = gdb.newest_frame()
    for _ in range(100000):
        if frame is None:
            return
        yield frame
        frame = frame.older()
    raise gdb.error("realm-gdb: too many frames")


def _dummy_frames():
    """The dummy frames (function calls made by gdb) of the selected
    thread's stack, newest first.  `==` tells them apart by gdb's frame ID
    (the call's stack address), which lasts as long as the call does, even
    while another context is shown over it."""
    frames = []
    try:
        with _AllFrames():
            for frame in _stack():
                if frame.type() == gdb.DUMMY_FRAME:
                    frames.append(frame)
    except gdb.error:
        _debug_exc()
    return frames


def _dummy_records():
    """gdb's records of the function calls it has made and not popped
    (`maint print dummy-frames`), each a (stack, code) pair: gdb keeps the
    call's breakpoint at its code address, on the stack, and inserts it
    whenever the program runs.  None if it can't tell."""
    try:
        out = _internal("maint print dummy-frames")
    except gdb.error:
        _debug_exc()
        return None
    return set((int(m.group(1), 16), int(m.group(2), 16)) for m in re.finditer(
        r"stack=(0x[0-9a-fA-F]+),\s*code=(0x[0-9a-fA-F]+)", out))


# Linux's PTRACE_SETOPTIONS, and its PTRACE_O_EXITKILL (see _ExitKill).
_PTRACE_SETOPTIONS = 0x4200
_PTRACE_O_EXITKILL = 0x100000
# The ptrace options gdb 8.2 to 17 set on a program they attached to
# (linux_nat_ptrace_options (1), and the TRACECLONE that
# linux_enable_event_reporting adds): TRACESYSGOOD, TRACEFORK, TRACEVFORK,
# TRACECLONE, TRACEEXEC and TRACEVFORKDONE.  (Linux can't read a thread's.)
_GDB_ATTACHED_OPTIONS = 0x3f


class _ExitKill(object):
    """While a window's host, in a program gdb attached to, has a parked
    thread's registers: have the kernel kill the program if gdb dies (kill
    -9, a crash), as it does a program gdb started (PTRACE_O_EXITKILL),
    rather than let it run on corrupted.  clear() takes the option off again
    once the host's own registers are back, from the host and from any
    thread made meanwhile (by a function call), which inherited it.  Linux,
    a native program only; elsewhere, or if the kernel refuses, nothing."""

    def __init__(self, thread, native_pid):
        self.host = None
        if not native_pid or not sys.platform.startswith("linux") or \
                not getattr(thread.inferior, "was_attached", False):
            return
        try:
            import ctypes
            self._ptrace = ctypes.CDLL(None, use_errno=True).ptrace
            self._ptrace.argtypes = (ctypes.c_long, ctypes.c_long, ctypes.c_void_p,
                                     ctypes.c_void_p)
            self._ptrace.restype = ctypes.c_long
            self.inferior = thread.inferior
            self.before = set(t.ptid[1] for t in self.inferior.threads())
            if self._set(thread.ptid[1], _GDB_ATTACHED_OPTIONS | _PTRACE_O_EXITKILL):
                self.host = thread.ptid[1]
        except Exception:
            _debug_exc()

    def _set(self, lwp, options):
        """ptrace(PTRACE_SETOPTIONS), from gdb's main thread (the tracer):
        whether it worked.  (Not on a running thread.)"""
        return self._ptrace(_PTRACE_SETOPTIONS, lwp, None, options) == 0

    def renew(self):
        """Set it again once a function call has stopped: gdb sets the
        options of a thread it attached to itself, at the first stop it
        sees of it, which can be the call's."""
        if self.host is not None:
            try:
                self._set(self.host, _GDB_ATTACHED_OPTIONS | _PTRACE_O_EXITKILL)
            except Exception:
                _debug_exc()

    def clear(self, ran):
        """Take the option off again; `ran`: whether the program may have
        run (a function call), and made threads."""
        if self.host is None:
            return
        try:
            lwps = set([self.host])
            if ran:
                lwps |= set(t.ptid[1] for t in self.inferior.threads()) - self.before
            for lwp in sorted(lwps):
                self._set(lwp, _GDB_ATTACHED_OPTIONS)
        except Exception:
            _debug_exc()
        self.host = None


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
        # Whether something resumed the program meanwhile, other than its
        # function calls (see _on_cont); whether the host was sent a SIGINT
        # to stop it before it ran (see _stop_before_run); and whether it may
        # have run, with a parked thread's registers.
        self.resumed = self.stopping = self.host_ran = False
        self.calls = 0
        self.breakpoints_skipped = False
        # Settings its function calls needed put back ("NAME VALUE"), and
        # whether a call was refused after all (see _on_inferior_call).
        self.settings_put_back = []
        self.refused = 0
        self.refusals = []
        # The entry points of the calls it refused (see _for_call), and the
        # threads other than its own those were made on, with the dummy
        # frames each had then (see _dummy_frames).
        self.refused_pcs = {}
        self.other_dummies = {}
        # The entry points of all its calls: one left stopped at its first
        # instruction ran none of it (gdb 8.2 that couldn't start it, say).
        self.call_entries = set()
        # The thread its last function call was made on, and the dummy
        # frames that thread had then (see _calls_left_up).
        self.last_call = None
        # The watchpoints gdb didn't mean to delete as its last call began,
        # and those it marked to delete since (see _note_lost_watchpoints).
        self.watchpoints = []
        self.lost_watchpoints = set()
        # The breakpoints disabled for a call that gdb deleted meanwhile (see
        # _enable_breakpoints).
        self.lost_breakpoints = set()
        # The function calls made in it that were left stopped, and popped
        # (see pop_calls): the user thread shown for each, and whether it
        # was one it refused (which didn't run); the calls made, and
        # refused, when it last looked; and, for one that couldn't be
        # popped, the user thread shown and gdb's error.
        self.abandoned = []
        self.looked = (0, 0)
        self.stuck = None
        self.showing = None
        # gdb's records of the function calls it had made and not popped as
        # its first call began (see _dummy_records); and the user thread
        # shown as each call began.
        self.dummy_records = None
        self.call_shown = []
        # What close() could not put back, and gdb's errors writing them.
        self.left = []
        self.left_errors = {}
        # The note to print if it found that calls must be refused (see
        # _calls_refused).
        self.calls_note = None
        # Whether the command it shows runs with Ctrl-C let through (see
        # _plain_before), and whether closing it leaves Ctrl-C blocked, for
        # whatever opened it to let through once it has printed what the
        # command left (see _let_sigint).
        self.interruptible = self.hold_sigint = False
        # The command that opened it, for messages (see _window_words).
        self.opener = _apply_opener if via is None else None
        # Whether close() is done, and __exit__; and the _ExitKill set on the
        # host while it is open.
        self.done = self.exited = False
        self.exitkill = None
        # Refused calls it couldn't stop (see _for_call), refused calls that
        # a Ctrl-C stopped (see _stop_at_entry), and `source`s in progress
        # in it (see _hook).
        self.unrefused = self.interrupted = self.sourcing = 0
        # Why it refused calls for the parked thread it showed last (see
        # _show_in_window), and the user threads it showed with too little
        # of their stack free for them.
        self.refused_why = None
        self.low_room = []
        # The warnings that closing it says once its command's collected
        # output is printed (see _warn), if a list.
        self.deferred = None
        # The inlined functions the host stopped at the start of, not entered
        # yet as gdb has it (see _skipped_inline).
        self.inline_skipped = ()
        # The _CallDeadline of its function call in progress, how many of its
        # calls it stopped, and why gdb didn't.
        self.deadline = None
        self.timed_out = 0
        self.timeout_why = self.deadline_after = None
        # (Whether SIGINT stops a call, and why gdb won't time one out, if it
        # won't: see _CallDeadline.)
        self.call_timing = None
        # Whether a Ctrl-C interrupted one of its calls (see _on_inferior_call),
        # which stops `realm-thread apply all` at the next parked thread, as
        # one held does.
        self.call_interrupted = False

    def __enter__(self):
        global _window
        if _window is not None:
            raise gdb.GdbError("realm-gdb: can't show a parked Realm user thread "
                               "from inside {}.".format(_window_words(_window)))
        _install_python_sigint()  # (In case something has replaced it.)
        _block_sigint(True)
        try:
            self._open()
        except BaseException:
            if _window is not self:
                _block_sigint(False)
            raise
        return self

    def _open(self):
        """__enter__'s work, with Ctrl-C blocked."""
        global _window
        _check_guard_hooks()
        _check_with_settings()
        self.thread = _stopped_thread()
        self.thread_num = self.thread.num
        self.native_pid = _native_pid(self.thread.inferior)
        _check_unrestored(self.thread)
        # (Function calls in progress, see _release: once a count that a call
        # which failed before it ran left up is dropped, see _call_running.
        # One kept, of a call it can't tell from one in progress, isn't one
        # of the window's: see _on_inferior_call.)
        _call_running()
        self.depth = _infcall_depth
        # (The calls already stopped on it, the user's own: not to be popped.
        # A parked thread's frames, shown over them, never reach them.)
        self.known_dummies = _dummy_frames()
        # (For its copies of the registers, on gdb < 8.3: see _keep_in_gdb.)
        used = set(w.tag for w in _unrestored)
        self.tag = min(set(range(len(used) + 1)) - used)
        self.selection = _KeepSelection().__enter__()
        frame = gdb.newest_frame()
        self.inline_skipped = _skipped_inline(frame)
        with _BigValues():
            self.saved = _read_regs(frame, _snapshot_names(frame))
        self.at_syscall = _at_syscall(self.thread, frame)
        # (Made first, so that a Ctrl-C while they take effect can't leave
        # either half done: each undoes itself on the way out.)
        self.signals = _HeldSignals()
        self.settings = _CallSettings(paging=self.via in (None, "rt"))
        try:
            self.signals.__enter__()
            self.settings.__enter__()
        except BaseException:
            try:
                self.settings.__exit__(None, None, None)
            finally:
                self.signals.__exit__(None, None, None)
            raise
        _window = self
        try:
            _keep_in_gdb(self.saved, self.tag)
            self.exitkill = _ExitKill(self.thread, self.native_pid)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def to_host(self):
        """Select the host again, if a command in the window selected
        another thread (`thread N`, `inferior N`): only the host's
        registers are kept, so only the host may show a parked thread.
        (Unless the host has gone: the program died in a function call.)"""
        if _selected_thread() != self.thread and self.thread.is_valid():
            self.thread.switch()

    def show(self, ut, regs, calls=True):
        """Put user thread `ut`'s context `regs` in place (see
        context_regs), with function calls refused unless `calls`."""
        self.to_host()
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
        global _sigint_call
        # (Let through by an interruptible command, or by a call that failed
        # before it ran: see _calls_over.  One pending is held here, while
        # the window is open: see _on_python_sigint.)
        _sigint_call = False
        _block_sigint(True)
        try:
            self.close()
        except KeyboardInterrupt:
            if not self.done:
                # (Left open, for the next prompt, see _on_before_prompt, or
                # gdb's event loop where there is none.)
                _post_close(self)
                raise
            self._release()
            raise
        except BaseException as e:
            if not self.done:
                self._fail_closed(e)
            self._release()
            raise
        self._release()
        return False

    def _fail_closed(self, error):
        """close() failed before the host's registers were back, other than
        by Ctrl-C (a bug in realm-gdb, say): rather than forget the window,
        keep the program from running, as when registers can't be put back,
        and let 'realm-thread back' try again."""
        self.done = True
        why = "realm-gdb failed: {}".format(str(error) or type(error).__name__)
        self.left = sorted(self.saved)
        self.left_errors = {n: why for n in self.left}
        if self not in _unrestored:
            _unrestored.append(self)
        _say("realm-gdb: WARNING: could not put back {}.  Don't let the program run; "
             "run 'realm-thread back' to try again.  (REALM_GDB_DEBUG=1 in gdb's "
             "environment shows where realm-gdb failed.)".format(_left_words(self)))

    def _release(self):
        global _window, _infcall_depth, _sigint_call
        if self.exited:
            return
        _sigint_call = False
        if _window is self:
            _window = None
        self.exited = True
        try:
            # (No call made in it is running now.  One that failed before it
            # ran (too few arguments, a stack it couldn't write) got no
            # inferior_call_post: its breakpoints would stay disabled, and
            # its refusal breakpoint set.)
            try:
                _end_refusals(self)
                if self.deadline is not None:  # (A call that failed before it ran.)
                    self.deadline.end()
                    self.deadline = None
                if _infcall_depth > self.depth:
                    _infcall_depth = self.depth
                    del _call_starts[self.depth:]
                _enable_breakpoints()  # (Only its calls disable them.)
            finally:
                self.settings.__exit__(None, None, None)
        finally:
            try:
                if self.hold_sigint:
                    # (SIGTERM, say, would have gdb end before it printed what
                    # the command left: see _let_sigint.)
                    _held_signals_after.append(self.signals)
                    self.signals.deferred = True
                self.signals.__exit__(None, None, None)
            finally:
                try:
                    _unrestored_calls(window=self)
                finally:
                    if not self.hold_sigint:
                        _block_sigint(False)

    def close(self):
        """Put the host's own registers back, and select what was selected.
        (Ctrl-C can't stop it before they are back: it is held till then.)"""
        if self.done:
            return
        thread = self.thread
        try:
            gone = not thread.is_valid()
        except RuntimeError:
            gone = True
        if not gone and self.stopping and not self.host_ran:
            if thread.is_running():
                # (Resumed in the background: it stops at the SIGINT once gdb
                # is back in its event loop.  Until then its registers are
                # left, and refused as any are, and then put back: see
                # _retry_unrestored.)
                self.done = True
                self.left = sorted(self.saved)
                self.left_errors = {n: "it is being stopped" for n in self.left}
                if self not in _unrestored:
                    _unrestored.append(self)
                self._warn("realm-gdb: thread {} gets its own registers back once it has "
                           "stopped at that SIGINT (it was resumed in the background).".format(
                               self.thread_num))
                self.selection.__exit__(None, None, None)
                return
            self.to_host()
            self._stopped_before_run("; its own registers are put back")
        if gone or self.host_ran or thread.is_running():
            self.done = True
            self._clear_exitkill()  # (Not on a thread that is running.)
            if not gone and not self.host_ran:  # (_on_cont has said so.)
                self._warn("realm-gdb: WARNING: the program ran while thread {} had user "
                           "thread {:#x}'s registers (something resumed it without a gdb "
                           "command that realm-gdb could refuse): it may crash.".format(
                               self.thread_num, self.showing or 0))
            return
        interrupted = False
        left = list(self.saved)
        for _ in range(10):
            try:
                self.to_host()
                self.pop_calls()
                self.left_errors = {}
                left = _put_back(self.saved, self.left_errors, self.tag)
                break
            except KeyboardInterrupt:
                interrupted = True
        self.left = left
        if self.stuck is None:
            self._check_records()
        if (left or self.stuck) and self not in _unrestored:
            _unrestored.append(self)
        self.done = True
        if self not in _unrestored:
            self._clear_exitkill()
        if left:
            self._warn("realm-gdb: WARNING: could not put back {}.  Don't let the program "
                       "run; run 'realm-thread back' to try again.".format(_left_words(self)))
        else:
            self._note_inline_entered()
        if hasattr(gdb, "set_convenience_variable"):
            gdb.set_convenience_variable("_realm_gdb_reg", None)
        self.selection.__exit__(None, None, None)
        abandoned = [ut for ut, refused in self.abandoned if not refused]
        if abandoned:  # (Those refused didn't run.)
            uts = sorted(set(abandoned))
            self._warn("realm-gdb: WARNING: {} made on thread {} stopped, and {} "
                       "abandoned (showing user thread{} {}): the program may be left "
                       "inconsistent (a lock the function held, say).".format(
                           "a function call" if len(abandoned) == 1 else
                           "{} function calls".format(len(abandoned)), self.thread_num,
                           "was" if len(abandoned) == 1 else "were",
                           "s" if len(uts) > 1 else "",
                           ", ".join("{:#x}".format(u or 0) for u in uts)))
        if self.stuck is not None:
            ut, error = self.stuck
            self._warn("realm-gdb: WARNING: a function call made on thread {} while it "
                       "showed {} couldn't be popped ({}): gdb would put the call's "
                       "breakpoint into that user thread's stack whenever the program "
                       "ran.  Don't let it run: 'kill' it.".format(
                           self.thread_num, _ut_words(ut or 0), error.rstrip(".")))
        if interrupted:
            raise KeyboardInterrupt

    def _stopped_before_run(self, then=""):
        """Once the selected host, sent a SIGINT to stop it before it ran (see
        _stop_before_run), has stopped: whether it stopped there (else it
        may have run, which it says), and has its registers as they were."""
        self.stopping = False
        if _our_sigint():
            self._warn("realm-gdb: thread {} stopped at that SIGINT before it ran{}.  (The "
                       "program's other threads ran meanwhile, until gdb stopped them; "
                       "thread {} may stop at its breakpoint again.)".format(
                           self.thread_num, then, self.thread_num))
            return True
        self.host_ran = True
        self._warn("realm-gdb: WARNING: thread {} didn't stop at the SIGINT before it ran, "
                   "and may have run with user thread {:#x}'s registers: the program may "
                   "crash.".format(self.thread_num, self.showing or 0))
        return False

    def _note_inline_entered(self):
        """Once the host's registers are back: if it stopped at the start of
        inlined functions that gdb showed as not entered yet, gdb now shows
        them as its newest frames (writing a thread's registers loses that),
        which the note says.  (The frame the user had selected stays so: see
        _KeepSelection.)"""
        skipped = self.inline_skipped
        if not skipped or not _inline_entered(skipped):
            return
        names = " and ".join(key[0] for key in skipped)
        self._warn("realm-gdb: thread {} now shows inlined {} as frame{}: it stopped at the "
                   "start of {}, which gdb shows as not entered yet only until the thread's "
                   "registers are written (as they were, to show a parked thread).".format(
                       self.thread_num, names,
                       " #0" if len(skipped) == 1 else "s #0-#{}".format(len(skipped) - 1),
                       names))

    def _warn(self, text):
        """Say `text` as it closes: once what its command's output was
        collected for is printed, if so (see _apply_in), else now."""
        if self.deferred is not None:
            self.deferred.append(text)
        else:
            _say(text)

    def _clear_exitkill(self):
        if self.exitkill is not None:
            self.exitkill.clear(self.calls or self.resumed)

    def _check_records(self):
        """After its calls are popped: whether gdb still keeps a record of
        one that it didn't have before (see _dummy_records), which it lost
        the frame of: a Ctrl-C (one another thread took, say) quitting gdb
        in the middle of popping the call, after it had put the registers
        back, leaves the call's breakpoint in the parked thread's stack,
        where gdb would insert it whenever the program ran, and then put
        back what it found there before.  That can't be popped, nor taken
        away: it is stuck (see pop_calls)."""
        if self.dummy_records is None:
            return
        now = _dummy_records()
        if not now or not now - self.dummy_records:
            return
        uts = [ut for ut in self.call_shown if ut is not None] or [self.showing]
        self.stuck = (uts[-1], "gdb lost its frame but kept its breakpoint, at {}".format(
                          ", ".join("{:#x}".format(code) for _, code in
                                    sorted(now - self.dummy_records))))

    def pop_calls(self):
        """Pop each function call made in the window, since it last looked,
        that was left stopped on the host (it shouldn't be: see "Register
        windows"), as `return` in its dummy frame does, which comes before
        the shown thread's frames.  That puts back gdb's state of the host
        from before the call: otherwise gdb would keep the call's breakpoint
        in the parked thread's stack, inserting it whenever the program
        runs, and the host would stop again at once where the call stopped,
        or get the signal it stopped with.  Each goes in self.abandoned; one
        that can't be popped, in self.stuck."""
        if (self.calls, self.refused) == self.looked or _thread_running(self.thread):
            return
        with _AllFrames():  # (`return` must find the dummy frame too.)
            self.to_host()
            self._pop_here(self.known_dummies)
            # (A call it refused on another thread, which a command in the
            # window selected (`thread apply 2 print f()`), was stopped all
            # the same.)
            others = self.other_dummies
            for thread in list(self.thread.inferior.threads()):
                if thread.ptid in others and thread.is_valid() and thread != self.thread:
                    thread.switch()
                    self._pop_here(others[thread.ptid])
            others.clear()
            self.to_host()
        self.looked = (self.calls, self.refused)

    def _pop_here(self, known):
        """pop_calls's work on the selected thread, whose dummy frames
        `known` (see _dummy_frames) are of calls already stopped before
        the window made any there (the user's own, left alone).  The
        window's are newer."""
        def window_call():
            frames = _dummy_frames()
            return frames[0] if frames and frames[0] not in known else None
        # (Two Ctrl-Cs at once, one stopping a call, can quit gdb as it
        # unwinds the call's stop, which leaves its frame cache cut short at
        # frame #0, hiding the call's dummy frame: start afresh.  Not while a
        # call is in progress, whose frames gdb holds on to.)
        gdb.invalidate_cached_frames()
        error = None
        frame = window_call()
        for _ in range(16):
            if frame is None:
                break
            # (One refused stopped at its function's first instruction: why,
            # see _stop_at_entry.)
            try:
                called = frame.newer()
                pc = int(called.pc()) & _U64 if called is not None else None
                refused = self.refused_pcs.get(pc) or (
                    "not started" if pc in self.call_entries else None)
            except gdb.error:
                refused = None
            self.abandoned.append((self.showing, refused))
            try:
                # (gdb writes back only the registers that differ from what
                # it has read of them: so read them all first, or it would
                # write the x87, SSE and AVX ones too, unchanged by a call
                # stopped at its first instruction, which gdb 13 and older
                # can't with AMX on.)
                newest = gdb.newest_frame()
                _read_regs(newest, _snapshot_names(newest))
                frame.select()
                _internal("return")
            except gdb.error as e:
                _debug_exc()
                error = str(e)
                break
            frame = window_call()
        if frame is not None:
            self.stuck = (self.showing, error or "it is still there")


# (_probe_extended couldn't tell.)
_NOT_CHECKED = object()


def _symbol_key(symbol):
    """(name, file, line) of gdb.Symbol `symbol` (None for None): one
    symbol's, whatever gdb.Symbol object has it."""
    if symbol is None:
        return None
    symtab = symbol.symtab
    return (symbol.print_name, symtab.filename if symtab is not None else None, symbol.line)


def _skipped_inline(frame):
    """The inlined functions ((name, file, line) each, innermost first) that
    newest `frame` of the selected thread is stopped at the start of, which
    gdb shows as not entered yet (the thread stopped at a breakpoint on their
    caller, say, which `step` then steps into): they come before the frame's
    own function in the blocks at its PC, each starting there."""
    try:
        pc = frame.pc()
        own = _symbol_key(frame.function())
        block = gdb.block_for_pc(pc)
        skipped = []
        while block is not None and own is not None:
            if block.function is not None:
                key = _symbol_key(block.function)
                if key == own:
                    return tuple(skipped)
                if block.start != pc:
                    break
                skipped.append(key)
            block = block.superblock
    except (gdb.error, RuntimeError):
        _debug_exc()
    return ()


def _inline_entered(skipped):
    """Whether the selected thread's newest frame is now the first of the
    inlined functions `skipped` (see _skipped_inline), which gdb then shows
    as frames."""
    try:
        frame = gdb.newest_frame()
        return frame.type() == gdb.INLINE_FRAME and _symbol_key(frame.function()) == skipped[0]
    except (gdb.error, RuntimeError):
        return False


def _probe_extended():
    """Write _arch.probe_reg of the selected thread, changed, then back:
    gdb's error if it can't, else None; _NOT_CHECKED if it couldn't read it
    to begin with."""
    gdb.newest_frame().select()  # (An outer frame's would be its saved copy.)
    expr = "$" + _arch.probe_reg
    try:
        with _CLanguage():  # (Fortran, for one, can't parse it.)
            old = int(gdb.parse_and_eval(expr))
    except gdb.error:
        _debug_exc()
        return _NOT_CHECKED
    try:
        with _CLanguage():
            for value in (old ^ 1, old):
                _internal("set var {} = {:#x}".format(expr, value))
    except gdb.error as e:
        return str(e)
    return None


def _calls_unwritable(window):
    """gdb's error writing the selected thread's extended registers, if it
    can't (see "Register windows"), else None.  Checked once per process;
    the window that finds it gets the note to print (calls_note)."""
    if _arch.probe_reg is None:
        return None
    inf = gdb.selected_inferior()
    key = (inf.num, inf.pid)
    if key not in _extended_unwritable:
        if _parameter("may-write-registers") is False:
            return None  # (The user's setting says nothing of gdb's ability.)
        error = _probe_extended()
        if error is _NOT_CHECKED:
            return None  # (Not kept: the next window tries again.)
        _extended_unwritable[key] = error
        if error is not None:
            window.calls_note = _unwritable_note(error)
    return _extended_unwritable[key]


# Why a second UI (`new-ui`) is open, once realm-gdb has seen one (see
# _calls_second_ui); and whether it has said what that means for calls.
_second_ui = globals().get("_second_ui")
_second_ui_noted = globals().get("_second_ui_noted", False)


def _calls_second_ui(window):
    """Why a second UI may be open, or None.  gdb blocks only the UI that
    makes a function call while the call runs: another UI's commands run
    meanwhile, MI's -exec-continue and -target-detach among them, which no
    hook sees, while the host has the parked thread's registers and the
    call is using its stack.  So calls are refused in windows then.  The
    first window that finds it, in the session, gets the note to print."""
    global _second_ui_noted
    why = _second_ui
    other = _other_terminal() if why is None else None
    if other is not None:
        why = ("gdb has {} open, a terminal or named pipe (a second UI, perhaps; if not, "
               "start gdb without it)".format(other))
    if why is not None and not _second_ui_noted:
        _second_ui_noted = True
        window.calls_note = _second_ui_note(why)
    return why


def _second_ui_note(why):
    while_call = ("gdb runs another UI's commands even while a function call runs, "
                  "which could let the program run, or let go of it, with the parked "
                  "thread's registers in place")
    if not _can_refuse_calls:
        return ("realm-gdb: WARNING: {}, and {}; this gdb can't refuse function calls, "
                "so don't make any while a parked Realm user thread is shown.".format(
                    why[0].upper() + why[1:], while_call))
    return ("realm-gdb: function calls are refused while a parked Realm user thread "
            "is shown, in this session: {}, and {}.".format(why, while_call))


# FreeBSD: the threads a syscall catchpoint stopped, ptid -> (pc, sp) as
# they stopped (see _at_syscall), noted from gdb's stop events (gdb 13 and
# later give Python catchpoints).
_syscall_stops = globals().get("_syscall_stops", {})


def _on_stop(event):
    """FreeBSD: note a thread that a syscall catchpoint stopped (see
    _at_syscall)."""
    try:
        bps = [bp for bp in getattr(event, "breakpoints", None) or ()
               if bp.is_valid() and bp.type == gdb.BP_CATCHPOINT]
        if not any(re.search(r"\bcatchpoint\b.*\bsyscall\b",
                             _execute("info breakpoints {}".format(bp.number)))
                   for bp in bps):
            return
        thread = getattr(event, "inferior_thread", None) or _selected_thread()
        frame = gdb.newest_frame()
        _syscall_stops[thread.ptid] = (int(frame.pc()) & _U64, _frame_sp(frame))
    except Exception:
        _debug_exc()


def _at_syscall(thread, frame):
    """Whether `thread`, selected, with its own newest frame `frame`, is
    stopped at a system call by a syscall catchpoint (at its entry or its
    return: gdb doesn't say which).  A function call made at the entry
    changes the system call the thread is making: on x86 Linux gdb cancels
    it, so that the program sees it fail (ENOSYS); elsewhere the kernel
    makes it with the call's arguments first (aarch64 Linux; on FreeBSD the
    program died of it).  So windows refuse calls on such a thread.  Linux
    marks such a stop's siginfo (SIGTRAP | 0x80: gdb asks for
    PTRACE_O_TRACESYSGOOD); FreeBSD gives none there, so its stop events
    are noted (_on_stop)."""
    import signal
    try:
        with _CLanguage():
            info = gdb.parse_and_eval("$_siginfo")
            if int(info["si_signo"]) == signal.SIGTRAP and \
                    int(info["si_code"]) == signal.SIGTRAP | 0x80:
                return True
    except (gdb.error, RuntimeError, ValueError):
        pass  # (No siginfo for it: not Linux, or stopped for gdb.)
    stop = _syscall_stops.get(thread.ptid)
    if stop is None:
        return False
    try:
        if (int(frame.pc()) & _U64, _frame_sp(frame)) == stop:
            return True
    except gdb.error:
        return False
    del _syscall_stops[thread.ptid]  # (It has run since.)
    return False


def _calls_refused(window):
    """Why function calls must be refused in `window` (for gdb's
    "may-call-functions is off" error), or None: where gdb can't write the
    extended registers, or a second UI may be open, or the host is stopped
    at a system call (see _at_syscall)."""
    if window.at_syscall:
        return ("on thread {}: a syscall catchpoint stopped it at a system call, which "
                "a call made there would change (it would fail, or be made with the "
                "call's arguments); continue past it first".format(window.thread_num))
    if _calls_unwritable(window) is not None:
        return ("in this program: this gdb can't write the x87, SSE and AVX "
                "registers, which a call changes")
    if _calls_second_ui(window) is not None:
        return "in this session: a second UI may be open, whose commands run during a call"
    return None


def _unwritable_note(error):
    why = ("this gdb can't write its threads' x87, SSE and AVX registers (\"{}\"), "
           "so it couldn't undo a function call's changes to them".format(
               error.rstrip(".")))
    gdb_too = "  (gdb's own calls fail here too.{})".format(
        "  With AMX on, gdb 13 and older can't write them; gdb 14 and later can."
        if "extended state" in error else "")
    if not _can_refuse_calls:
        return ("realm-gdb: WARNING: {}, and it can't refuse function calls: don't make "
                "any while a parked Realm user thread is shown, or the program can't "
                "run again until they are put back.{}".format(why, gdb_too))
    return ("realm-gdb: function calls are refused while a parked Realm user thread "
            "is shown, in this program: {}.{}".format(why, gdb_too))


def _reg_list(names, most=6):
    """"$r8, $r9, $r10 and 4 more": register `names`, in their numbers'
    order, at `most` of them."""
    names = sorted(names, key=lambda n: [int(p) if p.isdigit() else p
                                         for p in re.split(r"(\d+)", n)])
    words = ", ".join("$" + n for n in names[:most])
    return words + (" and {} more".format(len(names) - most) if len(names) > most else "")


def _left_words(window):
    """"thread N's own registers ($a, $b: gdb's error)", for a window that
    could not put them all back."""
    left, errors = window.left, window.left_errors
    why = sorted(set(errors[n].rstrip(".") for n in left if n in errors))
    if set(left) >= set(window.saved):
        names = ""  # (All of them.)
    else:
        names = _reg_list(left) + (": " if why else "")
    return "thread {}'s own registers{}".format(
        window.thread_num, " ({}{})".format(names, "; ".join(why)) if names or why else "")


def _stuck_words(window):
    """What a window that could not pop a function call left (see
    _Window.pop_calls)."""
    ut, error = window.stuck
    return ("thread {}'s state from before a function call, made while it showed user "
            "thread {:#x}, that couldn't be popped ({})".format(
                window.thread_num, ut or 0, error.rstrip(".")))


def _unrestored_words(window):
    stuck = window.stuck
    words = [_left_words(window)] if window.left or not stuck else []
    return " and ".join(words + ([_stuck_words(window)] if stuck else []))


def _not_back_words(window):
    """What a window left that keeps the program from running, as a clause:
    "thread N's own registers aren't all back (...)", or that a function
    call couldn't be popped."""
    words = []
    if window.left or not window.stuck:
        words.append(re.sub(r"^(thread \S+ own registers)", r"\1 aren't all back",
                            _left_words(window)))
    if window.stuck:
        ut, error = window.stuck
        words.append("a function call made on thread {} while it showed {} couldn't be "
                     "popped ({})".format(window.thread_num, _ut_words(ut or 0),
                                          error.rstrip(".")))
    return " and ".join(words)


class _Unrestored(gdb.GdbError):
    """A thread whose own registers an earlier window couldn't put back."""


def _check_unrestored(thread):
    """Refuse to show a parked thread on `thread` if an earlier window
    couldn't put its own registers back (trying again first): it would save
    the parked thread's as the thread's own."""
    if _retry_unrestored():
        return
    for w in _unrestored:
        if w.thread == thread:
            raise _Unrestored(
                "realm-gdb: can't show a parked Realm user thread on thread {}: {}.  "
                "{}".format(
                    thread.num, _not_back_words(w),
                    "Run 'realm-thread back' to try again, or 'kill' the program."
                    if w.left else "'kill' the program."))


def _retry_unrestored(report=False):
    """Try again to put back the registers of hosts whose windows couldn't
    (see _Window.close).  Returns whether none are left.  With `report`,
    says so for those still left.  (A function call that could not be
    popped can't be tried again: its dummy frame is in the parked thread's
    stack, which the host no longer shows.)"""
    for window in list(_unrestored):
        try:
            thread = window.thread
            if not thread.is_valid():
                _unrestored.remove(window)
                continue
            if window.stopping and thread.is_running():
                continue  # (Not stopped yet: see _stop_before_run.)
            with _KeepSelection():
                thread.switch()
                if window.stopping and not window._stopped_before_run() or \
                        window.host_ran:
                    # (Its registers aren't what they were: leave them.)
                    _unrestored.remove(window)
                    window._clear_exitkill()
                    continue
                errors = {}
                left = _put_back(window.saved, errors, window.tag)
                window.left, window.left_errors = left, errors
            if not left and window.stuck is None:
                _unrestored.remove(window)
                window._clear_exitkill()
                print("realm-gdb: put back thread {}'s own registers.".format(
                    window.thread_num))
            elif report:
                _say("realm-gdb: WARNING: still could not put back {}.  Don't let the "
                     "program run: 'kill' it{}.".format(
                         _unrestored_words(window),
                         ", or run 'realm-thread back' to try again" if left else ""))
        except Exception as e:
            _debug_exc()
            if report:
                _say("realm-gdb: WARNING: still could not put back thread {}'s own "
                     "registers ({}).  Don't let the program run: 'kill' it, or run "
                     "'realm-thread back' to try again.".format(
                         window.thread_num, e))
    _unrestored_calls(say=True)
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
    global _infcall_depth, _sigint_call
    # (Counts left up when the window opened, by calls that failed before
    # they ran, aren't calls of its: see _call_running.)
    base = _window.depth if _window is not None else 0
    if isinstance(event, getattr(gdb, "InferiorCallPreEvent", ())):
        if _window is not None:
            # (Until the call is set up: in a window that lets Ctrl-C through,
            # a plain command's typed at a terminal, one could land before a
            # call the window refuses is stopped, see _for_call.)
            _block_sigint(True)
        _infcall_depth += 1
        _note_call_start(event.ptid)
        if _window is not None and _infcall_depth == base + 1:
            _window.calls += 1
            if _window.call_timing is None:  # (Once a window.)
                _window.call_timing = (bool(_window.native_pid) and _sigint_stops(),
                                       _no_call_timeouts())
            stops, no_timeouts = _window.call_timing
            if stops:
                # (Also where gdb has timeouts, a while after its own should
                # have stopped the call: on FreeBSD, gdb 15's don't.)
                _window.deadline = _CallDeadline(
                    _window.native_pid, event.ptid[1],
                    _CALL_TIMEOUT + (0 if no_timeouts else _CALL_TIMEOUT_GRACE))
                _window.timeout_why = no_timeouts or "gdb's own call timeout didn't stop it"
            _window.watchpoints = _disable_breakpoints()
            if _window.dummy_records is None:
                _window.dummy_records = _dummy_records()
            if _window.showing not in _window.call_shown:
                _window.call_shown.append(_window.showing)
        if _window is not None:
            frames = _dummy_frames()
            _window.last_call = (event.ptid, frames)
            try:
                _window.call_entries.add(int(event.address) & _U64)
            except Exception:
                _debug_exc()
            _for_call(_window, event, frames)
            # (Last: Ctrl-C must be able to interrupt the call, see
            # _block_sigint, and one pending since stops it as it begins:
            # not passed to it, as a call that then failed before it ran,
            # writing its arguments, say, would leave the signal for the
            # program.  If it can't be stopped, the Ctrl-C is held, to land
            # once the window is closed, and the call runs.  Not let through
            # for one made from Python, which runs on beneath it, where that
            # would kill gdb: see _SIGINT_IN_PYTHON_SAFE.)
            held = False
            if _sigint_pending() and _take_sigint() and \
                    not _window.settings.calls_refused() and \
                    not _stop_at_entry(_window, event, frames, "interrupted"):
                held = _hold_sigint()
            if not held and (_SIGINT_IN_PYTHON_SAFE or not _python_beneath(sys._getframe())):
                _let_sigint_for_call()
    elif isinstance(event, getattr(gdb, "InferiorCallPostEvent", ())):
        # (First, as the window's call ends, in any window: gdb pops the
        # call's frame once this is over, and a Ctrl-C that landed in the
        # middle of that would leave gdb's record of the call, and its
        # breakpoint, in the parked thread's stack, see close(); and it would
        # land in this handler's work.  One that does is held, from here: see
        # _on_python_sigint.  Not as a call ends that gdb made while it set up
        # the window's, to put a string in the program's memory, say; the
        # window's own has yet to run.)
        if _window is not None and _infcall_depth <= base + 1:
            _sigint_call = False
            _block_sigint(True)
        _infcall_depth = max(0, _infcall_depth - 1)
        del _call_starts[_infcall_depth:]
        if _window is not None and _window.exitkill is not None:
            _window.exitkill.renew()
        if _window is not None and _disabled_breakpoints and not _window.refusals:
            # (The call ran.)
            _window.breakpoints_skipped = frozenset(num for num, _, _ in _disabled_breakpoints)
        if _infcall_depth <= base:
            fired = False
            if _window is not None and _window.deadline is not None:
                fired = _window.deadline.end()
                _window.timed_out += fired
                _window.deadline_after = _window.deadline.timeout
                _window.deadline = None
            # (Only where a parked thread's command could follow: as little
            # as can be done here, where a Ctrl-C typed meanwhile, which goes
            # to the program, would be left pending for it.)
            if _window is not None and _window.via is None and not fired and \
                    _stopped_by_sigint():
                _window.call_interrupted = True  # (Ctrl-C.)
            _calls_ended(_window)


def _for_call(window, event, frames):
    """Just before a function call in `window`, on a thread with dummy
    frames `frames` (see _dummy_frames): put back the settings the
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
        if window.settings.calls_refused() and not _stop_at_entry(window, event, frames):
            window.unrefused += 1
    except Exception:
        _debug_exc()


def _stop_at_entry(window, event, frames, why="refused"):
    """Have the function call that `event` (its inferior_call_pre) begins,
    in `window`, on a thread with dummy frames `frames`, stop at its first
    instruction, to be abandoned as the window closes (see pop_calls), so
    that nothing of it runs: one the window refuses ("refused", see
    _for_call), or one a Ctrl-C held meanwhile was pressed before
    ("interrupted": see _on_inferior_call).  Returns whether it could."""
    global _making_stop
    try:
        addr = int(event.address) & _U64
        _making_stop = True  # (Not to be disabled: see _on_breakpoint_created.)
        with _CLanguage():  # (Modula-2 can't parse `*ADDR`.)
            bp = gdb.Breakpoint("*{:#x}".format(addr), internal=True)
    except Exception:
        _debug_exc()
        return False
    finally:
        _making_stop = False
    bp.silent = True
    window.refusals.append(bp)
    window.refused += 1
    if why == "interrupted":
        window.interrupted += 1
        window.call_interrupted = True
    window.refused_pcs[addr] = why
    if event.ptid != window.thread.ptid:
        # (With the calls already stopped on it, not to be popped.)
        window.other_dummies.setdefault(event.ptid, frames)
    return True


# Whether _stop_at_entry is making its breakpoint.
_making_stop = False


def _on_breakpoint_created(bp):
    """A breakpoint made while a window's function call runs, by Python code
    that runs meanwhile (another extension's handler of inferior_call, say),
    is disabled for it too, as the others were as it began (see
    _disable_breakpoints): the call can't stop (see "Register windows")."""
    if _window is None or _infcall_depth <= _window.depth or _making_stop:
        return
    try:
        if bp.is_valid() and bp.enabled:
            bp.enabled = False
            _disabled_breakpoints.append((bp.number, bp, 0))
    except (gdb.error, RuntimeError):
        _debug_exc()


def _calls_ended(window):
    """The function calls made in `window` (if not None) are over, or the
    last of them, that ran: enable the breakpoints disabled for them, put
    back the settings for while one runs, stop refusing them at their entry,
    and note the watchpoints gdb is to delete."""
    _enable_breakpoints()
    if window is None:
        return
    try:
        window.settings.end_call()
    except Exception:
        _debug_exc()
    _end_refusals(window)
    _note_lost_watchpoints(window)


def _end_refusals(window):
    refusals = window.refusals
    for bp in refusals:
        try:
            if bp.is_valid():
                bp.delete()
        except Exception:
            _debug_exc()
    del refusals[:]


# The breakpoints the last note that calls ignored them was about (see
# _call_notes).
_skipped_noted = globals().get("_skipped_noted")


def _call_notes(window):
    """The notes on the function calls made in `window`, or refused in it,
    to print once it is closed."""
    global _skipped_noted
    notes = []
    if window.calls_note:  # (Once: see _calls_refused.)
        notes.append(window.calls_note)
        window.calls_note = None
    if window.low_room:
        notes.append("realm-gdb: function calls are refused on {}: under "
                     "{} KiB of {} stack is free below {} stack pointer.".format(
                         _uts_words(window.low_room),
                         _CALL_HEADROOM >> 10,
                         "their" if len(window.low_room) > 1 else "its",
                         "the" if len(window.low_room) > 1 else "its"))
        window.low_room = []
    if window.breakpoints_skipped and window.breakpoints_skipped != _skipped_noted:
        # (Once for the same breakpoints: the one the program stopped at is
        # always there.)
        _skipped_noted = window.breakpoints_skipped
        notes.append("realm-gdb: breakpoints are disabled while a function call runs on a "
                     "parked Realm user thread (a call there can't stop).")
    lost = sorted(window.lost_watchpoints)
    if lost:
        notes.append("realm-gdb: WARNING: gdb deletes watchpoint{} {} at the next stop: "
                     "during the function call it looked for {} frame among the parked "
                     "Realm user thread's.  Set {} again ('watch -l' has no "
                     "frame).".format(
                         "s" if len(lost) > 1 else "", ", ".join(map(str, lost)),
                         "their" if len(lost) > 1 else "the watchpoint's",
                         "them" if len(lost) > 1 else "it"))
    gone = sorted(window.lost_breakpoints, key=lambda n: (n < 0, abs(n)))
    if gone:
        # (One made from Python with internal=True has a negative number.)
        notes.append("realm-gdb: WARNING: gdb deleted breakpoint{} {} (a Python "
                     "FinishBreakpoint, say) as a function call made on the parked Realm "
                     "user thread stopped: it looked for {} frame among that thread's.  Set "
                     "{} again afterwards.".format(
                         "s" if len(gone) > 1 else "",
                         ", ".join(str(n) if n > 0 else "{} (internal)".format(n)
                                   for n in gone),
                         "their" if len(gone) > 1 else "the breakpoint's",
                         "them" if len(gone) > 1 else "it"))
    put = window.settings_put_back
    if put:
        notes.append("realm-gdb: the function calls ran with {} all the same, as calls "
                     "on a parked Realm user thread must.".format(
                         ", ".join(put[:-1]) + " and " + put[-1] if len(put) > 1 else put[0]))
    why = ("  (They are refused while a parked Realm user thread is shown, {}.)".format(
        window.refused_why) if window.refused_why else "")
    if window.unrefused:
        notes.append("realm-gdb: WARNING: a function call ran that realm-gdb refuses: "
                     "may-call-functions was turned on in the command, and the call "
                     "couldn't be stopped.{}".format(why))
    # (Those that failed before they ran weren't stopped.)
    interrupted = sum(1 for _, why in window.abandoned if why == "interrupted")
    if window.refused > window.interrupted:
        notes.append("realm-gdb: function calls were refused all the same, with "
                     "may-call-functions turned on in the command: each was stopped at its "
                     "first instruction, before any of it ran.{}".format(why))
    if window.timed_out:
        notes.append("realm-gdb: {} stopped after {} seconds, with a SIGINT from realm-gdb, "
                     "as {}.".format(
                         "a function call was" if window.timed_out == 1 else
                         "{} function calls were".format(window.timed_out),
                         window.deadline_after, window.timeout_why))
        window.timed_out = 0
    not_started = sum(1 for _, why in window.abandoned if why == "not started")
    if not_started:
        notes.append("realm-gdb: {} stopped at {} first instruction, and abandoned before "
                     "any of {} ran.".format(
                         "a function call was" if not_started == 1 else
                         "{} function calls were".format(not_started),
                         "its" if not_started == 1 else "their",
                         "it" if not_started == 1 else "them"))
    if interrupted:
        notes.append("realm-gdb: {} stopped at {} first instruction, and abandoned, so "
                     "that nothing of {} ran: Ctrl-C was pressed before {} began.".format(
                         "a function call was" if interrupted == 1 else
                         "{} function calls were".format(interrupted),
                         "its" if interrupted == 1 else "their",
                         "it" if interrupted == 1 else "them",
                         "it" if interrupted == 1 else "they"))
    return notes


def _breakpoint_table():
    """{number: [enabled, enable count, disposition, whether a watchpoint]}
    of the user's breakpoints of every kind, from `info breakpoints`: gdb.breakpoints() leaves some out
    (dprintfs; before gdb 11, hardware breakpoints and catchpoints, at one
    of which a call would be abandoned, perhaps in a system call, which
    putting the registers back doesn't undo), and doesn't tell an enable
    count (`enable count N`), which enabling one drops."""
    try:
        # (At annotation level 2 and up, its rows come apart, among
        # annotations: `set annotate 2`, as some front ends have.)
        with _Temporarily("annotate", "0", lambda old: isinstance(old, int) and old > 1):
            out = _execute("info breakpoints")
    except gdb.error:
        return {}
    table = {}
    row = None
    for line in out.splitlines():
        m = re.match(r"(\d+)\s+(.*?)\s+(keep|del|dis|dstp)\s+([yn])", line)
        if m:
            row = table[int(m.group(1))] = [m.group(4) == "y", 0, m.group(3),
                                            "watchpoint" in m.group(2)]
            continue
        m = re.search(r"\sdisable after (?:next|additional) (\d+) hits", line)
        if m and row is not None:
            row[1] = int(m.group(1))
    return table


def _disable_breakpoints():
    """Disable the user's enabled breakpoints, for a function call in a
    window, to be enabled again as they were (_enable_breakpoints).  Returns
    the watchpoints that gdb doesn't mean to delete (see
    _note_lost_watchpoints)."""
    objects = {}
    for bp in _breakpoints():
        try:
            if bp.is_valid():
                objects[bp.number] = bp
        except Exception:
            _debug_exc()
    table = _breakpoint_table()
    for num in sorted(set(objects) | set(table)):
        bp = objects.get(num)
        enabled, count = table.get(num, (bp is not None and bp.enabled, 0))[:2]
        try:
            if bp is not None and bp.enabled:
                bp.enabled = False
            elif bp is None and enabled:
                gdb.execute("disable {}".format(num), to_string=True)
            else:
                continue
            _disabled_breakpoints.append((num, bp, count))
        except (gdb.error, RuntimeError):
            _debug_exc()
    return [num for num, row in table.items() if row[3] and row[2] != "dstp"]


def _note_lost_watchpoints(window):
    """After a function call in `window`: note the watchpoints gdb marked to
    delete meanwhile.  As a call resumes the host, gdb checks each hardware
    watchpoint set on the host's own frames, disabled or not, looks for its
    frame among the frames the host now has, the parked thread's, and
    deletes it at the next stop when it doesn't find it.  Nothing stops
    that."""
    if not window.watchpoints:
        return
    table = _breakpoint_table()
    for num in window.watchpoints:
        if num not in table or table[num][2] == "dstp":
            window.lost_watchpoints.add(num)


def _enable_breakpoints():
    """Enable again the breakpoints _disable_breakpoints disabled.  Those
    gdb deleted meanwhile are noted in the open window: as a call stopped
    (one the window refused, let through), gdb looked for a Python
    FinishBreakpoint's frame among the frames the host had then, the parked
    thread's, and deleted it, as it does watchpoints (_note_lost_watchpoints)."""
    lost = []
    while _disabled_breakpoints:
        num, bp, count = _disabled_breakpoints[-1]
        for attempt in range(10):  # (Retried if interrupted, as _set is.)
            try:
                if bp is not None and not bp.is_valid():
                    lost.append(num)
                elif count > 1:
                    gdb.execute("enable count {} {}".format(count, num), to_string=True)
                elif bp is not None:
                    bp.enabled = True
                else:
                    gdb.execute("enable {}".format(num), to_string=True)
                break
            except (gdb.error, RuntimeError) as e:
                if "No breakpoint number" in str(e):
                    lost.append(num)
                _debug_exc()
                break
            except KeyboardInterrupt:
                if attempt == 9:
                    raise
        _disabled_breakpoints.pop()  # (Once it is enabled.)
    if lost and _window is not None:
        _window.lost_breakpoints.update(lost)


def _signal_thread(pid, lwp, sig):
    """Send signal `sig` to thread `lwp` of native process `pid` alone
    (Linux's tgkill, FreeBSD's thr_kill2, where `lwp` -1 is each of its
    threads): whether it could.  (Not without ctypes.)"""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform.startswith("freebsd"):
            return libc.thr_kill2(ctypes.c_int(pid), ctypes.c_long(lwp), ctypes.c_int(sig)) == 0
        if not sys.platform.startswith("linux"):
            return False
        if hasattr(libc, "tgkill"):  # (glibc >= 2.30)
            return libc.tgkill(ctypes.c_int(pid), ctypes.c_int(lwp), ctypes.c_int(sig)) == 0
        nr = {"x86_64": 234, "aarch64": 131, "ppc64le": 250, "ppc64": 250}.get(
            platform.machine())
        return nr is not None and libc.syscall(
            ctypes.c_long(nr), ctypes.c_long(pid), ctypes.c_long(lwp), ctypes.c_long(sig)) == 0
    except Exception:
        _debug_exc()
        return False


def _stop_before_run(thread, pid):
    """`thread`, of native process `pid`, with registers that aren't its own,
    is being resumed by something that got past realm-gdb's refusals (from
    gdb.events.cont, which gdb sends before it resumes the program): send it
    a SIGINT of its own, which the kernel stops it at before it runs an
    instruction, and gdb then stops the program at, as for a Ctrl-C (only
    with `handle SIGINT stop nopass`, gdb's default).  Whether it could."""
    try:
        if not pid or not thread.is_valid() or not thread.is_running() or \
                not _sigint_stops():
            return False
        import signal
        return _signal_thread(pid, thread.ptid[1], int(signal.SIGINT))
    except Exception:
        _debug_exc()
        return False


def _stopped_by_sigint():
    """Whether the selected thread last stopped at a SIGINT (from its
    siginfo: Linux and FreeBSD)."""
    import signal
    try:
        # (In any language: no member of it is parsed.)
        return int(gdb.parse_and_eval("$_siginfo")["si_signo"]) == int(signal.SIGINT)
    except Exception:
        return False


def _sigint_stops():
    """Whether gdb stops the program at a SIGINT it gets, and doesn't pass
    it on (`handle SIGINT stop nopass`, gdb's default)."""
    try:
        m = re.search(r"^SIGINT\s+(Yes|No)\s+(?:Yes|No)\s+(Yes|No)",
                      _internal("info signals SIGINT"), re.M)
    except gdb.error:
        return False
    return m is not None and m.groups() == ("Yes", "No")


class _CallDeadline(object):
    """A window's call that runs `timeout` seconds (_CALL_TIMEOUT, where gdb
    doesn't time function calls out, see _no_call_timeouts; a while longer
    where it should have) is stopped all the same, and unwound (see
    _CallSettings): a thread of Python's sends the call's thread a SIGINT
    then, as a Ctrl-C would.  (Not while gdb holds Python's lock in the
    call, as gdb 8.2 does in one made from Python.)"""

    def __init__(self, pid, lwp, timeout):
        import threading
        self.pid, self.lwp, self.timeout = pid, lwp, timeout
        self.lock = threading.Lock()
        self.over = threading.Event()
        self.fired = False
        watch = threading.Thread(target=self._watch, name="realm-gdb call deadline")
        watch.daemon = True
        watch.start()

    def _watch(self):
        import signal
        if not self.over.wait(self.timeout):
            with self.lock:  # (Not once the call is over: see end().)
                if not self.over.is_set() and not self._stopped():
                    self.fired = _signal_thread(self.pid, self.lwp, int(signal.SIGINT))

    def _stopped(self):
        """Whether the call's thread has stopped already (its call over, as
        gdb has yet to see), as Linux's /proc has it: the SIGINT would be
        left for after the call."""
        try:
            with open("/proc/{}/task/{}/stat".format(self.pid, self.lwp)) as f:
                return f.read().rpartition(")")[2].split()[0] in ("t", "T")
        except (OSError, IndexError):
            return False

    def end(self):
        """The call is over: whether it was stopped."""
        with self.lock:
            self.over.set()
        return self.fired


# Whether this gdb times function calls out itself (see _CallDeadline), and
# how long after its timeout realm-gdb stops a call all the same.
_CALL_TIMEOUTS = _has_parameter("direct-call-timeout")
_CALL_TIMEOUT_GRACE = 2


def _no_call_timeouts():
    """Why gdb won't time a function call out itself (see _CallDeadline):
    it has no timeouts, or (as `show direct-call-timeout` says) they don't
    work on this target, which doesn't run in async mode (FreeBSD's native
    one, gdb 15); else None."""
    if not _CALL_TIMEOUTS:
        return "this gdb has no call timeout"
    try:
        if "does not support async" in _internal("show direct-call-timeout"):
            return "gdb's call timeout doesn't work on this target"
    except gdb.error:
        _debug_exc()
    return None


def _our_sigint():
    """Whether the selected thread stopped at a SIGINT that _stop_before_run
    sent it: one sent to it alone, by gdb's process."""
    import signal
    try:
        with _CLanguage():
            info = gdb.parse_and_eval("$_siginfo")
            fields = [f.name for f in info.type.fields()]
            if "si_pid" in fields:  # (FreeBSD's, sent by thr_kill2: SI_LWP)
                pid, code = info["si_pid"], 0x10007
            else:  # (Linux's, by tgkill: SI_TKILL)
                pid, code = info["_sifields"]["_kill"]["si_pid"], -6
            return (int(info["si_signo"]) == int(signal.SIGINT) and
                    int(info["si_code"]) == code and int(pid) == os.getpid())
    except Exception:
        _debug_exc()
        return False


def _on_cont(event):
    """Something is resuming the program.  Inside a window, that is only
    expected of function calls; anything else got past the hooks (see
    _stop_before_run)."""
    if _window is not None and _infcall_depth == 0:
        _window.resumed = True
        words = (_window.thread_num, _window.showing or 0)
        if _window.host_ran:
            pass  # (Said already.)
        elif _stop_before_run(_window.thread, _window.native_pid):
            _window.stopping = True
            _say("realm-gdb: WARNING: the program is resuming while thread {} has user "
                 "thread {:#x}'s registers (something got past realm-gdb's refusals): "
                 "stopping thread {} before it runs, with a SIGINT.".format(
                     *(words + (_window.thread_num,))))
        elif _thread_running(_window.thread):
            _window.host_ran = True
            _say("realm-gdb: WARNING: the program is resuming while thread {} has user "
                 "thread {:#x}'s registers, and realm-gdb can't put its own back now: "
                 "the program may crash.".format(*words))
        else:
            # (Its own are put back as usual.)
            _say("realm-gdb: WARNING: other threads are resuming while thread {} has user "
                 "thread {:#x}'s registers (something got past realm-gdb's refusals), and "
                 "could run that user thread meanwhile: the program may crash.".format(*words))
    elif _window is None and _infcall_depth == 0 and _unrestored:
        for w in _unrestored:
            if w.host_ran or not _thread_running(w.thread):
                continue
            if _stop_before_run(w.thread, _native_pid(w.thread.inferior)):
                w.stopping = True
                _say("realm-gdb: WARNING: the program is resuming while realm-gdb couldn't "
                     "put back {} (something got past its refusals): stopping thread {} "
                     "before it runs, with a SIGINT.".format(
                         _unrestored_words(w), w.thread_num))
            else:
                w.host_ran = True
                _say("realm-gdb: WARNING: the program is resuming while realm-gdb couldn't "
                     "put back {}: the program may crash.".format(_unrestored_words(w)))
    elif _window is not None:
        # (In a function call, which runs the host alone: something else is
        # resuming another thread, which could resume the parked one onto the
        # stack the call is using.  In all-stop mode the event names no
        # thread.)
        try:
            others = [t.num for t in _window.thread.inferior.threads()
                      if t.is_valid() and t != _window.thread and t.is_running()]
        except Exception:
            others = []
        if others:
            _say("realm-gdb: WARNING: thread{} {} resuming during a function call made "
                 "while thread {} has {}'s registers (another UI's command, or a "
                 "thread the function started, which gdb 14 and older let run): the "
                 "program may crash.".format(
                     "s" if len(others) > 1 else "", ", ".join(map(str, others)) +
                     (" are" if len(others) > 1 else " is"), _window.thread_num,
                     _ut_words(_window.showing or 0)))
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
    _syscall_stops.clear()
    _drop_views(lambda v: inferior is None or v["inferior"] == inferior.num)
    _drop_selections(lambda sel: inferior is None or sel["inferior"] == inferior.num)
    for window in list(_unrestored):
        try:
            if not window.thread.is_valid():
                _unrestored.remove(window)
        except Exception:
            _unrestored.remove(window)
    _unrestored_calls()
    if _infcall_depth:
        _infcall_depth = 0
        del _call_starts[:]
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
    if not _retry_unrestored():
        _kill_unrestored()


def _unrestored_calls(say=False, window=None):
    """Refuse function calls while a host's registers couldn't be put back
    (see _unrestored), turning may-call-functions off (gdb >= 9), as a call
    lets the program run (every thread, unless scheduler-locking is on; the
    host perhaps on a parked thread's stack); and turn it back on as they
    are put back (saying so, with `say`), unless it has been changed
    meanwhile.  Outside windows, whose own settings come and go (see
    _CallSettings).  (Said after the warnings of the `window` closing, if
    any.)"""
    global _calls_off_unrestored
    if _window is not None:
        return
    say_it = window._warn if window is not None else _say
    try:
        if _unrestored and _calls_off_unrestored is None:
            if not _can_refuse_calls:
                if not getattr(_unrestored[-1], "calls_warned", False):
                    _unrestored[-1].calls_warned = True
                    say_it("realm-gdb: WARNING: until they are put back, don't make "
                           "function calls either: a call lets the program run.")
                return
            before = _parameter("may-call-functions")
            if before:
                _set("may-call-functions", "off")
                _calls_off_unrestored = True
                say_it("realm-gdb: until they are put back, function calls are refused "
                       "(may-call-functions is off), as a call lets the program run.")
        elif not _unrestored and _calls_off_unrestored is not None:
            _calls_off_unrestored = None
            if _parameter("may-call-functions") is False:
                _set("may-call-functions", "on")
                if say:
                    print("realm-gdb: function calls are allowed again "
                          "(may-call-functions is on).")
    except Exception:
        _debug_exc()


# Whether realm-gdb turned may-call-functions off (see _unrestored_calls).
# (Kept if the script is loaded again.)
_calls_off_unrestored = globals().get("_calls_off_unrestored")


def _kill_unrestored():
    """As gdb lets go of the programs: kill each attached one whose host's
    registers still couldn't be put back, rather than let it run on
    corrupted.  (gdb kills a program it started itself.)"""
    for window in list(_unrestored):
        try:
            if not window.thread.is_valid():
                continue
            inferior = window.thread.inferior
            if not getattr(inferior, "was_attached", True):
                continue
            _say("realm-gdb: WARNING: killing process {} as gdb lets go of it, rather "
                 "than let it run corrupted: realm-gdb could not put back {}.".format(
                     inferior.pid, _unrestored_words(window)))
            with _Temporarily("confirm", "off"), _KillNudge(inferior):
                _internal("kill inferiors {}".format(inferior.num))
            _unrestored.remove(window)
        except Exception:
            _debug_exc()
    _unrestored_calls()


class _KillNudge(object):
    """On FreeBSD (15.1, with gdb 15.1), gdb's kill of a program it attached
    to, and hasn't let run since, lets the program run on unkilled, and then
    waits for it to end, for good: after the kernel's transparent attach
    (debug.ptrace_attach_transparent), no thread has reported a stop for
    PT_KILL to give its SIGKILL to.  A SIGKILL sent to the program while it
    is stopped is no help: the kernel leaves it to one thread, which may
    sleep on while the others run.  So first send each of its threads a
    SIGKILL of its own, while it is stopped (thr_kill2 to every thread, or
    else to each one gdb lists): the kernel then ends the program as the
    first of them would return to the program's code, before it runs any.
    And until the kill is over, send the program SIGKILL every
    few milliseconds (INTERVAL): one sent once gdb has let it run wakes a
    thread, which dies of it, where every thread might otherwise sleep on.
    (A native process only: a remote one's pid isn't ours to signal.
    gdb.execute lets other Python threads run from gdb 12.)"""

    INTERVAL = 0.005

    def __init__(self, inferior):
        self.pid = None
        self.inferior = inferior
        # (Inferior.connection only there: gdb 12 to 14 die of it as they
        # exit, see _native_pid.)
        if sys.platform.startswith("freebsd") and inferior.pid > 0 and \
                getattr(getattr(inferior, "connection", None), "type", None) == "native":
            self.pid = inferior.pid

    def __enter__(self):
        if self.pid is not None:
            import signal
            import threading
            if not _signal_thread(self.pid, -1, int(signal.SIGKILL)):
                try:
                    threads = self.inferior.threads()
                except Exception:
                    _debug_exc()
                    threads = ()
                for thread in threads:
                    try:
                        _signal_thread(self.pid, thread.ptid[1], int(signal.SIGKILL))
                    except Exception:
                        _debug_exc()
            self.lock = threading.Lock()
            self.done = threading.Event()
            nudge = threading.Thread(target=self._nudge, name="realm-gdb kill")
            nudge.daemon = True
            nudge.start()
        return self

    def _nudge(self):
        import signal
        while not self.done.wait(self.INTERVAL):
            # (Under the lock, so not once the kill is over.  Between gdb
            # reaping the process and the kill returning, the pid would have
            # to be reused at once to be another's.)
            with self.lock:
                if self.done.is_set():
                    return
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except OSError:
                    return  # (Gone.)

    def __exit__(self, *exc):
        if self.pid is not None:
            with self.lock:
                self.done.set()
        return False


def _install_atexit():
    """Python's SystemExit (`python sys.exit()`, or a sourced script's
    `exit()`) ends gdb 14 and older at once, without gdb.events.gdb_exiting
    (or before gdb 12, which has none): have Python's own exit handling put
    back any host's registers then, as _on_gdb_exiting does.  (Once per gdb
    session; elsewhere it finds nothing to do, as gdb has let go of the
    program by the time Python exits.)"""
    if getattr(gdb, "_realm_gdb_atexit", False):
        return
    import atexit
    namespace = globals()
    atexit.register(lambda: _guarded(namespace["_on_gdb_exiting"])(None))
    gdb._realm_gdb_atexit = True


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------
#
# For the commands below, realm-gdb defines a hook-COMMAND that runs
# `realm-gdb-guard COMMAND`, and for some a hookpost-COMMAND that runs
# `realm-gdb-guard --after COMMAND`.  gdb runs a command's hooks however
# the command comes: by an alias or an abbreviation, from a script, a
# user-defined command, Python's gdb.execute or MI's `-interpreter-exec
# console`.  It doesn't run them for MI's own commands.  A hook gets no
# arguments, and a hook that fails stops its command.  If the user already
# has such a hook, realm-gdb's check is added to it, and realm-gdb says so
# (see _define_hook).
#
# Inside a window, CLI commands that would let the program run, or let go of
# it, are refused (_GUARDED_COMMANDS), and so are those that read the
# terminal themselves (_TERMINAL_COMMANDS).  Outside windows they aren't,
# unless a host's registers couldn't be put back.  MI commands bypass CLI
# hooks, but inside a window they can only come from the CLI's
# `interpreter-exec` (refused) or Python's gdb.execute_mi (which realm-gdb
# has refuse them too: see _install_execute_mi).
#
# While a parked thread is selected, commands that look at the selected
# thread or frame show it, and commands that would act on the thread as it
# really is are refused (see "Plain commands on a selected parked thread").

# (Each that this gdb has; and every `target` subcommand, since connecting
# to a new target kills the program, which a window's `confirm off` doesn't
# ask about; nor does `record full restore`, which loads a core file the
# same way.  `queue-signal` too, which would give the signal to the host;
# and on a remote target `monitor` and `maint packet`, which can do anything.)
_GUARDED_COMMANDS = (
    "continue", "signal", "step", "stepi", "next", "nexti", "finish", "until",
    "advance", "jump", "return", "reverse-step", "reverse-stepi", "reverse-next",
    "reverse-nexti", "reverse-finish", "reverse-continue", "detach",
    "detach inferiors", "disconnect", "kill", "kill inferiors", "quit", "run",
    "start", "starti", "attach", "core-file", "generate-core-file", "load",
    "checkpoint", "restart", "record", "record full", "record btrace",
    "interpreter-exec", "queue-signal", "monitor", "maintenance packet",
    "record full restore")
# Commands refused in a window only, each for the reason given.
_ASKING = "without asking, as every question gets its default answer there"
_DROPS_LIBREALM = "the shared libraries' symbols, librealm's among them, in the middle of " \
    "the command"
_REGISTERS_CHANGE = "it would change which registers that thread has{}, and so keep its own " \
    "from being put back"
_WINDOW_GUARDED = {
    "new-ui": "gdb runs the new UI's commands even during a function call, which "
              "could let the program run, or let go of it, with the parked thread's "
              "registers in place",
    "maintenance dump-me": "gdb would abort there, leaving that thread with the parked "
                           "thread's registers",
    "file": "it would replace the program's file " + _ASKING,
    "exec-file": "it would replace the program's file " + _ASKING,
    "symbol-file": "it would replace or drop the program's symbols " + _ASKING,
    "add-symbol-file": "it would add symbols " + _ASKING,
    "add-symbol-file-from-memory": "it would add symbols " + _ASKING,
    "remove-symbol-file": "it would drop symbols " + _ASKING,
    "nosharedlibrary": "it would drop " + _DROPS_LIBREALM,
    "delete checkpoint": "it may make a function call in the program, on the thread "
                         "that has the parked thread's registers",
    "info static-tracepoint-markers": "gdb lets the whole program run while it asks "
                                      "for them",
    "strace": "gdb lets the whole program run while it asks for static tracepoint "
              "markers ('strace -m')",
    "set sysroot": "it can drop " + _DROPS_LIBREALM,
    "set solib-search-path": "it can drop " + _DROPS_LIBREALM,
    "set endian": "it would change how gdb reads the registers that thread must get "
                  "back",
    "set tdesc filename": _REGISTERS_CHANGE.format(""),
    "unset tdesc filename": _REGISTERS_CHANGE.format(""),
    "set architecture": _REGISTERS_CHANGE.format(", and their sizes"),
    "set osabi": _REGISTERS_CHANGE.format(""),
    # (Python's strings, realm-gdb's commands among them, are passed to gdb in
    # the host charset: in EBCDIC, none could put that thread's registers
    # back, nor kill the program as gdb let go of it.)
    "set host-charset": "realm-gdb couldn't then put that thread's own registers back",
    "set charset": "it sets the host charset too, and realm-gdb couldn't then put that "
                   "thread's own registers back",
    "flash-erase": "it would erase the program's flash memory, with the parked thread's "
                   "registers in place",
    "info threads": "it would list that thread at the parked thread's frame, not its own",
}
# (tfind selects a trace frame, whose registers gdb reads instead, and won't
# write: that thread's own couldn't be put back.)
_WINDOW_GUARDED.update(
    (name, "it would show a trace frame's registers instead of that thread's, and "
           "keep its own from being put back")
    for name in ("tfind", "tfind start", "tfind end", "tfind none", "tfind tracepoint",
                 "tfind pc", "tfind line", "tfind range", "tfind outside"))
# The settings of them that `with SETTING VALUE -- COMMAND` can change around
# a command that opens a window (`with architecture X -- bt`, a parked
# thread selected), running no `set` hook: as they were as the outermost
# `with` in progress began (hook-with), if one is.  No window opens with
# one changed (see _check_with_settings).  Forgotten at hookpost-with; after
# a `with` that failed, which runs no hookpost, at gdb's prompt, or from gdb
# 15 as soon as gdb is back in its event loop (in MI, say: see _post_close);
# and a setting, at a `set` of it (_WITH_SET), which isn't `with`'s doing.
_WITH_GUARDED = ("architecture", "endian", "osabi", "tdesc filename", "host-charset")
_WITH_SET = dict([("set " + name, name) for name in _WITH_GUARDED] + [
    ("unset tdesc filename", "tdesc filename"), ("set charset", "host-charset")])
_with_settings = globals().get("_with_settings")
_with_depth = globals().get("_with_depth", 0)


def _with_guarded_settings():
    # (Before gdb 12, gdb.parameter has "" for "auto" until it is set.)
    return dict((name, _parameter(name, _NO_SETTING) or "auto") for name in _WITH_GUARDED)


def _with_hook(after):
    """hook-with (hookpost-with, `after`): see _with_settings."""
    global _with_settings, _with_depth
    if after:
        _with_depth = max(0, _with_depth - 1)
    else:
        if _with_settings is None:
            _with_settings = {}
            _post_with_end(_with_settings)
        # (Those an outer `with` has changed are kept as they were before it;
        # one a `set` has dropped since, as it is now.  And the settings a
        # window gives its function calls, to tell those the `with` changed:
        # see _CallSettings.with_changed.)
        for name, value in _with_guarded_settings().items():
            _with_settings.setdefault(name, value)
        for name in _WINDOW_SETTINGS:
            old = _get_setting(name)
            if old is not _NO_SETTING:
                _with_settings.setdefault("call: " + name, _setting_words(old))
        _with_depth += 1
    if not _with_depth:
        _with_settings = None


def _post_with_end(settings):
    """Have gdb forget `settings` (_with_settings, as a `with` began) once it
    is back in its event loop, if they are still the ones kept then (see
    _with_settings)."""
    def end():
        global _with_settings, _with_depth
        if _with_settings is settings:
            _with_settings, _with_depth = None, 0
    _post(end)


def _post(event):
    """gdb.post_event(event), where gdb can be left with one posted: not
    where it dies as it exits with one left, as it would in -batch, or after
    a script's `quit` (gdb 9 to 14, before gdb 15's fix, whose gdb.interrupt
    came with it)."""
    if hasattr(gdb, "interrupt"):
        try:
            code = getattr(event, "__code__", None)
            if code is not None:
                _posted_codes.add(code)
            gdb.post_event(event)
        except Exception:
            _debug_exc()


def _check_with_settings():
    """Before a command shows a parked thread on the selected thread (and as
    a window opens): refuse if a `with` has changed one of _WITH_GUARDED
    around it.  (Before the registry is read, which the setting may keep
    gdb from reading.)"""
    if _with_settings is None:
        return
    for name, value in sorted(_with_guarded_settings().items()):
        if value != _with_settings.get(name, value):
            thread = _selected_thread()
            raise gdb.GdbError(
                "realm-gdb: can't show a parked Realm user thread on thread {} "
                "inside 'with {}': {}.".format(thread.num if thread is not None else "?",
                                               name, _WINDOW_GUARDED["set " + name]))


# Those of them that make function calls, or let the program run: refused
# too while registers are left that couldn't be put back (see _guard).
_RUNS_PROGRAM = ("delete checkpoint", "info static-tracepoint-markers", "strace")
# Commands that read the terminal themselves (Python's REPL), refused in a
# window too: in `rt`, the output is collected until the command is over,
# so what they ask wouldn't show; and the window would stay open while they
# wait.  (Python's input() and sys.stdin are _NoTerminalInput's.)
_TERMINAL_COMMANDS = ("python-interactive", "guile-repl")

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
    "info address", "info scope", "info display", "list", "print", "call", "output",
    "printf", "x", "ptype", "whatis", "display", "disassemble", "find", "set", "set variable",
    "eval", "init-if-undefined", "print-object", "dump *", "append *", "restore",
    "mem", "memory-tag *", "info vtbl", "info common", "info macro", "info macros",
    "macro expand", "macro expand-once", "break", "tbreak", "hbreak", "thbreak",
    "dprintf", "clear", "break-range", "trace", "ftrace", "tvariable", "skip",
    "skip function", "skip file")
# Commands refused while a parked thread is selected: they would act on the
# selected thread as it really is.  `rt` runs those of _RT_COMMANDS on the
# parked thread.  (`thread apply` and `frame apply` run other commands: see
# "Plain commands on a selected parked thread".  So does `pipe COMMAND |
# SHELL`, whose COMMAND, shown the parked thread, would keep its window
# open for as long as SHELL took to read COMMAND's output; `rt pipe` runs
# SHELL once the window is closed.)
_ACTING_COMMANDS = (
    "signal", "queue-signal", "step", "stepi", "next", "nexti",
    "finish", "until", "advance", "jump", "return", "reverse-step",
    "reverse-stepi", "reverse-next", "reverse-nexti", "reverse-finish",
    "reverse-continue", "watch", "rwatch", "awatch", "thread apply",
    "interpreter-exec", "explore", "explore *")
_RT_COMMANDS = ("frame apply", "frame apply *", "faas", "pipe")
# Commands that run the user's own code (scripts, and code compiled into the
# program) on the selected thread as it really is: while a parked thread is
# selected, they run all the same, with a note.  Refused, a bare `python`
# typed at a terminal would leave the lines of its body to run as gdb
# commands, a `continue` among them.  (A Python script run by `source`, or
# from a user-defined command, has always run so.)
_NOTED_COMMANDS = ("python", "guile", "compile", "compile *", "python-interactive",
                   "guile-repl")
# Commands that end the selection (all of the inferior's) before they run,
# rather than being refused: the user means to leave it.
_LEAVING_COMMANDS = ("continue",)
# Commands that end it once they have switched to another thread (`thread
# N`; `thread` alone, or naming the thread already selected, keeps it).
_SWITCHING_COMMANDS = ("thread",)
# Commands after which plain commands no longer show a selected parked
# thread.
_SESSION_COMMANDS = ("new-ui",)
# Commands that only close a failed plain command's window (see _close_stale)
# when run after it: a Python script sourced then would otherwise count as
# part of a command typed at the top level (see _inside), and `inferior N`'s
# switch would be undone when the window closed later.
_CLOSING_COMMANDS = ("source", "inferior")
# The settings a window changes (see _CallSettings): `set` of one, and
# `with`, which saves the setting's value to put back, are closing commands
# too, or the window would later undo the user's `set`, or `with` put back
# the window's value once it was gone.
_WINDOW_SETTINGS = ("scheduler-locking", "unwindonsignal", "unwind-on-signal",
                    "unwind-on-terminating-exception", "stop-on-solib-events",
                    "may-insert-breakpoints", "follow-fork-mode", "detach-on-fork", "confirm",
                    "multiple-symbols", "pagination", "direct-call-timeout",
                    "indirect-call-timeout", "unwind-on-timeout", "may-call-functions")
_CLOSING_COMMANDS += ("with", "maintenance with") + tuple(
    "set " + name for name in _WINDOW_SETTINGS)
# Commands that read the lines after them (see _BODY_COMMANDS), refused in a
# window whose command's output is collected (`rt`, `realm-thread apply`),
# however they come there (`rt with ... -- define`, `rt eval "commands 2"`,
# a user-defined command), which can't pass those lines on.
_BODY_HOOKED = ("commands", "define", "document", "while-stepping", "actions", "while",
                "if")
# Commands that switch gdb's output to another stream (a log file, the TUI's
# windows) or back, refused while realm-gdb collects a command's output
# (`rt`, `realm-thread apply`, in core files too: _collecting), however
# they come there: gdb keeps the stream it switched from, the one collecting
# the output, to switch back to later, once it is gone, and then crashes (at
# the next `set logging enabled off` or `tui disable`; gdb 18 fixed
# logging's).  Each of the TUI's commands but `tui new-layout` turns the TUI
# on first.  (Also `+`, `-`, `<` and `>`, which can't be hooked: refused by
# name, see _check_command.)
_STREAM_COMMANDS = ("set logging enabled", "set logging on", "set logging off",
                    "set logging redirect", "set logging debugredirect", "tui enable",
                    "tui disable", "tui reg", "tui layout", "tui layout *", "layout",
                    "layout *", "tui focus", "focus", "tui refresh", "refresh",
                    "tui window height", "tui window width", "winheight", "winwidth",
                    "tabset")
_STREAM_COMMANDS_UNHOOKED = ("+", "-", "<", ">")
# Commands that start another program (see "Commands that start another
# program"), refused inside a window: `rt` runs them without one, but in a
# command that has one (a user-defined command run with `rt`, `eval` shown a
# selected parked thread), the program would run with the host holding the
# parked thread's registers.
_EXTERNAL_COMMANDS = ("shell", "make", "pipe", "edit")

# The commands hooked, by what their hooks do: "guarded" (_GUARDED_COMMANDS
# and _TERMINAL_COMMANDS), "acting", "rt" (a subset of "acting"), "leaving",
# "switching", "showing", "session", "external", "closing", "body" and
# "stream".
# (Kept if the script is loaded again: until the hooks are all defined
# again, those of the earlier load, the same, are in place.)
_hooked = globals().get("_hooked", {})
# The hooks of realm-gdb's that keep lines of the user's own: hook -> the
# lines (see _define_hook).
_user_hooks = globals().get("_user_hooks", {})

# Where a switching command was run from, while it runs: (thread, inferior
# number) of the selected thread then.
_switch_from = None


def _existing_commands(names):
    """Those of `names` that this gdb has, found without running a command
    that fails (see _install_hooks): `complete NAME` lists NAME itself.
    (With completions limited below gdb's default, `set max-completions 0`
    for one, it would list nothing: they are unlimited meanwhile.)"""
    with _Temporarily("max-completions", "unlimited",
                      lambda old: isinstance(old, int) and 0 <= old < 200):
        found = []
        for name in names:
            if name.startswith("set ") and name[len("set "):] in _WINDOW_SETTINGS:
                # (`complete set X` completes an expression too, once there
                # are symbols: `set unwind-on-signal` gets `signal`'s.)
                if _has_parameter(name[len("set "):]):
                    found.append(name)
                continue
            try:
                out = _execute("complete " + name)
            except gdb.error:
                continue
            if name in [line.strip() for line in out.splitlines()]:
                found.append(name)
        return found


# Commands to guard that gdb keeps out of `complete` and `help`, as
# deprecated: `target record`, which starts record-full as `target
# record-full` does (a command of its own, with hooks of its own).
_DEPRECATED_GUARDED = ("target record",)


def _deprecated_commands(names):
    """Those of deprecated commands `names` that this gdb has: `help NAME`
    finds them."""
    found = []
    for name in names:
        try:
            _execute("help " + name)
            found.append(name)
        except gdb.error:
            pass
    return found


def _is_alias(name):
    """Whether command `name` is an alias of another (gdb >= 10, which lists
    a command's names first in its help, its own first: `tui focus, fs,
    focus`).  Its hooks would never run: gdb runs those of the command it
    stands for."""
    try:
        first = _execute("help " + name).split("\n", 1)[0]
    except gdb.error:
        return False
    names = [n.strip() for n in first.split(",")]
    return name in names and names[0] != name


def _subcommands(prefix):
    """gdb's subcommands of `prefix` (none if it isn't a prefix command):
    not hooks, nor other user-defined commands, which `help` lists among
    them; realm-gdb's own hooks of an earlier load would otherwise be hooked
    in turn, and so on at each load."""
    try:
        out = _execute("help " + prefix)
    except gdb.error:
        return []
    return sorted(set(prefix + " " + word for word, rest in re.findall(
        r"^{} ([\w-]+)((?: --|,).*)$".format(re.escape(prefix)), out, re.M)
        if not word.startswith(("hook-", "hookpost-")) and
        not rest.rstrip().endswith("-- User-defined.")))


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


def _refuse_stream(cmd, where=None):
    """Refuse `cmd`, of _STREAM_COMMANDS, where realm-gdb collects a command's
    output (`where`, for messages)."""
    if where is None:
        where = (_window_words(_window) if _window is not None else
                 "'realm-thread apply'")
    raise gdb.GdbError(
        "realm-gdb: refusing '{}' in {}, whose output gdb collects: gdb would later "
        "switch {} back to that collection, gone by then, and crash.  Run it on its "
        "own.".format(cmd, where, "logging" if "logging" in cmd else "the TUI"))


def _window_words(window):
    """What opened `window`, for messages."""
    if window.via is None:
        return "'{}'".format(window.opener or "realm-thread apply")
    if window.via == "rt":
        return "'rt'"
    return "'{}', which shows the selected one".format(window.via)


# Why a guarded command is refused in a window, where it isn't for the
# reason the others are (see _guard).
_GUARDED_WHY = {
    "queue-signal": "the signal would go to thread {thread}, not to the user thread",
    "generate-core-file": "the core file would have them as thread {thread}'s own",
    "checkpoint": "the checkpoint, a copy of the program, would keep them in thread "
                  "{thread}",
}
_GUARDED_WHY.update((name, "gdb would record the program from them, as thread {thread}'s "
                           "own") for name in ("record", "record full", "record btrace"))


def _refusal(cmd, why, hint=""):
    """Refusing `cmd` in the window open now: `why`, and what to do."""
    whose = (_ut_words(_window.showing) + "'s" if _window.showing else
             "a parked Realm user thread's")
    if _window.via not in (None, "rt"):  # (A plain command's.)
        return ("realm-gdb: refusing '{}' inside '{}': thread {} has {} registers until "
                "'{}' is over: {}.{}".format(cmd, _window.via, _window.thread_num, whose,
                                             _window.via, why, hint))
    return ("realm-gdb: refusing '{}' in {}, where thread {} has {} registers until the "
            "command is over: {}.{}".format(
                cmd, _window_words(_window), _window.thread_num, whose, why, hint))


def _guard(cmd, like=None):
    """Refuse command `cmd` where it is refused (as `like`, if given: the CLI
    command that MI command `cmd` is the same as)."""
    global _pi_allowed
    if cmd == "python-interactive" and _pi_allowed:
        _pi_allowed = False  # (`rt pi EXPR`, which reads no terminal: see _check_command.)
        return
    if _window is not None and cmd in _TERMINAL_COMMANDS:
        raise gdb.GdbError(_refusal(
            cmd, "it reads the terminal, and {}".format(
                "its output is only shown once the command is over"
                if _window.via in (None, "rt") else "the thread would keep them while it "
                "waits")))
    if cmd in _TERMINAL_COMMANDS:
        return
    if (like or cmd) in _WINDOW_GUARDED:
        if _window is not None:
            raise gdb.GdbError(_refusal(cmd, _WINDOW_GUARDED[like or cmd]))
        if (like or cmd) not in _RUNS_PROGRAM:
            return
    if _window is not None:
        hint = ""
        if _window.via == "rt":
            # (Plain commands that would act on the thread as it really is
            # are refused while they show the selection.)
            hint = ("  Run it after 'realm-thread back', without 'rt'."
                    if _plain_off_why is None and cmd in _hooked.get("acting", ()) else
                    "  Run it without 'rt'.")
        raise gdb.GdbError(_refusal(cmd, _GUARDED_WHY.get(
            cmd, "the program mustn't run, or be let go of, meanwhile").format(
                thread=_window.thread_num), hint))
    # (Killing the program is the way out, as the message says.)
    if _unrestored and cmd.split()[0] not in ("kill", "queue-signal") and \
            not _retry_unrestored():
        if cmd == "quit" and (not _stdin_is_terminal() or _batch_session() or
                              not _attached_unrestored()):
            # (gdb kills a program it started as it quits.  And from a
            # script, rather than a user who can do as the message says, a
            # `quit` would come again at every end of its input; in batch
            # mode, gdb lets go of the program as the script ends.)
            _kill_unrestored()
            return
        retry = any(w.left for w in _unrestored)
        raise gdb.GdbError(
            "realm-gdb: refusing '{}': {}, so {}.  {}.".format(
                cmd, "; ".join(_not_back_words(w) for w in _unrestored),
                "a core file written now would have the parked thread's"
                if cmd == "generate-core-file" else
                "the program would run, or be let go of, corrupted",
                "Run 'realm-thread back' to try again, or 'kill' the program" if retry
                else "'kill' the program"))


def _batch_session():
    """Whether gdb runs in batch mode (-batch, -batch-silent), judged from
    its command line (see _read_session; once: without /proc, that runs
    `ps`)."""
    if not _batch:
        batch = _read_session()[1]
        if batch is not None:
            _batch.append(batch)
    return bool(_batch and _batch[0])


_batch = []


def _attached_unrestored():
    """Whether a program whose host's registers couldn't be put back was
    attached to, rather than started by gdb."""
    for window in _unrestored:
        try:
            if window.thread.is_valid() and getattr(window.thread.inferior,
                                                    "was_attached", True):
                return True
        except RuntimeError:
            pass
    return False


def _stdin_is_terminal():
    try:
        return os.isatty(0)
    except OSError:
        return False


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
    """realm-gdb-guard: hook-CMD's work (hookpost-CMD's, `after`), run
    from Python frames `outer` (see _outer_frames)."""
    global _switch_from, _second_ui
    if _internal_depth:
        # (A command run meanwhile by something else, say another extension's
        # handler of gdb.events.register_changed, is still guarded, and one
        # that starts another program refused in a window.)
        if after or _internal_commands[-1].startswith(cmd):
            return
        if cmd in _hooked.get("guarded", ()):
            _guard(cmd)
        elif cmd in _hooked.get("external", ()) and _window is not None:
            _refuse_external(cmd)
        return
    if after:
        if cmd in _hooked.get("switching", ()):
            _switched(cmd)
        if cmd == "define":
            _check_guard_hooks()  # (A hook of the user's that replaced realm-gdb's.)
        if cmd == "source" and _window is not None and _window.sourcing:
            _window.sourcing -= 1
        if cmd == "with":
            _with_hook(True)
        _plain_after(cmd, outer)
        return
    _close_stale(outer)
    if cmd == "with":
        _with_hook(False)
    if cmd in _WITH_SET and _with_settings is not None:
        _with_settings.pop(_WITH_SET[cmd], None)
    if cmd in _hooked.get("guarded", ()):
        _guard(cmd)
    if cmd == "source" and _window is not None:
        # (A file it reads gives its blocks' lines: `rt source FILE`.)
        _window.sourcing += 1
    if cmd in _hooked.get("body", ()) and _window is not None and \
            _window.via in (None, "rt") and not _window.sourcing:
        raise gdb.GdbError(
            "realm-gdb: refusing '{}' in {}: it reads the lines after it, up to "
            "'end', which can't come with a command run there.  Run it on its "
            "own.".format(cmd, _window_words(_window)))
    if cmd in _hooked.get("external", ()) and _window is not None:
        _refuse_external(cmd)
    if cmd in _hooked.get("stream", ()) and (_collecting or _window is not None and
                                              _window.via in (None, "rt")):
        _refuse_stream(cmd)
    if cmd in _hooked.get("session", ()):
        _second_ui = "a second UI was opened ('{}')".format(cmd)
        _plain_off(_second_ui)
    if cmd in _hooked.get("acting", ()):
        _refuse_acting(cmd)
    if cmd in _hooked.get("noted", ()):
        _note_host_command(cmd)
        if cmd == "python-interactive":
            _install_repl_prompts()
    if cmd in _hooked.get("leaving", ()) and _plain_shows() is not None:
        inferior = gdb.selected_inferior().num
        _drop_selections(lambda sel: sel["inferior"] == inferior, "continuing")
    if cmd in _hooked.get("switching", ()):
        _switch_from = None
        if _plain_shows() is not None:
            _switch_from = (_selected_thread(), gdb.selected_inferior().num)
        elif _views:
            _switch_from = (None, None)  # (A core file's: see _switched.)
    if cmd in _hooked.get("showing", ()):
        _plain_before(cmd, outer)


def _refuse_external(cmd, error=gdb.GdbError):
    raise error(_refusal(cmd, "the program it starts would run meanwhile",
                         "  Run it on its own."))


# Python's audit events (Python >= 3.8) of starting another program.
_SPAWN_EVENTS = frozenset(("os.system", "subprocess.Popen", "os.exec", "os.posix_spawn",
                           "os.spawn", "os.fork", "os.forkpty", "pty.spawn"))


def _audit(event, args):
    """Python code that starts another program in a window (`rt python
    os.system(...)`, a pretty-printer's subprocess): refused, as the
    commands that do are (see "Commands that start another program"),
    however it got the function.  (Not from another thread of Python's,
    which doesn't keep the window open; nor after a plain command that
    failed, whose window is closed first, as a hooked command would.)"""
    if _window is not None:
        import threading
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            _close_stale(_outer_frames(sys._getframe(1)))
        except Exception:
            _debug_exc()
        if _window is not None:
            _refuse_external(event, RuntimeError)


def _install_audit():
    """Have _audit see Python's audit events.  Once per gdb (a hook can't be
    taken away again): it calls whatever _audit is now, so loading the
    script again updates it.  And _posixsubprocess.fork_exec, which
    multiprocessing starts programs with, raising no event."""
    g = globals()
    if g.get("_audit_installed") or not hasattr(sys, "addaudithook"):
        return

    def hook(event, args):
        if event in g.get("_SPAWN_EVENTS", ()):
            g["_audit"](event, args)
    sys.addaudithook(hook)
    g["_audit_installed"] = True
    try:
        import _posixsubprocess
        real = _posixsubprocess.fork_exec
    except (ImportError, AttributeError):
        return

    def fork_exec(*args, **kwargs):
        g["_audit"]("_posixsubprocess.fork_exec", args)
        return real(*args, **kwargs)
    _posixsubprocess.fork_exec = fork_exec


def _install_thread_sigmask():
    """Have each thread of Python's started from now on (threading's, as
    extensions start theirs) block Ctrl-C (SIGINT) first thing, as gdb's
    own threads and gdb.Thread do: else windows can't be opened while it
    runs (see _sigint_takers).  (Python handles signals on its main thread
    only, so the thread loses nothing.)  Once per gdb."""
    import threading
    real = getattr(threading.Thread, "_bootstrap", None)
    if real is None or getattr(real, "_realm_gdb", False):
        return

    def _bootstrap(self, *args, **kwargs):
        try:
            import signal
            signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
        except Exception:
            pass
        return real(self, *args, **kwargs)
    _bootstrap._realm_gdb = True
    threading.Thread._bootstrap = _bootstrap


class RealmThreadGuard(gdb.Command):
    """Check a command for realm-gdb.py's hooks (internal).

    Usage: realm-gdb-guard [--after] COMMAND

    Used by realm-gdb.py's hook-COMMAND and hookpost-COMMAND definitions.
    Fails, which stops COMMAND from running, if COMMAND would let the program
    run or let go of it while a thread has a parked Realm user thread's
    registers (inside `rt`, `realm-thread apply` or `thread apply all bt`);
    or, while a parked thread is selected (by `realm-thread <ptr>`), if it
    would act on the selected thread as it really is.  If COMMAND looks at
    the selected thread or frame, a selected parked thread is shown for its
    duration."""

    def __init__(self):
        super(RealmThreadGuard, self).__init__("realm-gdb-guard",
                                               gdb.COMMAND_MAINTENANCE)

    def invoke(self, arg, from_tty):
        if not arg.strip():
            return  # (See _hide_guard.)
        after = arg.startswith("--after ")
        cmd = (arg[len("--after "):] if after else arg).strip()
        try:
            _hook(cmd, after, _outer_frames(sys._getframe().f_back))
        except gdb.GdbError:
            raise
        except KeyboardInterrupt:
            # (gdb 8 takes a command's KeyboardInterrupt for an error, and says
            # "Python Exception <class 'KeyboardInterrupt'> Quit:": "Quit".
            # Later gdbs quit, as they should.  One that lands before the
            # first line here, as gdb runs it, isn't caught.)
            if _COMMANDS_QUIT:
                raise
            raise gdb.GdbError("Quit")
        except Exception as e:
            _debug_exc()
            # (Not let through for want of the check, where it may matter.)
            if not after and (_window is not None or _unrestored or _selections) and \
                    any(cmd in _hooked.get(k, ()) for k in ("guarded", "external", "acting")):
                raise gdb.GdbError("realm-gdb: refusing '{}': realm-gdb's check of it "
                                   "failed ({}: {}).  (REALM_GDB_DEBUG=1 in gdb's "
                                   "environment shows where.)".format(
                                       cmd, type(e).__name__, e))


_entry_codes.add(RealmThreadGuard.invoke.__code__)


def _switched(cmd):
    """hookpost-CMD of a switching command: if it switched to another
    thread while plain commands showed a selection, end the selection (all
    of the inferior's), saying so; if not, say it is kept (gdb may have
    printed the thread's own frame, which the next command won't show)."""
    global _switch_from
    if _switch_from is None:
        return
    (old, inferior), _switch_from = _switch_from, None
    new = _selected_thread()
    if inferior is None:
        # (A core file's views stay on their threads, the one now selected
        # perhaps: gdb printed its own frame #0.)
        view = _current_view()
        if view is not None and not view["temporary"]:
            print("realm-thread: thread {} still shows {}: its stack starts "
                  "at frame #1 (#0 is {}), until 'realm-thread back'.".format(
                      view["thread_num"], _ut_words(view["user_thread"]),
                      _frame0_words(view["thread_num"])))
        return
    if new is not None and new == old:
        sel = _plain_shows()
        if sel is not None:
            print("realm-thread: still selecting {}: plain commands show "
                  "it, not thread {} as it really is, until 'realm-thread back'.".format(
                      _ut_words(sel["user_thread"]), new.num))
        return
    if new is None:
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
    bodies = _user_command_bodies()
    with _Temporarily("confirm", "off"):  # (Redefining a hook asks at a terminal.)
        _define_hooks(bodies)


def _user_command_bodies():
    """{name: [line, ...]} of the user-defined commands, hooks among them,
    from one `show user`, without realm-gdb's own check (an earlier
    load's) in them."""
    try:
        out = _execute("show user")
    except gdb.error:
        return {}
    return dict((name, _users_lines(body.splitlines()))
                for name, body in _show_user_bodies(out).items())


def _show_user_bodies(out):
    """{name: body} of `show user`'s output `out`, as it shows each body
    (indented)."""
    return dict((m.group(1), m.group(2)) for m in re.finditer(
        r'^User command "([^"]+)":\n((?:  .*\n?|\n)*)', out, re.M))


def _users_lines(lines):
    """The user's own of a hook's `lines`, as `show user` shows them
    (indented): not realm-gdb's check (realm-thread-guard, in an older
    realm-gdb.py), nor empty ones."""
    lines = [line[2:] if line.startswith("  ") else line for line in lines]
    return [line for line in lines if line.strip() and
            not line.startswith(("realm-gdb-guard ", "realm-thread-guard "))]


def _define_hooks(defined):
    """_install_hooks's work, given _user_command_bodies()."""
    kinds = {
        "guarded": set(_existing_commands(_GUARDED_COMMANDS + _TERMINAL_COMMANDS +
                                          tuple(sorted(_WINDOW_GUARDED))) +
                       _subcommands("target") + _deprecated_commands(_DEPRECATED_GUARDED)),
        "rt": set(_existing_commands(_expand(_RT_COMMANDS))),
        "acting": set(_existing_commands(_expand(_ACTING_COMMANDS))),
        "leaving": set(_existing_commands(_LEAVING_COMMANDS)),
        "switching": set(_existing_commands(_SWITCHING_COMMANDS)),
        "session": set(_existing_commands(_SESSION_COMMANDS)),
        "external": set(_existing_commands(_EXTERNAL_COMMANDS)),
        "closing": set(_existing_commands(_CLOSING_COMMANDS)),
        "noted": set(_existing_commands(_expand(_NOTED_COMMANDS))),
        "body": set(_existing_commands(_BODY_HOOKED)),
        "stream": set(c for c in _existing_commands(_expand(_STREAM_COMMANDS))
                      if not _is_alias(c)),
    }
    if "set logging enabled" in kinds["stream"]:
        # (gdb 12 made `set logging on` and `off` its aliases, which run its
        # hooks.)
        kinds["stream"] -= set(["set logging on", "set logging off"])
    kinds["acting"] |= kinds["rt"]
    kinds["showing"] = (set(_existing_commands(_expand(_SHOWING_COMMANDS))) -
                        kinds["acting"] - kinds["noted"])
    # (Built here, then put in place: an interrupted load leaves the guards
    # counted missing, and windows refused, rather than half there.)
    hooked = {}
    for c in sorted(set().union(*kinds.values())):
        hooks = _hook_names(c, kinds)
        # The user's own hooks are kept, after realm-gdb's check.
        try:
            for hook in hooks:
                _define_hook(hook, c, defined.get(hook, []), kinds)
        except gdb.error as e:
            print("realm-gdb: could not define {}: {}".format(hook, e))
            continue
        for kind, commands in kinds.items():
            if c in commands:
                hooked.setdefault(kind, set()).add(c)
    _hooked.clear()
    _hooked.update(hooked)
    _unguarded[:] = [c for c in _REQUIRED_GUARDS if c not in _hooked.get("guarded", ())]
    _keep_abbreviations(hooked)


# The aliases _keep_abbreviations has made: (prefix, abbreviation).  (Kept if
# the script is loaded again: making one again would fail, and gdb 12 stops a
# function call whose loading of a library, as it loads the script, runs a
# command that fails, even one that Python catches.)
_abbreviations = globals().get("_abbreviations", set())


def _keep_abbreviations(hooked):
    """Abbreviations that gdb took for one subcommand, which realm-gdb's
    hooks in the same list (`info hook-address`) would make ambiguous (`info
    h`, `set ho`, `tui window h`): each made an alias of it, as gdb's own
    abbreviations are (`alias -a`), which gdb takes over the hooks.  (gdb 10
    and later list it among the command's names in `help`.)"""
    prefixes = set(c.rsplit(" ", 1)[0] for c in set().union(*hooked.values()) if " " in c)
    for prefix in sorted(prefixes):
        names = _subcommand_names(prefix)
        for short in ("h", "ho", "hoo", "hook"):
            matches = [n for n in names if n.startswith(short)]
            if len(matches) != 1 or matches[0] == short or (prefix, short) in _abbreviations:
                continue
            _abbreviations.add((prefix, short))
            try:
                _internal("alias -a {0} {1} = {0} {2}".format(prefix, short, matches[0]))
            except gdb.error:
                _debug_exc()  # (The user's own, say.)


def _subcommand_names(prefix):
    """The names of `prefix`'s subcommands that `help PREFIX` lists, aliases
    among them, but realm-gdb's hooks."""
    try:
        out = _execute("help " + prefix)
    except gdb.error:
        return []
    names = set()
    for line in out.splitlines():
        for part in line.split(" -- ", 1)[0].split(", "):
            if part.startswith(prefix + " "):
                word = part[len(prefix) + 1:]
                if word and " " not in word and not word.startswith(("hook-", "hookpost-")):
                    names.add(word)
    return names


def _hook_names(c, kinds):
    """The hooks realm-gdb defines for command `c` (`kinds`: kind -> its
    commands): hook-C, and hookpost-C for some."""
    hooks = [_hook_name(c, "hook")]
    if c in kinds.get("showing", ()) or c in kinds.get("switching", ()) or \
            c in ("define", "source", "with"):
        hooks.append(_hook_name(c, "hookpost"))
    return hooks


def _define_hook(hook, c, body, kinds):
    """Define `hook` of command `c` (`kinds`: kind -> its commands):
    realm-gdb's check, and the lines of the user's own `body`, if any
    (saying so).  The check comes first, but last in hook-C of a showing
    command: its window would be closed by a hooked command among the
    user's lines (`output`, say), and C would then run on the host."""
    check = "realm-gdb-guard {}{}\n".format("--after " if "hookpost-" in hook else "", c)
    last = body and "hookpost-" not in hook and c in kinds.get("showing", ())
    lines = "".join(line + "\n" for line in body)
    _internal("define {}\n{}end\n".format(hook, lines + check if last else check + lines))
    said = _user_hooks.pop(hook, None)
    if body:
        _user_hooks[hook] = tuple(body)
    else:
        # (Out of completion and `help`, which list a deprecated command
        # only when nothing else matches.  gdb runs a hook without looking
        # it up, so with no warning of it.)
        try:
            _internal("maintenance deprecate " + hook)
        except gdb.error:
            _debug_exc()
    if body and said != tuple(body):  # (Not again as the script loads again.)
        print("realm-gdb: your {} now runs realm-gdb's check of '{}' {}.".format(
            hook, c, "after its own commands" if last else "first"))


def _check_guard_hooks():
    """Put realm-gdb's check back in any hook of its that has lost it: a
    `define hook-CMD` run where gdb runs no hooks, in a block (`if`, a
    user-defined command, breakpoint commands) or by Python with its body,
    replaces realm-gdb's hook unseen, as does a typed one.  One `show
    user`, a few milliseconds: as a window opens, after each `define`, and
    at a prompt while registers are left that couldn't be put back."""
    try:
        out = _execute("show user")
    except gdb.error:
        return
    bodies = _show_user_bodies(out)
    lost = []
    for c in sorted(set().union(*_hooked.values()) if _hooked else ()):
        for hook in _hook_names(c, _hooked):
            body = bodies.get(hook)
            if body is None or "realm-gdb-guard " not in body:
                lost.append((hook, c, _users_lines((body or "").splitlines())))
    if not lost:
        return
    with _Temporarily("confirm", "off"):
        for hook, c, body in lost:
            try:
                _define_hook(hook, c, body, _hooked)
            except gdb.error:
                _debug_exc()


# Commands every gdb has, without a guard on which windows mustn't open (see
# _how_shown); and those of them that have none, if realm-gdb couldn't
# define it.  (Kept if the script is loaded again, as _hooked is.)
_REQUIRED_GUARDS = ("continue", "step", "next", "finish", "detach", "kill", "quit", "run")
_unguarded = globals().get("_unguarded", list(_REQUIRED_GUARDS))


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

# How many _ViewsOff are in force, which leave every thread as it is.
_views_off = 0

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
    if not view["temporary"] and not _selections and not _any_kept_views():
        _remove_prompt_hook()
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


class _ViewsOff(object):
    """Threads as they are, with no view shown (`thread apply all`'s own
    threads, as in a live process).  Frame #0, which a view leaves the
    host's own, is selected meanwhile: gdb can select it again either way
    as `thread apply` ends.  The frame selected before is afterwards."""

    def __enter__(self):
        global _views_off
        self.level = _selected_level()
        if _current_view() is not None:
            _select_level(0)
        _views_off += 1
        gdb.invalidate_cached_frames()
        return self

    def __exit__(self, *exc):
        global _views_off
        _views_off -= 1
        _refresh_frames(self.level)
        return False


class _FrameId(object):
    def __init__(self, sp, pc):
        self.sp = sp
        self.pc = pc


class _ParkedThreadUnwinder(gdb.unwinder.Unwinder):
    def __init__(self):
        super(_ParkedThreadUnwinder, self).__init__("realm-parked-thread")

    def __call__(self, pending_frame):
        global _last_unwound_thread
        if _views_off:
            return None
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


def _close_at_prompt():
    """At a top-level prompt, which is never inside a window: close one left
    open, by a command that failed (see "Plain commands ...") or by a Ctrl-C
    as it closed (see _Window.__exit__).  (Not another UI's prompt during a
    function call, made in a window that is open.)"""
    if _call_running():
        return
    if _plain is not None:
        _close_plain(_plain, idle=True)
    elif _window is not None:
        try:
            _window.__exit__(None, None, None)
        finally:
            _let_held(idle=True)  # (See _Window.hold_sigint.)
    elif _sigint_is_blocked():
        _let_held(idle=True)  # (Never blocked outside a window.)


def _post_close(window):
    """Have gdb close `window`, if it is left open, as soon as it is back in
    its event loop, between commands (see _close_at_prompt): there may be no
    prompt event to do it, in MI and DAP, at a stop in the background
    (`continue &`) whose breakpoint commands ran a command that failed, or
    after a posted event's.  (During a function call made in it, gdb runs
    its event loop too: not then.)"""
    def close():
        try:
            # (Not by is_running(), false for a thread running a call, nor
            # between its stop and gdb's inferior_call_post, before gdb pops
            # it, putting back the registers it had then, the parked
            # thread's: see _calls_left_up.)
            if _window is window and _between_calls(window):
                _close_at_prompt()
        except Exception:
            _debug_exc()
    _post(close)


def _between_calls(window):
    """Whether no function call made in `window` is in progress (one that
    failed before it ran left its count up), so that it can be closed."""
    return (not _infcall_depth or _calls_left_up(window)) and not _host_in_call(window)


def _host_in_call(window):
    """Whether `window`'s host, selected, is running a function call, which
    only the command that made it can be in the middle of, whatever the
    count of calls says (see _call_running): gdb won't read its registers
    then, nor write them back."""
    try:
        if _selected_thread() == window.thread:
            gdb.newest_frame()
    except gdb.error as e:
        return "running" in str(e)
    except Exception:
        _debug_exc()
    return False


def _note_call_start(ptid):
    """At a function call's inferior_call_pre, on thread `ptid`: note the
    frame it is made from, the thread's newest, which the call's dummy frame
    will have next (see _call_running).  A call counted before, made from
    that frame too, can't be in progress (one is made from inside another's
    frames): it failed before it ran."""
    frame = None
    try:
        frame = gdb.newest_frame()
        for i, (p, f) in enumerate(_call_starts):
            if p == ptid and f is not None and f == frame:
                _call_starts[i] = (p, None)
    except Exception:
        _debug_exc()
    _call_starts.append((ptid, frame))


def _call_running():
    """Whether a function call is in progress: one counted by
    gdb.events.inferior_call, made on the selected thread, which is running
    it (gdb pretends otherwise, as a call runs: is_running() is false, but
    gdb won't read its registers) or stopped in it (at a dprintf, say), with
    the call's dummy frame on its stack, just newer than the frame it was
    made from (see _note_call_start).  A count that a call which failed
    before it ran left up (gdb sends no post event then) is dropped, and its
    breakpoints enabled: the dummy frame of a call the user left stopped is
    not one of the calls counted.  (Where those can't be told, as counted
    by an earlier load, any dummy frame keeps it up.)"""
    global _infcall_depth
    if not _infcall_depth:
        return False
    try:
        starts = None
        if len(_call_starts) == _infcall_depth:
            thread = _selected_thread()
            ptid = thread.ptid if thread is not None and thread.is_valid() else None
            starts = [f for p, f in _call_starts if f is not None and p == ptid]
        with _AllFrames():
            for frame in _stack():
                if frame.type() == gdb.DUMMY_FRAME:
                    older = frame.older()
                    if starts is None or (older is not None and older in starts):
                        return True
    except gdb.error as e:
        if "running" in str(e):
            return True
        _debug_exc()  # (No thread, or no registers: none is running.)
    except Exception:
        _debug_exc()
        return True
    _infcall_depth = 0
    del _call_starts[:]
    _enable_breakpoints()
    return False


def _calls_left_up(window):
    """Whether the function calls counted since `window` opened all failed
    before they ran, which left their count up (see _call_running), as told
    between commands: the thread the window's last call was made on,
    selected, isn't running one, and has no dummy frame it didn't have as
    that call began (one stopped, before gdb's inferior_call_post, has).
    (More closely than _call_running, which a window's own call left
    stopped would mislead.)"""
    last = window.last_call
    if last is None:
        return False
    try:
        thread = _selected_thread()
        if thread is None or thread.ptid != last[0]:
            return False  # (Not without selecting it.)
        # (Not _dummy_frames, which takes an error for the end of the stack:
        # gdb won't read the registers of a thread running a call.)
        with _AllFrames():
            return not any(frame.type() == gdb.DUMMY_FRAME and frame not in last[1]
                           for frame in _stack())
    except Exception:
        _debug_exc()
    return False


def _calls_over(window):
    """A command run in `window` is over, and so is every function call it
    made: drop the count that one which failed before it ran left up (gdb
    sends no post event then), doing what the end of the window's calls
    does (see _on_inferior_call), and block Ctrl-C again, which the call let
    through (see _block_sigint)."""
    global _infcall_depth, _sigint_call
    _sigint_call = False
    depth = window.depth
    if _infcall_depth > depth:
        _infcall_depth = depth
        del _call_starts[depth:]
        _calls_ended(window)
    if not window.interruptible:
        _block_sigint(True)


def _on_before_prompt():
    """Tidy up after a command (see "the marker line"), and on gdb < 11 show
    the selected thread's view again if gdb rebuilt its frames without the
    unwinder knowing (e.g. `thread apply 1 2 3 print 1`), going back to the
    frame the user had selected."""
    global _infcall_depth, _with_settings, _with_depth
    _close_at_prompt()
    _with_settings, _with_depth = None, 0  # (A `with` that failed: see there.)
    if any(w.stopping for w in _unrestored):
        _retry_unrestored()  # (A host stopped before it ran: see _stop_before_run.)
    if _unrestored:
        _check_guard_hooks()  # (Their refusals are the hooks'.)
    if _selections or _any_kept_views():
        _install_prompt_hook()  # (Around one set since: see _realm_prompt.)
    _printing_over()
    if _window is None:
        _call_running()  # (Drops a count of calls left up: see there.)
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


def _connect(registry_name, handler_name, tag=""):
    """Connect a handler once per gdb session (and `tag`), even if this
    script is loaded again (by another copy of librealm, or by hand).  The
    connection looks the handler up by name, so the most recently loaded
    copy runs."""
    flag = "_realm_gdb_connected_" + registry_name + tag
    registry = getattr(gdb.events, registry_name, None)
    if registry is None or getattr(gdb, flag, False):
        return
    namespace = globals()

    def dispatch(*args):
        return _guarded(namespace[handler_name])(*args)
    _dispatch_codes.add(dispatch.__code__)
    _entry_codes.add(dispatch.__code__)
    registry.connect(dispatch)
    setattr(gdb, flag, True)


# The code of the functions that _connect connects (see _python_beneath).
# (Kept if the script is loaded again: the first load's stay connected.)
_dispatch_codes = globals().get("_dispatch_codes", set())


def _python_beneath(frame):
    """Whether Python code runs beneath the event handler running in
    `frame` (a gdb.execute whose command made gdb send the event), rather
    than gdb alone, past the frames of _connect's dispatch.  (True where it
    can't tell: a handler connected by an earlier load, without these.)"""
    while frame is not None and frame.f_code not in _dispatch_codes:
        frame = frame.f_back
    return frame is None or frame.f_back is not None


def _install_event_handlers():
    gdb.unwinder.register_unwinder(None, _ParkedThreadUnwinder(), replace=True)
    gdb.frame_filters["realm-parked-thread"] = _frame_filter
    _connect("cont", "_on_cont")
    if sys.platform.startswith("freebsd") and hasattr(gdb, "BP_CATCHPOINT"):
        _connect("stop", "_on_stop")  # (See _at_syscall.)
    _connect("exited", "_on_exited")
    _connect("inferior_call", "_on_inferior_call")
    _connect("breakpoint_created", "_on_breakpoint_created")
    _connect("before_prompt", "_on_before_prompt")
    _connect("gdb_exiting", "_on_gdb_exiting")
    _connect("tui_enabled", "_on_tui_enabled")  # (gdb >= 16)
    # (Twice: on gdb 12 and 13, the quit a SIGTERM makes raises
    # KeyboardInterrupt as the first one starts, and gdb carries on to the
    # next.  When the first one has run, the second finds nothing to do.)
    _connect("gdb_exiting", "_on_gdb_exiting", "_again")


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
    if word not in ("i", "info") and _command_named(word, ("backtrace",)):
        return rest.strip()  # (An alias of the user's.)
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
    return "bt " + " ".join(options + [_number_words(n)])


def _unwind_words(view, limit):
    """Why `view`'s parked stack wasn't unwound, given `set backtrace
    limit`."""
    if limit and limit <= view["host_level"] + 1:
        return "within 'set backtrace limit {}'".format(limit)
    return "(is the realm-parked-thread unwinder disabled? see 'info unwinder')"


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
        with _Temporarily("backtrace limit", str((limit or 0) + view["host_level"] + 1),
                          lambda old: bool(limit)):
            _refresh_frames("parked")
            if not _shown(view):
                raise _Unshowable("it can't be unwound {}".format(_unwind_words(
                    view, limit)))
            if callable(command):  # (see _apply_external)
                return command()
            return _apply_execute(_view_command(command, view))
    finally:
        _remove_view(view)
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
    if word not in _FRAME_WORDS + _UP_WORDS + _DOWN_WORDS:
        # (The user's alias, and the arguments it gives: `alias u2 = up 2`.)
        expanded = _alias_expansion(word)
        if expanded is not None:
            more = expanded.split()
            word, args = more[0], more[1:] + args
        else:
            word = _command_named(word, ("frame", "up", "down")) or word
    if word in _FRAME_WORDS:
        if args[:1] == ["level"]:
            args = args[1:]
        how = [sub for sub in ("address", "apply", "function", "level", "view")
               if args and sub.startswith(args[0])]
        if how in (["address"], ["function"], ["view"]) and _select_frame_forms():
            # (Selected as `select-frame`, which prints nothing, does.)
            _execute("select-frame " + " ".join([how[0]] + args[1:]))
            return _execute("frame")
        if len(args) != 1 or not re.match(r"\d+$", args[0]):
            return None
        level = int(gdb.parse_and_eval(args[0]))  # (As gdb reads it: input-radix.)
        frame = gdb.newest_frame()
        for _ in range(level):
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


def _select_frame_forms():
    """Whether this gdb's `select-frame` takes `function`, `address` and
    `view`, as its `frame` does (gdb >= 9)."""
    global _select_frame_function
    if _select_frame_function is None:
        _select_frame_function = bool(_existing_commands(["select-frame function"]))
    return _select_frame_function


_select_frame_function = None


def _show_in_window(window, ut, selection=None):
    """Show user thread `ut` in register window `window` (see "Register
    windows"), with `selection`'s frame selected (see "Selecting a parked
    thread").  Returns what _window_done needs."""
    window.to_host()  # (Reading `ut` through the host's inferior, too.)
    room = _call_room(ut)
    if not room and not _can_refuse_calls:
        raise _Unshowable("too little of its stack is free below its stack pointer "
                          "for function calls, which this gdb can't refuse")
    if window.at_syscall and not _can_refuse_calls:
        raise _Unshowable("a syscall catchpoint stopped thread {} at a system call, which "
                          "a function call made there would change, and this gdb can't "
                          "refuse function calls".format(window.thread_num))
    # (Even with the user's own may-call-functions off: a command in the
    # window can turn it on for a call, `with may-call-functions on -- ...`,
    # which the window must then stop.)
    refused = _calls_refused(window) if room else (
        "on {}: under {} KiB of its stack is free below its stack "
        "pointer".format(_ut_words(int(ut)), _CALL_HEADROOM >> 10))
    window.refused_why = refused
    regs, ut_notes = _arch.context_regs(ut)
    window.show(ut, regs, calls=refused is None)
    if selection is not None:
        _select_level(selection["level"])
    return {"ut": ut, "selection": selection, "room": room, "notes": ut_notes,
            "regs": _raw_regs(gdb.newest_frame()), "refused": refused}


class _ChangedNote(str):
    """The note that a command changed the registers shown (see
    _window_done): left out where the window couldn't put the thread's own
    back, which it says instead (see _kept_notes)."""


def _kept_notes(window, notes):
    """`notes`, but for those that say a register change doesn't last, if
    `window` couldn't put some registers back."""
    if window.left or window.resumed:
        return [n for n in notes if not isinstance(n, _ChangedNote)]
    return notes


def _window_done(window, shown, what):
    """After command `what` (its first word) has run with _show_in_window's
    `shown` in `window`: remember the frame it left selected, and return the
    notes to print once the window is closed."""
    ut = shown["ut"]
    notes = []
    if _selected_thread() == window.thread and not window.resumed:
        changed = sorted(n for n, v in _raw_regs(gdb.newest_frame()).items()
                         if n in shown["regs"] and v != shown["regs"][n])
        if changed and (window.calls, window.refused) != window.looked and \
                any(f not in window.known_dummies for f in _dummy_frames()):
            # (They are a function call's, left stopped, which the window
            # abandons, saying so: see pop_calls.)
            changed = []
        if changed:
            # (The window puts the thread's own registers back.)
            notes.append(_ChangedNote(
                "realm-gdb: {}: '{}' changed {}, which doesn't last: a "
                "parked thread's saved registers aren't changed, and thread {}'s own "
                "are put back.".format(_ut_words(int(ut)), what, _reg_list(changed),
                                       window.thread_num)))
        if shown["selection"] is not None:
            shown["selection"]["level"] = _selected_level()
    notes += ["realm-gdb: {}: {}.".format(_ut_words(int(ut)), n) for n in shown["notes"]]
    sel = shown["selection"]
    if not shown["room"] and not (sel is not None and sel.get("room_noted")):
        # (Once for all of them, see _call_notes; and once for a selection.
        # A call refused says why itself, see _run_in_window.)
        window.low_room.append(int(ut))
        if sel is not None:
            sel["room_noted"] = True
    return notes


def _run_in_window(window, ut, command, notes, selection=None):
    """Run `command` with user thread `ut` shown in register window
    `window`, returning its output (see "Register windows").  With a
    `selection` (see "Selecting a parked thread"), its frame is selected
    first, and the frame `command` leaves selected is remembered."""
    shown = _show_in_window(window, ut, selection)
    try:
        try:
            if callable(command):  # (see _apply_external)
                out = command()
            else:
                out = _frame_motion(command)
                if out is None:
                    out = _interruptibly(command)
        except gdb.error as e:
            if shown["refused"] is None or "may-call-functions" not in str(e):
                raise
            # (gdb's message alone would leave the user wondering why.)
            note, window.calls_note = window.calls_note, None
            raise gdb.error("{}\n{}".format(e, note) if note else (
                "{}  (realm-gdb: function calls are refused while a parked Realm user "
                "thread is shown, {}.)".format(e, shown["refused"])))
        notes += _window_done(window, shown,
                              "edit" if callable(command) else command.split()[0])
    finally:
        _calls_over(window)
        # (Before the next parked thread is shown over it: see pop_calls.)
        window.pop_calls()
    return out


def _interruptibly(command):
    """_apply_execute(command) in the open window, with Ctrl-C let through
    for it, as for a plain command typed at a terminal (see _plain_before),
    where Ctrl-C can land while Python runs a command (see
    _SIGINT_IN_PYTHON_SAFE): gdb quits the command, and gdb.execute raises
    KeyboardInterrupt (see _apply_each); in Python code that the command
    runs (a pretty-printer's, `rt python`'s), it is raised there, as Python
    would; in realm-gdb's own it is held (see _on_python_sigint).  A
    function call's end blocks it again, for the rest of the command, as
    gdb pops the call (see _on_inferior_call)."""
    global _sigint_letting
    if not _SIGINT_IN_PYTHON_SAFE:
        return _apply_execute(command)
    if _sigint_pending() and _take_sigint():
        raise KeyboardInterrupt  # (Pressed before it began.)
    frame = sys._getframe()
    _sigint_letting = (frame, "rt")
    try:
        _block_sigint(False)
        return _apply_execute(command)
    finally:
        _block_sigint(True)
        if _sigint_letting is not None and _sigint_letting[0] is frame:
            _sigint_letting = None


class _Unshowable(Exception):
    """A parked user thread that turns out not to be showable."""


# Commands that read the lines after them, up to `end`, which a command run
# by `rt` or `realm-thread apply` can't have: run from Python, gdb >= 13
# takes them from the command's own text, so there are none (`rt define
# foo` would empty foo), and older gdbs read them from the terminal, with
# the window open.  Those after them read lines only when given no argument.
_BODY_COMMANDS = ("commands", "define", "document", "while", "if", "while-stepping",
                  "actions")
_BODY_COMMANDS_BARE = ("python", "guile", "compile")
_WATCH_COMMANDS = ("watch", "rwatch", "awatch")
# Whether `python-interactive` (`pi`), refused in a window as it reads the
# terminal, may run there now: given an expression, by `rt` (see _guard).
_pi_allowed = False


def _command_named(word, names):
    """Which of commands `names` gdb takes command word `word` for (an
    abbreviation or an alias of it, the user's too), if any: the one whose
    `help` it gets.  (Not cached: an alias defined later changes `help`'s
    text.  It takes a tenth of a millisecond.)"""
    if not re.match(r"[\w-]+$", word):
        return None
    try:
        text = _execute("help " + word)
    except gdb.error:
        return None  # (Ambiguous, or no such command.)
    for name in names:
        try:
            if _execute("help " + name) == text:
                return name
        except gdb.error:
            pass
    return None


def _alias_expansion(word):
    """What command word `word` stands for if it is an alias (the user's,
    `alias NAME = COMMAND [ARGS]`, gdb >= 10): its command's full name, and
    the default arguments it was given, as `help` shows them (`alias u2 = up
    2`, `alias fa = frame apply all`); None if it isn't one."""
    if not re.match(r"[\w-]+$", word):
        return None
    try:
        text = _execute("help " + word)
    except gdb.error:
        return None  # (Ambiguous, or no such command.)
    # (gdb >= 10 lists a command's names first, its own first: see _is_alias.)
    names = [n.strip() for n in text.split("\n", 1)[0].split(",")]
    if len(names) < 2 or word not in names[1:] or not re.match(r"[\w -]+$", names[0]):
        return None
    m = re.search(r"^\s+alias (?:-a )?{} = (\S.*)$".format(re.escape(word)), text, re.M)
    return m.group(1).strip() if m else names[0]


def _expand_alias(command):
    """`command`, with its first word, if an alias, written as what it
    stands for (see _alias_expansion)."""
    words = command.split(None, 1)
    expanded = _alias_expansion(words[0]) if words else None
    if expanded is None:
        return command
    return expanded + (" " + words[1] if len(words) > 1 else "")


def _inner_command(command, withs=None):
    """(COMMAND, its first word's name among those _check_command looks
    for, or None, the rest) of `command`, or of COMMAND in `with SETTING --
    COMMAND`, `eval "COMMAND"`, `frame apply ... COMMAND` and the like (see
    _applied_command); None if there is none.  Each `with`'s SETTING (and
    value) goes in list `withs`, if given."""
    if not command.split() or "\n" in command:
        return None
    while True:
        # (An alias of the command that runs another, as typed: `alias fa =
        # frame apply all`, `fa python`.)
        words = _expand_alias(command).split(None, 1)
        if len(words) < 2:
            break
        applied = _applied_command(words)
        if " -- " in words[1] and _command_named(words[0], ("with",)):
            setting, command = words[1].split(" -- ", 1)
            if withs is not None:
                withs.append(setting)
        elif applied is not None:
            command = applied
        else:
            # (`eval "TEXT"`, with nothing to format: TEXT's.  With
            # arguments, what it runs can't be told from here: see §7.2.)
            literal = re.match(r'"((?:[^"\\]|\\.)*)"\s*$', words[1].strip())
            if literal is None or not _command_named(words[0], ("eval",)):
                break
            command = re.sub(r"\\(.)", r"\1", literal.group(1))
        if not command.split():
            return None
    words = command.split(None, 1)
    name = _command_named(words[0], _BODY_COMMANDS + _BODY_COMMANDS_BARE +
                          _WATCH_COMMANDS + ("python-interactive",))
    if name not in _BODY_COMMANDS + _BODY_COMMANDS_BARE:
        # (With the arguments the user's alias gives it, `alias wl = watch
        # -l`.  Not for a command that reads a block, whose alias's arguments
        # gdb drops where it reads the block itself, as from gdb.execute, a
        # script or a definition: `alias pyx = python print(1)` reads one.)
        words = _expand_alias(command).split(None, 1)
    rest = words[1].strip() if len(words) > 1 else ""
    return command, name, rest


# What may come between `frame apply` or `thread apply` (or `faas`, `taas`,
# `tfaas`) and the command they run: which frames or threads (`all`, `level
# 1 3-4`, COUNT, -COUNT, `1.2-3`, `$var`, `1.*`), `-ascending`, and their
# options (`-q`, `-c`, `-s`, `-past-main on`, `--`).
_APPLY_WORD = re.compile(r"(all|level|on|off|-\S*|\$\S+|\d\S*|\*)$")


def _applied_command(words):
    """COMMAND of `frame apply ... COMMAND`, `thread apply ... COMMAND`,
    `faas COMMAND`, `taas COMMAND` or `tfaas COMMAND` (`words`: the first
    word, and the rest), as abbreviated or aliased; or None."""
    first, rest = words
    expanded = _alias_expansion(first)  # (`alias fa = frame apply all`, say.)
    if expanded is None:
        # (Where gdb's `help` doesn't name an alias's command, before gdb 10,
        # which has no arguments for one either: by its help.)
        expanded = _command_named(first, ("frame apply all", "frame apply level",
                                          "frame apply", "thread apply all",
                                          "thread apply"))
    if expanded is not None:
        first, rest = (expanded + " " + rest).split(None, 1)
    name = _command_named(first, ("frame", "thread", "faas", "taas", "tfaas"))
    if name is None:
        return None
    if name in ("frame", "thread"):
        sub = rest.split(None, 1)
        if len(sub) < 2 or not re.match(r"[\w-]+$", sub[0]):
            return None
        try:
            if _execute("help {} {}".format(name, sub[0])) != \
                    _execute("help {} apply".format(name)):
                return None
        except gdb.error:
            return None
        rest = sub[1]
    for m in re.finditer(r"\S+", rest):
        if not _APPLY_WORD.match(m.group()):
            return rest[m.start():]
    return None


def _reads_body(name, rest):
    """Whether command `name` (of _inner_command), with arguments `rest`,
    reads the lines after it, up to `end`."""
    if name == "compile":
        return _compile_reads_body(rest)
    return name in _BODY_COMMANDS or name in _BODY_COMMANDS_BARE and not rest


def _compile_reads_body(rest):
    """Whether `compile REST` (or `expression REST`) is left with no code to
    compile, and so reads it from the lines after it: `compile`, `compile
    code -raw`, `compile print /x`, but not `compile file FILE`."""
    words = rest.split()
    if words and re.match(r"f(i(le?)?)?$", words[0]):
        return False
    if words and re.match(r"c(o(de?)?)?$|p(r(i(nt?)?)?)?$", words[0]):
        words.pop(0)
    while words and words[0].startswith("-"):  # (-raw, print's options, --)
        if words.pop(0) == "--":
            break
    if words and words[0].startswith("/"):  # (print's /FMT)
        words.pop(0)
    return not words


def _drop_body(command, from_tty, where):
    """`rt COMMAND` or `realm-thread apply PTR COMMAND` (`where`), where
    COMMAND reads the lines after it (see _BODY_COMMANDS), is refused, but
    its lines would then run as gdb commands of their own (a `continue`
    among a Python block's, say).  Typed at a terminal, read them first, up
    to their `end`, with Python's input(), saying so, and return how many
    there were; else None.  (A terminal gives a line at a time while gdb runs a command; from
    a pipe, input() would read through a buffer of Python's own, taking
    gdb's later commands with it.  A failed command in a script or a
    user-defined command ends it, and its lines with it.)"""
    inner = _inner_command(command)
    if inner is None or not _reads_body(*inner[1:]) or not from_tty or \
            not _stdin_is_terminal():
        return None
    script = inner[1] in _BODY_COMMANDS_BARE or inner[1] == "document"
    print("realm-gdb: {} refuses '{}', which reads the lines after it: type them, up "
          "to 'end', to have them dropped.".format(where, command))
    try:
        return _read_body(script)
    except (EOFError, KeyboardInterrupt):
        return None


# Commands whose lines, up to `end`, are gdb commands (see _read_body), and
# those whose lines aren't (when bare: see _reads_body).
_BLOCK_WORDS = ("while", "if", "define", "commands", "while-stepping", "ws", "stepping",
                "actions")
_SCRIPT_WORDS = ("python", "py", "guile", "gu", "compile", "document")


def _read_body(script):
    """Read the lines of a block, up to its `end`, as gdb would: those of a
    `script` block (Python's, say) are taken as they are; in others, a line
    that starts a block reads that block's lines too.  Returns how many
    lines were read."""
    count = 0
    while True:
        line = input(">").strip()
        count += 1
        if line == "end":
            return count
        if script:
            continue
        words = line.split(None, 1)
        word = words[0] if words else ""
        if word in _BLOCK_WORDS:
            count += _read_body(False)
        elif word in _SCRIPT_WORDS and (len(words) == 1 or word == "document" or
                                        re.match(r"c(o(de?)?)?$", words[1])):
            count += _read_body(True)


def _check_command(command, where, dropped=None):
    """Refuse a command that `rt` or `realm-thread apply` (`where`) can't
    run as it would run on its own (see _BODY_COMMANDS), or that would be
    undone at once (a watchpoint tied to a parked frame).  `dropped`: how
    many lines of a refused command's body _drop_body read."""
    global _pi_allowed
    withs = []
    inner = _inner_command(command, withs)
    for setting in withs:
        # (`with` sets it itself, and runs no `set` hook: see _STREAM_COMMANDS
        # and _WINDOW_GUARDED.  Before gdb 12 it doesn't put back some of them
        # either, such as the architecture, which kept the thread's own
        # registers from being put back.)
        name = _with_refused(setting)
        if name == "set logging enabled":
            _refuse_stream("with logging enabled", where)
        elif name is not None:
            raise gdb.GdbError(
                "realm-gdb: refusing 'with {}' in {}, which shows a parked thread on "
                "thread {}: {}.".format(name[len("set "):], where, _host_words(),
                                         _WINDOW_GUARDED[name]))
    if inner is None:
        return
    command, name, rest = inner
    words = command.split(None, 1)
    if _reads_body(name, rest):
        raise gdb.GdbError(
            "realm-gdb: refusing '{}' in {}: it reads the lines after it, up to "
            "'end', which {} can't pass on{}.  {}".format(
                command, where, where,
                " (the lines typed were dropped)" if dropped else "",
                "Give it its code on one line, or run '{} source FILE'.".format(
                    where.strip("'").split()[0] if where == "'rt'" else
                    "realm-thread apply N") if name in _BODY_COMMANDS_BARE else
                "Run it without {}.".format(where)))
    if name in _WATCH_COMMANDS and not re.match(r"-l(o(c(a(t(i(on?)?)?)?)?)?)?(\s|$)", rest):
        raise gdb.GdbError(
            "realm-gdb: refusing '{}' in {}: gdb would delete it at once, as it looks "
            "for its locals' frame on the thread it checks, where the parked thread's "
            "isn't.  '{} {} -l {}' watches the memory instead; a watchpoint on globals "
            "needs no {}.".format(
                command, where, "rt" if where == "'rt'" else "realm-thread apply N",
                words[0], rest.strip(), where))
    if re.search(r"\$_shell\s*\(", command):
        raise gdb.GdbError(
            "realm-gdb: refusing '{}' in {}: its shell command would run while thread {} "
            "holds the parked thread's registers.  Run it on its own.".format(
                command, where, _host_words()))
    if words[0] in _STREAM_COMMANDS_UNHOOKED:
        try:
            _execute("help " + words[0])  # (Not `complete +`, which lists the rest.)
        except gdb.error:
            pass  # (No TUI.)
        else:
            _refuse_stream(words[0], where)
    if name == "python-interactive" and rest:
        _pi_allowed = True  # (Last: once nothing refuses it.)


def _host_words():
    """The number of the thread that shows a parked one, for messages: the
    window's host, or (as `rt` or `realm-thread apply` checks its command)
    the selected thread."""
    if _window is not None:
        return str(_window.thread_num)
    host = _selected_thread()
    return str(host.num) if host is not None else "?"


def _with_refused(setting):
    """The `set` command whose setting `with SETTING [VALUE] -- ...`
    (`setting`: what comes before `--`) sets, as abbreviated, if `set` of it
    is refused in a window (see _WINDOW_GUARDED), or is `set logging
    enabled` (gdb >= 13, see _STREAM_COMMANDS); else None."""
    words = setting.split()
    names = [name for name in sorted(_WINDOW_GUARDED) if name.startswith("set ")]
    if _has_parameter("logging enabled"):
        names.append("set logging enabled")
    texts = {}
    for name in names:
        size = len(name.split()) - 1
        if len(words) < size:
            continue
        try:
            if size not in texts:
                texts[size] = _execute("help set " + " ".join(words[:size]))
            if texts[size] == _execute("help " + name):
                return name
        except gdb.error:
            texts.setdefault(size, None)  # (Not a setting of that many words.)
    return None


# How _apply's command was run (see _apply_execute): whether it was typed
# (from_tty), and whether its output is printed (rather than given to `rt
# pipe`'s SHELL); and the command, for messages (see _window_words).
_apply_from_tty = _apply_prints = False
_apply_opener = None
# Whether gdb.execute can keep the output's styling (gdb >= 17), once known.
_execute_styles = globals().get("_execute_styles")


def _apply_execute(command):
    """_execute(command), for _apply: a backtrace typed at the terminal says
    "(More stack frames follow...)" when it stops short, as gdb's own does
    (nothing else of a backtrace's depends on from_tty); and output to be
    printed is styled as gdb would print it, where gdb can."""
    global _execute_styles
    from_tty = _apply_from_tty and _bt_args(command) is not None
    if _apply_prints and _execute_styles is not False:
        try:
            out = gdb.execute(command, from_tty, True, styling=True)
        except TypeError:  # (gdb < 17)
            _execute_styles = False
        else:
            _execute_styles = True
            return _cli_output(command, out)
    return _execute(command, from_tty)


def _apply(targets, command, header, selection=None, collect=None, flags=(),
           from_tty=False, opener=None):
    """Run `command` on each parked user thread in `targets` ([(ut,
    problem)] of the selected inferior), shown on the selected thread, and
    print its output (or, with a list `collect`, append it there).  With
    `header`, each output gets a "User thread PTR:" line first, and `flags`
    are `thread apply`'s -q, -c and -s (see _apply_each).  A `selection` is
    for `rt` (see _run_in_window).  Returns the first error, if any
    (gdb.error).  (The output is printed once the window is closed:
    gdb's pager may wait for input, and so let another UI run a command; see
    "Register windows".)  A command that starts another program is
    _apply_external's."""
    global _pi_allowed, _apply_from_tty, _apply_prints, _apply_opener
    if not callable(command):
        _check_command(command, "'rt'" if selection is not None else
                       "'realm-thread apply'")
    saved = _apply_from_tty, _apply_prints, _apply_opener
    _apply_from_tty, _apply_prints = from_tty, collect is None
    _apply_opener = opener or _apply_opener
    try:
        return _apply_checked(targets, command, header, selection, collect,
                              flags if header else ())
    finally:
        _pi_allowed = False
        _apply_from_tty, _apply_prints, _apply_opener = saved


def _apply_checked(targets, command, header, selection, collect, flags):
    external = _external(command) if collect is None and not callable(command) else None
    if external is not None:
        return _apply_external(targets, command, header, selection, external, flags)
    how = _how_shown()
    if isinstance(how, _NotShown):
        raise how.error()
    what = "edit" if callable(command) else (command.split() or [command])[0]
    window = None
    if how == "window":
        window = _Window(via="rt" if selection is not None else None)
        # (Ctrl-C is blocked until what was collected, and the notes, are
        # printed: else it would land as the window closed, and lose them.)
        window.hold_sigint = True
    try:
        return _apply_in(window, targets, command, header, selection, collect, what, flags)
    finally:
        if window is not None:
            _let_held()


# How many commands realm-gdb runs with their output collected (`rt`,
# `realm-thread apply`, in core files too: see _apply_in), where gdb's
# output may not be switched to another stream (see _STREAM_COMMANDS).
# (Kept if the script is loaded again, as by a library a call loads.)
_collecting = globals().get("_collecting", 0)


def _apply_in(window, targets, command, header, selection, collect, what, flags):
    """_apply_checked's work, in `window` (None for core-file views).  A
    Ctrl-C stops it at the next parked thread."""
    global _collecting
    first_error = []
    notes = []
    printed = []
    unshowable = None
    _collecting += 1
    if window is not None:
        window.deferred = []  # (Its warnings after what was collected: see _warn.)
    try:
        with (window if window is not None else _NoWindow()), \
                _NoTerminalInput(what, window):
            _apply_each(window, targets, command, header, selection, what, first_error,
                        notes, printed, flags)
    except _ApplyUnshowable as e:
        unshowable = e.error
    except BaseException:
        if window is not None:  # (Its warnings, all the same: see _warn.)
            deferred, window.deferred = window.deferred or [], None
            for note in deferred:
                _say(note)
        raise
    finally:
        _collecting -= 1
    try:
        if collect is not None:
            collect.append("".join(printed))
        else:
            if window is not None and _SIGINT_IN_PYTHON_SAFE and not _sigint_pending():
                # (The window is closed: Ctrl-C at gdb's pager can stop the
                # output now.  One pressed meanwhile is held until the notes
                # are printed, as it would lose them, landing first.)
                _let_sigint()
            try:
                gdb.write("".join(printed))
                # (Output that doesn't end a line, such as `output`'s, would
                # otherwise wait until after the prompt, or the next command's.)
                gdb.flush()
            except KeyboardInterrupt:
                # (`q` at gdb's pager, or Ctrl-C: "Quit", as gdb's own commands
                # say, which gdb 8 would put as a Python exception.)
                raise gdb.GdbError("Quit")
        if unshowable is not None:
            raise unshowable
    finally:
        # (Also once `q` at gdb's pager, or Ctrl-C, has cut the output short:
        # the warnings among them must come.)
        if window is not None:
            notes = (window.deferred or []) + _kept_notes(window, notes) + _call_notes(window)
            window.deferred = None  # (A close still to come says them itself.)
        for note in notes:
            _say(note)
    return first_error[0] if first_error else None


class _ApplyUnshowable(Exception):
    """A parked thread _apply_each couldn't show, ending its work."""

    def __init__(self, error):
        Exception.__init__(self, str(error))
        self.error = error


def _apply_each(window, targets, command, header, selection, what, first_error, notes,
                printed, flags):
    """_apply_in's loop over `targets`.  As `thread apply`'s: -q leaves out
    the headers, -s every thread whose command fails or prints nothing, and
    with -c (or -s) its error isn't the command's.  (Without them, the
    command goes on to the others, then fails with the first error.)"""
    for i, (ut, problem) in enumerate(targets):
        if i and window is not None and (_sigint_pending() or window.call_interrupted):
            break  # (Ctrl-C: one held lands once the window is closed.)
        if i and problem is None and window is not None:
            # (Again: the command run on the last one may have written this
            # one's saved context, or its state.  No thread has run since.
            # Read through the host's inferior, which that may have left.)
            window.to_host()
            if _recording():
                # (It started recording, which would log the next one's
                # registers as they were put in place: see _how_shown.)
                how = _how_shown()
                raise _ApplyUnshowable(gdb.GdbError(
                    "realm-thread: can't show the other parked user threads: " +
                    (how.sentence() if isinstance(how, _NotShown) else "recording began")))
            problem = _parked_problem(ut, {})
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
                if str(e).startswith("realm-gdb: refusing "):
                    # (realm-gdb's own refusal of the command, the same on
                    # every parked thread: once.)
                    raise _ApplyUnshowable(gdb.GdbError(str(e)))
                out = e
            except KeyboardInterrupt:
                # (Ctrl-C quit the command, see _interruptibly: it lands again
                # once the window is closed, and what was collected, and the
                # notes, are printed.)
                if window is None or not _hold_sigint():
                    raise
                break
        if problem is not None and not header:
            raise _ApplyUnshowable(gdb.GdbError("realm-thread: can't show {}: {}.".format(
                _ut_words(int(ut)), problem)))
        if "-s" in flags and (problem is not None or isinstance(out, gdb.error) or not out):
            continue
        if header and "-q" not in flags:
            printed.append("\n{}:\n".format(_ut_words(int(ut), capital=True)))
        if problem is not None:
            printed.append("  <{}>\n".format(_problem_words(problem)))
        elif isinstance(out, gdb.error):
            if not first_error and "-c" not in flags:
                first_error.append(out)
            if header:  # (A single thread's error is the command's.)
                printed.append("  <{} failed: {}>\n".format(what, out))
        else:
            printed.append(out)


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
    # (The user's alias, and the arguments it gives: `alias p2 = pipe -d X`.)
    expanded = _alias_expansion(m.group(1))
    if expanded is not None:
        command = expanded + " " + command[m.end():]
        m = _COMMAND_WORD.match(command)
    word, rest = m.group(1), command[m.end():].strip()

    def abbreviates(name, least):
        return name.startswith(word) and word.startswith(least) or alias == name
    alias = _command_named(word, ("shell", "make", "edit", "pipe"))  # (The user's.)
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


def _apply_external(targets, command, header, selection, external, flags):
    """_apply for a command that starts another program (see above).  (Its
    output can't be told apart: -s leaves out only threads that can't be
    shown.)"""
    first_error = None
    for ut, problem in targets:
        if problem is not None and external[0] != "direct":
            if not header:
                raise gdb.GdbError("realm-thread: can't show {}: {}.".format(
                    _ut_words(int(ut)), problem))
            if "-s" in flags:
                continue
        if header and "-q" not in flags:
            gdb.write("\n{}:\n".format(_ut_words(int(ut), capital=True)))
            gdb.flush()
        if problem is not None and external[0] != "direct":
            gdb.write("  <{}>\n".format(_problem_words(problem)))
            continue
        try:
            if external[0] == "direct":
                _run_unshown(command)
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
            if first_error is None and not ("-c" in flags or "-s" in flags):
                first_error = e
            if header and "-s" not in flags:
                gdb.write("  <{} failed: {}>\n".format(command.split()[0], e))
    return first_error


def _run_unshown(command):
    """Run `command` as it is, with plain commands showing nothing (`rt pipe
    shell ... | ...` mustn't be refused as a plain `pipe` is)."""
    global _plain_suppressed
    _plain_suppressed += 1
    try:
        gdb.execute(command, to_string=False)
    finally:
        _plain_suppressed -= 1


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
        self.in_force = False

    def let(self, real, *args, **kwargs):
        """Python's input() or a read of sys.stdin: `real` (what it would
        have run) if that is now no part of the command it is in force for:
        once that is over (a pdb made meanwhile keeps sys.stdin), or for code
        run after a plain command that failed (a `code.interact()` console's
        next line), whose window closes first, as gdb's prompt would close
        it (see _close_stale).  Else it refuses."""
        entry = _plain
        if self.in_force and entry is not None and entry["input"] is self:
            try:
                _close_stale(_outer_frames(sys._getframe().f_back.f_back))
            except Exception:
                _debug_exc()
        if not self.in_force:
            return real(*args, **kwargs)
        self.refuse()

    def _input(self, *args, **kwargs):
        return self.let(self.input, *args, **kwargs)

    def refuse(self):
        if self.window is not None and self.window.via not in (None, "rt"):
            raise gdb.GdbError(
                "realm-gdb: '{}' reads the terminal, which can't be done while "
                "thread {} shows a parked Realm user thread: it would keep the "
                "parked thread's registers while it waits.".format(
                    self.what, self.window.thread_num))
        raise gdb.GdbError(
            "realm-gdb: refusing to read the terminal in '{}' inside {}: its question "
            "would show only once it was over{}.".format(
                self.what, _window_words(self.window) if self.window is not None else
                "'rt' or 'realm-thread apply'",
                ", and thread {} has a parked thread's registers until then".format(
                    self.window.thread_num) if self.window else ""))

    def __enter__(self):
        import builtins
        self.input, self.stdin = builtins.input, sys.stdin
        builtins.input = self._input
        sys.stdin = _RefusedInput(self, self.stdin)
        self.in_force = True
        return self

    def __exit__(self, *exc):
        import builtins
        self.in_force = False
        builtins.input, sys.stdin = self.input, self.stdin
        return False


class _RefusedInput(object):
    """sys.stdin while _NoTerminalInput `owner` is in force (see its let)."""

    def __init__(self, owner, real):
        self._owner, self._real = owner, real
        self.encoding = getattr(real, "encoding", None)

    def read(self, *args):
        return self._owner.let(self._real.read, *args)

    def readline(self, *args):
        return self._owner.let(self._real.readline, *args)

    def readlines(self, *args):
        return self._owner.let(self._real.readlines, *args)

    def __iter__(self):
        return self._owner.let(self._real.__iter__)

    def isatty(self):
        return self._real.isatty() if not self._owner.in_force else False


class _NoWindow(object):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _problem_words(problem):
    """How a problem is shown in place of a parked thread's output."""
    if problem.startswith("not started yet"):
        return problem
    return "not shown: " + problem


# --------------------------------------------------------------------------
# thread apply all
# --------------------------------------------------------------------------

def _dump_parked_threads(command, flags=(), from_tty=False):
    """Run backtrace `command` on every parked user thread of each inferior,
    in a section that begins and ends with a line of its own, with `thread
    apply`'s `flags` (see _apply_each).  (The caller restores the
    selection.)"""
    for inferior in sorted(gdb.inferiors(), key=lambda inf: inf.num):
        if _select_inferior(inferior):
            _dump_inferior_parked_threads(command, flags, from_tty)


def _dump_inferior_parked_threads(command, flags, from_tty):
    """_dump_parked_threads for the selected inferior, on the selected
    thread."""
    where = ""
    if len(gdb.inferiors()) > 1:
        where = " of inferior {}".format(gdb.selected_inferior().num)
    end = "=== End of parked Realm user threads{} ===".format(where)
    how = _how_shown()
    try:
        sps, _ = _host_sps()
        parked, loop = _parked_threads()
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
        words = _no_registry_words(no_realm=False)
        if words is not None:  # (Not left out silently.)
            print("\n=== Parked Realm user threads{}: not shown ===".format(where))
            print(words)
            print(end)
        return
    if isinstance(how, _NotShown):
        _print_not_shown(where, len(parked), how)
        if loop is not None:
            print("realm-gdb: {}; there may be more parked threads.".format(loop))
        print(end)
        return
    host = _selected_thread()
    if host is None:
        how = _NotShown("no thread is selected", "select one ('thread N')")
        _print_not_shown(where, len(parked), how)
        print(end)
        return
    if how == "window":
        try:
            # (Before the header: what a window would refuse at once.)
            _check_with_settings()
            _check_unrestored(host)
        except gdb.GdbError as e:
            print("\n=== Parked Realm user threads{} ({}): not shown ===".format(
                where, len(parked)))
            print(e)
            print(end)
            return
        print("\n=== Parked Realm user threads{} ({}), each shown on thread {} (whose "
              "thread-local variables they show) ===".format(where, len(parked),
                                                            _thread_words(host)))
    else:
        print("\n=== Parked Realm user threads{} ({}), shown on thread {}: each stack "
              "starts at frame #1 (#0 is {}) ===".format(
                  where, len(parked), _thread_words(host), _frame0_words(host.num)))
    if "-ascending" in flags:  # (Oldest first, as `thread apply all -ascending`.)
        parked.reverse()
        flags = [f for f in flags if f != "-ascending"]
    try:
        _apply([(ut, _parked_problem(ut, sps)) for ut in parked], command, header=True,
               flags=flags, from_tty=from_tty,
               opener="thread apply all " + (command.split() or ["bt"])[0])
    except (gdb.error, gdb.GdbError) as e:
        print(str(e) if str(e).startswith("realm-") else "realm-gdb: {}".format(e))
    if loop is not None:
        print("\nrealm-gdb: {}; there may be more parked threads.".format(loop))
    print("\n" + end)


def _thread_words(thread):
    """gdb's name for `thread`: N, or I.N with more than one inferior."""
    if len(gdb.inferiors()) > 1:
        return "{}.{}".format(thread.inferior.num, thread.num)
    return str(thread.num)


def _print_not_shown(where, count, how):
    print("\n=== Parked Realm user threads{}{}: not shown ===".format(
        where, "" if count is None else " ({})".format(count)))
    print(how.sentence(capital=True))


def _thread_apply_flags():
    """Whether this gdb's `thread apply` takes -q, -c and -s (gdb >= 8.3), as
    its `help` says."""
    global _has_thread_apply_flags
    if _has_thread_apply_flags is None:
        try:
            _has_thread_apply_flags = "-q" in _execute("help thread apply")
        except gdb.error:
            _has_thread_apply_flags = False
    return _has_thread_apply_flags


_has_thread_apply_flags = None


def _split_flags(arg):
    """`thread apply`'s flags at the start of `arg`, and the command after
    them, verbatim."""
    flags = []
    rest = arg.lstrip()
    while rest.startswith("-"):
        m = re.match(r"(\S+)\s*(.*)$", rest, re.S)
        flags.append(m.group(1))
        rest = m.group(2)
        if m.group(1) == "--":
            break
    return flags, rest


class _Gdb8Quit(object):
    """Around a realm-gdb command's work: a KeyboardInterrupt out of it (a
    Ctrl-C a window held, landing once it is closed) is a quit, "Quit", as
    gdb 9 and later take it, where gdb 8 would say "Python Exception <class
    'KeyboardInterrupt'>" (see _COMMANDS_QUIT)."""

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is not None and issubclass(kind, KeyboardInterrupt) and not _COMMANDS_QUIT:
            raise gdb.GdbError("Quit")
        return False


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

    def invoke(self, arg, from_tty):
        _close_stale(_outer_frames(sys._getframe().f_back))
        with _Gdb8Quit():  # (See there.)
            _printing_over()
            flags, rest = _split_flags(arg.strip())
            if not rest:
                raise gdb.GdbError("Please specify a command at the end of "
                                   "'thread apply all'")
            others = [f for f in flags if f != "--" and not (
                len(f) > 1 and "-ascending".startswith(f))]
            if others and not _thread_apply_flags():
                # (As gdb 8.2's own: it takes them for the command.)
                raise gdb.GdbError("Undefined command: \"{}\".  Try \"help\".".format(others[0]))
            bt = _bt_args(rest) is not None
            if bt and _window is not None:
                raise gdb.GdbError("realm-gdb: refusing 'thread apply all {}' inside {}: it "
                                   "can't show parked Realm user threads there.".format(
                                       rest, _window_words(_window)))
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
                # (Threads as they are, even with a parked thread selected, or
                # shown in a core file.)
                _plain_suppressed += 1
                try:
                    with (_ViewsOff() if any(_views.values()) else _NoWindow()):
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
            _dump_parked_threads(rest, [f for f in flags if f != "--"] +
                                 (["-ascending"] if ascending else []), from_tty)
        if error is not None:
            raise gdb.GdbError(str(error))

    def complete(self, text, word):
        # As gdb's own `thread apply ID... COMMAND` would, after the flags.
        # (gdb >= 13 first asks with word=None, to find the word breaks.)
        if word is None:
            word = re.split(r"[ \t]", text)[-1]
        flags, rest = _split_flags(text)
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
    line = "complete " + prefix + rest
    try:
        if line == line.rstrip() or "hook-eval" in _user_hooks:
            out = _execute(line)
        else:
            # (gdb.execute drops a line's trailing whitespace, which says
            # the last word is done: `complete info ` lists info's
            # subcommands, `complete info` completes `info`.  `eval` passes
            # its string on as it is.  Not with the user's own hook-eval,
            # which would run then.)
            line = 'eval "{}"'.format(
                line.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%"))
            out = _internal(line)
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


def _split_targets(arg):
    """([user thread expressions], command) of `realm-thread apply`'s
    arguments, which name them as `thread apply` takes an ID list: numbers,
    ranges N-M, pointers, convenience variables, and expressions in
    parentheses (the first, any word: see _split_target); or (["all"],
    command)."""
    targets = []
    rest = arg.lstrip()
    while rest:
        target, after = _split_target(rest)
        if not targets and target == "all":
            return ["all"], after
        if targets and not target.startswith("(") and \
                not re.match(r"(\d+(-\d+)?|0[xX][0-9a-fA-F]+|\$\S*)$", target):
            break  # (The command.)
        m = re.match(r"(\d+)-(\d+)$", target)
        if m:
            low, high = int(m.group(1)), int(m.group(2))
            if low > high:
                raise gdb.GdbError("realm-thread: inverted range: {}".format(target))
            targets += [str(n) for n in range(low, high + 1)]
        else:
            targets.append(target)
        rest = after.lstrip()
    return targets, rest


def _lookup_user_thread(expr):
    """The parked-or-not Realm::UserThread* that `expr` evaluates to, or
    that `expr` numbers (as `realm-thread` lists them), in the selected
    inferior.  Raises gdb.GdbError saying what is wrong."""
    decimal = re.match(r"\s*\d+\s*$", expr) is not None
    if decimal:
        uts = _registry_addrs()
        if not uts:
            words = _no_registry_words()  # (Say why no user thread can be found.)
            if words is not None:
                raise gdb.GdbError(words)
        numbers = _numbered(uts)
        for addr, number in numbers.items():
            if number == int(expr):
                if addr not in uts:
                    raise gdb.GdbError("realm-thread: user thread {} ({:#x}) is no longer "
                                       "in the registry.".format(number, addr))
                expr = "{:#x}".format(addr)
                break
        else:
            if int(expr) < 65536:  # (No user thread is that low in memory.)
                raise gdb.GdbError("realm-thread: no user thread {} (run 'realm-thread' "
                                   "to list them).".format(int(expr)))
    try:
        with _CLanguage():  # (A pointer as `realm-thread` lists it: 0x...)
            addr = int(gdb.parse_and_eval(expr)) & _U64
    except gdb.error as e:
        raise gdb.GdbError("realm-thread: cannot use {!r} as a UserThread "
                           "pointer: {}".format(expr, e))
    if addr < 65536 and not decimal:
        return _lookup_user_thread(str(addr))  # (`realm-thread $n`, as `thread $n`.)
    try:
        ut = _find_user_thread(addr)
    except gdb.error as e:
        raise gdb.GdbError("realm-thread: cannot read the Realm user thread "
                           "registry: {}".format(e))
    if ut is not None:
        _numbered(_registry_addrs())  # (For the messages that name it: see _ut_words.)
    if ut is None:
        words = _no_registry_words()
        if words is not None:
            raise gdb.GdbError(words)
        for num, addrs in (_user_threads_elsewhere() if len(gdb.inferiors()) > 1 else []):
            if addr in addrs:
                raise gdb.GdbError(
                    "realm-thread: {:#x} is a user thread of inferior {}: select it "
                    "('inferior {}') and rerun this command.".format(addr, num, num))
        raise gdb.GdbError("realm-thread: {} is not a Realm user thread{} (run "
                           "'realm-thread' to list them).".format(
                               "{} ({:#x})".format(expr.strip(), addr) if decimal else
                               "{:#x}".format(addr), "'s number, nor one's address"
                               if decimal else ""))
    return ut


def _showable_user_thread(expr, sps):
    """_lookup_user_thread(expr), which must be parked and showable (see
    _parked_problem, given `sps`); else gdb.GdbError saying why not."""
    ut = _lookup_user_thread(expr)
    problem = _parked_problem(ut, sps)
    if problem is not None:
        raise gdb.GdbError("realm-thread: can't show {}: {}.".format(
            _ut_words(int(ut)), problem))
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
# Which of `realm-thread N`'s explanations it has given in full (see _select):
# True once that plain commands show the selection, False that they don't,
# None for neither yet.
_selection_explained = globals().get("_selection_explained")

# The prompt hook realm-gdb installed while a thread is selected, and the one
# it replaced (called first).
_prompt_hooks = globals().get("_prompt_hooks", {"ours": None, "theirs": None})

# The reminder that realm-gdb's prompt hook puts in front of the prompt.  gdb
# makes the prompt a hook returns its `prompt` setting, so the prompt the
# hook gets next time has the reminder already, as `show prompt` does
# meanwhile; once nothing is selected, the setting is put back without it.
_PROMPT_REMINDER = re.compile(r"^(?:\[realm-thread (?:0x[0-9a-f]+|\d+)\] )+")


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
        print("realm-thread: {}; no longer selecting {}.".format(
            why, _uts_words([sel["user_thread"] for sel in dropped])))
    if not _selections and not _any_kept_views():
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
        why = ("can't be found: " + _REGISTRY_DROPPED if _registry_dropped()
               else "is no longer a Realm user thread")
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
        raise gdb.GdbError("realm-thread: {} {}; no longer selecting "
                           "it.".format(_ut_words(sel["user_thread"]), why))
    return ut


def _realm_prompt(current):
    """The prompt (gdb.prompt_hook) while a parked thread is selected: the
    user's, after a reminder of the selection."""
    global _plain_suppressed
    prompt = current = _PROMPT_REMINDER.sub("", current)
    theirs = _prompt_hooks["theirs"]
    try:
        # (The last Python code to run before the prompt: a before_prompt
        # handler that runs after realm-gdb's may have left a window open.
        # Closed before the user's hook runs, which sees threads as they
        # are, and whose failure mustn't leave it open; and after it.)
        _close_at_prompt()
        if theirs is not None:
            _plain_suppressed += 1
            try:
                got = theirs(current)
            finally:
                _plain_suppressed -= 1
            if got is not None:
                prompt = got
    except Exception:
        _debug_exc()
    try:
        _close_at_prompt()
        sel = _selection() or _kept_view()
        if sel is not None:
            n = _ut_number(sel["user_thread"])
            prompt = "[realm-thread {}] {}".format(
                n if n is not None else "{:#x}".format(sel["user_thread"]), prompt)
    except Exception:
        _debug_exc()
    return prompt


def _view_elsewhere_words():
    """" (thread N shows user thread M (0x...): 'thread N' goes back to it)",
    if another thread of the selected inferior has a view kept (a core
    file's), else ""."""
    current = _selected_thread()
    for stack in _views.values():
        for view in stack[::-1]:
            if not view["temporary"] and view["thread_object"] != current and \
                    view["inferior"] == gdb.selected_inferior().num:
                return " (thread {} shows {}: 'thread {}' goes back to it)".format(
                    view["thread_num"], _ut_words(view["user_thread"]), view["thread_num"])
    return ""


def _kept_view():
    """The view of a parked thread on the selected thread of a core file
    that `realm-thread N` keeps until 'realm-thread back' (see "Core-file
    views"), if any."""
    view = _current_view()
    return view if view is not None and not view["temporary"] else None


def _any_kept_views():
    return any(not v["temporary"] for stack in _views.values() for v in stack)


def _install_prompt_hook():
    if _prompt_hooks["ours"] is not None and gdb.prompt_hook is _prompt_hooks["ours"]:
        _idle_codes.add(_prompt_hooks["ours"].__code__)  # (An earlier load's, say.)
        return
    namespace = globals()
    _prompt_hooks["theirs"] = gdb.prompt_hook
    _prompt_hooks["ours"] = lambda current: namespace["_realm_prompt"](current)
    _idle_codes.add(_prompt_hooks["ours"].__code__)
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
        raise gdb.GdbError("realm-thread: can't {} from inside {}.".format(
            what, _window_words(_window)))
    if not _has_target():
        raise gdb.GdbError("realm-thread: no process or core file.")
    how = _how_shown()
    if isinstance(how, _NotShown):
        raise how.error()
    _check_with_settings()
    _stopped_thread()
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
# `thread apply`, ...) is refused, saying to run `realm-thread back` first,
# or `rt COMMAND` where that works; `python` runs, with a note (see
# _NOTED_COMMANDS).  Two end the
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
# failed command's window first, and refuse, in a window, the MI commands
# that would let the program run or let go of it (_install_execute_mi).

# The window a plain command opened, if any: a dict
#   window:   the _Window
#   command:  the command
#   outer:    the Python frames it was run from (_outer_frames), and
#   lasti:    the innermost one's f_lasti then
#   shown:    _show_in_window's
#   input:    its _NoTerminalInput
_plain = globals().get("_plain")

# Plain commands on realm-gdb's behalf that show threads as they are (in
# `thread apply all`).  (Kept if the script is loaded again in the middle of
# one, whose end then counts down.)
_plain_suppressed = globals().get("_plain_suppressed", 0)

# Why plain commands don't show a selected parked thread in this session,
# if they don't (see _plain_off).
_plain_off_why = globals().get("_plain_off_why")


def _new_ui_command(text):
    """Whether gdb command `text` is `new-ui` (or an abbreviation gdb takes
    for it: `new mi ...`)."""
    word = (text.split() or [""])[0]
    return len(word) >= 3 and "new-ui".startswith(word)


class _SplitArgs(list):
    """gdb's command line from `ps`, whose arguments with spaces came apart."""


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
        # (Arguments with spaces come apart: see _session_problem.)
        return _SplitArgs(out.decode("utf-8", "replace").split()) or None
    except Exception:
        return None


# gdb's command-line options (its getopt_long_only table, gdb 8.2 to 18):
# those that take a value, and those that don't; and the names that are
# one option.
_GDB_OPTIONS_WITH_VALUE = (
    "annotate", "b", "baud", "c", "cd", "command", "core", "d", "D", "data-directory",
    "directory", "e", "early-init-command", "early-init-eval-command", "editor-command",
    "eiex", "eix", "eval-command", "ex", "exec", "i", "iex", "init-command",
    "init-eval-command", "interpreter", "ix", "l", "p", "pid", "s", "se", "symbols",
    "tclcommand", "tty", "ui", "x")
_GDB_OPTIONS_WITHOUT_VALUE = (
    "args", "batch", "batch-silent", "binary-output", "configuration", "dbx",
    "enable-external-editor", "f", "fullname", "help", "n", "nh", "no-escape-args",
    "nowindows", "nw", "nx",
    "q", "quiet", "r", "readnever", "readnow", "return-child-result", "silent",
    "statistics", "tui", "version", "w", "windows", "write")
_GDB_OPTION_NAMES = {
    "i": "interpreter", "ui": "interpreter", "ex": "eval-command",
    "iex": "init-eval-command", "eiex": "early-init-eval-command", "x": "command",
    "ix": "init-command", "eix": "early-init-command", "b": "baud", "c": "core",
    "d": "directory", "D": "data-directory", "e": "exec", "p": "pid", "s": "symbols",
    "f": "fullname", "n": "nx", "q": "quiet", "silent": "quiet", "r": "readnow",
    "nw": "nowindows", "w": "windows"}


def _gdb_option(word):
    """The option that `word` (an argument without its dashes and value)
    names, as gdb's getopt_long_only takes it: an exact name, else the one
    option it abbreviates (`-interp`, `--batch-s`); None if none (an
    ambiguous or unknown one, which gdb refuses to start with)."""
    names = _GDB_OPTIONS_WITH_VALUE + _GDB_OPTIONS_WITHOUT_VALUE
    if word in names:
        return _GDB_OPTION_NAMES.get(word, word)
    found = set(_GDB_OPTION_NAMES.get(n, n) for n in names if word and n.startswith(word))
    return found.pop() if len(found) == 1 else None


# (_session_problem's word for a `new-ui` among gdb's `-ex` commands.)
_NEW_UI_ON_COMMAND_LINE = "a second UI is open ('new-ui')"


def _session_problem():
    """Why plain commands can't show a selected parked thread in this gdb
    session, judged from gdb's command line; or None."""
    return _read_session()[0]


def _read_session():
    """(_session_problem's answer, whether gdb runs in batch mode or None
    if it can't tell), from gdb's command line."""
    args = _gdb_args()
    if args is None:
        return "realm-gdb can't tell whether this is an MI session", None
    takes_value = set(_GDB_OPTION_NAMES.get(n, n) for n in _GDB_OPTIONS_WITH_VALUE)
    interpreter, batch, problem = None, False, None
    i = 1
    while i < len(args):
        a = args[i]
        i += 1
        if a == "--":
            break
        if not a.startswith("-") or a == "-":
            continue  # (the program, a core file, a process)
        word, eq, value = a.lstrip("-").partition("=")
        option = _gdb_option(word)
        if option is None:
            problem = problem or ("realm-gdb can't tell what kind of session this is "
                                  "(gdb's {!r})".format(a))
            continue
        if option in ("args", "no-escape-args"):  # (gdb 18's other --args)
            break
        if option in takes_value and not eq:
            value = args[i] if i < len(args) else ""
            i += 1
        if option == "interpreter":
            interpreter = value
        elif option in ("batch", "batch-silent"):
            batch = True
        elif option in ("eval-command", "init-eval-command", "early-init-eval-command") \
                and _new_ui_command(value):
            problem = problem or _NEW_UI_ON_COMMAND_LINE
    if problem is not None:
        return problem, batch
    if isinstance(args, _SplitArgs):
        # (`ps` splits an argument with spaces, so an option's value may
        # have taken a real option after it for its own, `-ex 'echo -s'
        # --interpreter mi`: look at every word for those that turn plain
        # commands off, which can find too many, never too few.)
        for k, a in enumerate(args[1:], 1):
            word, eq, value = a.lstrip("-").partition("=")
            if a == "new-ui" or a.startswith("-") and a != "-" and \
                    _gdb_option(word) == "interpreter" and not (
                        value if eq else (args[k + 1] if k + 1 < len(args) else "")
                    ).startswith(("console", "tui")):
                return ("realm-gdb can't tell from gdb's command line whether this is "
                        "an MI session or has a second UI ('ps' splits arguments with "
                        "spaces, and one says {!r})".format(a)), batch
    if interpreter and not interpreter.startswith(("console", "tui")):
        return "this is {} {} session".format(
            "a" if interpreter.startswith("dap") else "an",
            re.match(r"[A-Za-z]*", interpreter).group(0).upper() or interpreter), batch
    if batch and not hasattr(gdb.events, "gdb_exiting"):
        return ("in batch mode, this gdb (< 12) could let go of the program at the "
                "end with a parked thread's registers in place"), batch
    return None, batch


def _other_terminal():
    """A terminal, or a named pipe, that gdb has open besides its standard
    input, output and error (a second UI, 'new-ui', most likely), as "PATH
    (file descriptor N)"; else None.  (Linux only.  Not a pseudo-terminal's
    master side, which a UI isn't on, nor an unnamed pipe, "pipe:[N]", of
    gdb's own.)"""
    import stat
    try:
        fds = sorted(int(fd) for fd in os.listdir("/proc/self/fd"))
    except (OSError, ValueError):
        return None
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
                    not path.endswith("/ptmx") and os.isatty(fd)):
                # (A pseudo-terminal's master side, /dev/ptmx or, in a
                # container, /dev/pts/ptmx, is no terminal of a UI's.)
                return "{} (file descriptor {})".format(path, fd)
            if fd > 2 and path.startswith("/") and path not in own and \
                    stat.S_ISFIFO(os.fstat(fd).st_mode):
                return "{} (file descriptor {})".format(path, fd)
        except OSError:
            pass
    return None


# Whether realm-gdb has said what the TUI shows while a parked thread is
# selected (once per session).
_tui_noted = globals().get("_tui_noted", False)


def _tui_note(on=False):
    """If the TUI is on as a parked thread is selected (`on`: it has just
    been turned on), say once what its windows show.  (It draws them from
    the selected thread and frame between commands, when the host has its
    own registers, though its source window can be left at a location a
    command showed.)  Checked as a thread is selected, and as the TUI is
    turned on where gdb tells (gdb.events.tui_enabled); not at each prompt:
    before gdb 10, a command run then (`info win`) would be the one an empty
    line repeats."""
    global _tui_noted
    if _tui_noted or _selection() is None:
        return
    if not on:
        try:
            out = _execute("info win")  # (Its windows, while the TUI is on.)
        except gdb.error:
            return
        if not out.strip() or "not active" in out:
            return
    _tui_noted = True
    thread = _selected_thread()
    print("realm-gdb: the TUI's windows show thread {} as it really is, not the "
          "selected parked Realm user thread (its source and assembly windows may be left "
          "at a location a command showed); only commands' output shows the user "
          "thread.".format(
              thread.num if thread is not None else "N"))


def _on_tui_enabled(event):
    if getattr(event, "enabled", False):
        _tui_note(on=True)


def _plain_off(why):
    """From now on, plain commands don't show a selected parked thread."""
    global _plain_off_why
    if _plain_off_why is not None:
        return
    _plain_off_why = why
    if _selections:
        print("realm-gdb: {}, so from now on plain commands show threads as they "
              "really are: use 'rt COMMAND' for a selected parked Realm user "
              "thread.".format(why))


def _plain_shows():
    """The selection that plain commands show now, if any."""
    if _plain_off_why is not None or _plain_suppressed > 0 or _window is not None:
        return None
    if not _has_target():
        return None
    sel = _selection()
    if sel is not None and _selected_thread() is None:
        # (gdb has lost its selected thread: see _selected_thread.)
        _drop_selections(lambda s: s["inferior"] == sel["inferior"],
                         "no thread is selected")
        return None
    return sel


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
    elif cmd == "pipe":
        why = ("thread {} would keep the user thread's registers while the shell "
               "command runs".format(gdb.selected_thread().num))
    elif cmd.startswith("explore"):
        why = ("it asks its questions at the terminal, which can't be done while the user "
               "thread is shown")
    else:
        why = "it would act on thread {} as it really is".format(
            gdb.selected_thread().num)
    raise gdb.GdbError(
        "realm-gdb: refusing '{}' while {} is selected: {}.  Run "
        "'realm-thread back' first{}.".format(
            cmd, _ut_words(sel["user_thread"]), why,
            ", or 'rt {} ...' to run it on the user thread".format(cmd)
            if cmd in _hooked.get("rt", ()) else
            ", or 'rt {} -l EXPR' to watch the memory EXPR is in".format(cmd)
            if cmd in _WATCH_COMMANDS else
            ", or use 'print' or 'ptype' instead" if cmd.startswith("explore") else
            ", or use 'thread apply all', which works with one selected"
            if cmd == "thread apply" else ""))


def _note_host_command(cmd):
    """hook-CMD of a command that runs the user's own code on the selected
    thread as it really is (see _NOTED_COMMANDS): say so, once a selection,
    while plain commands show a selected parked thread."""
    sel = _plain_shows()
    if sel is None or sel.get("noted"):
        return
    sel["noted"] = True
    word = cmd.split()[0]
    print("realm-thread: '{}' runs on thread {} as it really is, not on {}{}.".format(
              word, gdb.selected_thread().num, _ut_words(sel["user_thread"]),
              " (the gdb commands it runs show the user thread; 'rt {} ...' runs it "
              "all on the user thread)".format(word) if word != "compile" else ""))


class _ReplPrompt(object):
    """A prompt of Python's REPL (`pi`), as sys.ps1 or sys.ps2, that closes
    a window left open, by a plain command that failed, before the REPL
    reads a line, as gdb's own prompt does (see _close_at_prompt): the REPL
    gives no prompt event, and before gdb 15 no posted closer runs (see
    _post_close).  (Python's REPL leaves sys.ps1 and sys.ps2 set anyway.)"""

    _realm_gdb = True

    def __init__(self, text):
        self.text = text

    def __str__(self):
        try:
            if _plain is not None or _window is not None:
                _close_at_prompt()
        except Exception:
            _debug_exc()
        return str(self.text)


def _install_repl_prompts():
    for name, default in (("ps1", ">>> "), ("ps2", "... ")):
        prompt = getattr(sys, name, None)
        if getattr(prompt, "_realm_gdb", False):
            prompt = prompt.text  # (An earlier load's.)
        setattr(sys, name, _ReplPrompt(default if prompt is None else prompt))


def _plain_before(cmd, outer):
    """hook-CMD of a command that looks at the selected thread or frame
    (run from Python frames `outer`): open a window showing a selected
    parked thread for it."""
    global _plain
    if _plain is not None:
        return  # (CMD is part of the command that opened it)
    if _infcall_depth and (_window is not None or _call_running()):
        # (A dprintf's `printf`, in a function call: threads are running.
        # With no window open, a count that a call which failed before it
        # ran left up is dropped: see _call_running.)
        return
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
        # (What to do depends on what is selected now: _check_selection
        # drops a selection that is no longer good.)
        host, now = _selected_thread(), _plain_shows()
        if host is None or isinstance(e, _Unrestored):
            advice = ""  # (Its own advice is what to do.)
        elif now is sel:
            advice = ("; run 'realm-thread back' first to run it on thread {} as it really "
                      "is".format(host.num))
        elif now is None:
            advice = "; run it again to run it on thread {} as it really is".format(host.num)
        else:
            advice = "; run it again to run it on {}".format(_ut_words(now["user_thread"]))
        raise gdb.GdbError("{}  ('{}' was not run{}.)".format(e, cmd, advice))
    except Exception as e:  # (shouldn't be: but CMD mustn't run on the host)
        _debug_exc()
        raise gdb.GdbError("realm-thread: can't show {}: {}  ('{}' "
                           "was not run.)".format(_ut_words(sel["user_thread"]), e, cmd))
    try:
        entry = {"window": window, "command": cmd, "outer": outer,
                 "lasti": outer[0].f_lasti if outer else None, "input": None,
                 "bpnum": _bpnum() if cmd in _PENDING_COMMANDS else None}
        entry["shown"] = _show_in_window(window, ut, sel)
        entry["input"] = _NoTerminalInput(cmd, window).__enter__()
        _plain = entry
        _post_close(window)  # (In case CMD fails where no prompt follows.)
        if not outer and _stdin_is_terminal() and not _batch_session() and \
                _SIGINT_IN_PYTHON_SAFE:
            # (Typed, or in breakpoint commands, with a prompt to come: gdb
            # runs CMD itself, not Python (see _block_sigint), so Ctrl-C
            # can quit it as usual, leaving the window for that prompt.
            # Not where Ctrl-C kills gdb if it lands as Python runs, as a
            # pretty-printer may: see _SIGINT_IN_PYTHON_SAFE.)
            window.interruptible = True
            _hand_sigint_to_gdb()
    except BaseException as e:
        # (A Ctrl-C as Ctrl-C is let through for CMD, say: its window
        # closes here, and is no longer CMD's to close later, comparing
        # the host's own registers with the parked thread's.)
        if _plain is entry:
            _plain = None
        if entry["input"] is not None:
            entry["input"].__exit__(None, None, None)
        window.__exit__(None, None, None)
        if isinstance(e, (_Unshowable, gdb.error)):
            raise gdb.GdbError("realm-thread: can't show {}: {}.  ('{}' "
                               "was not run.)".format(_ut_words(int(ut)), e, cmd))
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
    if not mine:
        # (After a command run from no Python, at the top level: a Python
        # script run by gdb (`-x FILE.py`, which runs no `source` hook)
        # comes after it, and its outermost frame is the script's own.)
        return outer[-1].f_code.co_name != "<module>"
    return mine[0].f_lasti == entry["lasti"]


def _plain_after(cmd, outer):
    """hookpost-CMD: close the window CMD opened."""
    entry = _plain
    if entry is not None and entry["command"] == cmd and _same_frames(entry, outer):
        entry["finished"] = True
        _close_plain(entry)


def _close_stale(outer):
    """Close a plain command's window if the code now running (from Python
    frames `outer`) isn't part of that command, which then failed.  (Not in
    the middle of a function call, which only a command can be making: one
    that failed before it ran, as the command did then, doesn't count.)"""
    entry = _plain
    if entry is not None and not _inside(entry, outer) and _between_calls(entry["window"]):
        _close_plain(entry)


# MI commands that would let the program run, or let go of it (refused by
# gdb.execute_mi, as the CLI's are by their hooks: see _install_execute_mi).
_MI_GUARDED = re.compile(r"-(exec-(continue|finish|jump|next|next-instruction|run|step|"
                         r"step-instruction|until)|target-(attach|detach|disconnect|download|flash-erase|select)|"
                         r"gdb-exit|interpreter-exec|file-exec-and-symbols|file-exec-file|file-symbol-file|"
                         r"trace-list-static-markers|trace-find)$")
# Those of them refused as a CLI command is (see _WINDOW_GUARDED).
_MI_LIKE_CLI = {"-trace-find": "tfind"}


def _install_execute_mi():
    """Python's gdb.execute_mi (gdb >= 14) runs MI commands, which run no
    hooks: have it close a failed plain command's window first, as a hooked
    command would (a script that catches the failure, then resumes the
    program with -exec-continue, would otherwise resume it with the window
    open), and refuse those that would let the program run or let go of it
    where the CLI's are refused (in a window: one left open by a failed
    command that can't be told from the code now running, say)."""
    _wrap_gdb("execute_mi", "_before_execute_mi")


def _before_execute_mi(caller, args):
    _before_parameter(caller, args)
    if args and isinstance(args[0], str) and _MI_GUARDED.match(args[0]):
        try:
            _guard(args[0], _MI_LIKE_CLI.get(args[0]))
        except gdb.GdbError as e:
            raise gdb.error(str(e))  # (As execute_mi's own errors are.)


def _install_parameter():
    """Have Python's gdb.parameter close a failed plain command's window
    first, as a hooked command would: else a script that catches the
    failure, then saves a setting and sets it back (gdb.with_parameter
    does), would save the window's value (`confirm off`, say), and leave it
    for good."""
    _wrap_gdb("parameter", "_before_parameter")


def _before_parameter(caller, args):
    try:
        _close_stale(_outer_frames(caller))
    except Exception:
        _debug_exc()


def _wrap_gdb(name, before):
    """Have Python's gdb.NAME run realm-gdb's function `before` (looked up
    by name, so that the latest load's runs) first, given the frame that
    calls it and its arguments.  Also in gdb's C module, which the gdb
    module takes it from, and gdb.with_parameter looks gdb.parameter up in
    (gdb 15 and 16)."""
    old = getattr(gdb, name, None)
    if old is None:
        return
    real = getattr(old, "_realm_gdb_real", old)  # (An earlier load's wraps it.)
    namespace = globals()

    def wrapper(*args, **kwargs):
        check = namespace.get(before)
        if check is not None:  # (None: a later load of another version.)
            check(sys._getframe().f_back, args)
        return real(*args, **kwargs)
    wrapper._realm_gdb_real = real
    wrapper.__doc__ = real.__doc__
    setattr(gdb, name, wrapper)
    try:
        import _gdb
        if getattr(_gdb, name, None) in (real, old):
            setattr(_gdb, name, wrapper)
    except ImportError:
        pass
    _rebind(real, old, wrapper)


def _rebind(real, old, wrapper):
    """Point the references to gdb's function `real` (or to an earlier
    load's wrapper of it, `old`) that Python code took for itself before
    realm-gdb loaded (`from gdb import execute_mi`, in a module, or in
    gdb's `python`), and closures over them, at `wrapper`.  (Not those in
    a running function's variables, an object's attributes, or a
    functools.partial.)"""
    import gc
    try:
        modules = set(id(vars(m)) for m in list(sys.modules.values())
                      if hasattr(m, "__dict__"))
        mine = set(id(c) for f in (wrapper, old)
                   for c in (getattr(f, "__closure__", None) or ()))  # (Which hold `real`.)
        for obj in (real, old):
            for ref in (gc.get_referrers(obj) if obj not in (None, wrapper) else ()):
                if isinstance(ref, dict) and id(ref) in modules:
                    for key, value in list(ref.items()):
                        if value is obj and not key.startswith("_realm_gdb"):
                            ref[key] = wrapper
                elif type(ref).__name__ == "cell" and id(ref) not in mine:
                    try:
                        if ref.cell_contents is obj:
                            ref.cell_contents = wrapper
                    except (AttributeError, ValueError):
                        pass  # (Python < 3.7 can't.)
    except Exception:
        _debug_exc()


def _close_plain(entry, idle=False):
    """Close the window a plain command opened, and print its notes (with
    gdb `idle`, about to prompt: see _let_held)."""
    global _plain, _sigint_call
    window, notes = entry["window"], []
    # (Until the notes are printed: see _apply_checked.  No call is running.)
    _sigint_call = False
    _block_sigint(True)
    window.hold_sigint = True
    try:
        try:
            if _plain is entry:
                _plain = None
            if entry["input"] is not None:
                entry["input"].__exit__(None, None, None)
            if not window.resumed and not window.done:
                try:
                    notes = _window_done(window, entry["shown"], entry["command"])
                except gdb.error:
                    _debug_exc()
        finally:
            window.__exit__(None, None, None)
        shown = entry.get("shown")
        if not entry.get("finished") and shown is not None and shown["refused"] and \
                shown["selection"] is not None and \
                shown["selection"].get("refused_noted") != shown["refused"]:
            # (It failed, perhaps of a call refused, which gdb's error doesn't
            # explain: once for the selection.)
            shown["selection"]["refused_noted"] = shown["refused"]
            notes.append("realm-gdb: function calls are refused while a parked Realm user "
                         "thread is shown, {} (gdb says \"may-call-functions is "
                         "off\").".format(shown["refused"]))
        if entry.get("finished") and entry["bpnum"] is not None and \
                entry["bpnum"] == _bpnum() and _parameter("breakpoint pending", False) is None:
            # (It ran, and made no breakpoint: gdb's question whether to make
            # it pending, as its location isn't loaded, took its default
            # answer in the window, see _CallSettings.)
            notes.append("realm-gdb: no breakpoint was made: with a parked Realm user "
                         "thread shown, gdb's question whether to make it pending took "
                         "its default answer, no.  Run '{}' again after 'realm-thread "
                         "back', or after 'set breakpoint pending on'.".format(
                             entry["command"]))
        for note in _kept_notes(window, notes) + _call_notes(window):
            _say(note)
    finally:
        _let_held(idle)


# Plain commands that ask whether to make a breakpoint pending (see
# _close_plain).
_PENDING_COMMANDS = ("break", "tbreak", "hbreak", "thbreak", "dprintf")


def _bpnum():
    """$bpnum, the number of the last breakpoint made, of any kind (gdb 8's
    Python doesn't see hardware breakpoints); 0 if none."""
    try:
        value = gdb.parse_and_eval("$bpnum")
        return 0 if value.type.code == gdb.TYPE_CODE_VOID else int(value)
    except gdb.error:
        return 0


class RealmThreadRun(gdb.Command):
    """Run a command on the selected parked Realm user thread.

    Usage: rt COMMAND

    Runs COMMAND (`bt`, `frame 3`, `up`, `info locals`, `print VAR`, ...) on
    the parked user thread that `realm-thread N` selected, as `realm-thread
    apply N COMMAND` would, but with the frame last selected on it (by an
    earlier `rt frame N`, `rt up`, ...) selected first.  In a console gdb
    session, plain commands that work with the selected frame do the same
    without `rt` (see `help realm-thread`); `rt` also runs those refused
    there that work on a parked thread (`rt python ...`, `rt frame apply
    ...`), and works in MI and DAP sessions too.  Commands that would let
    the program run or let go of it are refused: run them without `rt`
    (after `realm-thread back`, for those refused there too), after which
    the parked thread is no longer selected: Realm may resume it.

    Commands that start another program and look at nothing of the parked
    thread's (`shell`, `!`, `make`, `edit`) run as they would without `rt`;
    `pipe COMMAND | SHELL` runs COMMAND on it, then SHELL, and `edit *EXPR`
    works out EXPR on it, then edits (as `realm-thread apply` does too).
    In a core file, where `realm-thread N` keeps the parked thread shown,
    `rt COMMAND` is just COMMAND."""

    def __init__(self):
        super(RealmThreadRun, self).__init__("rt", gdb.COMMAND_STACK)

    def invoke(self, arg, from_tty):
        _close_stale(_outer_frames(sys._getframe().f_back))
        with _Gdb8Quit():  # (See there.)
            _printing_over()
            command = arg.strip()
            if not command:
                raise gdb.GdbError("usage: rt COMMAND (after 'realm-thread N')")
            # (First: a body of COMMAND's mustn't be left to run, whatever fails.)
            dropped = _drop_body(command, from_tty, "'rt'")
            if dropped is not None:
                _check_command(command, "'rt'", dropped)
            if _has_target() and _window is None and _how_shown() == "view":
                view = _current_view()
                if view is None or view["temporary"]:
                    raise gdb.GdbError("realm-thread: no parked user thread is shown on "
                                       "this thread{}: run 'realm-thread N' first.".format(
                                           _view_elsewhere_words()))
                # (A command that reads the lines after it would have none, from
                # gdb.execute: `rt define foo` would empty foo.)
                _check_command(command, "'rt'")
                global _pi_allowed
                _pi_allowed = False  # (No window here: see _check_command.)
                try:
                    gdb.execute(command, from_tty=from_tty)
                except gdb.error as e:
                    raise gdb.GdbError(str(e))  # (gdb's error, not a Python exception.)
                return
            _live_checks("run 'rt'")
            sel = _selection()
            if sel is None:
                elsewhere = sorted(set(s["inferior"] for s in _selections))
                raise gdb.GdbError(
                    "realm-thread: no parked user thread is selected{}: run "
                    "'realm-thread N' first, or 'realm-thread apply N COMMAND'"
                    ".".format(" in this inferior (inferior {} has one)".format(
                        elsewhere[-1]) if elsewhere else ""))
            with _KeepSelection():
                sps, _ = _host_sps()
                ut = _check_selection(sel, sps)
                error = _apply([(ut, None)], command, header=False, selection=sel,
                               from_tty=from_tty)
            if error is not None:
                raise gdb.GdbError(str(error))

    def complete(self, text, word):
        if word is None:
            word = re.split(r"[ \t]", text)[-1]
        return _complete_command(text, word)


class RealmThread(gdb.Command):
    """List parked Realm user threads, select one, or run a command on them.

    Usage: realm-thread [N|<ptr> | apply N...|all [-q] [-c] [-s] COMMAND | back]

    With no argument, list the Realm user threads of the selected inferior,
    numbered in the order they were made (as `info threads` numbers gdb's
    threads), with the selected one marked `*`.  A user thread is given by
    that number N or by its UserThread pointer <ptr> (one word, or an
    expression in parentheses).

    `realm-thread apply N COMMAND` runs COMMAND on that parked user thread,
    for example `realm-thread apply 3 bt full`, or (gdb >= 8.3)
    `realm-thread apply 3 frame apply level 3 print VAR`.  As `thread
    apply`, it takes a list of them too (`realm-thread apply 3 5-7 bt`), and
    `realm-thread apply all COMMAND` runs it on each parked user thread;
    each with a "User thread N (<ptr>):" line first.  `thread apply`'s flags
    come after the list: -q leaves those lines out, -s leaves out the
    threads where COMMAND fails or prints nothing, and -c goes on past a
    failure without failing at the end (without -c or -s, it goes on, then
    fails with the first error); after `all`, -ascending goes from the
    oldest.

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

    `realm-thread N` selects that parked user thread, until `realm-thread
    back` (or `restore`); selections nest.  In a live process this changes
    nothing in the program: `rt COMMAND` then runs COMMAND on it, as
    `realm-thread apply N COMMAND` does, with the frame selected by an
    earlier `rt frame N` (or `rt up`, ...) selected first.  The prompt shows
    the selection.  It is dropped once the program runs.  If realm-gdb
    couldn't put a thread's own registers back, `realm-thread back` tries
    again instead.

    In a console gdb session, while a parked thread is selected, plain
    commands that work with the selected frame (`bt`, `frame N`, `up`,
    `print VAR`, `info locals`, `x`, `list`, `set var`, `break`, ...) show
    it, as `rt` would; gdb's pager is off in them (`rt COMMAND` pages), and
    its queries take their default answers (so `break` makes no pending
    breakpoint unless `set breakpoint pending on`).  A change to a variable
    kept in a register lasts only for that command, and a thread-local
    variable is the selected thread's own.  Commands that would act on the
    thread as it really is (`step`, `next`, `finish`, `return`, `watch`,
    `thread apply`, `frame apply`, ...) are refused; `python` runs on the
    thread as it really is, with a note.  `continue`, or `thread N` to
    another thread, ends the selection.  Not in MI or DAP sessions, after
    `new-ui`, or in batch mode with gdb < 12: use `rt` there.

    In a core file, a Python unwinder shows the parked thread's stack on the
    selected thread instead: frame #0 stays the thread's own frame (which
    backtraces show as a `<parked Realm user thread ...>` line), and the
    user thread's stack starts at frame #1.  There `realm-thread N` keeps it
    shown, so that `bt`, `frame N`, `info locals` and `print VAR` (and `rt
    COMMAND`, which is the same there) operate on it."""

    def __init__(self):
        # (A prefix, for its subcommands `apply`, `back` and `restore`, which
        # gdb abbreviates and completes; its own `invoke` takes the rest.)
        super(RealmThread, self).__init__("realm-thread", gdb.COMMAND_RUNNING, prefix=True)

    def invoke(self, arg, from_tty):
        _close_stale(_outer_frames(sys._getframe().f_back))
        with _Gdb8Quit():  # (See there.)
            _printing_over()
            arg = arg.strip()
            if _window is not None:
                # (Listing them: the thread showing one has its stack pointer,
                # which would make it look as if it were switching to it.)
                raise gdb.GdbError("realm-thread: can't {} from inside {}.".format(
                    "list the user threads" if not arg else "change what is selected",
                    _window_words(_window)))
            if not arg:
                self._list()
            else:
                self.dont_repeat()  # (an empty line mustn't select it again)
                self._show_thread(arg)

    def complete(self, text, word):
        return []  # (gdb completes the subcommands.)

    @staticmethod
    def complete_apply(text, word):
        # (gdb >= 13 first asks with word=None, to find the word breaks.)
        if word is None:
            word = re.split(r"[ \t]", text)[-1]
        after = text.lstrip()
        try:
            targets, rest = _split_targets(after)
        except gdb.GdbError:
            return []
        if not rest and not after[-1:].isspace():
            return [w for w in ("all",) if w.startswith(word) and len(targets) <= 1]
        return _complete_command(rest, word)

    def _list(self):
        if not _has_target():
            print("realm-thread: no process or core file.")
            self._list_elsewhere(False)
            return
        sps = {}
        others_running = False
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
        rows = []
        try:
            # (Read the registry through a stopped thread if there is one:
            # gdb 9 can't read memory through a running thread.)
            with _KeepSelection():
                _select_inferior(gdb.selected_inferior())
                try:
                    for ut in _iter_user_threads():
                        rows.append((int(ut), self._state(ut, sps, shown, selected)))
                finally:
                    self._print_rows(rows)
        except _RegistryLoop as e:
            print("realm-thread: {}; the list above may be incomplete.".format(e))
        except gdb.error as e:
            if rows:
                print("realm-thread: cannot read the rest of the Realm user thread "
                      "registry: {}; the list above may be incomplete.".format(e))
            elif not others_running:
                raise gdb.GdbError("realm-thread: {}".format(e) if isinstance(e, _RegistryLoop)
                                   else "realm-thread: cannot read the Realm user thread "
                                   "registry: {}".format(e))
            else:
                print("realm-thread: cannot read the Realm user thread registry while "
                      "the program is running.  To list them, stop it ('interrupt') "
                      "and rerun this command.")
        else:
            if not rows:
                print(_no_registry_words() or
                      "realm-thread: no Realm user threads in the registry.")
            else:
                how = _how_shown()
                if isinstance(how, _NotShown):
                    print("realm-thread: can't show parked user threads now: " +
                          how.sentence())
        self._list_elsewhere(bool(rows))

    @staticmethod
    def _list_elsewhere(listed):
        """Say which other inferiors have user threads (`listed`: whether
        this one's were)."""
        if len(gdb.inferiors()) > 1:
            for num, addrs in _user_threads_elsewhere():
                print("realm-thread: inferior {} has {} Realm user thread{}{}: "
                      "select it ('inferior {}') to list {}.".format(
                          num, len(addrs), "s" if len(addrs) > 1 else "",
                          " too" if listed else "", num, "them" if len(addrs) > 1 else "it"))

    @staticmethod
    def _state(ut, sps, shown, selected):
        executing = _executing_on(ut, sps)
        if bool(ut["running"]):
            t = _host_thread_of(ut)
            return "RUNNING" + ("  (on thread {})".format(t.num) if t else "")
        if executing is not None:
            return "RUNNING  (on thread {}, switching contexts)".format(executing)
        if int(ut) in shown:
            return "parked  (shown on thread {})".format(
                ", ".join(str(n) for n in sorted(shown[int(ut)])))
        if int(ut) in selected:
            current = _selection()
            return "parked  (selected{})".format(
                "" if current is not None and current["user_thread"] == int(ut)
                else " earlier")
        try:
            if _not_started(_arch.unwind_regs_from_ctx(ut)[_arch.pc_reg]) is not None:
                return "not started"
        except gdb.error:
            pass
        return "parked"

    @staticmethod
    def _print_rows(rows):
        """The list, as `info threads` lists threads: by number, the selected
        one marked with "*"."""
        if not rows:
            return
        numbers = _numbered([addr for addr, _ in rows])
        width = max(2, len(str(max(numbers[addr] for addr, _ in rows))))
        ptr = max(len("{:#x}".format(addr)) for addr, _ in rows)
        view = _kept_view()  # (A core file's: the one shown, as a selection.)
        print("  {:<{}}  {:<{}}  State".format("Id", width, "UserThread", ptr))
        for addr, state in sorted(rows, key=lambda row: numbers[row[0]]):
            print("{} {:<{}}  {:<#{}x}  {}".format(
                "*" if state.endswith("(selected)") or
                view is not None and view["user_thread"] == addr else " ",
                numbers[addr], width, addr, ptr, state))

    def _apply(self, arg, from_tty):
        targets, command = _split_targets(arg)
        flags, command = _split_flags(command.strip())
        command = command.strip()
        # (See RealmThreadRun.invoke.)
        dropped = _drop_body(command, from_tty, "'realm-thread apply'")
        if dropped is not None:
            _check_command(command, "'realm-thread apply'", dropped)
        ascending = False
        for flag in flags:
            if len(flag) > 1 and "-ascending".startswith(flag) and targets == ["all"]:
                ascending = True
            elif flag not in ("-q", "-c", "-s", "--"):
                raise gdb.GdbError("Unrecognized option at: {} {}".format(flag, command))
        flags = [f for f in flags if not (len(f) > 1 and "-ascending".startswith(f))]
        if not targets or not command:
            raise gdb.GdbError("usage: realm-thread apply N...|all [-ascending] [-q] [-c] "
                               "[-s] COMMAND")
        _live_checks("nest 'realm-thread apply'")
        with _KeepSelection():
            sps, _ = _host_sps()
            if targets == ["all"]:
                try:
                    parked, broken = _parked_threads()  # (Those before a break are shown.)
                except gdb.error as e:
                    raise gdb.GdbError("realm-thread: {}".format(e) if isinstance(
                        e, _RegistryLoop) else "realm-thread: cannot read the Realm user "
                        "thread registry: {}".format(e))
                if not parked:
                    print(_no_registry_words() or
                          "realm-thread: no parked Realm user threads.")
                    return
                if ascending:
                    parked.reverse()
                try:
                    error = _apply([(ut, _parked_problem(ut, sps)) for ut in parked],
                                   command, header=True, flags=flags, from_tty=from_tty)
                finally:
                    if broken is not None:
                        print("\nrealm-thread: {}; there may be more parked "
                              "threads.".format(broken))
            elif len(targets) == 1:
                error = _apply([(_showable_user_thread(targets[0], sps), None)], command,
                               header=False, from_tty=from_tty)
            else:
                # (As `thread apply 2 3`: each with its header, one that can't
                # be shown saying why.)
                uts = [_lookup_user_thread(t) for t in targets]
                error = _apply([(ut, _parked_problem(ut, sps)) for ut in uts], command,
                               header=True, flags=flags, from_tty=from_tty)
        if error is not None:
            raise gdb.GdbError(str(error))

    def _show_thread(self, expr):
        if not _has_target():
            raise gdb.GdbError("realm-thread: no process or core file.")
        how = _how_shown()
        if how == "window":
            return self._select(expr)
        if isinstance(how, _NotShown):
            raise how.error()
        thread = _stopped_thread()
        ut = _showable_user_thread(expr, _host_sps()[0])
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
            raise gdb.GdbError("realm-thread: can't show {}: it can't be "
                               "unwound {}.".format(_ut_words(int(ut)), _unwind_words(
                                   view, _parameter("backtrace limit"))))
        print("realm-thread: showing {} on thread {}: its stack "
              "starts at frame #1 (#0 is {}).".format(
                  _ut_words(int(ut)), thread.num, _frame0_words(thread.num)))
        print("Use 'bt', 'frame N', 'info locals', 'print VAR' as normal.  "
              "Run 'realm-thread back' to undo.")
        _install_prompt_hook()
        gdb.execute("frame")  # (As `thread N` prints its frame.)

    def _select(self, expr):
        """`realm-thread <ptr>` in a live process (see "Selecting a parked
        thread")."""
        _live_checks("select a parked user thread")
        _check_unrestored(gdb.selected_thread())  # (Rather than select one not shown.)
        with _KeepSelection():
            ut = _showable_user_thread(expr, _host_sps()[0])
            inferior = gdb.selected_inferior()
            sel = {"inferior": inferior.num, "pid": inferior.pid,
                   "user_thread": int(ut), "regs": _arch.unwind_regs_from_ctx(ut),
                   "level": 0}
        other = _other_terminal() if _plain_off_why is None else None
        if other is not None:
            _plain_off("a second UI may be open (gdb has {} open, a terminal or named "
                       "pipe)".format(other))
        current = _selection()
        if current is not None and current["user_thread"] == sel["user_thread"] and \
                current["regs"] == sel["regs"]:
            _selections.remove(current)  # (Selected again: from frame #0, not twice.)
        _selections.append(sel)
        _install_prompt_hook()
        host = gdb.selected_thread().num
        global _selection_explained
        if _selection_explained == (_plain_off_why is None):
            # (Said in full once: `help realm-thread` has it too.)
            print("[Selecting {}, shown on thread {}{}]".format(
                _ut_words(int(ut)), host, "" if _plain_off_why is None else
                "; 'rt COMMAND' runs commands on it"))
        elif _plain_off_why is None:
            print("realm-thread: selected {}, shown on thread {} one command at a time "
                  "(the program isn't changed).  'bt', 'frame N', 'print VAR', ... show "
                  "it, and 'step', 'finish', 'watch', ... are refused, until 'realm-thread "
                  "back', 'continue', or 'thread N' to another thread.".format(
                      _ut_words(int(ut)), host))
        else:
            print("realm-thread: selected {}.  Run commands on it with "
                  "'rt COMMAND' ('rt bt', 'rt frame 2', 'rt print VAR'), which shows it on "
                  "thread {} for that command (the program isn't changed).  Plain "
                  "commands show thread {} as it really is: {}.  'realm-thread back' "
                  "undoes it.".format(_ut_words(int(ut)), host, host, _plain_off_why))
        _selection_explained = _plain_off_why is None
        with _KeepSelection():
            error = _apply([(ut, None)], "frame", header=False, selection=sel)
        if error is not None:
            print("realm-thread: {}".format(error))
        _tui_note()

    def _back(self):
        if _unrestored:
            _retry_unrestored(report=True)
            sel = _selection()
            if sel is not None:
                print("realm-thread: {} is still selected ('realm-thread "
                      "back' again goes back from it).".format(_ut_words(sel["user_thread"])))
            return
        view = _current_view()
        if view is None:
            sel = _selection()
            if sel is None:
                print("realm-thread: no parked user thread is selected{}.".format(
                    _view_elsewhere_words()))
                return
            _drop_selections(lambda s: s is sel)
            sel = _selection()
            if sel is not None:
                print("realm-thread: back to {}.".format(_ut_words(sel["user_thread"])))
            else:
                print("realm-thread: no parked user thread is selected now.")
            self._print_frame(sel)
            return
        _remove_view(view)
        _refresh_frames("parked")
        view = _current_view()
        if view is not None:
            print("realm-thread: back to {}.".format(_ut_words(view["user_thread"])))
        else:
            print("realm-thread: no parked user thread is shown now.")
        self._print_frame(None)

    @staticmethod
    def _print_frame(sel):
        """The frame selected now, as `thread N` prints it: on the parked
        thread selected now (`sel`), or the thread itself.  (gdb's source
        position, for `list`, is then there too.)"""
        try:
            if sel is None:
                if _has_target() and _selected_thread() is not None:
                    gdb.execute("frame")
                return
            with _KeepSelection():
                sps, _ = _host_sps()
                error = _apply([(_check_selection(sel, sps), None)], "frame", header=False,
                               selection=sel)
            if error is not None:
                print("realm-thread: {}".format(error))
        except gdb.error as e:
            print(str(e))


# What realm-gdb.py says as it loads.
_LOAD_BANNER = """\
realm-gdb.py loaded: 'thread apply all bt' also shows parked Realm user
threads, 'realm-thread' lists them, 'realm-thread N' selects one; see 'realm-help'."""

_HELP_BANNER = """\
realm-gdb.py: commands for Realm user threads.

  thread apply all bt       gdb's 'thread apply all bt', plus the stack
                            of every parked Realm user thread.  Also for
                            'backtrace', 'where' and 'info stack', and any
                            bt options and flags ('thread apply all bt full').

  realm-thread              List the Realm user threads, numbered.

  realm-thread apply N...|all [-q] [-c] [-s] COMMAND
                            Run COMMAND on parked user threads (or each of
                            them), given by number (or range N-M) or
                            UserThread pointer: 'bt full', 'frame apply
                            level 3 print VAR' (gdb >= 8.3).

  realm-thread N|<ptr>      Select a parked user thread: then plain 'bt',
  realm-thread back         'frame N', 'up', 'print VAR', 'info locals', ...
                            show it, keeping the frame you select, and
                            'step', 'finish', 'return', 'watch', ... are
                            refused, until 'back' undoes the selection
                            ('continue' and 'thread N' undo it too).
  rt COMMAND                Run COMMAND on the selected parked thread
                            (also in MI and DAP sessions, where plain
                            commands show threads as they really are).

  realm-help                Print this summary.

In a live process, a parked user thread is shown by putting its saved
registers into the selected thread for the duration of one command, then
putting that thread's own back, so that at gdb's prompt nothing in the
program has changed.  During such a command, commands that would let the
program run or let go of it (continue, step, detach, quit, ...) are
refused, and function calls hold the other threads, can't stop at
breakpoints, and are unwound if they get a signal.  Parked user threads
can't be shown in non-stop mode or while the program runs.  See 'help
realm-thread' and 'help rt' for more.

Linux / FreeBSD on x86-64 or aarch64.  Source: src/realm/realm-gdb.py."""


class RealmThreadApply(gdb.Command):
    """Run a command on parked Realm user threads.

    Usage: realm-thread apply N...|all [-ascending] [-q] [-c] [-s] COMMAND

    As `thread apply`, for the user threads `realm-thread` lists: N is a
    number, a range N-M, or a UserThread pointer.  See `help realm-thread`."""

    def __init__(self):
        super(RealmThreadApply, self).__init__("realm-thread apply", gdb.COMMAND_RUNNING)

    def invoke(self, arg, from_tty):
        _close_stale(_outer_frames(sys._getframe().f_back))
        with _Gdb8Quit():  # (See there.)
            _printing_over()
            if _window is not None:
                raise gdb.GdbError("realm-thread: can't run 'realm-thread apply' inside "
                                   "{}.".format(_window_words(_window)))
            _realm_thread._apply(arg, from_tty)

    def complete(self, text, word):
        return RealmThread.complete_apply(text, word)


class RealmThreadBack(gdb.Command):
    """Undo the latest `realm-thread N`, back to the user thread selected
    before it, if any.

    Usage: realm-thread back

    Also tries again to put back a thread's own registers that realm-gdb
    couldn't.  See `help realm-thread`."""

    def __init__(self, name="realm-thread back"):
        super(RealmThreadBack, self).__init__(name, gdb.COMMAND_RUNNING)
        self.name = name

    def invoke(self, arg, from_tty):
        _close_stale(_outer_frames(sys._getframe().f_back))
        _printing_over()
        self.dont_repeat()  # (an empty line mustn't go back again)
        if arg.strip():
            raise gdb.GdbError("realm-thread: '{}' takes no argument.".format(self.name))
        if _window is not None:
            raise gdb.GdbError("realm-thread: can't change what is selected from inside "
                               "{}.".format(_window_words(_window)))
        _realm_thread._back()


def _hide_guard():
    """Keep `realm-gdb-guard`, which only realm-gdb's hooks run, out of
    completion (`realm-<TAB>`, `rt <TAB>`): gdb leaves out a deprecated
    command where others match.  gdb warns once that one is deprecated, as
    it is run: here, unseen.  (And select-frame's forms, found once: see
    _select_frame_forms.)"""
    try:
        _execute("maintenance deprecate realm-gdb-guard")
        _execute("realm-gdb-guard")
    except gdb.error:
        _debug_exc()
    _select_frame_forms()


class RealmHelp(gdb.Command):
    """Print realm-gdb.py's summary of its commands.

    Usage: realm-help"""

    def __init__(self):
        super(RealmHelp, self).__init__("realm-help", gdb.COMMAND_SUPPORT)

    def invoke(self, arg, from_tty):
        print(_HELP_BANNER)


def _load():
    """Set realm-gdb up: its commands, hooks and event handlers."""
    global _plain_off_why, _second_ui, _banner_printed, _realm_thread
    import inspect
    for command in (ThreadApplyAll, RealmThread, RealmThreadApply, RealmThreadBack,
                    RealmThreadRun, RealmThreadGuard, RealmHelp):
        # (gdb 13 and later dedent a command's docstring for `help`.)
        command.__doc__ = inspect.cleandoc(command.__doc__)
    ThreadApplyAll()
    _realm_thread = RealmThread()
    RealmThreadApply()
    RealmThreadBack()
    RealmThreadBack("realm-thread restore")
    RealmThreadRun()
    RealmThreadGuard()
    _hide_guard()
    RealmHelp()
    _install_hooks()
    _install_execute_mi()
    _install_parameter()
    _install_atexit()
    _install_audit()
    _install_thread_sigmask()
    _install_python_sigint()
    if _plain_off_why is None:
        _plain_off_why = _session_problem()
        if _plain_off_why == _NEW_UI_ON_COMMAND_LINE:
            _second_ui = _plain_off_why
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
    elif not _batch_session():  # (Not into a batch run's output: `gdb -batch -p PID`.)
        _banner_printed = True
        try:
            print(_LOAD_BANNER)
        except KeyboardInterrupt:
            # (`q` at gdb's pager, which gdb's own messages can bring up as
            # it attaches to a process (`gdb -p PID`) at a short terminal:
            # no more of it, and realm-gdb is set up by then.)
            pass


if _supported():
    try:
        _load()
    except BaseException:  # (Ctrl-C, say, as librealm loads.)
        print("realm-gdb: loading realm-gdb.py was interrupted, so it doesn't show "
              "parked Realm user threads in a live process: load it again ('source' "
              "it, from src/realm in Realm's sources) to finish.")
        raise
else:
    # Other platforms / architectures: do nothing rather than provide a
    # half-working command.
    pass
