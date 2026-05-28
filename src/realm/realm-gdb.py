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
# Linux x86-64 only.  On other targets the script loads as a no-op.

import gdb
import platform


# glibc x86-64 ucontext gregs[] indices (see /usr/include/sys/ucontext.h).
REG_RBP = 10
REG_RSP = 15
REG_RIP = 16


def _supported():
    return platform.system() == "Linux" and platform.machine() in ("x86_64", "amd64")


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
    gregs = ut["ctx"]["uc_mcontext"]["gregs"]
    return {
        "rip": int(gregs[REG_RIP]) & ((1 << 64) - 1),
        "rsp": int(gregs[REG_RSP]) & ((1 << 64) - 1),
        "rbp": int(gregs[REG_RBP]) & ((1 << 64) - 1),
    }


def _current_regs():
    return {
        "rip": int(gdb.parse_and_eval("(unsigned long)$rip")),
        "rsp": int(gdb.parse_and_eval("(unsigned long)$rsp")),
        "rbp": int(gdb.parse_and_eval("(unsigned long)$rbp")),
    }


def _set_regs(regs):
    gdb.execute("set $rip = {:#x}".format(regs["rip"]), to_string=True)
    gdb.execute("set $rsp = {:#x}".format(regs["rsp"]), to_string=True)
    gdb.execute("set $rbp = {:#x}".format(regs["rbp"]), to_string=True)


# Stack of saved register snapshots for nested `realm-thread`/`realm-thread back`.
_impersonate_stack = []


def _restore_all_impersonations():
    """Pop every outstanding realm-thread impersonation, restoring the
    currently selected pthread's registers to the original snapshot.

    Invoked from gdb hook- commands defined in _install_safety_hooks() so
    that the user can't accidentally `continue`, `step`, or `thread N` with
    clobbered registers."""
    if not _impersonate_stack:
        return
    count = len(_impersonate_stack)
    while _impersonate_stack:
        regs = _impersonate_stack[-1]
        try:
            _set_regs(regs)
        except gdb.error as e:
            print("realm-gdb: WARNING: failed to restore registers: {}".format(e))
            break
        _impersonate_stack.pop()
    print(
        "realm-thread: auto-restored {} impersonation(s) before resume/switch.".format(
            count
        )
    )


def _install_safety_hooks():
    """Install gdb hook-<cmd> commands that auto-restore impersonations
    before any command that resumes the inferior or switches threads.  Each
    hook just calls _restore_all_impersonations() via the Python interpreter
    -- harmless when nothing is impersonating."""
    cmds = (
        "continue", "step", "next", "finish", "run", "start",
        "until", "advance", "jump", "stepi", "nexti", "signal",
        "thread",
    )
    body = "python _restore_all_impersonations()"
    for c in cmds:
        try:
            gdb.execute(
                "define hook-{c}\n{body}\nend\n".format(c=c, body=body),
                to_string=True,
            )
        except gdb.error as e:
            print("realm-gdb: WARNING: could not install hook-{}: {}".format(c, e))


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
    try:
        for ut in parked:
            try:
                regs = _saved_regs(ut)
            except gdb.error as e:
                print("\nUser thread {:#x}: <cannot read ctx: {}>".format(int(ut), e))
                continue
            print("\nUser thread {:#x}:".format(int(ut)))
            try:
                _set_regs(regs)
                gdb.execute("bt", to_string=False)
            except gdb.error as e:
                print("  <bt failed: {}>".format(e))
    finally:
        try:
            _set_regs(saved)
        except gdb.error as e:
            print("realm-gdb: WARNING: failed to restore registers: {}".format(e))


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
    `info locals`, and `print VAR` operate on its stack.
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
            regs = _saved_regs(ut)
        except gdb.error as e:
            print("realm-thread: cannot read ctx: {}".format(e))
            return
        _impersonate_stack.append(_current_regs())
        try:
            _set_regs(regs)
        except gdb.error as e:
            _impersonate_stack.pop()
            print("realm-thread: failed to set registers: {}".format(e))
            return
        print("realm-thread: now impersonating user thread {:#x}.".format(int(ut)))
        print("Use 'bt', 'frame N', 'info locals', 'print VAR' as normal.")
        print("Run 'realm-thread back' to restore.")

    def _restore(self):
        if not _impersonate_stack:
            print("realm-thread: nothing to restore.")
            return
        try:
            _set_regs(_impersonate_stack[-1])
        except gdb.error as e:
            print("realm-thread: failed to restore: {}".format(e))
            return
        _impersonate_stack.pop()
        print("realm-thread: restored.")


if _supported():
    ThreadApply()
    RealmThread()
    _install_safety_hooks()
    print(
        "realm-gdb.py loaded: 'thread apply all bt' includes parked user threads; "
        "use 'realm-thread' to inspect parked stacks."
    )
else:
    # Other platforms / architectures: do nothing rather than provide a
    # half-working command.
    pass
