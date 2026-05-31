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
    """Snapshot the currently-selected thread's registers AND record which
    thread we snapshot from, so the restore can target the same thread
    even if the selected thread has changed in the meantime (e.g., during
    gdb teardown when `gdb.events.gdb_exiting` fires)."""
    t = gdb.selected_thread()
    return {
        "thread_num": t.num if (t is not None and t.is_valid()) else None,
        "rip": int(gdb.parse_and_eval("(unsigned long)$rip")),
        "rsp": int(gdb.parse_and_eval("(unsigned long)$rsp")),
        "rbp": int(gdb.parse_and_eval("(unsigned long)$rbp")),
    }


def _set_regs(regs):
    """Write rip/rsp/rbp to the currently-selected thread."""
    gdb.execute("set $rip = {:#x}".format(regs["rip"]), to_string=True)
    gdb.execute("set $rsp = {:#x}".format(regs["rsp"]), to_string=True)
    gdb.execute("set $rbp = {:#x}".format(regs["rbp"]), to_string=True)


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


# Stack of impersonation entries for nested `realm-thread`/`realm-thread back`.
# Each entry is a dict:
#   host_regs:        snapshot of the host pthread's registers BEFORE impersonation
#                     (used to restore on `realm-thread back`, quit, continue, etc.)
#   parked_regs:      the parked user thread's saved context (re-applied after an
#                     inferior function call, which temporarily swaps host_regs back)
#   user_thread_ptr:  the UserThread pointer the user named (informational)
# Both host_regs and parked_regs are snapshots as returned by _current_regs:
# {thread_num, rip, rsp, rbp}.  thread_num identifies which gdb thread the
# entry applies to so restore writes to the right thread even if the
# selected thread has shifted.
_impersonate_stack = []


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
        if not _apply_snapshot(_impersonate_stack[-1]["host_regs"]):
            break
        _impersonate_stack.pop()
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
    try:
        for ut in parked:
            try:
                regs = _saved_regs(ut)
            except gdb.error as e:
                print("\nUser thread {:#x}: <cannot read ctx: {}>".format(int(ut), e))
                continue
            parked_tagged = {
                "thread_num": saved["thread_num"],
                "rip": regs["rip"],
                "rsp": regs["rsp"],
                "rbp": regs["rbp"],
            }
            # Push a synthetic impersonation entry so the inferior_call
            # event handler will swap to safe_host if a pretty-printer,
            # display expression, or conditional breakpoint expression
            # fires a function call while we're walking the parked stack.
            _impersonate_stack.append({
                "host_regs": safe_host,
                "parked_regs": parked_tagged,
                "user_thread_ptr": int(ut),
            })
            print("\nUser thread {:#x}:".format(int(ut)))
            try:
                _set_regs(parked_tagged)
                gdb.execute("bt", to_string=False)
            except gdb.error as e:
                print("  <bt failed: {}>".format(e))
            finally:
                _impersonate_stack.pop()
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
        parked_with_thread = {
            "thread_num": host["thread_num"],
            "rip": parked["rip"],
            "rsp": parked["rsp"],
            "rbp": parked["rbp"],
        }
        entry = {
            "host_regs": host,
            "parked_regs": parked_with_thread,
            "user_thread_ptr": int(ut),
        }
        _impersonate_stack.append(entry)
        try:
            _set_regs(parked_with_thread)
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
        # Use _apply_snapshot so the restore writes back to the originating
        # thread by number, not to whatever thread is currently selected.
        if not _apply_snapshot(_impersonate_stack[-1]["host_regs"]):
            return
        _impersonate_stack.pop()
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

Linux x86-64 only.  Source: src/realm/realm-gdb.py in the Realm tree.
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
    print(_HELP_BANNER)
else:
    # Other platforms / architectures: do nothing rather than provide a
    # half-working command.
    pass
