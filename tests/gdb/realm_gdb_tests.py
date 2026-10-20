# Copyright 2025 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for src/realm/realm-gdb.py.

This module runs inside gdb, against realm_gdb_test.cc.  It is also imported
by run_realm_gdb_tests.py (outside gdb) for the list of tests, so nothing at
module level may need gdb.

Each test is a generator taking a Context.  Every `yield` hands control back
to gdb's prompt, as if the user had typed the next command, so that
prompt-driven behaviour (gdb.events.before_prompt) is exercised the way it is
interactively.  The driver feeds `rgt-step` lines to gdb's stdin, one per
step (a command of its own, since realm-gdb refuses `python` while a parked
thread is selected).

Tests describe what a user should see, not how realm-gdb.py achieves it:
frames are found by function name rather than by number, and the program
itself checks, once resumed, that nothing the debugger did corrupted it.

In a live process a parked user thread is shown for one command at a time
(`realm-thread apply PTR COMMAND`, or `rt COMMAND` after `realm-thread PTR`
selected it), so a test looks at it from inside such a command with
Context.within, which runs a Python function there.  In a core file
`realm-thread PTR` keeps it shown (Context.show and back).

Output protocol (parsed by the driver): lines starting with "RGT: ".
"""

import atexit
import json
import os
import re
import sys
import traceback

try:
    import gdb
except ImportError:  # imported by the driver, outside gdb
    gdb = None

_U64 = (1 << 64) - 1

# Must match realm_gdb_test.h.
NUM_CHAIN = 3
NUM_HOT = 5
NUM_FHOT = 4
RECURSE_DEPTH = 40


# --------------------------------------------------------------------------
# Test registry
# --------------------------------------------------------------------------
#
# A scenario is how the program is brought under the debugger:
#   live      gdb runs the program to a breakpoint in realm_gdb_test_stop
#   attach    gdb attaches to a program spinning in realm_gdb_test_wait_for_debugger
#   freeze    gdb attaches to a program frozen by REALM_FREEZE_ON_ERROR
#   core      gdb loads a core file written by gcore at the breakpoint above
#   kcore     gdb loads a core file written by the kernel when the program crashed
#   noprocess gdb loads the program but doesn't start it
#
# `ending` tells the driver how gdb is expected to finish and what to check
# about the program afterwards:
#   exits     (live)   the test ran the program to completion itself
#   release   (attach) the test released the program; it must exit cleanly
#   alive     (freeze) the process must still be alive (and frozen) afterwards
#   killed    (attach) realm-gdb must have killed the program as gdb left
#   none      nothing to check about the program
#
# `interaction` is "interactive" (commands on stdin, so gdb shows a prompt
# between steps), "batch" (all steps run from one -ex command under -batch;
# gdb never shows a prompt and exits without running `quit`), "mi" (gdb
# runs the MI interpreter, as under an IDE: commands typed by a test are MI
# commands, and gdb.events.before_prompt never fires), or "mi_new_ui" (MI
# on a second UI of a console gdb, made with `new-ui`, while the console
# sits idle at its prompt).

TESTS = {}

ENDINGS = {
    "live": "exits",
    "attach": "release",
    "freeze": "alive",
    "core": "none",
    "kcore": "none",
    "noprocess": "exits",
}


class TestInfo(object):
    def __init__(self, name, func, scenario, interaction, ending, variant, doc,
                 init, stop_at, cpus):
        self.name = name
        self.func = func
        self.scenario = scenario
        self.interaction = interaction
        self.ending = ending
        self.variant = variant
        self.doc = doc
        self.init = init
        self.stop_at = stop_at
        self.cpus = cpus


def test(scenarios=("live",), interaction="interactive", ending=None,
         variants=None, init=None, stop_at="realm_gdb_test_stop", cpus=1):
    """Register a test for each scenario (and each variant, if any).  `init`
    is gdb commands to run before the program (and so realm-gdb.py) is
    loaded, like a user's ~/.gdbinit.  `stop_at` is where a live program
    stops for the test.  With cpus=2 the probe task (which stops) runs on a
    second CPU processor, so that the parked tasks can run meanwhile."""
    def register(func):
        base = func.__name__
        if base.startswith("test_"):
            base = base[len("test_"):]
        for scenario in scenarios:
            for variant in (variants or [None]):
                # A variant is a string, or (label, string) where the
                # string alone wouldn't make a readable unique test name.
                label = variant
                if isinstance(variant, tuple):
                    label, variant = variant
                name = "{}.{}".format(scenario, base)
                if variant is not None:
                    name += "." + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
                assert name not in TESTS, name
                TESTS[name] = TestInfo(name, func, scenario, interaction,
                                       ending or ENDINGS[scenario], variant,
                                       (func.__doc__ or "").strip(), init,
                                       stop_at, cpus)
        return func
    return register


# --------------------------------------------------------------------------
# Harness (runs inside gdb)
# --------------------------------------------------------------------------

class TestFailure(Exception):
    pass


class TestSkip(Exception):
    pass


_protocol = None  # file the driver reads the harness's report from


def _out(msg):
    # gdb's prompt has no newline after it, so start a fresh line.
    sys.stdout.write("\nRGT: {}\n".format(msg))
    sys.stdout.flush()
    if _protocol:
        with open(_protocol, "a") as f:
            f.write("RGT: {}\n".format(msg))


class UserThreadInfo(object):
    """A Realm::UserThread from the registry, read without realm-gdb.py."""

    def __init__(self, value):
        self.ptr = int(value)
        self.running = bool(value["running"])
        self.stack_base = int(value["stack_base"])
        self.stack_size = int(value["stack_size"])

    def contains(self, addr):
        return self.stack_base <= addr < self.stack_base + self.stack_size

    def __repr__(self):
        return "<UserThread {:#x} {}>".format(
            self.ptr, "running" if self.running else "parked")


class _Patched(object):
    def __init__(self, owner, name, fake):
        self.owner, self.name, self.fake = owner, name, fake

    def __enter__(self):
        self.real = getattr(self.owner, self.name)
        setattr(self.owner, self.name, self.fake)
        return self.real

    def __exit__(self, *exc):
        setattr(self.owner, self.name, self.real)
        return False


class Context(object):
    """Everything a test needs: gdb helpers plus facts about the program."""

    def __init__(self, info, config):
        self.info = info
        self.config = config
        self.scenario = info.scenario
        self.variant = info.variant
        self.exit_code = None
        self.exit_codes = []
        self.exited = False
        self.last_stop = None
        self.stop_thread = None
        self.roles = {}
        gdb.events.exited.connect(self._on_exited)
        gdb.events.stop.connect(self._on_stop)

    # ---- events ----

    def _on_exited(self, event):
        self.exited = True
        self.exit_code = getattr(event, "exit_code", None)
        self.exit_codes.append(self.exit_code)

    def _on_stop(self, event):
        desc = type(event).__name__
        if isinstance(event, gdb.SignalEvent):
            desc = "signal " + event.stop_signal
        elif isinstance(event, gdb.BreakpointEvent):
            desc = "breakpoint {}".format(
                ", ".join(str(b.location) for b in event.breakpoints))
        self.last_stop = desc

    # ---- assertions ----

    def fail(self, msg):
        raise TestFailure(msg)

    def check(self, cond, msg):
        if not cond:
            raise TestFailure(msg)

    def check_in(self, needle, haystack, msg=None):
        if needle not in haystack:
            raise TestFailure("{}: {!r} not found in:\n{}".format(
                msg or "missing output", needle, haystack))

    def check_not_in(self, needle, haystack, msg=None):
        if needle in haystack:
            raise TestFailure("{}: {!r} unexpectedly found in:\n{}".format(
                msg or "unexpected output", needle, haystack))

    def check_eq(self, actual, expected, msg):
        if actual != expected:
            raise TestFailure("{}: expected {!r}, got {!r}".format(
                msg, expected, actual))

    # ---- simulating ----

    @staticmethod
    def patched(name, fake):
        """realm-gdb's function `name` (in gdb's __main__) replaced by `fake`
        for a `with` block."""
        return _Patched(sys.modules["__main__"], name, fake)

    # ---- running commands ----

    def run(self, cmd):
        """Run a gdb command, returning its output.  Any error fails the
        test."""
        try:
            return _cli_text(gdb.execute(cmd, to_string=True))
        except gdb.error as e:
            raise TestFailure("command {!r} failed: {}".format(cmd, e))

    def try_run(self, cmd):
        try:
            return True, _cli_text(gdb.execute(cmd, to_string=True))
        except gdb.error as e:
            return False, str(e)

    def expect_error(self, cmd, pattern=None):
        """Run a command that must fail (cleanly) and return its message."""
        ok, msg = self.try_run(cmd)
        if ok:
            raise TestFailure("command {!r} should have failed, but printed:\n{}"
                              .format(cmd, msg))
        self.check_clean_error(cmd, msg)
        if pattern is not None and not re.search(pattern, msg):
            raise TestFailure("command {!r} failed with {!r}, expected /{}/"
                              .format(cmd, msg, pattern))
        return msg

    def check_clean_error(self, cmd, msg):
        if "Error occurred in Python" in msg:
            raise TestFailure("command {!r} raised a Python exception: {}"
                              .format(cmd, msg))

    def check_no_python_error(self, cmd, output):
        for bad in ("Traceback (most recent call last)", "Python Exception",
                    "Error occurred in Python"):
            if bad in output:
                raise TestFailure("command {!r} hit a Python error:\n{}"
                                  .format(cmd, output))

    def run_capturing(self, cmd, quits=False):
        """Run a command, returning (its error message, or None if it
        succeeded, and what it printed), which gdb.execute(to_string=True)
        doesn't return for a command that fails.  With `quits`, one that
        quits (Ctrl-C) too, with the error "Quit"."""
        path = os.path.join(os.path.dirname(_protocol), "capture.log")
        # (Not to_string: gdb would log to that string.)
        for c in ("set logging file " + path, "set logging overwrite on",
                  "set logging redirect on"):
            gdb.execute(c)
        self._set_logging("on")
        error = None
        try:
            gdb.execute(cmd)
        except gdb.error as e:
            error = str(e)
        except KeyboardInterrupt:
            if not quits:
                raise
            error = "Quit"
        finally:
            self._set_logging("off")
        with open(path) as f:
            return error, f.read()

    @staticmethod
    def _set_logging(value):
        try:
            gdb.execute("set logging enabled " + value)
        except gdb.error:  # gdb < 12
            gdb.execute("set logging " + value)

    def allow_output(self, text):
        """Let gdb print `text`, which the driver fails a test for otherwise
        (its BAD_OUTPUT)."""
        _out("ALLOW-OUTPUT " + text)

    def require_output(self, text):
        """Have the driver fail the test if gdb doesn't print `text` (once
        the test has typed its final command, say)."""
        _out("REQUIRE-OUTPUT " + text)

    def await_exit(self):
        """Have the driver, once the test is over, wait for gdb to end by
        itself (say, on a signal the test sends it at its prompt) rather than
        close gdb's stdin, which gdb treats as `quit`."""
        _out("AWAIT-EXIT")

    def supports(self, cmd):
        """Whether this gdb accepts `cmd` (run it on a plain thread, before
        realm-gdb.py is involved), for syntax that older gdbs lack."""
        return self.try_run(cmd)[0]

    def run_clean(self, cmd):
        """Run a command that may legitimately succeed or fail, as long as
        it doesn't trip over a Python error.  Returns (ok, text)."""
        ok, text = self.try_run(cmd)
        if ok:
            self.check_no_python_error(cmd, text)
        else:
            self.check_clean_error(cmd, text)
        return ok, text

    def eval_int(self, expr):
        try:
            return int(gdb.parse_and_eval(expr))
        except gdb.error as e:
            raise TestFailure("cannot evaluate {!r}: {}".format(expr, e))

    # ---- program state ----

    @property
    def is_core(self):
        return self.scenario in ("core", "kcore")

    def user_threads(self):
        out = []
        h = gdb.parse_and_eval("Realm::user_thread_registry_head")
        while int(h) != 0:
            out.append(UserThreadInfo(h))
            if len(out) > 10000 or any(u.ptr == int(h["registry_next"]) for u in out):
                break  # (a test made the registry loop)
            h = h["registry_next"]
        return out

    def user_thread_for(self, addr_expr):
        addr = self.eval_int("(unsigned long)({})".format(addr_expr))
        for ut in self.user_threads():
            if ut.contains(addr):
                return ut
        raise TestFailure("no user thread's stack contains {} ({:#x})".format(
            addr_expr, addr))

    def find_roles(self):
        """Work out which user thread is which, from the addresses the
        program published."""
        r = {}
        for i in range(NUM_CHAIN):
            r["chain{}".format(i)] = self.user_thread_for(
                "realm_gdb_test_state.chain_locals[{}]".format(i))
        r["recurse"] = self.user_thread_for("realm_gdb_test_state.recurse_local")
        r["hot"] = self.user_thread_for("realm_gdb_test_state.hot_local")
        r["fhot"] = self.user_thread_for("realm_gdb_test_state.fhot_local")
        r["opt"] = self.user_thread_for("realm_gdb_test_state.opt_local")
        r["probe"] = self.user_thread_for("realm_gdb_test_state.probe_local")
        for name, ut in r.items():
            want_running = (name == "probe") and self.scenario != "noprocess"
            if ut.running != want_running:
                self.fail("{} user thread {:#x} has running={}".format(
                    name, ut.ptr, ut.running))
        self.roles = r
        return r

    def ut(self, role):
        return self.roles[role].ptr

    def utw(self, role_or_ptr):
        """"user thread N (0x...)", as realm-gdb names one in its messages
        (by the number `realm-thread` lists it by: see _ut_words)."""
        return sys.modules["__main__"]._ut_words(self.ptr(role_or_ptr))

    def remind(self, role_or_ptr):
        """The prompt's reminder of a selection: "[realm-thread N] "."""
        ptr = self.ptr(role_or_ptr)
        n = sys.modules["__main__"]._ut_number(ptr)
        return "[realm-thread {}] ".format(n if n is not None else "{:#x}".format(ptr))

    def chain_marker(self, i):
        return self.eval_int("realm_gdb_test_state.chain_markers[{}]".format(i))

    # ---- frames, threads and registers ----

    def frames(self, limit=200):
        out = []
        f = gdb.newest_frame()
        while f is not None and len(out) < limit:
            out.append(f)
            try:
                f = f.older()
            except gdb.error:
                break
        return out

    @staticmethod
    def frame_name(f):
        try:
            n = f.name()
        except gdb.error:
            n = None
        return n or ""

    def frame_names(self, limit=200):
        return [self.frame_name(f) for f in self.frames(limit)]

    def find_frame(self, name):
        """(level, frame) of the innermost frame whose function is `name`,
        or (None, None)."""
        for level, f in enumerate(self.frames()):
            if self.frame_name(f) == name:
                return level, f
        return None, None

    def select_frame(self, name):
        level, f = self.find_frame(name)
        if f is None:
            self.fail("no frame for {} in:\n{}".format(
                name, "\n".join(self.frame_names())))
        self.run("frame {}".format(level))
        self.check_eq(self.frame_name(gdb.selected_frame()), name,
                      "selected frame after 'frame {}'".format(level))
        return level

    def selected_level(self):
        sel = gdb.selected_frame()
        for level, f in enumerate(self.frames()):
            if f == sel:
                return level
        return None

    def check_names_in_order(self, names, frame_names, what, consecutive=True):
        """Check that `names` appear in `frame_names`, innermost first and
        (by default) consecutively, with no other frames in between."""
        try:
            start = frame_names.index(names[0])
        except ValueError:
            self.fail("{}: {} not in backtrace:\n{}".format(
                what, names[0], "\n".join(frame_names)))
        if consecutive:
            got = frame_names[start:start + len(names)]
        else:
            rest = iter(frame_names[start:])
            got = [n for n in names if any(f == n for f in rest)]
        if got != list(names):
            self.fail("{}: expected frames {} but found {}".format(
                what, names, frame_names[start:start + len(names) + 2]))

    def check_chain_stack(self, i, what="chain stack"):
        """The stack of chain task `i` (its rgt_chain_level3 frame has its
        chain_marker)."""
        names = self.frame_names()
        self.check_names_in_order(["rgt_chain_level3", "rgt_chain_level2",
                                   "rgt_chain_level1", "chain_task"], names, what)
        frame = self.find_frame("rgt_chain_level3")[1]
        try:
            marker = int(frame.read_var("chain_marker"))
        except (gdb.error, ValueError) as e:
            self.fail("{}: cannot read chain_marker: {}".format(what, e))
        self.check_eq(marker, self.chain_marker(i), what + ": chain_marker")
        return names

    def check_host_stack(self, what="host stack"):
        """The stopped thread's own stack, with no parked thread shown."""
        names = self.frame_names()
        if self.scenario in ("live", "core"):
            self.check_eq(names[0], "realm_gdb_test_stop", what + ": frame #0")
            self.check_names_in_order(["realm_gdb_test_stop",
                                       "realm_gdb_test_wait_for_debugger",
                                       "rgt_hot_probe", "probe_task"],
                                      names, what)
        for n in ("rgt_chain_level3", "rgt_recurse", "rgt_hot_wait"):
            self.check(n not in names,
                       "{}: unexpected parked frame {} in {}".format(what, n, names))

    def thread_by_num(self, num):
        for t in gdb.selected_inferior().threads():
            if t.num == num:
                return t
        return None

    def switch_thread(self, num):
        """Switch threads without going through the `thread` command (and
        so without any hook on it)."""
        t = self.thread_by_num(num)
        self.check(t is not None, "no thread {}".format(num))
        t.switch()

    def find_thread_with_frame(self, name):
        cur = gdb.selected_thread()
        found = None
        try:
            for t in sorted(gdb.selected_inferior().threads(), key=lambda t: t.num):
                t.switch()
                if self.find_frame(name)[1] is not None:
                    found = t.num
                    break
        finally:
            cur.switch()
        return found

    def reg_names(self):
        """Every register gdb saves around an inferior call (floating-point
        and vector ones included).  gdb < 10 can't list them: there, those
        of gdb 10's list that this gdb has."""
        arch = gdb.selected_frame().architecture()
        try:
            names = [r.name for r in arch.registers("save")]
            if names:
                return names
        except AttributeError:
            pass
        if "x86-64" in arch.name() or "x86_64" in arch.name():
            names = (["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"]
                     + ["r{}".format(i) for i in range(8, 16)]
                     + ["rip", "eflags", "cs", "ss", "ds", "es", "fs", "gs"]
                     + ["st{}".format(i) for i in range(8)]
                     + ["fctrl", "fstat", "ftag", "fiseg", "fioff", "foseg", "fooff", "fop"]
                     + ["xmm{}".format(i) for i in range(32)] + ["mxcsr"]
                     + ["ymm{}h".format(i) for i in range(32)]
                     + ["k{}".format(i) for i in range(8)]
                     + ["zmm{}h".format(i) for i in range(32)]
                     + ["pkru", "fs_base", "gs_base", "orig_rax"])
        elif "aarch64" in arch.name():
            names = (["x{}".format(i) for i in range(31)] + ["sp", "pc", "cpsr"]
                     + ["v{}".format(i) for i in range(32)]
                     + ["z{}".format(i) for i in range(32)]
                     + ["p{}".format(i) for i in range(16)] + ["ffr", "fpsr", "fpcr"])
        else:
            raise TestSkip("unsupported architecture " + arch.name())
        frame = gdb.newest_frame()
        have = []
        for name in names:
            try:
                frame.read_register(name)
                have.append(name)
            except (gdb.error, ValueError):
                pass
        return have

    @staticmethod
    def reg_value(v):
        """A register's value: an int, or its bytes in hex if it isn't an
        integer (vector, floating-point) or can't be read."""
        try:
            if v.type.code in (gdb.TYPE_CODE_INT, gdb.TYPE_CODE_PTR,
                               gdb.TYPE_CODE_FLAGS) and v.type.sizeof <= 8:
                return int(v) & _U64
            n = v.type.sizeof
            vec = v.cast(gdb.lookup_type("unsigned char").vector(n - 1))
            return "bytes " + "".join("{:02x}".format(int(vec[i]) & 0xff)
                                      for i in range(n))
        except (gdb.error, TypeError, ValueError):
            return str(v)

    def regs(self, thread_num=None):
        """The thread's actual registers (its innermost frame's)."""
        cur = gdb.selected_thread()
        if thread_num is not None and thread_num != cur.num:
            self.switch_thread(thread_num)
        try:
            f = gdb.newest_frame()
            return {r: self.reg_value(f.read_register(r)) for r in self.reg_names()}
        finally:
            if gdb.selected_thread().num != cur.num:
                cur.switch()

    def check_regs(self, before, thread_num=None, what="registers"):
        after = self.regs(thread_num)
        # (But for the bits the kernel keeps itself: aarch64's PSTATE.SS, set
        # by a single step that a signal cut short.)
        bits = getattr(getattr(sys.modules["__main__"], "_arch", None), "kernel_bits", {})

        def same(r, a, b):
            mask = bits.get(r)
            if mask and isinstance(a, int) and isinstance(b, int):
                return a & ~mask == b & ~mask
            return a == b
        diff = ["{}: {} -> {}".format(r, _hex(before[r]), _hex(after[r]))
                for r in before if not same(r, before[r], after.get(r))]
        if diff:
            self.fail("{} changed:\n  {}".format(what, "\n  ".join(diff)))

    # ---- showing parked threads ----

    def ptr(self, role_or_ptr):
        return self.ut(role_or_ptr) if isinstance(role_or_ptr, str) else role_or_ptr

    def apply(self, role_or_ptr, cmd):
        """`realm-thread apply PTR CMD`'s output; any error fails the test."""
        out = self.run("realm-thread apply {:#x} {}".format(self.ptr(role_or_ptr), cmd))
        self.check_no_python_error("realm-thread apply", out)
        return out

    def within(self, role_or_ptr, fn):
        """Run fn() while the parked user thread is shown (from inside
        `realm-thread apply PTR python ...`), returning fn's result.  A
        failure inside fails the test."""
        box = {}
        _within_fns.append((fn, box))
        try:
            out = self.apply(role_or_ptr, "python rgt._within()")
        finally:
            if _within_fns and _within_fns[-1][1] is box:
                _within_fns.pop()
        if "error" in box:
            raise box["error"]
        self.check("result" in box, "the function given to within() didn't run:\n" + out)
        self.last_within_output = out
        return box["result"]

    def select(self, role_or_ptr):
        """`realm-thread PTR` in a live process (a selection for `rt`, and for
        plain commands where they show it: self.plain, as it says)."""
        out = self.run("realm-thread {:#x}".format(self.ptr(role_or_ptr)))
        self.check_no_python_error("realm-thread", out)
        if re.search(r"^\[Selecting ", out, re.M):  # (Said in full once.)
            self.plain = "'rt COMMAND' runs commands on it" not in out
        else:
            self.plain = "show it, and 'step'" in out
            self.check(self.plain or "Plain commands show thread" in out,
                       "what plain commands show, on selecting:\n" + out)
        return out

    def rt(self, cmd):
        """`rt CMD`'s output; any error fails the test."""
        out = self.run("rt " + cmd)
        self.check_no_python_error("rt", out)
        return out

    def show(self, role_or_ptr):
        """Core files: `realm-thread PTR`."""
        out = self.run("realm-thread {:#x}".format(self.ptr(role_or_ptr)))
        self.check_no_python_error("realm-thread", out)
        return out

    def back(self):
        out = self.run("realm-thread back")
        self.check_no_python_error("realm-thread back", out)
        return out

    # ---- endings ----

    def continue_to_exit(self, cmd="continue"):
        """Resume with `cmd` and require the program to run to a clean
        exit (it checks its own state on the way)."""
        self.check(self.scenario in ("live", "noprocess"),
                   "continue_to_exit only makes sense for a live process")
        ok, out = self.try_run(cmd)
        sys.stdout.write(out if ok else ("error: " + out + "\n"))
        if not self.exited:
            where = ""
            if gdb.selected_inferior().pid != 0:
                where = self.try_run("bt 15")[1]
            self.fail("program did not exit after {!r} (last stop: {})\n{}"
                      .format(cmd, self.last_stop, where))
        self.check_eq(self.exit_code, 0, "program exit code")
        return out

    def wait_for_stop(self, seconds=60):
        """Generator: wait for the selected thread to stop again (yield after
        each item)."""
        import time
        deadline = time.time() + seconds
        while time.time() < deadline:
            thread = gdb.selected_thread()
            if thread is None or not thread.is_running():
                return
            time.sleep(0.1)
            yield

    def wait_for_exit(self, seconds=120):
        """Generator: give gdb's event loop a chance to notice the program
        exiting (yield after each item)."""
        import time
        deadline = time.time() + seconds
        while not self.exited and time.time() < deadline:
            time.sleep(0.1)
            yield

    def mi_result(self):
        """(result class, rest) of the latest MI command a test typed, e.g.
        ("done", ',value="5"').  The driver records it (see drive())."""
        with open(_protocol) as f:
            lines = [l for l in f if l.startswith("RGT-MI-RESULT ")]
        self.check(lines, "no MI result recorded")
        cls, _, rest = lines[-1][len("RGT-MI-RESULT "):].rstrip("\n").partition(" ")
        return cls, rest

    def check_mi_result(self, cls, pattern=None):
        got, rest = self.mi_result()
        self.check_eq(got, cls, "MI result class (rest: {})".format(rest))
        if pattern is not None and not re.search(pattern, rest):
            self.fail("MI result {}{} doesn't match /{}/".format(got, rest, pattern))

    def shown_on(self, role_or_ptr):
        """The thread (number) `realm-thread` says the user thread is shown
        on, or None."""
        ptr = self.ut(role_or_ptr) if isinstance(role_or_ptr, str) else role_or_ptr
        m = re.search(r"{:#x}\s+\S+.*shown on thread (\d+)".format(ptr),
                      self.run("realm-thread"))
        return int(m.group(1)) if m else None

    def check_exited(self, what):
        """After resuming with a typed command: the program must have run
        to a clean exit."""
        if not self.exited:
            where = ""
            if gdb.selected_inferior().pid != 0:
                where = self.try_run("bt 15")[1]
            self.fail("program did not exit after {} (last stop: {})\n{}"
                      .format(what, self.last_stop, where))
        self.check_eq(self.exit_code, 0, "program exit code")

    def program_args(self):
        """The program's arguments, as the driver gave them."""
        return " ".join(self.config["program_args"])

    def run_program(self):
        """`run` the program, with its arguments and no stdin (gdb's stdin
        carries the test's commands).  `run ARGS` replaces the arguments, so
        they are given again."""
        return self.run("run {} < /dev/null".format(self.program_args()))

    def release(self):
        """Let an attached program carry on once gdb lets go of it.  (Written
        from Python: `set var` is refused while a parked thread is
        selected.)"""
        v = gdb.parse_and_eval("realm_gdb_test_release")
        gdb.selected_inferior().write_memory(
            int(v.address), (1).to_bytes(v.type.sizeof, sys.byteorder))
        self.check_eq(self.eval_int("realm_gdb_test_release"), 1, "released")


def _have_pending_level():
    return hasattr(getattr(gdb, "PendingFrame", None), "level")


# (fn, result box) for Context.within, run by _within.
_within_fns = []


def _within():
    """Run inside `realm-thread apply PTR python rgt._within()`."""
    fn, box = _within_fns[-1]
    try:
        box["result"] = fn()
    except (TestFailure, TestSkip) as e:
        box["error"] = e
    except Exception:
        box["error"] = TestFailure("exception while the parked thread was shown:\n"
                                   + traceback.format_exc())


def _hex(v):
    return "{:#x}".format(v) if isinstance(v, int) else str(v)


# --------------------------------------------------------------------------
# Running tests (inside gdb)
# --------------------------------------------------------------------------

_current = None


class Type(object):
    """Yielded by a test to have the driver type `cmd` at gdb's prompt, as
    a user would, instead of the test running it from Python.  (Some
    commands, e.g. `interpreter-exec mi`, abort older gdbs when run from
    Python.)  With final=True the test is over once the command has been
    typed; use it for commands that may end the gdb session."""

    def __init__(self, cmd, final=False):
        self.cmd = cmd
        self.final = final


class _Runner(object):
    def __init__(self, info, config):
        self.info = info
        self.ctx = Context(info, config)
        self.gen = None
        self.done = False
        self.nsteps = 0
        self.batch = info.interaction == "batch"

    def step(self):
        """Run the test up to its next yield.  Ends by telling the driver
        what to do next:
          STEP-END n           send the next step
          STEP-END n TYPE cmd  type cmd at the prompt, then send the next step
          DONE n               the test is over; close gdb's stdin"""
        self.nsteps += 1
        n = self.nsteps
        if self.done:
            _out("DONE {}".format(n))
            return
        typed = None
        try:
            if self.gen is None:
                _setup(self.ctx)
                self.gen = self.info.func(self.ctx)
                if self.gen is None or not hasattr(self.gen, "send"):
                    raise StopIteration
            req = next(self.gen)
            if isinstance(req, Type):
                typed = req
                if req.final:
                    self._finish()
        except StopIteration:
            self._finish()
        except TestSkip as e:
            self.done = True
            _out("SKIP {}".format(e))
        except TestFailure as e:
            self.done = True
            _out("FAIL {}".format(e))
            _out("FAIL at step {}".format(n))
        except Exception:
            self.done = True
            _out("FAIL harness exception:\n" + traceback.format_exc())
        if typed is not None and (not self.done or typed.final):
            if self.batch:
                gdb.execute(typed.cmd)
            else:
                _out("STEP-END {} TYPE {}".format(n, typed.cmd))
        elif self.done:
            _out("DONE {}".format(n))
        else:
            _out("STEP-END {}".format(n))

    def _finish(self):
        self.done = True
        ctx = self.ctx
        if ctx.info.ending == "exits" and not ctx.exited:
            _out("FAIL test finished without running the program to exit")
            return
        _out("PASS {}".format(self.info.name))


def _setup(ctx):
    """Bring the program to the state every test starts from."""
    s = ctx.scenario
    if s == "live":
        where = ctx.info.stop_at
        ctx.run("break " + where)
        ctx.run_program()
        ctx.check(ctx.frame_name(gdb.selected_frame()) == where,
                  "did not stop in {} (last stop: {})".format(where, ctx.last_stop))
        ctx.stop_thread = gdb.selected_thread().num
        ctx.find_roles()
    elif s in ("attach", "freeze"):
        frame = "rgt_hot_probe"
        num = ctx.find_thread_with_frame(frame)
        ctx.check(num is not None, "no thread is in " + frame)
        ctx.stop_thread = num
        ctx.find_roles()
    elif s in ("core", "kcore"):
        num = ctx.find_thread_with_frame("rgt_hot_probe")
        ctx.check(num is not None, "no thread in the core is in rgt_hot_probe")
        ctx.run("thread {}".format(num))
        ctx.stop_thread = num
        ctx.find_roles()
    elif s == "noprocess":
        pass
    else:
        raise TestFailure("unknown scenario " + s)


def _exit_handler(path):
    """A Python exit handler registered after realm-gdb.py was loaded, as
    by another extension, which must run as gdb exits (the driver also
    registers one before; see its check_gdb_output)."""
    with open(path, "a") as f:
        f.write("RGT-EXIT-HANDLER after\n")


def start(name, config_json):
    """Called by the driver (via -ex) to pick the test to run."""
    global _current, _protocol
    config = json.loads(config_json)
    _protocol = config.get("protocol")
    if _protocol:
        atexit.register(_exit_handler, _protocol)
    info = TESTS[name]
    _out("BEGIN {} ({})".format(name, info.doc.splitlines()[0] if info.doc else ""))
    _current = _Runner(info, config)
    _StepCommand()


if gdb is not None:
    class _StepCommand(gdb.Command):
        """rgt-step: run the test up to its next yield (see step)."""

        def __init__(self):
            super(_StepCommand, self).__init__("rgt-step", gdb.COMMAND_OBSCURE)

        def invoke(self, arg, from_tty):
            step()


def step():
    if _current is not None:
        _current.step()


def run_all():
    """Batch mode: run every step back to back, without a prompt."""
    while _current is not None and not _current.done:
        _current.step()


def _cli_text(out):
    """Output gdb.execute(..., to_string=True) collected, as the CLI shows
    it: under MI, gdb 18 returns some of it, line by line, as MI stream
    records (~"text\\n")."""
    def unquote(m):
        text = bytearray()
        for piece in re.findall(r"\\[0-7]{1,3}|\\.|[^\\]+", m.group(1), re.S):
            if piece[0] != "\\":
                text += piece.encode("utf-8", "surrogateescape")
            elif piece[1] in "01234567":
                text.append(int(piece[1:], 8) & 0xff)
            else:
                text += {"n": "\n", "t": "\t", "r": "\r", "e": "\x1b"}.get(
                    piece[1], piece[1]).encode("utf-8")
        return text.decode("utf-8", "replace")
    if not re.search(r'(?:^|\n)[~@&]"', out):
        return out
    return "".join(re.sub(r'^[~@&]"((?:[^"\\]|\\.)*)"\n?$', unquote, line, flags=re.S)
                   for line in out.splitlines(True))


def typed(cmd):
    return Type(cmd)


def typed_final(cmd):
    return Type(cmd, final=True)




# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------

def _finish(t):
    """End a test as its scenario expects."""
    if t.scenario in ("live", "noprocess"):
        t.continue_to_exit()
    elif t.scenario == "attach":
        t.release()


def _arch_name():
    return gdb.selected_frame().architecture().name()


def _is_aarch64():
    return "aarch64" in _arch_name()


def _is_x86_64():
    name = _arch_name()
    return "x86-64" in name or "x86_64" in name


def _is_freebsd():
    return sys.platform.startswith("freebsd")


def _ctx_field(ut, reg):
    """The expression for the saved "pc" or "sp" in user thread `ut`'s
    context (a ucontext_t, laid out per CPU and OS)."""
    if _is_freebsd():
        field = {"pc": "mc_rip", "sp": "mc_rsp"} if _is_x86_64() else {
            "pc": "mc_gpregs.gp_elr", "sp": "mc_gpregs.gp_sp"}
    else:
        field = {"pc": "gregs[16]", "sp": "gregs[15]"} if _is_x86_64() else {
            "pc": "pc", "sp": "sp"}
    return "((Realm::UserThread *){:#x})->ctx.uc_mcontext.{}".format(int(ut), field[reg])


def _starts(name, prefix):
    return name.startswith(prefix)


USER_SWITCH = "Realm::UserThread::user_switch"
# The parked thread's innermost frame: user_switch, where it called
# swapcontext; on FreeBSD, whose swapcontext is a system call, libc's stub
# making it (with libthr's wrapper's frame between them), as for any thread
# there blocked in a system call.
PARKED_FRAME0 = "_swapcontext" if _is_freebsd() else USER_SWITCH


def _check_parked_frame0(t, what):
    """Inside a register window, frame #0 is the parked thread's own
    (see PARKED_FRAME0); in a core-file view it is the host's."""
    if not t.is_core:
        t.check(_starts(t.frame_name(gdb.newest_frame()), PARKED_FRAME0),
                "{}: frame #0 is {}, not {}".format(
                    what, t.frame_name(gdb.newest_frame()), PARKED_FRAME0))


# ---- loading ------------------------------------------------------------

@test(scenarios=("live",))
def test_loaded(t):
    """The script is auto-loaded from librealm's .debug_gdb_scripts and
    registers its commands."""
    out = t.run("info auto-load python-scripts")
    if t.config.get("embedded", True):
        t.check(re.search(r"Yes\s+realm-gdb\.py", out) is not None,
                "realm-gdb.py not auto-loaded:\n" + out)
    for cmd in ("realm-help", "help realm-thread", "help thread apply",
                "help thread apply all", "help realm-gdb-guard"):
        t.check_no_python_error(cmd, t.run(cmd))
    t.check_in("realm-thread apply", t.run("realm-help"), "realm-help")
    # (realm-gdb's hooks are deprecated, which keeps them out of help and
    # completion: see _define_hook.)
    t.check_not_in("hook-", t.run("help user-defined"), "help user-defined")
    t.check_not_in("hook-", t.run("complete info h"), "complete info h")
    t.check_in("realm-gdb-guard continue", t.run("show user hook-continue"),
               "realm-gdb's hook-continue, all the same")
    # (Nor do they make abbreviations ambiguous: see _keep_abbreviations.)
    t.check_in("SIGUSR1", t.run("info h SIGUSR1"), "'info h', for 'info handle'")
    t.run("set ho ASCII")
    t.check_in("ASCII", t.run("show host-charset"), "'set ho', for 'set host-charset'")
    t.run("set host-charset auto")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_old_guard_dropped(t):
    """A hook of an older realm-gdb.py's, whose check was named
    realm-thread-guard (as a gdb that loaded one has it), gets realm-gdb's
    check alone, rather than keeping that line as one of the user's, which
    would fail; and `realm-th` abbreviates `realm-thread`."""
    # (Defined from Python, which runs no hooks: the next window puts the
    # check back, see _check_guard_hooks.)
    t.run("define hook-continue\nrealm-thread-guard continue\nend")
    t.apply("chain0", "bt 1")
    body = t.run("show user hook-continue")
    t.check_not_in("realm-thread-guard", body, "hook-continue")
    t.check_eq(body.count("realm-gdb-guard continue"), 1, "realm-gdb's check in hook-continue")
    t.check_in("List parked Realm user threads", t.run("help realm-th"), "help realm-th")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_loaded_twice(t):
    """Sourcing the script a second time (e.g. an installed copy on top of
    the embedded one) leaves everything working, with no duplicated output
    or notices about its own hooks."""
    t.select("chain1")
    out = t.run("source {}".format(t.config["script"]))
    t.check_not_in("you defined", out, "notices on reloading")
    t.check_not_in("could not define", out, "notices on reloading")
    t.check_in("realm-gdb.py loaded again.", out, "what reloading says")
    t.check_not_in("debugging commands", out, "the banner again")
    yield
    t.check_in("rgt_chain_level3", t.rt("bt"), "rt bt after reloading")
    t.check_eq(gdb.prompt_hook("(gdb) "), t.remind("chain1") + "(gdb) ",
               "prompt after reloading")
    t.run("realm-thread back")
    t.check(gdb.prompt_hook is None, "a prompt hook left in place")
    before = t.regs()
    out = t.run("thread apply all bt")
    t.check_eq(out.count("Parked Realm user threads"), 1,
               "parked-thread sections in 'thread apply all bt'")
    yield

    def inside():
        t.check_chain_stack(0)
        t.select_frame("rgt_chain_level3")
        t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
                   t.chain_marker(0) + 1, "inferior call")
    t.within("chain0", inside)
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), init="define hook-step\necho user-hook-step\\n\nend\n"
                                "define hook-continue\necho user-hook-continue\\n\nend\n"
                                "define hookpost-print\necho user-hookpost-print\\n\nend\n"
                                "define hook-print\noutput 7\necho user-hook-print\\n\nend\n")
def test_user_hook_kept(t):
    """Hooks the user defined first are kept, with realm-gdb's check run
    before their own commands (after them in hook-CMD of a command that
    shows a selection, whose window a hooked command among them, `output`,
    would close), and realm-gdb says so; loading it again doesn't add the
    check twice.  So the commands are refused as usual, and plain commands
    show a selection, the user's commands running too."""
    hooks = (("hook-step", "step"), ("hook-continue", "continue"),
             ("hookpost-print", "print"), ("hook-print", "print"))

    def check_hooks(when):
        for hook, cmd in hooks:
            body = t.run("show user " + hook)
            t.check_in("user-" + hook, body, "the user's {}, {}".format(hook, when))
            t.check_eq(body.count("realm-gdb-guard"), 1,
                       "realm-gdb's checks in {}, {}".format(hook, when))
    check_hooks("as loaded")
    t.check_in("realm-gdb-guard", t.run("show user hook-next"), "realm-gdb's hook-next")
    # What it said when it loaded (load it again to see, as if for the
    # first time: it doesn't say it again for the same hook).
    main = sys.modules["__main__"]
    t.check_not_in("now runs realm-gdb's check", t.run("source {}".format(
        t.config["script"])), "the notes, loaded again")
    main._user_hooks.clear()
    out = t.run("source {}".format(t.config["script"]))
    for hook, cmd in hooks:
        t.check_in("your {} now runs realm-gdb's check of '{}' {}".format(
            hook, cmd, "after its own commands" if hook == "hook-print" else "first"),
            out, "the note about " + hook)
    check_hooks("loaded again")
    body = t.run("show user hook-print")
    t.check(body.index("user-hook-print") < body.index("realm-gdb-guard"),
            "realm-gdb's check last in hook-print:\n" + body)
    yield
    for cmd in ("next", "step", "continue"):
        t.expect_error("realm-thread apply {:#x} {}".format(t.ut("chain0"), cmd), "refusing")
    t.select("chain0")
    out = t.run("print $pc")
    t.check_in(PARKED_FRAME0, out, "plain print, shown the selection")
    t.check_in("7user-hook-print", out, "the user's hook-print, before it")
    t.check_in("user-hookpost-print", out, "the user's hookpost-print, after it")
    t.check_in("rgt_chain_level3", t.run("bt"), "plain bt")
    t.run("realm-thread back")
    _check_parked_section(t, t.run("thread apply all bt"))
    yield
    t.continue_to_exit()


@test(scenarios=("live",), init="set trace-commands on\n"
                                "define hook-continue\necho user-hook-continue\\n\nend\n")
def test_user_hook_traced(t):
    """With `set trace-commands on` (which echoes each command, "+CMD",
    into its captured output too), the user's hooks are kept as they are,
    without the echo of realm-gdb's `show user` read as a line of theirs
    (which then made `continue` and `quit` fail: "Undefined command:
    User"), at load and after a later `define`; and `rt` still knows `py`
    for `python`, refusing it bare."""
    main = sys.modules["__main__"]
    path = os.path.join(t.config["tmp"], "traced-hook.gdb")
    with open(path, "w") as f:
        f.write("define hook-next\necho user-hook-next\\n\nend\n")
    t.run("source " + path)
    for hook in ("hook-continue", "hook-next"):
        body = t.run("show user " + hook)
        t.check_eq(body.count('User command "'), 1, "{}'s header, once:\n{}".format(hook, body))
        t.check_in("user-" + hook, body, "the user's " + hook)
        t.check_eq(body.count("realm-gdb-guard"), 1, "realm-gdb's check in " + hook)
    t.check_eq(main._command_named("py", ("python",)), "python", "'py' taken for 'python'")
    t.expect_error("realm-thread apply {:#x} py".format(t.ut("chain0")), "reads the lines after it")
    t.expect_error("realm-thread apply {:#x} continue".format(t.ut("chain0")), "refusing")
    yield
    t.check_in("user-hook-continue", t.continue_to_exit(), "continue, with the user's hook")


@test(scenarios=("live",))
def test_user_hook_after_load(t):
    """A hook the user defines after realm-gdb has loaded (here from a
    sourced file; typed, too), replacing realm-gdb's own, gets realm-gdb's
    check put back in front of its commands (from hookpost-define), saying
    so; so the command is still refused in a window, and the user's commands
    still run.  (gdb runs no hooks for a `define` that Python runs with its
    body.)"""
    path = os.path.join(t.config["tmp"], "late-hook.gdb")
    with open(path, "w") as f:
        f.write("define hook-continue\necho user-late-hook\\n\nend\n"
                "define hook-frame\noutput 0\necho user-late-frame\\n\nend\n"
                # (In a block, gdb runs no hooks for the define.)
                "if 1\n  define hook-step\n    echo user-if-hook\\n\n  end\nend\n")
    out = t.run("source " + path)
    t.check_in("your hook-continue now runs realm-gdb's check of 'continue' first", out,
               "the note")
    t.check_in("your hook-frame now runs realm-gdb's check of 'frame' after its own "
               "commands", out, "the note about hook-frame")
    body = t.run("show user hook-continue")
    t.check_in("user-late-hook", body, "the user's commands")
    t.check_eq(body.count("realm-gdb-guard"), 1, "realm-gdb's check")
    t.expect_error("realm-thread apply {:#x} continue".format(t.ut("chain0")), "refusing")
    t.expect_error("realm-thread apply {:#x} step".format(t.ut("chain0")), "refusing")
    t.check_in("user-if-hook", t.run("show user hook-step"), "the user's hook-step, kept")
    # (From Python, too, of a command that isn't a guard's: `shell`'s.)
    gdb.execute("define hook-shell\necho user-py-shell\\n\nend", to_string=True)
    t.check_not_in("realm-gdb-guard", t.run("show user hook-shell"), "hook-shell, defined")
    t.check_in("your hook-shell now runs realm-gdb's check", t.apply("chain0", "bt 1"),
               "the note, as a window opens")
    body = t.run("show user hook-shell")
    t.check_in("user-py-shell", body, "the user's hook-shell")
    t.check_eq(body.count("realm-gdb-guard"), 1, "realm-gdb's check in hook-shell")
    t.expect_error('realm-thread apply {:#x} eval "shell true"'.format(t.ut("chain0")),
                   "refusing 'shell'")
    t.select("chain0")
    if t.plain:
        out = t.run("frame 0")
        t.check_in("0user-late-frame", out, "the user's hook-frame")
        t.check_in(PARKED_FRAME0, out, "frame 0, shown the selection")
    t.run("realm-thread back")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_script_names(t):
    """realm-gdb.py defines each of its top-level names once (a second
    definition would replace the first everywhere)."""
    import ast
    with open(t.config["script"]) as f:
        tree = ast.parse(f.read())
    seen = {}
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        for name in names:
            t.check(name not in seen, "{} is defined at lines {} and {}".format(
                name, seen.get(name), node.lineno))
            seen[name] = node.lineno
    yield
    t.continue_to_exit()


@test(scenarios=("noprocess",))
def test_before_run(t):
    """Commands are harmless before the program runs and before Realm has
    created any threads, and work once it has."""
    # With a shared librealm the script isn't loaded until the library is.
    for cmd in ("realm-thread", "realm-thread back", "realm-help",
                "thread apply all bt", "realm-thread apply all bt"):
        t.run_clean(cmd)
    yield
    t.run("break main")
    t.run("break realm_gdb_test_stop")
    t.run_program()
    t.check_eq(t.frame_name(gdb.selected_frame()), "main", "stopped in")
    yield
    ok, out = t.run_clean("realm-thread")
    t.check(ok, "realm-thread failed before Runtime::init: " + out)
    t.check_in("no Realm user threads", out)
    out = t.run("thread apply all bt")
    t.check_not_in("Parked Realm user threads", out)
    t.expect_error("realm-thread apply 0x1234 bt")
    t.check_in("no parked", t.run("realm-thread apply all bt"))
    yield
    t.run("continue")
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop",
               "stopped in")
    t.scenario = "live"
    t.find_roles()
    t.within("chain1", lambda: t.check_chain_stack(1))
    yield
    t.continue_to_exit()


# ---- realm-thread (list) ------------------------------------------------

@test(scenarios=("live", "attach", "core", "freeze"))
def test_list(t):
    """`realm-thread` lists every registered user thread and whether it is
    running."""
    out = t.run("realm-thread")
    for ut in t.user_threads():
        m = re.search(r"{:#x}\s+(\S+)".format(ut.ptr), out)
        t.check(m is not None, "{:#x} not listed:\n{}".format(ut.ptr, out))
        t.check(m.group(1).lower() in (("running",) if ut.running else
                                       ("parked", "not")),  # ("not started")
                "state of {:#x}: {}".format(ut.ptr, m.group(1)))
    t.check_not_in("can't be shown", out, "realm-thread list")
    yield
    _finish(t)


# ---- thread apply all bt ------------------------------------------------

PARKED_END = "=== End of parked Realm user threads ==="


# The start of a backtrace line, up to the function's name.  (gdb marks a
# return address signed for pointer authentication, as on aarch64, "[PAC]".)
_BT_LINE = r"^#\d+\s+(?:0x[0-9a-f]+(?: \[PAC\])? in )?"


def _parked_section(t, out):
    """The parked threads' section of `thread apply all bt` output, from its
    first line to its last, which must be there."""
    marker = "Parked Realm user threads"
    t.check_in(marker, out, "'thread apply all bt' output")
    t.check_eq(out.count(marker), 1, "parked sections")
    start = out.index(marker)
    end = re.compile(r"^=== End of parked Realm user threads( of inferior \d+)? ===$",
                     re.M).search(out, start)
    t.check(end, "the parked section has no last line:\n" + out[start:])
    return out[start:end.end()]


def _split_parked(section):
    """{user thread pointer: backtrace text} from the parked section."""
    parts = re.split(r"^User thread (?:\d+ \()?(0x[0-9a-f]+)\)?:?.*$", section, flags=re.M)
    return {int(parts[i], 16): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


CHAIN_FRAMES = ["rgt_chain_level3", "rgt_chain_level2", "rgt_chain_level1", "chain_task"]


def _check_parked_section(t, out, full=False, chain_frames=CHAIN_FRAMES,
                          consecutive=True):
    section = _parked_section(t, out)
    if not t.is_core:
        # (In a live process each parked stack starts at frame #0.)
        t.check_not_in("<parked Realm user thread", section, "parked section")
    stacks = _split_parked(section)
    for i in range(NUM_CHAIN):
        bt = stacks.get(t.ut("chain{}".format(i)))
        t.check(bt is not None, "chain{} missing from parked section".format(i))
        names = re.findall(_BT_LINE + r"([\w:~<>]+)", bt, re.M)
        t.check_names_in_order(chain_frames, names,
                               "chain{} in parked section".format(i), consecutive)
        if full:
            t.check_in("chain_marker = {}".format(t.chain_marker(i)), bt,
                       "locals of chain{}".format(i))
    rec = stacks.get(t.ut("recurse"))
    t.check(rec is not None and rec.count("rgt_recurse") >= RECURSE_DEPTH,
            "recursive task's stack in parked section:\n{}".format(rec))
    hot = stacks.get(t.ut("hot"))
    t.check(hot is not None and "rgt_hot_wait" in hot,
            "hot task's stack in parked section:\n{}".format(hot))
    t.check(t.ut("probe") not in stacks,
            "the running user thread is listed as parked")
    return stacks


@test(scenarios=("live", "attach", "core", "kcore", "freeze"))
def test_thread_apply_all_bt(t):
    """`thread apply all bt` adds every parked user thread's stack, and
    leaves the debugger's own state (thread, frame, registers) as it was."""
    t.run("frame 1")
    thread = gdb.selected_thread().num
    before = t.regs()
    yield
    out = t.run("thread apply all bt")
    t.check_no_python_error("thread apply all bt", out)
    _check_parked_section(t, out)
    host = out[:out.index("Parked Realm user threads")]
    if t.scenario in ("live", "core"):
        t.check_in("realm_gdb_test_stop", host, "host threads' stacks")
    yield
    t.check_eq(gdb.selected_thread().num, thread, "selected thread")
    t.check_eq(t.selected_level(), 1, "selected frame")
    t.check_regs(before)
    t.check_host_stack()
    yield
    _finish(t)


@test(scenarios=("live", "core"), variants=[
    "backtrace", "where", "bt full", ("bt_dash_full", "bt -full"), "-q bt", "bt 3",
    "bt -frame-arguments all", "info stack"])
def test_thread_apply_all_bt_forms(t):
    """Other spellings and options of `thread apply all bt` also show
    parked threads, and the bt options apply to them too."""
    v = t.variant
    probe = {"bt -full": "bt -full 1", "bt -frame-arguments all":
             "bt -frame-arguments all 1",
             "-q bt": "thread apply {} -q print 1".format(t.stop_thread)}.get(v)
    if probe and not t.supports(probe):
        raise TestSkip("this gdb does not support '{}'".format(v))
    out = t.run("thread apply all " + v)
    t.check_no_python_error(v, out)
    if v == "-q bt":
        # (As `thread apply`'s -q leaves out the "Thread N" lines: no "User
        # thread PTR" lines, but every parked thread's stack.)
        section = _parked_section(t, out)
        t.check_not_in("User thread ", section, "-q: the headers")
        t.check_eq(section.count("rgt_chain_level3"), NUM_CHAIN, "-q: the chain tasks' stacks")
    elif v == "bt 3":
        # three of each parked thread's frames (after the marker line, in a
        # core file)
        section = _parked_section(t, out)
        stacks = _split_parked(section)
        for ptr, bt in stacks.items():
            t.check(_parked_frame_count(bt) == 3,
                    "'bt 3' printed {} parked frames for {:#x}:\n{}".format(
                        _parked_frame_count(bt), ptr, bt))
    else:
        _check_parked_section(t, out, full=v in ("bt full", "bt -full"))
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_realm_thread_numbers(t):
    """`realm-thread` numbers the user threads, as `info threads` does
    gdb's (in the order they were made), and marks the selected one; a
    number works where a UserThread pointer does, and one that isn't a user
    thread's is refused, saying so."""
    out = t.run("realm-thread")
    # (After any warnings of gdb's, the first command's in a core file.)
    t.check(re.search(r"^\s*Id\s+UserThread\s+State$", out, re.M), "the header:\n" + out)
    numbers = dict((int(p, 16), int(n)) for n, p in re.findall(
        r"^[ *] (\d+)\s+(0x[0-9a-f]+)\s", out, re.M))
    uts = [ut.ptr for ut in t.user_threads()]
    t.check_eq(sorted(numbers), sorted(uts), "the threads numbered")
    t.check_eq([numbers[p] for p in reversed(uts)], list(range(1, len(uts) + 1)),
               "the numbers, in the order the threads were made")
    n = numbers[t.ut("chain0")]
    t.check_in("rgt_chain_level3", t.run("realm-thread apply {} bt".format(n)),
               "realm-thread apply N")
    t.expect_error("realm-thread apply 9999 bt", "no user thread 9999")
    if t.scenario == "live":
        t.run("realm-thread {}".format(n))
        t.check(re.search(r"^\* {}\s+{:#x}\s+parked\s+\(selected\)$".format(
            n, t.ut("chain0")), t.run("realm-thread"), re.M), "the selected one, marked")
        t.check_in("rgt_chain_level3", t.rt("bt"), "rt bt on the thread selected by number")
        t.back()
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_realm_thread_apply_lists(t):
    """As `thread apply` does gdb's threads, `realm-thread apply` takes a
    list of user threads (numbers, ranges N-M, pointers, `$n`), each with its
    "User thread N (0x...):" header, one that can't be shown saying why;
    `apply`, `back` and `restore` are subcommands, which gdb abbreviates
    (`realm-thread ap`), completes and has help for; `realm-thread $n` takes
    a number, as `thread $n` does; and -ascending (for `all`, and in `thread
    apply all bt`) goes from the oldest."""
    out = t.run("realm-thread")
    numbers = dict((int(p, 16), int(n)) for n, p in re.findall(
        r"^[ *] (\d+)\s+(0x[0-9a-f]+)\s", out, re.M))
    chain = [numbers[t.ut("chain{}".format(i))] for i in range(3)]
    want = sorted(chain)
    out = t.run("realm-thread ap {}-{} {:#x} print 7".format(
        want[0], want[1], t.ut("chain{}".format(chain.index(want[2])))))
    got = [int(n) for n in re.findall(r"^User thread (\d+) \(0x[0-9a-f]+\):$", out, re.M)]
    t.check(set(want) <= set(got), "the headers, for {}:\n{}".format(want, out))
    t.check(got[0] == want[0] and len(re.findall(r"^\$\d+ = 7$", out, re.M)) >= 3,
            "the list, in order:\n" + out)
    t.expect_error("realm-thread apply {}-{} print 1".format(want[1], want[0]),
                   "inverted range")
    t.run("set $rgt_n = {}".format(chain[0]))
    t.check_in("rgt_chain_level3", t.run("realm-thread apply $rgt_n bt"), "$n")
    t.check_in("realm-thread apply", t.run("complete realm-thread a"), "completion")
    t.check_in("Usage: realm-thread apply", t.run("help realm-thread apply"), "help")
    t.check_not_in("realm-gdb-guard", t.run("complete realm-"), "the guard, completed")
    t.check_not_in("deprecated", t.run("bt 1"), "the guard's warning")
    desc = re.findall(r"^User thread (\d+) ", t.run("realm-thread apply all print 1"), re.M)
    asc = re.findall(r"^User thread (\d+) ", t.run("realm-thread apply all -asc print 1"),
                     re.M)
    t.check(desc and asc == list(reversed(desc)), "-ascending: {} {}".format(desc, asc))
    out = t.run("thread apply all -ascending bt 1")
    asc = re.findall(r"^User thread (\d+) ", out, re.M)
    t.check(asc == list(reversed(desc)), "thread apply all -ascending bt: {}".format(asc))
    if t.scenario == "live":
        t.run("realm-thread $rgt_n")
        t.check_in(t.remind("chain0"), gdb.prompt_hook("(gdb) "), "realm-thread $n")
        t.check(re.match(r"\[realm-thread \d+\] ", t.remind("chain0")), "the prompt's number")
        t.check_in("[Selecting {}, shown on thread".format(t.utw("chain1")),
                   t.run("realm-thread {}".format(chain[1])), "selecting again, briefly")
        t.check_in("back to " + t.utw("chain0"), t.run("realm-thread b"), "realm-thread b")
        t.run("realm-thread restore")
    else:
        # (A core file's view is kept until 'back', as a selection is.)
        out = t.run("realm-thread $rgt_n")
        t.check(re.search(r"^#1 ", out, re.M), "the frame, on selecting:\n" + out)
        t.check_eq(gdb.prompt_hook("(gdb) ") if gdb.prompt_hook else None,
                   t.remind("chain0") + "(gdb) ", "the prompt")
        t.check(re.search(r"^\* {}\s".format(chain[0]), t.run("realm-thread"), re.M),
                "the one shown, marked")
        t.run("realm-thread back")
        t.check(gdb.prompt_hook is None, "a prompt hook left in place")
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_realm_thread_apply_flags(t):
    """`realm-thread apply all` takes `thread apply`'s flags: -q leaves out
    the "User thread" lines, -s the threads where the command fails or
    prints nothing, and -c goes on past failures without failing at the end
    (without either, it goes on, then fails with the first error).  An
    unknown flag is refused, as gdb's own does."""
    out = t.run("realm-thread apply all -q print 1")
    t.check_not_in("User thread", out, "-q: the headers")
    t.check(len(re.findall(r"^\$\d+ = 1$", out, re.M)) > NUM_CHAIN, "-q: the output:\n" + out)
    if t.supports("frame apply all -s print 1"):
        out = t.run("realm-thread apply all -s frame apply all -s print chain_marker")
        t.check_eq(len(re.findall(r"^User thread ", out, re.M)), NUM_CHAIN,
                   "-s: only the threads that have chain_marker:\n" + out)
    ok, out = t.try_run("realm-thread apply all -c print rgt_nosuch")
    t.check(ok, "-c: the command doesn't fail: " + out)
    t.check(out.count("<print failed: No symbol") > NUM_CHAIN, "-c: the errors:\n" + out)
    t.expect_error("realm-thread apply all print rgt_nosuch", "No symbol")
    t.expect_error("realm-thread apply all -x print 1", "Unrecognized option at: -x")
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live",))
def test_thread_apply_other_commands(t):
    """Overriding `thread apply` must not break its other uses."""
    threads = sorted(th.num for th in gdb.selected_inferior().threads())
    out = t.run("thread apply all print $_thread")
    t.check_not_in("Parked Realm user threads", out)
    for n in threads:
        t.check(re.search(r"= {}\b".format(n), out) is not None,
                "thread {} missing from 'thread apply all print $_thread':\n{}"
                .format(n, out))
    yield
    # quoting and spacing must survive (and nothing here may make an
    # inferior call, e.g. to put a string literal in memory, which would
    # let the stopped thread run)
    out = t.run('thread apply all printf "x  \\"y\\" %d\\n", 7')
    t.check_in('x  "y" 7', out, "printf through thread apply")
    yield
    # a list of thread IDs, a range, and $_thread
    a, b = threads[0], threads[1]
    out = t.run("thread apply {} {} print $_thread".format(a, b))
    t.check(re.search(r"= {}\b".format(a), out) and re.search(r"= {}\b".format(b), out),
            "'thread apply {} {}':\n{}".format(a, b, out))
    out = t.run("thread apply {}-{} print $_thread".format(a, b))
    t.check(re.search(r"= {}\b".format(a), out) and re.search(r"= {}\b".format(b), out),
            "'thread apply {}-{}':\n{}".format(a, b, out))
    yield
    # a single thread's bt doesn't get the parked section
    out = t.run("thread apply {} bt".format(t.stop_thread))
    t.check_not_in("Parked Realm user threads", out)
    t.check_in("realm_gdb_test_stop", out)
    yield
    # -s suppresses errors; -c continues past them (gdb >= 9)
    if t.supports("thread apply {} -s print 1".format(t.stop_thread)):
        ok, out = t.try_run("thread apply all -s print no_such_variable_xyz")
        t.check(ok, "'thread apply all -s' failed: " + out)
        t.check_not_in("no_such_variable_xyz", out, "-s should silence errors")
        ok, out = t.try_run("thread apply all -c print no_such_variable_xyz")
        t.check(ok, "'thread apply all -c' failed: " + out)
    yield
    t.check_host_stack()
    t.continue_to_exit()


@test(scenarios=("core",))
def test_thread_apply_all_bt_while_showing(t):
    """In a core file, `thread apply all bt` while a parked thread is shown
    lists the parked threads and leaves that thread shown as it was."""
    t.show("chain0")
    t.select_frame("rgt_chain_level2")
    yield
    out = t.run("thread apply all bt")
    _check_parked_section(t, out)
    yield
    t.check_chain_stack(0, "stack after 'thread apply all bt'")
    t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2",
               "selected frame")
    t.back()
    t.check_host_stack()


@test(scenarios=("live", "core"), variants=[("all", "thread apply all print 1"),
                                            ("list", "thread apply LIST print 1"),
                                            ("bt", "thread apply all bt")])
def test_thread_apply_keeps_frame(t):
    """`thread apply` leaves the thread and frame the user had selected."""
    t.run("frame 2")
    thread = gdb.selected_thread().num
    yield
    ids = " ".join(str(n) for n in sorted((th.num for th in
                                           gdb.selected_inferior().threads()),
                                          reverse=True))
    yield typed(t.variant.replace("LIST", ids))
    t.check_eq(gdb.selected_thread().num, thread, "selected thread")
    t.check_eq(t.selected_level(), 2, "selected frame")
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live",))
def test_completion(t):
    """`thread apply all` completes like gdb's own, and so does the command
    in `realm-thread apply`."""
    for text, want in (("thread apply all bac", "thread apply all backtrace"),
                       ("thread apply all -as", "thread apply all -ascending"),
                       ("thread apply all print realm_gdb_test_ad",
                        "thread apply all print realm_gdb_test_add"),
                       ("thread apply all -q fr", "thread apply all -q frame"),
                       ("realm-thread ap", "realm-thread apply"),
                       ("realm-thread apply al", "realm-thread apply all"),
                       ("realm-thread apply all bac", "realm-thread apply all backtrace"),
                       ("realm-thread apply 0x1234 print realm_gdb_test_ad",
                        "realm-thread apply 0x1234 print realm_gdb_test_add"),
                       ("realm-thread ba", "realm-thread back"),
                       ("rt bac", "rt backtrace"),
                       ("rt print realm_gdb_test_ad", "rt print realm_gdb_test_add")):
        out = t.run("complete " + text)
        t.check_in(want, out, "completions of {!r}".format(text))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_thread_apply_all_whitespace(t):
    """`thread apply all` flags may be separated by any whitespace."""
    threads = sorted(th.num for th in gdb.selected_inferior().threads())
    if t.supports("thread apply {} -q print 1".format(t.stop_thread)):
        out = t.run("thread apply all -q\tprint $_thread")
        t.check_not_in("Thread {} (".format(threads[0]), out, "-q output")
        for n in threads:
            t.check(re.search(r"= {}\b".format(n), out), "-q<tab> output:\n" + out)
        out = t.run("thread apply all  -q \t -s\tprint $_thread")
        t.check(re.search(r"= {}\b".format(threads[0]), out), "mixed whitespace:\n" + out)
    if t.supports("thread apply {} -- print 1".format(t.stop_thread)):
        out = t.run("thread apply all --\tprint $_thread")
        t.check(re.search(r"= {}\b".format(threads[0]), out), "--<tab> output:\n" + out)
    out = t.run('thread apply all \tprintf "x \\"y\\"\\n"')
    t.check_in('x "y"', out, "printf after a tab")
    yield
    t.continue_to_exit()


def _parked_frame_count(section):
    """Frames of the parked stack in one parked thread's backtrace text:
    everything from its innermost frame (PARKED_FRAME0) on, or every frame
    if that isn't shown (`bt -N`)."""
    names = re.findall(_BT_LINE + r"(\S+)", section, re.M)
    starts = [i for i, n in enumerate(names) if PARKED_FRAME0.split("::")[-1] in n]
    return len(names) - (starts[0] if starts else 0)


def _check_bt_count(t, count, expected=3):
    out = t.run("thread apply all bt " + count)
    stacks = _split_parked(_parked_section(t, out))
    for i in range(NUM_CHAIN):
        bt = stacks[t.ut("chain{}".format(i))]
        t.check_eq(_parked_frame_count(bt), expected,
                   "parked frames of chain{} for 'bt {}':\n{}".format(i, count, bt))
    if not t.is_core:
        bt = t.apply("chain0", "bt " + count)
        t.check_eq(_parked_frame_count(bt), expected,
                   "parked frames for 'realm-thread apply PTR bt {}':\n{}".format(count, bt))


@test(scenarios=("live", "core"), variants=[
    ("literal", "3"), ("hex", "0x3"), ("plus", "+3"), ("sum", "1+2"),
    ("convenience", "$rgt_three"), ("no_filters", "no-filters 3"),
    ("dash_no_filters", "-no-filters 3"), ("full", "full 3"), ("outermost", "-3")])
def test_thread_apply_all_bt_count(t):
    """`thread apply all bt N` shows N of each parked thread's frames, for
    any count expression gdb accepts (`bt -N`: the N outermost)."""
    if t.variant.startswith("-no-filters") and not t.supports("bt -no-filters 1"):
        raise TestSkip("this gdb does not support 'bt -no-filters'")
    t.run("set $rgt_three = 3")
    _check_bt_count(t, t.variant)
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live",), stop_at="rgt_inline_stop", variants=["3", "no-filters 3"])
def test_thread_apply_all_bt_count_inline_host(t):
    """`thread apply all bt N` when the thread showing the parked threads has
    inline frames inside its innermost frame."""
    _check_bt_count(t, t.variant)
    yield
    t.run("delete")
    t.continue_to_exit()


def _check_bt_limit(t, args):
    """`set backtrace limit 3` limits each parked thread to 3 frames (or
    fewer, with `bt 2`), as for any thread."""
    if "no-filters" in args and not t.supports("bt -no-filters 1"):
        raise TestSkip("this gdb does not support 'bt -no-filters'")
    t.run("set backtrace limit 3")
    try:
        _check_bt_count(t, args, expected=2 if args.endswith("2") else 3)
        yield
        t.run("set backtrace limit 1")
        if t.is_core:
            # (The marker line takes the one frame.)
            t.expect_error("realm-thread {:#x}".format(t.ut("chain0")),
                           r"backtrace limit 1")
        else:
            _check_bt_count(t, "", expected=1)
    finally:
        t.run("set backtrace limit unlimited")
    t.check_not_in("rgt_chain_level3", t.frame_names(), "stack")


@test(scenarios=("live", "core"), variants=[("plain", ""), ("count", "2"),
                                            ("no_filters", "-no-filters")])
def test_thread_apply_all_bt_limit(t):
    """`thread apply all bt` with a backtrace limit (`set backtrace limit`):
    in a core file, the frames before each parked stack don't count."""
    for step in _check_bt_limit(t, t.variant):
        yield step
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live",), stop_at="rgt_inline_stop", variants=[
    ("plain", ""), ("no_filters", "-no-filters")])
def test_thread_apply_all_bt_limit_inline_host(t):
    """The same, on a thread with inline frames inside its innermost frame."""
    for step in _check_bt_limit(t, t.variant):
        yield step
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_multiple_inferiors(t):
    """A second (empty) inferior disturbs neither `realm-thread apply` in the
    first nor `thread apply all bt`."""
    t.run("add-inferior")
    t.run("inferior 2")
    t.check_in("no process", t.expect_error("realm-thread apply all bt"),
               "realm-thread apply in an empty inferior")
    yield
    t.run("thread 1.{}".format(t.stop_thread))
    out = t.run("thread apply all bt")
    t.check_in("of inferior 1", out, "parked section header")
    _check_parked_section(t, out)
    t.within("chain0", lambda: t.check_chain_stack(0))
    t.run("remove-inferiors 2")
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["one", "call_schedule_multiple"])
def test_two_inferiors(t):
    """Two live inferiors: `thread apply all bt` shows each one's parked
    threads, `realm-thread apply` points to the other inferior's, and a
    function call made while one of them shows a parked thread lets nothing
    else run, even with schedule-multiple."""
    stop1 = t.stop_thread
    chain0_1 = t.ut("chain0")
    args = t.program_args()
    t.run("add-inferior -exec {}".format(t.config["program"]))
    t.run("inferior 2")
    t.run("set args " + args)
    t.run_program()
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop",
               "inferior 2 stopped in")
    stop2 = gdb.selected_thread().num
    chain0_2 = t.find_roles()["chain0"].ptr
    addrs2 = set(u.ptr for u in t.user_threads())
    yield
    t.run("thread 1.{}".format(stop1))
    t.find_roles()
    if t.variant == "one":
        out = t.run("thread apply all bt")
        for num, ptr in ((1, chain0_1), (2, chain0_2)):
            m = re.search(r"^=== Parked Realm user threads of inferior {0} \(\d+\), "
                          r".*?^=== End of parked Realm user threads "
                          r"of inferior {0} ===$".format(num), out, re.M | re.S)
            t.check(m, "inferior {}'s parked section:\n{}".format(num, out))
            t.check_in("({:#x}):".format(ptr), m.group(0),
                       "inferior {}'s parked section".format(num))
            t.check_in("rgt_chain_level3", m.group(0),
                       "inferior {}'s parked section".format(num))
        t.check(re.search(r"inferior 2 has \d+ Realm user threads too",
                          t.run("realm-thread")), "realm-thread list in inferior 1")
        only2 = addrs2 - set(u.ptr for u in t.user_threads())
        if only2:
            t.expect_error("realm-thread apply {:#x} bt".format(min(only2)),
                           "is a user thread of inferior 2")
        t.select(chain0_1)
        # (`thread 2.N` is refused while plain commands show the selection.)
        inferior2 = [i for i in gdb.inferiors() if i.num == 2][0]
        [th for th in inferior2.threads() if th.num == stop2][0].switch()
        t.expect_error("rt bt", r"inferior 1 has one")
        # (Inferior 2 runs to the end; inferior 1's selection stays.)
        t.run("continue")
        t.run("thread 1.{}".format(stop1))
        t.check_in("rgt_chain_level3", t.rt("bt"), "rt bt in inferior 1")
        t.run("realm-thread back")
    else:
        # A call in inferior 2 while it shows a parked thread holds
        # inferior 1 (whose stopped thread would otherwise finish).
        pc1 = gdb.newest_frame().pc()
        t.run("thread 2.{}".format(stop2))
        t.run("set schedule-multiple on")
        t.check_in("= 100000", t.apply(chain0_2, "print realm_gdb_test_nap(100000)"),
                   "call in inferior 2")
        t.run("set schedule-multiple off")
        t.run("thread 1.{}".format(stop1))
        t.check_eq(gdb.newest_frame().pc(), pc1, "inferior 1's PC")
        t.check_eq(t.eval_int("realm_gdb_test_state.num_finished"), 0,
                   "inferior 1's finished tasks")
    yield
    if t.variant != "one":
        t.run("thread 2.{}".format(stop2))
        t.run("continue")
        t.run("thread 1.{}".format(stop1))
    t.continue_to_exit()
    t.check_eq(sorted(t.exit_codes), [0, 0], "exit codes")


@test(scenarios=("live",))
def test_registry_loop(t):
    """A corrupt registry (one that loops) is reported, not silently cut
    short."""
    first = t.user_threads()[0]
    link = "((Realm::UserThread *){:#x})->registry_next".format(first.ptr)
    saved = t.eval_int("(unsigned long){}".format(link))
    t.run("set var {} = (Realm::UserThread *){:#x}".format(link, first.ptr))
    yield
    t.check_in("loops back to {:#x}".format(first.ptr), t.run("realm-thread"),
               "realm-thread list")
    t.check_in("loops back to {:#x}".format(first.ptr), t.run("thread apply all bt"),
               "thread apply all bt")
    t.run("set var {} = (Realm::UserThread *){:#x}".format(link, saved))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_registry_broken(t):
    """A registry link to unreadable memory (as a crash may leave): the
    parked threads before it are still shown, and the broken entry and the
    rest of the registry are reported, not silently left out."""
    uts = t.user_threads()
    victim = [u for u in uts if not u.running][0]
    link = "((Realm::UserThread *){:#x})->registry_next".format(victim.ptr)
    saved = t.eval_int("(unsigned long){}".format(link))
    t.run("set var {} = (Realm::UserThread *)0x10".format(link))
    yield
    out = t.run("thread apply all bt")
    stacks = _split_parked(_parked_section(t, out))
    for u in uts[:uts.index(victim) + 1]:
        if not u.running:
            t.check(u.ptr in stacks, "{:#x} missing from the parked section:\n{}".format(
                u.ptr, out))
    t.check(re.search(r"^User thread (\d+ \()?0x10\)?:\s+<not shown: cannot read it: ",
                      out, re.M),
            "the broken entry:\n" + out)
    t.check_in("cannot read the rest of the Realm user thread registry", out,
               "thread apply all bt")
    # (The list and `realm-thread apply all` too: the threads before it, and
    # what can't be read.)
    out = t.run("realm-thread")
    t.check_in("{:#x}".format(victim.ptr), out, "the list")
    t.check_in("cannot read the rest of the Realm user thread registry", out, "the list")
    t.check_in("may be incomplete", out, "the list")
    error, out = t.run_capturing("realm-thread apply all bt 1")
    t.check_in("({:#x}):".format(victim.ptr), out, "realm-thread apply all")
    t.check_in("there may be more parked threads", out, "realm-thread apply all")
    t.run("set var {} = (Realm::UserThread *){:#x}".format(link, saved))
    yield
    t.continue_to_exit()


# ---- realm-thread apply ------------------------------------------------

@test(scenarios=("live", "attach", "core", "kcore", "freeze"))
def test_apply_show(t):
    """`realm-thread apply PTR COMMAND` shows a parked thread's stack, with
    working frame selection, locals, arguments and registers, and leaves the
    selected thread exactly as it was (thread, frame, registers)."""
    t.run("frame 1")
    thread = gdb.selected_thread().num
    before = t.regs()
    for i in range(NUM_CHAIN):
        def inside(i=i):
            t.check_chain_stack(i)
            _check_parked_frame0(t, "chain{}".format(i))
            t.select_frame("rgt_chain_level3")
            t.check_eq(t.eval_int("chain_marker"), t.chain_marker(i), "chain_marker")
            t.check_in('"chain-{}"'.format(i), t.run("print chain_tag"), "chain_tag")
            t.check_in("chain_marker = {}".format(t.chain_marker(i)),
                       t.run("info locals"), "info locals")
            t.check_eq(t.eval_int("args.index"), i, "args.index")
            t.check_eq(t.eval_int("widget.id"), i, "widget.id")
            t.check_in("widget", t.run("info args"), "info args")
            t.check_eq(t.eval_int("$pc"), gdb.selected_frame().pc() & _U64, "$pc")
            for cmd in ("info frame", "x/2i $pc", "info registers", "list", "frame"):
                t.check_no_python_error(cmd, t.run(cmd))
            t.run("up")
            t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2", "after up")
            t.run("down")
            t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level3",
                       "after down")
        t.within("chain{}".format(i), inside)
        yield
        t.check_eq(gdb.selected_thread().num, thread, "selected thread")
        t.check_eq(t.selected_level(), 1, "selected frame")
        t.check_regs(before)
        t.check_host_stack()
    out = t.apply("chain0", "bt full")
    t.check_in("chain_marker = {}".format(t.chain_marker(0)), out, "bt full")
    if not t.is_core:
        t.check(re.match(r"#0\s+.*" + re.escape(PARKED_FRAME0), out),
                "frame #0 of 'realm-thread apply PTR bt full':\n" + out)
    if t.supports("frame apply 1 print 1"):
        out = t.apply("chain0", "frame apply all -q -s print chain_marker")
        t.check_in("= {}".format(t.chain_marker(0)), out, "frame apply")
    yield
    _finish(t)


@test(scenarios=("live", "attach", "core", "kcore"))
def test_apply_callee_saved(t):
    """Variables held in callee-saved registers in a parked frame show the
    parked thread's values, not the host thread's."""
    def inside():
        t.select_frame("rgt_hot_wait")
        for k in range(NUM_HOT):
            expected = t.eval_int("realm_gdb_test_state.hot_wait_expected[{}]".format(k))
            val = gdb.parse_and_eval("hot{}".format(k))
            if val.is_optimized_out:
                if k == NUM_HOT - 1:
                    continue  # (spilled, and the compiler may not say where)
                t.fail("hot{} is <optimized out>".format(k))
            t.check_eq(int(val), expected, "hot{}".format(k))
    t.within("hot", inside)
    yield
    _finish(t)


@test(scenarios=("live", "attach", "core", "kcore"))
def test_apply_callee_saved_fp(t):
    """Floating-point variables live across a parked thread's wait (on
    aarch64, in its callee-saved d8-d15) show the parked thread's values,
    not the host thread's.  A value may be <optimized out> (compilers don't
    always say where one is, and a core file's view has no floating-point
    registers to give on aarch64), but never wrong."""
    unreadable = []

    def inside():
        t.select_frame("rgt_fhot_wait")
        shown = 0
        for k in range(NUM_FHOT):
            expected = float(gdb.parse_and_eval(
                "realm_gdb_test_state.fhot_wait_expected[{}]".format(k)))
            try:
                val = gdb.parse_and_eval("fhot{}".format(k))
            except gdb.error as e:
                # (FreeBSD's clang says where a double is with DW_OP_deref_size
                # 16, which gdb 15 can't read on any thread.)
                if "more than 8 bytes" not in str(e):
                    raise
                unreadable.append(str(e))
                continue
            if val.is_optimized_out:
                continue
            t.check_eq(float(val), expected, "fhot{}".format(k))
            shown += 1
        return shown
    shown = t.within("fhot", inside)
    yield
    _finish(t)
    if not shown:
        raise TestSkip(unreadable[0] if unreadable else
                       "every fhot value is <optimized out> where rgt_fhot_wait waits")


@test(scenarios=("live", "core"))
def test_apply_recursion(t):
    """A deep parked stack unwinds completely, with the right locals at
    every level."""
    def inside():
        names = t.frame_names(400)
        t.check_eq(names.count("rgt_recurse"), RECURSE_DEPTH, "rgt_recurse frames")
        t.check_in("recurse_task", names, "recursive stack")
        level, _ = t.find_frame("rgt_recurse")
        for depth in (RECURSE_DEPTH, RECURSE_DEPTH - 7, 1):
            t.run("frame {}".format(level + (RECURSE_DEPTH - depth)))
            t.check_eq(t.eval_int("recurse_depth"), depth, "recurse_depth")
    t.within("recurse", inside)
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_apply_every_thread(t):
    """Every registered parked user thread can be shown, including Realm's
    own idle workers, alone or with `realm-thread apply all`."""
    before = t.regs()
    parked = [ut for ut in t.user_threads() if not ut.running]
    for ut in parked:
        ok, out = t.run_clean("realm-thread apply {:#x} bt".format(ut.ptr))
        t.check(ok, "bt of {:#x} failed: {}".format(ut.ptr, out))
        yield
    out = t.run("realm-thread apply all bt")
    stacks = _split_parked(out)
    t.check_eq(sorted(stacks), sorted(ut.ptr for ut in parked),
               "user threads in 'realm-thread apply all bt'")
    for ptr, bt in stacks.items():
        t.check(_parked_frame_count(bt) > 0, "{:#x}'s stack:\n{}".format(ptr, bt))
    t.check_regs(before)
    t.check_host_stack()
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_apply_all(t):
    """`realm-thread apply all COMMAND` runs COMMAND on each parked thread,
    under a line naming it; if it fails on some, it goes on, and then
    fails with the first error."""
    parked = [ut.ptr for ut in t.user_threads() if not ut.running]
    out = t.run("realm-thread apply all print $pc")
    stacks = _split_parked(out)
    t.check_eq(sorted(stacks), sorted(parked), "user threads listed")
    for ptr, text in stacks.items():
        t.check(re.search(r"^\$\d+ = ", text, re.M), "{:#x}: {}".format(ptr, text))
    yield
    error, out = t.run_capturing("realm-thread apply all print no_such_variable_xyz")
    t.check(error is not None and "no_such_variable_xyz" in error,
            "'realm-thread apply all' with a failing command: {}".format(error))
    t.check_eq(out.count("<print failed: "), len(parked), "failures listed:\n" + out)
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"), variants=[
    "0", "0x10", "no_such_symbol_xyz", ("address_of_state", "&realm_gdb_test_state"),
    ("function", "realm_gdb_test_add"), ("void_convenience", "$_nonexistent_convenience"),
    ("syntax_error", "(1 +)")])
def test_apply_bad_pointer(t):
    """Anything that isn't a parked user thread is refused cleanly, without
    changing anything, by `realm-thread apply PTR` and `realm-thread PTR`."""
    before = t.regs()
    t.expect_error("realm-thread apply {} bt".format(t.variant))
    t.expect_error("realm-thread " + t.variant)
    t.check_in("no parked user thread is selected", t.run("realm-thread back"))
    yield
    t.check_regs(before)
    t.check_host_stack()
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_apply_running_thread(t):
    """A running user thread's saved context is stale, so showing it
    is refused (its stack is already the host thread's)."""
    before = t.regs()
    t.expect_error("realm-thread apply {:#x} bt".format(t.ut("probe")), "running")
    yield
    t.check_regs(before)
    t.check_host_stack()
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_apply_expressions(t):
    """The pointer can be any expression: a value history entry, or one in
    parentheses with spaces in it."""
    t.run("print (void *){:#x}".format(t.ut("chain2")))
    out = t.run("realm-thread apply $ bt")
    t.check_in("rgt_chain_level3", out, "realm-thread apply $")
    out = t.run("realm-thread apply (realm_gdb_test_state.chain_locals[1] , (void *){:#x}) "
                "bt".format(t.ut("chain1")))
    t.check_in("rgt_chain_level3", out, "parenthesized expression")
    t.within(t.ut("chain1"), lambda: t.check_chain_stack(1))
    out = t.run("realm-thread apply {:#x} -- bt 1".format(t.ut("chain1")))
    t.check_eq(_parked_frame_count(out), 1, "'--' before the command")
    yield
    t.check_host_stack()
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live",))
def test_apply_usage(t):
    """`realm-thread apply` and `rt` without a command say how to use them."""
    for cmd in ("realm-thread apply", "realm-thread apply all",
                "realm-thread apply {:#x}".format(t.ut("chain0"))):
        t.expect_error(cmd, "usage: realm-thread apply")
    t.expect_error("rt", "usage: rt COMMAND")
    t.check_in("no parked user thread is selected", t.run("realm-thread back"))
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_apply_from_other_thread(t):
    """Showing a parked thread on a thread other than the one that stopped
    leaves both exactly as they were."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.run("thread {}".format(other))
    before_other = t.regs()
    before_stop = t.regs(t.stop_thread)
    yield
    t.within("chain0", lambda: t.check_chain_stack(0))
    _check_parked_section(t, t.run("thread apply all bt"))
    t.check_eq(gdb.selected_thread().num, other, "selected thread")
    yield
    t.check_regs(before_other, other, "other thread's registers")
    t.check_regs(before_stop, t.stop_thread, "stopped thread's registers")
    t.run("thread {}".format(t.stop_thread))
    t.check_host_stack()
    t.continue_to_exit()


@test(scenarios=("live",), stop_at="rgt_inline_stop")
def test_apply_inline_host_frame(t):
    """The stopped thread's innermost frame is an inline frame (common in
    optimized code): showing a parked thread, inferior calls, and the
    thread's own stack afterwards all still work."""
    host = t.frame_names()[:2]
    t.check_eq(host, ["rgt_inline_stop", "rgt_hot_probe"], "host stack")
    before = t.regs()

    def inside():
        t.check_chain_stack(0)
        t.select_frame("rgt_chain_level3")
        level = t.selected_level()
        t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
                   t.chain_marker(0) + 1, "add()")
        t.check_eq(t.selected_level(), level, "selected frame level after call")
    t.within("chain0", inside)
    yield
    _check_parked_section(t, t.run("thread apply all bt"))
    t.check_eq(t.frame_names()[:2], host, "host stack afterwards")
    t.check_regs(before)
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live", "attach"))
def test_apply_write_memory(t):
    """Assigning to a parked frame's variables keeps that frame selected and
    the parked stack shown (gdb rebuilds its frames after a write)."""
    def inside():
        level = t.select_frame("rgt_chain_level3")
        m = t.chain_marker(0)
        t.run("set var chain_marker = {}".format(m))
        t.check_eq(t.selected_level(), level, "selected frame level after the write")
        t.check_eq(t.eval_int("chain_marker"), m, "chain_marker")
        # The same through the Python API, which gdb doesn't follow with a
        # rebuild of its frames.
        inf = gdb.selected_inferior()
        addr = t.eval_int("&chain_marker")
        inf.write_memory(addr, inf.read_memory(addr, 4))
        t.check_chain_stack(0, "stack after writing memory")
    t.within("chain0", inside)
    yield
    t.check_host_stack()
    _finish(t)


@test(scenarios=("live",))
def test_apply_nested_refused(t):
    """A command run on a parked thread can't show parked threads itself
    (`realm-thread apply`, `thread apply all bt`): the thread showing one has
    its registers until the command is over.  Other `thread apply` and
    `frame apply` commands work."""
    before = t.regs()
    ptr = t.ut("chain0")
    for inner in ("realm-thread apply {:#x} bt".format(t.ut("chain1")),
                  "realm-thread apply all bt", "thread apply all bt"):
        error, out = t.run_capturing("realm-thread apply {:#x} {}".format(ptr, inner))
        t.check(error is not None and ("nest" in error or "inside" in error),
                "{!r} inside realm-thread apply: {} / {}".format(inner, error, out))
    out = t.apply("chain0", "thread apply {} bt 1".format(t.stop_thread))
    t.check_in(PARKED_FRAME0.split("::")[-1], out, "thread apply N bt inside")
    out = t.apply("chain0", "thread apply all print $_thread")
    t.check_not_in("Parked Realm user threads", out, "thread apply all print inside")
    yield
    t.check_regs(before)
    t.check_host_stack()
    t.continue_to_exit()


# ---- realm-thread <ptr> and rt -------------------------------------------

def _frame_names_of(t, role, count=2):
    """The names of a parked thread's innermost frames (they depend on how
    Realm was compiled)."""
    return re.findall(_BT_LINE + r"([\w:~<>]+)", t.apply(role, "bt {}".format(count)),
                      re.M)


def _rt_frame_name(t):
    """The function of the frame `rt` has selected on the selected thread."""
    m = re.match(r"#\d+\s+(?:0x[0-9a-f]+(?: \[PAC\])? in )?([\w:~<>]+)", t.rt("frame"))
    t.check(m, "no frame line from 'rt frame'")
    return m.group(1)


@test(scenarios=("live", "attach", "freeze"))
def test_select(t):
    """In a live process `realm-thread <ptr>` selects a parked thread, which
    `rt COMMAND` then works on, keeping the frame selected on it (`rt frame
    N`, `rt up`, `rt down`), function calls included, while nothing in the
    program changes between commands: plain commands act on the thread as
    it really is."""
    t.run("frame 1")
    before = t.regs()
    out = t.select("chain0")
    t.check(re.search(r"selected user thread \d+ \({:#x}\)".format(t.ut("chain0")), out),
            "realm-thread PTR:\n" + out)
    t.check(re.search(r"^#0\s+.*" + re.escape(PARKED_FRAME0), out, re.M),
            "frame #0 shown on selecting:\n" + out)
    t.check(re.search(r"{:#x}\s+parked\s+\(selected\)".format(t.ut("chain0")),
                      t.run("realm-thread")), "realm-thread list")
    t.check_regs(before)
    t.check_eq(t.selected_level(), 1, "the thread's own selected frame")
    yield
    names = re.findall(_BT_LINE + r"([\w:~<>]+)", t.rt("bt"), re.M)
    t.check_names_in_order(CHAIN_FRAMES, names, "rt bt")
    level = names.index("rgt_chain_level3")
    t.check_in("rgt_chain_level3", t.rt("frame {}".format(level)), "rt frame N")
    yield
    m = t.chain_marker(0)
    t.check_in("= {}".format(m), t.rt("print chain_marker"), "rt print, a command later")
    t.check_in("chain_marker = {}".format(m), t.rt("info locals"), "rt info locals")
    t.check_in('"chain-0"', t.rt("print chain_tag"), "rt print chain_tag")
    t.check_in("rgt_chain_level2", t.rt("up"), "rt up")
    t.check_in("= 0", t.rt("print widget.id"), "rt print, one frame up")
    t.check_in("rgt_chain_level3", t.rt("down"), "rt down")
    t.check_in("rgt_chain_level1", t.rt("up 2"), "rt up 2")
    t.check_in("rgt_chain_level3", t.rt("down 2"), "rt down 2")
    t.check_in(PARKED_FRAME0, t.rt("down 999"), "rt down 999 (as far as it goes)")
    t.expect_error("rt down", "Bottom \\(innermost\\) frame selected; you cannot go down")
    t.expect_error("rt frame 9999", "No frame at level 9999")
    t.check_in("rgt_chain_level3", t.rt("f {}".format(level)), "rt f N")
    t.check_in("= {}".format(m + 1), t.rt("print realm_gdb_test_add(chain_marker, 1)"),
               "a function call")
    t.check_eq(_rt_frame_name(t), "rgt_chain_level3", "rt's frame after the call")
    yield
    t.check_host_stack()
    t.check_regs(before)
    t.check_eq(t.selected_level(), 1, "the thread's own selected frame")
    t.check_in("no parked user thread is selected now", t.run("realm-thread back"))
    t.expect_error("rt bt", "no parked user thread is selected")
    t.check(gdb.prompt_hook is None, "a prompt hook left in place")
    yield
    _finish(t)


@test(scenarios=("live", "attach"))
def test_select_external(t):
    """Commands that start another program don't show the parked thread
    meanwhile, as /proc shows the host thread's registers: `rt shell` (`!`),
    `rt edit`, `rt make`, `realm-thread apply PTR shell`, and pipe's shell
    command.
    `rt pipe COMMAND | SHELL` still runs COMMAND on the parked thread, and
    `rt edit *$pc` edits where the parked thread's pc is."""
    tmp = t.config["tmp"]
    proc = "/proc/{}/task/{}/syscall".format(gdb.selected_inferior().pid,
                                             gdb.selected_thread().ptid[1])
    if not os.path.exists(proc):
        raise TestSkip("no " + proc)  # (Linux's)
    out = os.path.join(tmp, "external.out")

    def written(cmd):
        if os.path.exists(out):
            os.unlink(out)
        t.run(cmd.format(proc=proc, out=out))
        with open(out) as f:
            return f.read()

    def sp_pc(text):  # (/proc's syscall line ends with SP and PC)
        m = re.search(r"(0x[0-9a-f]+) (0x[0-9a-f]+)\s*$", text, re.M)
        return m.groups() if m else None
    own = sp_pc(written("shell cat {proc} > {out}"))
    t.check(own is not None, "the host thread's SP and PC, from " + proc)
    t.select("chain0")
    parked_sp = re.search(r"0x[0-9a-f]+", t.rt("print $sp")).group(0)
    t.check(parked_sp != own[0], "the parked thread's SP differs from the host's")
    yield
    for cmd in ("rt shell cat {proc} > {out}", "rt !cat {proc} > {out}",
                "realm-thread apply %#x shell cat {proc} > {out}" % t.ut("chain0")):
        t.check_eq(sp_pc(written(cmd)), own, "{}: the host's own SP and PC meanwhile".format(
            cmd.split(" cat")[0]))
    editor = os.path.join(tmp, "editor.sh")
    with open(editor, "w") as f:
        f.write('echo "$@" > {}; cat {} >> {}\n'.format(out, proc, out))
    saved = os.environ.get("EDITOR")
    os.environ["EDITOR"] = "sh " + editor
    try:
        t.check_eq(sp_pc(written("rt edit")), own, "rt edit: the host's own SP and PC")
        where = re.search(r"Line (\d+) of \"([^\"]+)\"", t.rt("info line *$pc"))
        shown = t.run("rt edit *$pc")
        t.check_in(USER_SWITCH, shown, "rt edit *$pc: where it says the pc is")
        with open(out) as f:
            text = f.read()
        t.check(where and text.startswith("+{} ".format(where.group(1))) and
                os.path.basename(where.group(2)) in text.splitlines()[0],
                "rt edit *$pc: the editor at the parked thread's pc ({} vs {!r})".format(
                    where and where.groups(), text))
        t.check_eq(sp_pc(text), own, "rt edit *$pc: the host's own SP and PC meanwhile")
    finally:
        if saved is None:
            os.environ.pop("EDITOR", None)
        else:
            os.environ["EDITOR"] = saved
    if any(os.access(os.path.join(d, "make"), os.X_OK)
           for d in os.environ.get("PATH", "").split(os.pathsep)):
        makefile = os.path.join(tmp, "external.mk")
        with open(makefile, "w") as f:
            f.write("all:\n\t@cat {} > {}\n".format(proc, out))
        t.check_eq(sp_pc(written("rt make -s -f " + makefile)), own,
                   "rt make: the host's own SP and PC meanwhile")
    yield
    if t.try_run("help pipe")[0]:
        sp_out = os.path.join(tmp, "external.sp")
        text = written("rt pipe print $sp | (cat > %s; cat {proc} > {out})" % sp_out)
        with open(sp_out) as f:
            t.check_in(parked_sp, f.read(), "rt pipe: COMMAND on the parked thread")
        t.check_eq(sp_pc(text), own, "rt pipe: the host's own SP and PC for SHELL")
        text = written("rt | print $sp | (cat > %s; cat {proc} > {out})" % sp_out)
        t.check_eq(sp_pc(text), own, "rt | ... |: the same")
        t.check_eq(sp_pc(written("rt pipe shell cat {proc} > {out} | cat")), own,
                   "rt pipe shell ... |: the host's own SP and PC")
        t.run("rt pipe -d XX print 1 XX sh -c 'cat > /dev/null; exit 3'")
        t.check_in("= 3", t.run("print $_shell_exitcode"), "rt pipe: $_shell_exitcode")
        t.expect_error("rt pipe | cat", "needs a COMMAND")
    yield
    t.check_eq(sp_pc(written("shell cat {proc} > {out}")), own,
               "shell, with the parked thread selected")
    if t.plain:
        # (A plain command's window is closed by the next command it runs
        # that has a hook: as for a command that failed, see _close_stale.)
        t.check_eq(sp_pc(written('eval "shell cat %s > %s"' % (proc, out))), own,
                   "eval 'shell ...', with the parked thread selected")
    yield
    t.check_regs(t.regs())
    t.run("realm-thread back")
    yield
    _finish(t)


@test(scenarios=("live",))
def test_external_nested_refused(t):
    """Inside a command that shows a parked thread (`rt eval ...`, a
    user-defined command run with `rt`, Python's gdb.execute in `rt`),
    commands that start another program are refused: it would run while the
    host has the parked thread's registers.  (`rt` itself runs them without
    showing it: see test_select_external.)"""
    t.run("define rgt-shell\nshell true\nend")
    t.select("chain0")
    nested = ['rt eval "shell true"', "rt rgt-shell", 'rt eval "edit"',
              'rt python gdb.execute("shell true")']
    if t.try_run("help pipe")[0]:
        nested.append('rt eval "pipe print 1 | cat"')
    for cmd in nested:
        if cmd.startswith("rt python"):  # (Its errors come by way of Python.)
            ok, msg = t.try_run(cmd)
            t.check(not ok, "{} ran".format(cmd))
        else:
            msg = t.expect_error(cmd)
        # (Before gdb 15, a command's error in Python is "Error while executing
        # Python code".)
        if msg.strip() != "Error while executing Python code.":
            t.check(re.search(r"refusing '(shell|edit|pipe)'.*the program it starts", msg),
                    "{}: {}".format(cmd, msg))
    t.run("shell true")  # (Outside them, it runs.)
    if hasattr(sys, "addaudithook"):  # (Python >= 3.8: see _audit.)
        for how in ("system", "popen", "multiprocessing"):
            out = t.rt("python rgt._spawn_in_window('{}')".format(how))
            t.check(re.search(r"rgt-refused: realm-gdb: refusing '(os\.system|"
                              r"subprocess\.Popen|_posixsubprocess\.fork_exec)'.*the "
                              r"program it starts", out),
                    "Python's {} in rt: {}".format(how, out))
            out = t.run("python rgt._spawn_in_window('{}')".format(how))
            t.check_in("rgt-spawned", out, "Python's {} outside rt".format(how))
        # (After a plain command that failed, its window is closed first, as
        # by a hooked command.)
        out = t.run("python rgt._spawn_in_window('system', 'print rgt_no_such_var')")
        t.check_in("rgt-spawned", out, "Python's os.system after a failed plain command")
    t.back()
    yield
    _finish(t)


def _spawn_in_window(how, failing=None):
    """Run inside a window (see test_external_nested_refused): start `true`
    from Python, and say whether it was refused.  First, command `failing`,
    which fails, if given."""
    import subprocess
    if failing is not None:
        try:
            gdb.execute(failing, to_string=True)
        except gdb.error:
            pass
    try:
        if how == "system":
            os.system("true")
        elif how == "multiprocessing":
            # (By _posixsubprocess.fork_exec, which raises no audit event.)
            import multiprocessing.util
            pid = multiprocessing.util.spawnv_passfds(b"/bin/true", ["true"], [])
            os.waitpid(pid, 0)
        else:
            subprocess.call(["true"])
        print("rgt-spawned")
    except RuntimeError as e:
        print("rgt-refused: {}".format(e))


@test(scenarios=("live", "attach", "core"))
def test_terminal_input_refused(t):
    """Inside `rt` or `realm-thread apply`, a command that reads the terminal
    (`explore`'s questions, Python's input() or sys.stdin, Python's REPL) is
    refused at once: its output only shows once it is over, and in a window
    the thread showing the parked one would keep its registers meanwhile."""
    ptr = "{:#x}".format(t.ut("chain0"))
    before = t.regs() if t.scenario != "core" else None
    t.expect_error("realm-thread apply {} explore realm_gdb_test_state".format(ptr),
                   "read the terminal")
    # (gdb <= 12 reports any error from the `python` command so.)
    passes_on = "rgt-probe" in (t.try_run('python raise gdb.GdbError("rgt-probe")')[1])
    for cmd in ("realm-thread apply {} python input()".format(ptr),
                "realm-thread apply {} python import sys; sys.stdin.readline()".format(ptr)):
        t.expect_error(cmd, "read the terminal" if passes_on
                       else "Error while executing Python code")
    if t.scenario != "core":
        t.select("chain0")
        names = re.findall(_BT_LINE + r"([\w:~<>]+)", t.rt("bt"), re.M)
        t.rt("frame {}".format(names.index("rgt_chain_level2")))
        t.expect_error("rt explore widget", "read the terminal")
        t.expect_error("rt pi", "refusing 'python-interactive'")
        t.check_in("= 1", t.rt("print 1"), "rt afterwards")
        t.run("realm-thread back")
    t.check_eq(t.run("python print(input.__name__)").strip(), "input",
               "Python's input() as it was afterwards")
    yield
    if before is not None:
        t.check_regs(before)
        t.check_host_stack()
    _finish(t)


@test(scenarios=("core",))
def test_core_external(t):
    """In a core file, `realm-thread apply PTR pipe COMMAND | SHELL` gives
    SHELL COMMAND's output for the parked thread."""
    if not t.try_run("help pipe")[0]:
        raise TestSkip("this gdb has no pipe")
    out = os.path.join(t.config["tmp"], "external.out")
    t.run("realm-thread apply {:#x} pipe bt 1 | cat > {}".format(t.ut("chain0"), out))
    with open(out) as f:
        t.check_in(PARKED_FRAME0, f.read(), "pipe's SHELL got the parked thread's bt")
    yield


@test(scenarios=("live",))
def test_select_nested(t):
    """Selections nest, each with its own selected frame: `realm-thread
    back` goes back to the one before."""
    t.select("chain0")
    level = re.findall(_BT_LINE + r"([\w:~<>]+)", t.rt("bt"), re.M).index("rgt_chain_level3")
    t.rt("frame {}".format(level))
    yield
    t.select("chain1")
    t.check_eq(_rt_frame_name(t), PARKED_FRAME0, "chain1's frame")
    t.rt("frame {}".format(level))
    t.check_in("= {}".format(t.chain_marker(1)), t.rt("print chain_marker"),
               "chain1's chain_marker")
    yield
    t.check_in("back to " + t.utw("chain0"), t.run("realm-thread back"))
    t.check_eq(_rt_frame_name(t), "rgt_chain_level3", "chain0's frame, remembered")
    t.check_in("= {}".format(t.chain_marker(0)), t.rt("print chain_marker"), "chain0's")
    t.check_in("no parked user thread is selected now", t.run("realm-thread back"))
    t.check_in("no parked user thread is selected", t.run("realm-thread back"))
    yield
    t.continue_to_exit()


@test(scenarios=("live",), interaction="mi", variants=[
    "-exec-next", "-exec-step-instruction", "-exec-finish", "tbreak_continue", "signal_0"])
def test_select_dropped_when_running(t):
    """Where plain commands don't show a selected parked thread (here, in an
    MI session), letting the program run (with any command but a function
    call) ends the selection, since Realm may then resume the parked thread.
    (In a console gdb, `continue` ends it first, and the rest are refused:
    test_plain_continue, test_plain_refused.)"""
    out = t.select("chain0")
    t.check(not t.plain, "plain commands show the selection in MI:\n" + out)
    v = t.variant
    yield
    cmd = v
    if v in ("tbreak_continue", "signal_0"):
        t.run("tbreak realm_gdb_test_opaque")
        cmd = "-exec-continue" if v == "tbreak_continue" else \
            '-interpreter-exec console "signal 0"'
    yield typed(cmd)
    t.check_mi_result("running")
    for _ in t.wait_for_stop():
        yield
    t.expect_error("rt bt", "no parked user thread is selected")
    t.check_not_in("(selected", t.run("realm-thread"), "realm-thread list")
    t.check(gdb.prompt_hook is None, "a prompt hook left in place")
    yield
    t.run("delete")
    t.continue_to_exit()


def _poke(lhs, rhs):
    """`set var LHS = RHS` in the program's memory, from Python, which works
    in the frame selected on the thread as it really is: plain commands
    would show a selected parked thread, whose frame #0 needn't be in C++
    (see PARKED_FRAME0)."""
    target = gdb.parse_and_eval(lhs)
    size = target.type.sizeof
    value = int(gdb.parse_and_eval(rhs).cast(target.type)) & ((1 << (8 * size)) - 1)
    gdb.selected_inferior().write_memory(int(target.address),
                                         value.to_bytes(size, sys.byteorder))


@test(scenarios=("live",), variants=["context", "running", "gone"])
def test_select_stale(t):
    """A selected parked thread that is no longer parked as it was (its saved
    context changed, it is running, it left the registry) isn't shown: `rt`
    says why and drops the selection."""
    ut = "((Realm::UserThread *){:#x})".format(t.ut("chain0"))
    t.select("chain0")
    if t.variant == "context":
        field = _ctx_field(t.ut("chain0"), "pc")
        saved = t.eval_int(field)
        _poke(field, "{:#x}".format(saved + 1))
        why = "has run since it was selected"
    elif t.variant == "running":
        field = "{}->running".format(ut)
        saved = t.eval_int(field)
        _poke(field, "1")
        why = "it is running"
    else:
        # (Unlink it: it is the head, or follows another user thread.)
        uts = t.user_threads()
        i = [u.ptr for u in uts].index(t.ut("chain0"))
        if i == 0:
            field = "Realm::user_thread_registry_head"
        else:
            field = "((Realm::UserThread *){:#x})->registry_next".format(uts[i - 1].ptr)
        saved = t.eval_int("(unsigned long){}".format(field))
        _poke(field, "{}->registry_next".format(ut))
        why = "is no longer a Realm user thread"
    yield
    msg = t.expect_error("rt bt")
    t.check_in(why, msg, "rt on a stale selection")
    t.check_in("no longer selecting it", msg, "rt on a stale selection")
    t.expect_error("rt bt", "no parked user thread is selected")
    t.run("set var {} = {:#x}".format(field, saved) if t.variant != "gone" else
          "set var {} = (Realm::UserThread *){:#x}".format(field, saved))
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["continue", "step", "detach", "return"])
def test_select_refused(t):
    """Commands that would let the program run (or act on the thread showing
    the parked one) are refused in `rt`, saying to leave out `rt`; the
    selection stays."""
    before = t.regs()
    t.select("chain0")
    msg = t.expect_error("rt " + t.variant)
    # (Plain `step` and `return` are refused too while plain commands show
    # the selection: test_plain_refused.)
    hint = ("after 'realm-thread back', without 'rt'"
            if t.plain and t.variant in ("step", "return") else "Run it without 'rt'")
    t.check(re.search(r"refusing '{}'.*{}".format(t.variant, hint), msg), "refusal: " + msg)
    t.check_in("rgt_chain_level3", t.rt("bt"), "rt bt after a refusal")
    yield
    t.check_regs(before)
    t.run("realm-thread back")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_select_prompt(t):
    """While a parked thread is selected, the prompt says which, after
    whatever prompt hook the user has; afterwards the user's hook is put
    back, and one set meanwhile is left alone."""
    mine = lambda current: "(mine) "
    gdb.prompt_hook = mine
    try:
        t.select("chain0")
        t.check_eq(gdb.prompt_hook("(gdb) "), t.remind("chain0") + "(mine) ",
                   "the prompt")
        yield
        t.select("chain1")
        t.check_eq(gdb.prompt_hook("(gdb) "), t.remind("chain1") + "(mine) ",
                   "the prompt, nested")
        t.run("realm-thread back")
        t.check_eq(gdb.prompt_hook("(gdb) "), t.remind("chain0") + "(mine) ",
                   "the prompt, back")
        t.run("realm-thread back")
        t.check(gdb.prompt_hook is mine, "the user's prompt hook afterwards")
        yield
        t.select("chain2")
        other = lambda current: "(other) "
        gdb.prompt_hook = other
        t.run("realm-thread back")
        t.check(gdb.prompt_hook is other, "a prompt hook set meanwhile")
    finally:
        gdb.prompt_hook = None
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_select_prompt_setting(t):
    """gdb makes the prompt a prompt hook returns its `prompt` setting: the
    reminder is in it once however many prompts go by (each step ends at
    one), and goes once nothing is selected, leaving the user's prompt as it
    was, spaces at either end and all; also when the program runs."""
    t.run("set prompt \\040(rgt)\\040")
    mine = " (rgt) "
    t.check_eq(gdb.parameter("prompt"), mine, "the prompt set")
    remind = t.remind
    t.select("chain0")
    yield
    yield
    t.check_eq(gdb.parameter("prompt"), remind("chain0") + mine, "the prompt, after prompts")
    t.select("chain1")
    yield
    t.check_eq(gdb.parameter("prompt"), remind("chain1") + mine, "the prompt, nested")
    t.run("realm-thread back")
    yield
    t.check_eq(gdb.parameter("prompt"), remind("chain0") + mine, "the prompt, back")
    t.run("realm-thread back")
    t.check_eq(gdb.parameter("prompt"), mine, "the prompt, nothing selected")
    yield
    t.check_eq(gdb.parameter("prompt"), mine, "the prompt, after a prompt")
    t.select("chain2")
    yield
    t.continue_to_exit()
    t.check_eq(gdb.parameter("prompt"), mine, "the prompt, once the program ran")


@test(scenarios=("live",))
def test_select_errors(t):
    """What can't be selected or run is refused cleanly, selecting nothing:
    a running user thread, `rt` without a selection, and selecting or `rt`
    from inside `rt` or `realm-thread apply`."""
    t.expect_error("rt bt", "no parked user thread is selected")
    t.expect_error("realm-thread {:#x}".format(t.ut("probe")), "running")
    t.check_not_in("(selected", t.run("realm-thread"))
    t.select("chain0")
    for cmd in ("rt rt bt", "rt realm-thread {:#x}".format(t.ut("chain1")),
                "realm-thread apply {:#x} realm-thread {:#x}".format(
                    t.ut("chain1"), t.ut("chain2")),
                "rt realm-thread back"):
        error, out = t.run_capturing(cmd)
        t.check(error is not None and "inside" in error, "{!r}: {} / {}".format(
            cmd, error, out))
    t.check_eq(_rt_frame_name(t), PARKED_FRAME0, "the selection afterwards")
    t.run("realm-thread back")
    t.check(gdb.prompt_hook is None, "a prompt hook left in place")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_select_from_other_thread(t):
    """`rt` shows the selected parked thread on whichever thread is selected,
    keeping its frame, and leaves every thread's registers as they were."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    before_stop = t.regs()
    before_other = t.regs(other)
    frame1 = _frame_names_of(t, "chain0")[1]
    t.select("chain0")
    t.rt("up")
    # (`thread N` is refused while plain commands show the selection.)
    t.switch_thread(other)
    yield
    t.check_eq(_rt_frame_name(t), frame1, "the frame selected on the other thread")
    t.check_in("rgt_chain_level3", t.rt("bt"), "rt bt on another thread")
    if t.plain:
        t.check_in("rgt_chain_level3", t.run("bt"), "plain bt on another thread")
    t.check_regs(before_other, other, "the other thread's registers")
    t.check_regs(before_stop, t.stop_thread, "the stopped thread's registers")
    t.switch_thread(t.stop_thread)
    t.run("realm-thread back")
    yield
    t.continue_to_exit()


@test(scenarios=("core",))
def test_select_core(t):
    """In a core file, `rt COMMAND` after `realm-thread <ptr>` is COMMAND,
    with the parked thread shown; without one shown it says so."""
    t.expect_error("rt bt", "no parked user thread is shown")
    t.show("chain0")
    out = t.rt("bt")
    t.check_in("<parked Realm user thread {:#x}>".format(t.ut("chain0")), out, "rt bt")
    t.check_in("rgt_chain_level3", out, "rt bt")
    t.select_frame("rgt_chain_level3")
    t.check_in("= {}".format(t.chain_marker(0)), t.rt("print chain_marker"), "rt print")
    t.back()
    t.expect_error("rt bt", "no parked user thread is shown")
    yield


@test(scenarios=("live",), interaction="mi")
def test_mi_select(t):
    """MI's -interpreter-exec console runs `realm-thread <ptr>` and `rt`, as
    an IDE's debug console would."""
    before = t.regs()
    frame1 = _frame_names_of(t, "chain0")[1]
    yield typed('-interpreter-exec console "realm-thread {:#x}"'.format(t.ut("chain0")))
    t.check_mi_result("done")
    yield typed('-interpreter-exec console "rt up"')
    t.check_mi_result("done")
    yield typed('-interpreter-exec console "rt print realm_gdb_test_add(2, 3)"')
    t.check_mi_result("done")
    t.check_eq(_rt_frame_name(t), frame1, "rt's frame")
    t.check_regs(before)
    t.run("realm-thread back")
    yield
    t.continue_to_exit()


# ---- plain commands on a selected parked thread -------------------------

@test(scenarios=("live", "attach"))
def test_plain_commands(t):
    """In a console gdb, after `realm-thread <ptr>`, commands that work with
    the selected frame (`bt`, `frame N`, `up`, `print VAR`, `info locals`,
    `x`, `eval`, `dump`, `skip`, `set var`, `break`, ...) show the parked
    thread as `rt` would,
    keeping the frame selected on it, while nothing in the program changes
    between commands but what they change.  `thread apply all bt` shows
    threads as they are."""
    before = t.regs()
    out = t.select("chain0")
    t.check(t.plain, "plain commands show the selection:\n" + out)
    names = re.findall(_BT_LINE + r"([\w:~<>]+)", t.run("bt"), re.M)
    t.check_names_in_order(CHAIN_FRAMES, names, "bt")
    t.check_eq(names[0], PARKED_FRAME0, "bt's frame #0")
    level = names.index("rgt_chain_level3")
    # (`frame N`, `up` and `down` print the frame they select straight to
    # gdb's console, so `frame` shows it here.)
    t.run("frame {}".format(level))
    t.check_in("rgt_chain_level3", t.run("frame"), "frame N")
    t.check_regs(before)
    yield
    m = t.chain_marker(0)
    t.check_in("= {}".format(m), t.run("print chain_marker"), "print, a command later")
    t.check_in("= {:#x}".format(m), t.run("p/x chain_marker"), "p/x")
    t.check_in(str(m), t.run("output chain_marker"), "output")
    t.check_in("chain_marker = {}".format(m), t.run("info locals"), "info locals")
    t.run("up")
    t.check_in("rgt_chain_level2", t.run("frame"), "up")
    t.run("down")
    t.check_in("rgt_chain_level3", t.run("frame"), "down")
    t.check_in("rgt_chain_level3", t.run("info frame"), "info frame")
    t.check_in(PARKED_FRAME0, t.run("where 1"), "where")
    t.check_in(PARKED_FRAME0, t.run("info stack 1"), "info stack")
    t.check_in("chain_marker", t.run("list"), "list")
    t.check_in("= {}".format(m + 1), t.run("print realm_gdb_test_add(chain_marker, 1)"),
               "a function call")
    t.check_in("= {}".format(m + 2), t.run("call realm_gdb_test_add(chain_marker, 2)"),
               "call")
    t.check_eq(_rt_frame_name(t), "rgt_chain_level3", "rt's frame: the one left selected")
    if "frame level" in t.run("help frame"):  # (gdb >= 8.3)
        t.run("frame level {}".format(level + 1))
        t.check_in("rgt_chain_level2", t.run("frame"), "frame level N")
        t.run("frame function rgt_chain_level3")
        t.check_in("rgt_chain_level3", t.run("frame"), "frame function F")
    sp = lambda out: re.search(r"= (0x[0-9a-f]+)", out).group(1)
    t.check_eq(sp(t.run("print/x $sp")), sp(t.rt("print/x $sp")), "$sp, as rt shows it")
    t.check_eq(re.findall(r"^0x[0-9a-f]+", t.run("x/2gx $sp"), re.M),
               re.findall(r"^0x[0-9a-f]+", t.rt("x/2gx $sp"), re.M), "x, as rt shows it")
    t.check_regs(before)
    t.check_host_stack("Python's view of the thread meanwhile")
    yield
    # Changing a parked frame's variable on its stack lasts; a register
    # doesn't, and realm-gdb says so.
    t.run("set var chain_marker = {}".format(m + 10))
    t.check_in("= {}".format(m + 10), t.run("print chain_marker"), "set var, then print")
    t.check_in("= {}".format(m + 10), t.rt("print chain_marker"), "set var, then rt print")
    t.run("set var chain_marker = {}".format(m))
    t.run("set $rgt_var = 41")
    t.check_in("= 42", t.run("print $rgt_var + 1"), "set $var")
    # (In frame #0: in an outer frame, gdb writes a register where an inner
    # frame saved it, on the stack, if one did.)
    t.run("frame 0")
    out = t.run("set var ${} = 0x1234".format(_callee_saved_reg()))
    t.check(re.search(r"'set variable' changed \${}, which doesn't last".format(
        _callee_saved_reg()), out), "the note on set var of a register:\n" + out)
    t.run("frame {}".format(level))
    # Commands that evaluate an expression in the parked frame's scope.
    t.check_in("= {}".format(m), t.run('eval "print %d", chain_marker'), "eval")
    t.run("init-if-undefined $rgt_init = chain_marker")
    t.check_in("= {}".format(m), t.run("print $rgt_init"), "init-if-undefined")
    dumped = os.path.join(t.config["tmp"], "chain_marker.bin")
    t.run("dump value {} chain_marker".format(dumped))
    with open(dumped, "rb") as f:
        t.check_eq(int.from_bytes(f.read(), sys.byteorder), m, "dump value")
    t.check_in("rgt_chain_level3", t.run("skip"), "skip (the selected frame's function)")
    t.run("skip delete")
    t.run("up")
    t.check_not_in("No symbol", t.run("info vtbl widget"), "info vtbl, one frame up")
    t.run("down")
    # A breakpoint without a location is at the parked frame's.
    breakpoints = len(gdb.breakpoints() or ())
    pc = re.match(r"#\d+\s+(0x[0-9a-f]+) in rgt_chain_level3", t.run("frame"))
    t.check(pc, "frame: rgt_chain_level3, at an address")
    t.run("break")
    bp = gdb.breakpoints()[-1]
    t.check_eq(len(gdb.breakpoints()), breakpoints + 1, "breakpoints after 'break'")
    t.check_in(pc.group(1), t.run("info breakpoints {}".format(bp.number)),
               "where 'break' put it: the parked frame's pc")
    t.run("delete {}".format(bp.number))
    t.run("tbreak realm_gdb_test_opaque")
    t.run("delete {}".format(gdb.breakpoints()[-1].number))
    t.check_regs(before)
    yield
    _check_parked_section(t, t.run("thread apply all bt"))
    t.check_in("rgt_chain_level3", t.run("frame"), "frame, after thread apply all bt")
    t.check_regs(before)
    t.run("realm-thread back")
    out = t.run("bt 1")
    t.check_not_in(PARKED_FRAME0, out, "bt after realm-thread back")
    yield
    _finish(t)


@test(scenarios=("live",))
def test_plain_showing(t):
    """In a console gdb, after `realm-thread <ptr>`, more of the commands
    that look at the selected frame show the parked thread's: `display` and
    `info display`, `ptype`, `whatis`, `printf`, `info address`, `info
    symbol`, `info line`, `info registers`, `disassemble`, `find`, `append`,
    `select-frame`, `up-silently`, `down-silently`, `tvariable`, and
    `thbreak` and `clear` without a location; nothing in the program
    changes."""
    before = t.regs()
    t.select("chain0")
    t.check(t.plain, "plain commands show the selection")
    names = re.findall(_BT_LINE + r"([\w:~<>]+)", t.run("bt"), re.M)
    level = names.index("rgt_chain_level3")
    t.run("frame {}".format(level))
    m = t.chain_marker(0)
    # (`display` prints straight to gdb's console, as `frame N` does.)
    t.run("display chain_marker")
    out = t.run("info display")
    shown = re.search(r"^(\d+):\s+y\s+chain_marker", out, re.M)
    t.check(shown, "info display:\n" + out)
    t.check_not_in("cannot be evaluated", out, "info display")
    t.run("undisplay {}".format(shown.group(1)))
    t.check_in("type = int", t.run("ptype chain_marker"), "ptype")
    t.check_in("type = int", t.run("whatis chain_marker"), "whatis")
    t.check_in(str(m), t.run('printf "%d\\n", chain_marker'), "printf")
    t.check_in('Symbol "chain_marker" is', t.run("info address chain_marker"),
               "info address")
    t.check_in("rgt_chain_level3", t.run("info symbol $pc"), "info symbol")
    t.check_in("rgt_chain_level3", t.run("info line *$pc"), "info line")
    t.check_in("rgt_chain_level3", t.run("info registers pc"), "info registers")
    t.check(re.search(r"code for function \w*rgt_chain_level3", t.run("disassemble")),
            "disassemble")
    t.check_in("1 pattern found", t.run("find &chain_marker, +sizeof(chain_marker), "
                                        "chain_marker"), "find")
    appended = os.path.join(t.config["tmp"], "chain_marker.bin")
    t.run("append value {} chain_marker".format(appended))
    with open(appended, "rb") as f:
        t.check_eq(int.from_bytes(f.read(), sys.byteorder), m, "append value")
    t.run("select-frame 0")
    t.check_in(PARKED_FRAME0, t.run("frame"), "select-frame N")
    t.run("select-frame {}".format(level))
    t.run("up-silently")
    t.check_in("rgt_chain_level2", t.run("frame"), "up-silently")
    t.run("down-silently")
    t.check_in("rgt_chain_level3", t.run("frame"), "down-silently")
    t.run("tvariable $rgt_tv = chain_marker")
    out = t.run("info tvariables")
    t.run("delete tvariable $rgt_tv")
    t.check(re.search(r"\$rgt_tv\s+{}\b".format(m), out), "tvariable:\n" + out)
    breakpoints = len(gdb.breakpoints() or ())
    pc = re.match(r"#\d+\s+(0x[0-9a-f]+) in rgt_chain_level3", t.run("frame"))
    t.check(pc, "frame: rgt_chain_level3, at an address")
    t.run("thbreak")
    # (Python doesn't see hardware breakpoints before gdb 11.)
    hw = re.search(r"^(\d+)\s+hw breakpoint\s+del\s+y\s+(0x[0-9a-f]+)",
                   t.run("info breakpoints"), re.M)
    t.check(hw and int(hw.group(2), 16) == int(pc.group(1), 16),
            "where 'thbreak' put it: the parked frame's pc")
    if hw:
        t.run("delete {}".format(hw.group(1)))
    t.run("break")
    t.run("clear")  # (Of the parked frame's pc, where `break` put one.)
    t.check_eq(len(gdb.breakpoints() or ()), breakpoints, "breakpoints after 'clear'")
    t.check_regs(before)
    t.check_host_stack()
    t.run("realm-thread back")
    yield
    t.continue_to_exit()


_PLAIN_REFUSED = [
    # (command, the name realm-gdb gives it, whether `rt` runs it)
    ("step", "step", False), ("next", "next", False), ("stepi", "stepi", False),
    ("nexti", "nexti", False), ("finish", "finish", False), ("return", "return", False),
    ("until", "until", False), ("advance realm_gdb_test_opaque", "advance", False),
    ("jump realm_gdb_test_opaque", "jump", False), ("signal 0", "signal", False),
    ("watch chain_marker", "watch", False),
    ("thread apply 1 bt", "thread apply", False),
    ("frame apply all print 1", "frame apply all", True), ("faas print 1", "faas", True),
    ('interpreter-exec mi "-exec-next"', "interpreter-exec", False),
    ("queue-signal SIGUSR1", "queue-signal", False), ("pipe print 1 | cat", "pipe", True),
    ("| print 1 | cat", "pipe", True)]


@test(scenarios=("live",), variants=[("bar" if c.startswith("|") else c.split()[0] if c.split()[0]
                                      not in ("set", "frame", "thread", "interpreter-exec")
                                      else c, c) for c, _, _ in _PLAIN_REFUSED])
def test_plain_refused(t):
    """In a console gdb, while a parked thread is selected, commands that
    would act on the selected thread as it really is are refused, saying to
    run `realm-thread back` first (or `rt COMMAND`, where that works); the
    selection stays."""
    name, rt = [(n, r) for c, n, r in _PLAIN_REFUSED if c == t.variant][0]
    if name in ("frame apply all", "faas") and not t.try_run("help faas")[0] or \
            name == "pipe" and not t.try_run("help pipe")[0]:
        raise TestSkip("this gdb has no {}".format(name))
    before = t.regs()
    breakpoints = len(gdb.breakpoints() or ())
    t.select("chain0")
    t.check(t.plain, "plain commands show the selection")
    msg = t.expect_error(t.variant, r"refusing '{}' while {} is selected"
                         .format(re.escape(name), re.escape(t.utw("chain0"))))
    t.check_in("Run 'realm-thread back' first", msg, "the refusal")
    if rt:
        t.check_in("'rt {} ...'".format(name), msg, "the refusal")
    elif name == "watch":
        t.check_in("'rt watch -l EXPR'", msg, "the refusal")
    else:
        t.check_not_in("'rt ", msg, "the refusal")
    t.check_in("rgt_chain_level3", t.run("bt"), "bt after a refusal")
    t.check_regs(before)
    t.check_eq(len(gdb.breakpoints() or ()), breakpoints, "breakpoints")
    yield
    t.run("realm-thread back")
    t.continue_to_exit()  # (the program checks chain_marker is as it was)


@test(scenarios=("live",))
def test_plain_continue(t):
    """In a console gdb, `continue` with parked threads selected ends the
    selections (as `realm-thread back` would), saying so, then runs the
    program as usual."""
    t.select("chain0")
    t.select("chain1")
    yield
    out = t.continue_to_exit()
    t.check(re.search(r"realm-thread: continuing; no longer selecting user threads "
                      r"\d+ \({:#x}\) and \d+ \({:#x}\)\.".format(t.ut("chain0"),
                                                           t.ut("chain1")), out),
            "the note:\n" + out)


@test(scenarios=("core",))
def test_core_thread_note(t):
    """In a core file, a view stays on its thread across `thread N`: `thread`
    to that thread prints gdb's frame #0 of it (the host's own), so it says
    the thread still shows the parked one; and not of a thread that shows
    none."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.run("realm-thread {:#x}".format(t.ut("chain0")))
    note = "thread {} still shows {}".format(t.stop_thread, t.utw("chain0"))
    t.check_in(note, t.run("thread"), "thread")
    t.check_not_in("still shows", t.run("thread {}".format(other)), "thread N, another")
    t.check_in(note, t.run("thread {}".format(t.stop_thread)), "thread N, back")
    t.run("realm-thread back")
    t.check_not_in("still shows", t.run("thread"), "thread, after realm-thread back")
    yield


@test(scenarios=("live",))
def test_plain_thread(t):
    """In a console gdb, `thread N` with parked threads selected switches to
    thread N as usual, then ends the selections (as `realm-thread back`
    would), saying so.  `thread` alone, `thread` of the thread already
    selected, and a `thread N` that fails keep them; the first two say so,
    as gdb may have printed the thread's own frame."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.select("chain0")
    t.select("chain1")
    for cmd in ("thread", "thread {}".format(t.stop_thread)):
        out = t.run(cmd)
        t.check_not_in("no longer selecting", out, cmd)
        t.check_in("still selecting {}: plain commands show it, not "
                   "thread {} as it really is".format(t.utw("chain1"), t.stop_thread),
                   out, cmd)
    t.expect_error("thread 9999", "thread")
    t.check_in(PARKED_FRAME0, t.run("bt 1"), "plain bt: still selected")
    yield
    out = t.run("thread {}".format(other))
    t.check(re.search(r"realm-thread: switched to thread {}; no longer selecting user "
                      r"threads \d+ \({:#x}\) and \d+ \({:#x}\)\.".format(
                          other, t.ut("chain0"), t.ut("chain1")), out),
            "the note:\n" + out)
    t.check_eq(gdb.selected_thread().num, other, "the selected thread")
    t.check_in(t.frame_name(gdb.newest_frame()), t.run("bt 1"), "plain bt on thread N")
    t.expect_error("rt bt", "no parked user thread is selected")
    t.check(gdb.prompt_hook is None, "a prompt hook left in place")
    t.switch_thread(t.stop_thread)
    yield
    t.continue_to_exit()


@test(scenarios=("live", "attach"))
def test_plain_failed(t):
    """A plain command that fails while it shows a selected parked thread
    (so that gdb doesn't run its hookpost) shows it no longer: the next
    prompt puts the thread's own registers back, and so does the next
    command, whichever comes first."""
    before = t.regs()
    t.select("chain0")
    names = re.findall(_BT_LINE + r"([\w:~<>]+)", t.run("bt"), re.M)
    t.run("frame {}".format(names.index("rgt_chain_level3")))
    m = t.chain_marker(0)
    t.expect_error("print rgt_nosuchvar", "No symbol")
    yield  # (the prompt)
    t.check_regs(before, what="registers at the next prompt")
    t.check_in("= {}".format(m), t.run("print chain_marker"), "print, after a failure")
    t.expect_error("x/x 0", "Cannot access memory")
    t.check_in("= {}".format(m), t.run("print chain_marker"), "the next command")
    t.check_regs(before, what="registers after the next command")
    t.expect_error("print rgt_nosuchvar", "No symbol")
    t.check_in("no parked user thread is selected now", t.run("realm-thread back"),
               "realm-thread back, next")
    t.check_regs(before, what="registers after realm-thread back")
    yield
    _finish(t)


@test(scenarios=("live",))
def test_plain_failed_execute_mi(t):
    """A Python script that catches a plain command's failure, then resumes
    the program with gdb.execute_mi('-exec-continue') (whose MI command runs
    no hooks), resumes it with the thread's own registers back."""
    if not hasattr(gdb, "execute_mi"):
        raise TestSkip("this gdb has no gdb.execute_mi")
    before = t.regs()
    t.select("chain0")
    t.expect_error("print rgt_nosuchvar", "No symbol")
    gdb.execute_mi("-exec-continue")
    for _ in t.wait_for_exit():
        yield
    t.check_exited("gdb.execute_mi('-exec-continue')")


@test(scenarios=("live", "attach"), interaction="batch")
def test_plain_batch(t):
    """In batch mode, plain commands show a selected parked thread if gdb
    has gdb.events.gdb_exiting (gdb >= 12: it puts a thread's own registers
    back if gdb exits while a failed command's window is open), and realm-gdb
    says why not otherwise.  There being no prompt, the next command closes a
    failed command's window; and so does gdb exiting, after which an attached
    program carries on cleanly."""
    before = t.regs()
    out = t.select("chain0")
    exiting = hasattr(gdb.events, "gdb_exiting")
    t.check_eq(t.plain, exiting, "plain commands show the selection in batch mode")
    if not exiting:
        t.check_in("in batch mode", out, "why plain commands don't show it")
    bt = t.run("bt 3")
    if exiting:
        t.check_in(USER_SWITCH, bt, "plain bt")
    else:
        t.check_not_in(USER_SWITCH, bt, "plain bt")
    t.expect_error("print rgt_nosuchvar", "No symbol")
    yield
    t.check_in("= 1", t.run("print 1"), "the next command")
    t.check_regs(before)
    t.expect_error("print rgt_nosuchvar", "No symbol")
    if t.scenario == "attach":
        t.release()  # (and gdb exits with the window open)
    else:
        t.continue_to_exit()  # (`continue` closes it first)
    yield


@test(scenarios=("live",), interaction="mi")
def test_mi_plain(t):
    """In an MI session (an IDE's own MI commands run no hooks), plain
    commands show threads as they really are, even with a parked thread
    selected, and nothing is refused; realm-gdb says so on selecting it, and
    `rt` shows it."""
    out = t.select("chain0")
    t.check(not t.plain, "plain commands show the selection in MI")
    t.check_in("this is an MI session", out, "why plain commands don't show it")
    t.check_not_in(USER_SWITCH, t.run("bt 3"), "plain bt")
    t.check_in(USER_SWITCH, t.rt("bt 3"), "rt bt")
    t.run("set var $rgt_var = 1")
    t.run("realm-thread back")
    yield
    t.continue_to_exit()


@test(scenarios=("live",), interaction="mi_new_ui")
def test_mi_new_ui_plain(t):
    """With a second UI open (`new-ui`), whose MI commands run no hooks,
    plain commands show threads as they really are; realm-gdb says so on
    selecting a parked thread."""
    out = t.select("chain0")
    t.check(not t.plain, "plain commands show the selection with a second UI")
    t.check_in("second UI", out, "why plain commands don't show it")
    t.check_not_in(USER_SWITCH, t.run("bt 3"), "plain bt")
    t.run("realm-thread back")
    yield
    t.continue_to_exit()


# ---- register windows ---------------------------------------------------

@test(scenarios=("live",))
def test_window_output_after(t):
    """realm-gdb prints what a command run on a parked thread printed only
    once the thread showing it has its own registers back: gdb's pager may
    wait for input, and meanwhile another UI could run a command."""
    main = sys.modules["__main__"]
    real = gdb.write
    open_writes = []

    def write(text, *args, **kwargs):
        if getattr(main, "_window", None) is not None:
            open_writes.append(text)
        return real(text, *args, **kwargs)

    gdb.write = write
    try:
        yield typed("realm-thread apply {:#x} bt 3".format(t.ut("chain0")))
        yield typed("realm-thread apply all bt 1")
        yield typed("thread apply all bt 1")
    finally:
        gdb.write = real
    t.check_eq(open_writes, [], "output printed while a parked thread was shown")
    t.continue_to_exit()


@test(scenarios=("live", "attach"))
def test_window_close_interrupted(t):
    """Ctrl-C as a window closes (here a KeyboardInterrupt raised in
    realm-gdb's code just before it puts the host's registers back) doesn't
    leave the host with the parked thread's registers: they are put back
    first, then the interruption goes on."""
    main = sys.modules["__main__"]
    name = next(n for n in ("pop_calls", "_pop_abandoned_call", "_abandoned_call")
                if hasattr(main._Window, n))
    real = getattr(main._Window, name)
    raised = []

    def interrupted(self, *args):
        if not raised and sys._getframe(1).f_code.co_name == "close":
            raised.append(True)
            raise KeyboardInterrupt
        return real(self, *args)
    before = t.regs()
    setattr(main._Window, name, interrupted)
    try:
        try:
            gdb.execute("realm-thread apply {:#x} bt 1".format(t.ut("chain0")),
                        to_string=True)
        except (KeyboardInterrupt, gdb.error):
            pass
    finally:
        setattr(main._Window, name, real)
    t.check(raised, "the interruption")
    t.check(main._window is None, "a window left open")
    t.check_regs(before, what="registers after the interrupted close")
    t.check_host_stack()
    yield
    _finish(t)


def _callee_saved_reg():
    """A callee-saved register of this architecture."""
    if _is_aarch64():
        return "x20"
    if _is_x86_64():
        return "r12"
    raise TestSkip("unsupported architecture " + _arch_name())


@test(scenarios=("live", "attach"), variants=[
    "dump", "call", "fault", "set_reg", "set_pc", "set_vector", "set_x87", "set_flags",
    "execute_mi_return", "error"])
def test_window_registers_exact(t):
    """Whatever happens while a parked thread is shown, the thread showing
    it gets back every one of its registers (floating-point and vector ones
    included): after a dump, a function call, one that faults, the user
    changing a register (the PC, a vector or x87 register, the flags), MI's
    -exec-return from Python (gdb >= 14), or a command that fails.  (In the
    attach scenario the thread is in the middle of a system call.)"""
    if t.variant == "execute_mi_return" and not hasattr(gdb, "execute_mi"):
        raise TestSkip("gdb.execute_mi is gdb >= 14")
    if _is_x86_64():
        # (Something no call would leave there.  xmm0 is dead here.)
        t.run("set var $xmm0.v2_int64[0] = 0x1122334455667788")
    before = t.regs()
    v = t.variant
    out = ""
    if v == "dump":
        t.run("thread apply all bt full")
    elif v == "call":
        t.within("chain0", lambda: t.check_eq(
            t.eval_int("realm_gdb_test_add(2, 3)"), 5, "add()"))
    elif v == "fault":
        t.within("chain0", lambda: t.expect_error("print realm_gdb_test_fault()",
                                                  "signal"))
    elif v == "set_reg":
        out = t.apply("chain0", "set var ${} = 0x1234".format(_callee_saved_reg()))
    elif v == "set_pc":
        out = t.apply("chain0", "set var $pc = realm_gdb_test_add")
    elif v == "set_vector":
        out = t.apply("chain0", "set var $xmm1.v2_int64[0] = 0x5" if _is_x86_64()
                      else "set var $v1.d.u[0] = 0x5")
    elif v == "set_x87":
        if not _is_x86_64():
            raise TestSkip("x87 registers are x86's")
        out = t.apply("chain0", "set var $st0 = 1.25")
    elif v == "set_flags":
        out = t.apply("chain0", "set var $eflags = $eflags ^ 0x1" if _is_x86_64()
                      else "set var $cpsr = $cpsr ^ 0x20000000")
    elif v == "execute_mi_return":
        t.apply("chain0", "python gdb.execute_mi('-exec-return')")
    else:
        t.expect_error("realm-thread apply {:#x} print no_such_variable_xyz".format(
            t.ut("chain0")), "no_such_variable_xyz")
    if v.startswith("set_"):
        t.check(re.search(r"changed \$\w+.*, which doesn't last", out),
                "a note that the register change doesn't last:\n" + out)
    else:
        t.check_not_in("doesn't last", out, "no note")
    yield
    t.check_regs(before, what="registers after " + v)
    t.check_host_stack()
    _finish(t)


_WINDOW_SETTINGS = ("scheduler-locking", "unwindonsignal", "unwind-on-terminating-exception",
                    "confirm", "may-call-functions", "direct-call-timeout",
                    "indirect-call-timeout", "unwind-on-timeout", "stop-on-solib-events",
                    "may-insert-breakpoints", "follow-fork-mode", "detach-on-fork")


def _settings(t):
    return {name: t.run("show " + name) for name in _WINDOW_SETTINGS
            if t.supports("show " + name)}


@test(scenarios=("live",), variants=["defaults", "user_values", "error", "interrupted"])
def test_window_settings_restored(t):
    """The settings realm-gdb changes while a parked thread is shown
    (scheduler-locking, unwindonsignal, confirm, call timeouts, ...) are the
    user's again afterwards, whatever they were, and however the command
    ended (an error, Ctrl-C)."""
    if t.variant == "user_values":
        for cmd in ("set scheduler-locking step", "set unwindonsignal on",
                    "set confirm off", "set direct-call-timeout 3",
                    "set unwind-on-timeout off", "set follow-fork-mode child",
                    "set detach-on-fork off"):
            if t.supports(cmd):
                t.run(cmd)
    before = _settings(t)
    ptr = t.ut("chain0")
    if t.variant in ("defaults", "user_values"):
        t.within("chain0", lambda: t.check_in("is on", t.run("show unwindonsignal"),
                                              "unwindonsignal inside"))
        t.run("thread apply all bt")
    elif t.variant == "error":
        t.expect_error("realm-thread apply {:#x} print no_such_variable_xyz".format(ptr))
    else:
        # (Ctrl-C: gdb's SIGINT handler makes the command quit.)
        try:
            t.try_run("realm-thread apply {:#x} python import os, signal; "
                      "os.kill(os.getpid(), signal.SIGINT); "
                      "gdb.execute('bt full')".format(ptr))
        except KeyboardInterrupt:
            pass
    yield
    t.check_eq(_settings(t), before, "settings")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_warnings_to_stderr(t):
    """realm-gdb's WARNINGs go to gdb's stderr, which `-batch-silent` leaves
    (it silences stdout only); its other messages go to stdout, as print()
    puts them."""
    main = sys.modules["__main__"]
    written = []
    real = gdb.write

    def write(text, stream=gdb.STDOUT, *args, **kwargs):
        written.append((text, stream))
        return real(text, stream, *args, **kwargs)
    gdb.write = write
    try:
        main._say("realm-gdb: WARNING: rgt-warning")
    finally:
        gdb.write = real
    t.check(("realm-gdb: WARNING: rgt-warning\n", gdb.STDERR) in written,
            "the warning, written: {}".format(written))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_fork_call(t):
    """A function call in a window that starts another program (`system()`,
    which forks), with the user's `detach-on-fork off`: gdb would keep the
    child stopped, a copy of the program in the middle of the call, while
    the call waited for it to start, until the call timed out (gdb >= 15)
    or for good.  The window's calls let the child go, and the user's
    setting is kept."""
    main = sys.modules["__main__"]
    if not t.supports("show detach-on-fork"):
        raise TestSkip("this gdb has no detach-on-fork")
    if main._arch.probe_reg is not None and main._probe_extended() is not None:
        raise TestSkip("calls are refused here: gdb can't write the extended registers")
    t.run("set detach-on-fork off")
    out = t.apply("chain0", 'print (int)system("true")')
    t.check_in("= 0", out, "system()'s result")
    t.check_eq(len(gdb.inferiors()), 1, "inferiors, afterwards")
    t.check_in("off", t.run("show detach-on-fork"), "detach-on-fork, afterwards")
    t.run("set detach-on-fork on")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["call", "printer"])
def test_window_breakpoint_in_call(t):
    """A function call made while a parked thread is shown can't stop at a
    breakpoint (it would be left on the parked thread's stack): breakpoints
    are ignored meanwhile, and realm-gdb says so; they are as they were
    afterwards."""
    t.run("break realm_gdb_test_opaque")
    t.run("break realm_gdb_test_widget_sum")
    t.run("disable {}".format(gdb.breakpoints()[-1].number))
    bps = [(bp.number, bp.enabled, bp.hit_count) for bp in gdb.breakpoints()]
    if t.variant == "call":
        def inside():
            t.select_frame("rgt_chain_level3")
            t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
                       t.chain_marker(0) + 1, "add()")
        t.within("chain0", inside)
        out = t.last_within_output
    else:
        # (Not `thread apply all bt full`: gdb's own part of it would make
        # calls on the stopped thread's frames, as gdb always does.)
        install_calling_printer("to_string_opaque")
        out = t.run("realm-thread apply all bt full")
        for i in range(NUM_CHAIN):
            t.check_in("widget sum={}".format(t.eval_int(
                "realm_gdb_test_state.widget_sums[{}]".format(i))), out, "widget")
    t.check_in("breakpoints are disabled while a function call runs", out, "the note")
    t.check_eq([(bp.number, bp.enabled, bp.hit_count) for bp in gdb.breakpoints()],
               bps, "breakpoints afterwards")
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",), cpus=2)
def test_window_holds_threads(t):
    """Function calls made while a parked thread is shown run nothing but
    the thread making them (Realm could resume the parked thread on any
    thread, onto the stack the call is using): here the stopped thread,
    which would otherwise go on and finish the program."""
    host = _main_thread(t)
    t.run("thread {}".format(host))

    def inside():
        for _ in range(3):
            t.check_eq(t.eval_int("realm_gdb_test_nap(100000)"), 100000, "nap()")
    t.within("chain0", inside)
    yield
    t.run("thread {}".format(t.stop_thread))
    t.check_eq(t.frame_name(gdb.newest_frame()), "realm_gdb_test_stop",
               "the stopped thread's frame #0")
    t.check_eq(t.eval_int("realm_gdb_test_state.num_finished"), 0, "finished tasks")
    t.continue_to_exit()


@test(scenarios=("live",), cpus=2, variants=[
    "with_locking", "define_locking", "python_locking", "plain_with_locking",
    "with_unwind", "define_unwind", "python_unwind"])
def test_window_settings_kept(t):
    """A command in a window can't take away what its function calls need:
    one that turns scheduler-locking or unwindonsignal off (with `with
    SETTING off -- print f()`, in a user-defined command, from Python)
    still holds the other threads during the call, and has a call that
    faults unwound, so the program runs on to a clean exit."""
    host = _main_thread(t)
    t.run("thread {}".format(host))
    how, _, which = t.variant.rpartition("_")
    setting = "scheduler-locking" if which == "locking" else "unwindonsignal"
    call = "realm_gdb_test_nap(100000)" if which == "locking" else "realm_gdb_test_fault()"
    if how.endswith("with") and not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb has no 'with'")
    before = t.run("show " + setting)
    if how == "define":
        t.run("define rgt-off-call\nset {} off\nprint {}\nend".format(setting, call))
        cmd = "rgt-off-call"
    elif how == "python":
        cmd = 'python gdb.execute("set {} off"); gdb.execute("print {}")'.format(
            setting, call)
    else:
        cmd = "with {} off -- print {}".format(setting, call)
    t.select("chain0")
    if how == "plain_with":
        if not t.plain:
            raise TestSkip("plain commands don't show parked threads here")
    else:
        cmd = "rt " + cmd
    outs = []
    for _ in range(3 if which == "locking" else 1):
        error, out = t.run_capturing(cmd)
        outs.append(out)
        if which == "locking":
            t.check(error is None, "{} failed: {}\n{}".format(cmd, error, out))
            t.check_in("100000", out, cmd)
        else:
            # (Before gdb 15, a command's error in Python is "Error while
            # executing Python code".)
            t.check(error is not None and ("signal" in error or how == "python"),
                    "{}: {}\n{}".format(cmd, error, out))
    t.back()
    if how != "define" and how != "python":
        t.check_eq(t.run("show " + setting), before, setting + " afterwards")
    t.run("set {} off".format(setting))
    t.run("thread {}".format(t.stop_thread))
    t.check_host_stack()
    t.check_eq(t.eval_int("realm_gdb_test_state.num_finished"), 0, "finished tasks")
    yield
    t.continue_to_exit()
    if not how.startswith("plain"):
        # (The first time: the setting the command made then stays the
        # user's, as it would without a window, and the window found it.)
        t.check_in("{} on".format(setting), outs[0], "the note")


@test(scenarios=("live",))
def test_window_catchpoint_in_call(t):
    """A function call in a window doesn't stop at a catchpoint either, even
    where Python can't see them (before gdb 11): it would be abandoned in
    the middle of a system call, which putting the registers back doesn't
    undo.  The program then runs on to a clean exit, without stopping again
    at once where it had stopped."""
    catch = None
    for names in ("nanosleep clock_nanosleep", "nanosleep"):
        ok, out = t.try_run("catch syscall " + names)
        if ok:
            catch = re.search(r"Catchpoint (\d+)", out).group(1)
            break
    if catch is None:
        raise TestSkip("catch syscall: " + out)
    t.select("chain0")
    out = t.rt("print realm_gdb_test_nap(1000)")
    t.check_in("1000", out, "nap()")
    t.check_in("breakpoints are disabled while a function call runs", out, "the note")
    t.back()
    t.check_in("catchpoint     keep y", t.run("info breakpoints " + catch),
               "the catchpoint afterwards")
    t.run("delete " + catch)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_bad_context(t):
    """A parked thread whose saved stack pointer isn't on its stack (a
    corrupt UserThread) isn't shown: a function call would run there."""
    field = _ctx_field(t.ut("chain0"), "sp")
    saved = t.eval_int(field)
    t.run("set var {} = 0x10".format(field))
    yield
    stacks = _split_parked(_parked_section(t, t.run("thread apply all bt")))
    t.check(re.search(r"<not shown: its saved stack pointer 0x10 is outside its stack",
                      stacks[t.ut("chain0")]), "chain0:\n" + stacks[t.ut("chain0")])
    t.expect_error("realm-thread apply {:#x} bt".format(t.ut("chain0")),
                   "stack pointer 0x10 is outside its stack")
    t.run("set var {} = {:#x}".format(field, saved))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_user_calls_off(t):
    """With the user's own may-call-functions off, a command in a window
    that turns it back on for a call (`with may-call-functions on --`) makes
    the call, as it would elsewhere: it isn't taken for a call the window
    refused."""
    if not t.supports("with print pretty -- echo"):
        raise TestSkip("`with` (gdb >= 9)")
    before = t.regs()
    t.run("set may-call-functions off")
    try:
        out = t.apply("chain0", "with may-call-functions on -- print "
                      "realm_gdb_test_add(2, 3)")
        t.check_in("= 5", out, "the call")
        t.check_not_in("refused", out, "a note")
        t.check_in("is off", t.run("show may-call-functions"), "may-call-functions after")
        t.expect_error("realm-thread apply {:#x} print realm_gdb_test_add(2, 3)".format(
            t.ut("chain0")), "may-call-functions is off")
        # A window that refuses calls (simulated here: AMX, or a second UI)
        # still stops such a call.
        main = sys.modules["__main__"]
        with t.patched('_calls_refused', lambda window: "simulated"):
            error, out = t.run_capturing(
                "realm-thread apply {:#x} with may-call-functions on -- print "
                "realm_gdb_test_add(2, 3)".format(t.ut("chain0")))
        t.check(error is not None, "a call made in a window that refuses them:\n" + out)
        t.check_in("refused all the same", out, "the note")
    finally:
        t.run("set may-call-functions on")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_low_headroom(t):
    """With little of its stack free below a parked thread's saved stack
    pointer, function calls are refused while it is shown (Realm's stacks
    have no guard page below them), and realm-gdb says so; on gdbs that
    can't refuse them (gdb 8), it isn't shown."""
    ut = "((Realm::UserThread *){:#x})".format(t.ut("chain0"))
    base = t.eval_int("(unsigned long){}->stack_base".format(ut))
    size = t.eval_int("(unsigned long){}->stack_size".format(ut))
    sp = None

    def find_sp():
        return int(gdb.newest_frame().read_register(
            "sp" if _is_aarch64() else "rsp")) & _U64
    sp = t.within("chain0", find_sp)
    fake = (sp - 4096) & ~0xfff
    t.run("set var {}->stack_base = (void *){:#x}".format(ut, fake))
    t.run("set var {}->stack_size = {}".format(ut, base + size - fake))
    yield
    if t.supports("show may-call-functions"):
        out = t.apply("chain0", "bt 3")
        t.check_eq(_parked_frame_count(out), 3, "bt 3")
        t.check_in("function calls are refused on", out, "the note")
        error, out = t.run_capturing("realm-thread apply {:#x} print realm_gdb_test_add"
                                     "(1, 2)".format(t.ut("chain0")))
        t.check(error is not None, "a call was made:\n" + out)
        t.check_in("under 64 KiB of its stack is free", str(error), "the reason, in the error")
        t.check_in("is on", t.run("show may-call-functions"), "may-call-functions after")
        # Nor if a command in the window turns calls back on: the call is
        # stopped before it starts, and abandoned.
        commands = ["python gdb.execute('set may-call-functions on'); "
                    "gdb.execute('print realm_gdb_test_add(1, 2)')"]
        if t.supports("with print pretty -- echo"):
            commands.append("with may-call-functions on -- print realm_gdb_test_add(1, 2)")
        for cmd in commands:
            error, out = t.run_capturing("realm-thread apply {:#x} {}".format(
                t.ut("chain0"), cmd))
            t.check(error is not None, "{}: a call was made:\n{}".format(cmd, out))
            t.check_in("refused all the same", out, "the note")
            t.check_not_in("WARNING", out, "a warning")
            t.check_not_in("breakpoints are disabled while", out, "the note for calls that ran")
        t.check_in("is on", t.run("show may-call-functions"), "may-call-functions after")
        if t.supports("with print pretty -- echo"):
            # Nor if the user has calls off too.
            t.run("set may-call-functions off")
            try:
                error, out = t.run_capturing(
                    "realm-thread apply {:#x} with may-call-functions on -- print "
                    "realm_gdb_test_add(1, 2)".format(t.ut("chain0")))
            finally:
                t.run("set may-call-functions on")
            t.check(error is not None, "user's calls off: a call was made:\n" + out)
            t.check_in("refused all the same", out, "the note, with the user's calls off")
        t.check_host_stack()
    else:
        t.expect_error("realm-thread apply {:#x} bt".format(t.ut("chain0")),
                       "too little")
    t.run("set var {}->stack_base = (void *){:#x}".format(ut, base))
    t.run("set var {}->stack_size = {}".format(ut, size))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_record(t):
    """While the program is being recorded, parked threads aren't shown
    (that would change a thread's registers, which spoils the execution
    log), and realm-gdb says so; once recording stops, they are."""
    ok, out = t.try_run("record full")
    if not ok:
        raise TestSkip("record full: " + out)
    try:
        _check_not_shown(t, t.run("thread apply all bt"),
                         len([u for u in t.user_threads() if not u.running]),
                         why="being recorded")
        msg = t.expect_error("realm-thread apply {:#x} bt".format(t.ut("chain0")))
        t.check_in("record stop", msg, "realm-thread apply while recording")
    finally:
        t.run("record stop")
    yield
    _check_parked_section(t, t.run("thread apply all bt"))
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_after_exit(t):
    """Once the program has exited, the commands say so cleanly."""
    t.continue_to_exit()
    yield
    for cmd in ("thread apply all bt", "realm-thread"):
        t.run_clean(cmd)
    t.check_in("no process", t.expect_error("realm-thread apply all bt"))


def _signal_gdb_later(sig, delay):
    """Have SIG<sig> sent to gdb in `delay` seconds (from another process:
    gdb doesn't release Python's lock while it waits for a call; started
    from another thread, as realm-gdb refuses to start a program from
    Python's main thread inside a window)."""
    import subprocess
    import threading
    start = threading.Thread(target=subprocess.Popen, args=(
        ["sh", "-c", "sleep {}; kill -{} {}".format(delay, sig, os.getpid())],))
    start.start()
    start.join()


def _interrupted_call():
    """Run inside a register window (see test_window_interrupted_call)."""
    _signal_gdb_later("INT", 0.5)
    try:
        gdb.execute("print realm_gdb_test_nap(5000000)", to_string=True)
        print("not interrupted")
    except (gdb.error, KeyboardInterrupt) as e:
        print("interrupted: {}".format(e))


def _pending_signals(pid):
    """The signals pending for process `pid` and its threads, as a mask
    (bit N-1 for signal N), from /proc (None where there is none)."""
    mask = 0
    try:
        paths = ["/proc/{}/status".format(pid)] + [
            "/proc/{}/task/{}/status".format(pid, tid)
            for tid in os.listdir("/proc/{}/task".format(pid))]
        for path in paths:
            with open(path) as f:
                for line in f:
                    if line.startswith(("SigPnd:", "ShdPnd:")):
                        mask |= int(line.split()[1], 16)
    except (OSError, ValueError):
        return None
    return mask


def _ctrl_c_then_call(failing=False):
    """Run inside a register window (see test_window_ctrl_c_before_call)."""
    import signal
    import time
    gdb.execute("print realm_gdb_test_add(1, 2)", to_string=True)  # (Held after it.)
    os.kill(os.getpid(), signal.SIGINT)  # (Held: the window has it blocked.)
    start = time.time()
    try:
        # (One that fails before it runs, as it can't write its argument, a
        # struct passed by value, on the stack: see window_call_failed_early.)
        gdb.execute("with may-write-memory off -- print ((int(*)(RealmGdbTestState))"
                    "realm_gdb_test_add)(realm_gdb_test_state)" if failing else
                    "print realm_gdb_test_nap(3000000)", to_string=True)
        result = "not interrupted"
    except (gdb.error, KeyboardInterrupt) as e:
        result = "interrupted ({})".format(str(e).splitlines()[0] if str(e) else "Quit")
    print("RGT-CALL {} after {:.1f}s".format(result, time.time() - start))


_quit_function = []


def _define_quit_function():
    """$_rgt_quit(): sets gdb's quit flag (gdb.interrupt), as a Ctrl-C that
    lands while gdb runs does, and returns 0.  gdb hands the quit to Python
    as it next runs Python: the handler of the next function call, say."""
    if _quit_function:
        return

    class Quit(gdb.Function):
        def __init__(self):
            super(Quit, self).__init__("_rgt_quit")

        def invoke(self):
            gdb.interrupt()
            return 0
    _quit_function.append(Quit())


_calls_on_function = []


def _define_calls_on_function():
    """$_rgt_calls_on(): turns may-call-functions on, and returns 0."""
    if _calls_on_function:
        return

    class CallsOn(gdb.Function):
        def __init__(self):
            super(CallsOn, self).__init__("_rgt_calls_on")

        def invoke(self):
            gdb.execute("set may-call-functions on")
            return 0
    _calls_on_function.append(CallsOn())


def _signal_gdb(sig):
    """Run inside a register window (see test_window_signals)."""
    import signal
    import time
    os.kill(os.getpid(), getattr(signal, "SIG" + sig))
    time.sleep(0.2)
    gdb.execute("bt full", to_string=True)


@test(scenarios=("attach",), variants=["TERM", "HUP", "TERM_in_call", "HUP_in_call",
                                       "TERM_output"])
def test_window_signals(t):
    """SIGTERM or SIGHUP to gdb while a parked thread is shown (as an IDE or
    a closed terminal may send), even in the middle of a function call: gdb
    first puts the thread's registers back, then exits, so the attached
    program carries on cleanly.  What the command printed meanwhile is
    printed first ("TERM_output")."""
    sig = t.variant.split("_")[0]
    t.release()
    if t.variant.endswith("in_call"):
        cmd = ("python rgt._signal_gdb_later('{}', 0.5); "
               "gdb.execute('print realm_gdb_test_nap(2000000)')".format(sig))
    elif t.variant.endswith("output"):
        cmd = "python rgt._signal_gdb('{}'); print('RGT-COLL' + 'ECTED')".format(sig)
        t.require_output("RGT-COLLECTED")
    else:
        cmd = "python rgt._signal_gdb('{}')".format(sig)
    yield typed_final("realm-thread apply {:#x} {}".format(t.ut("hot"), cmd))


@test(scenarios=("attach",))
def test_window_interrupted_call(t):
    """Ctrl-C in the middle of a function call made while a parked thread is
    shown: the call is unwound, the thread's registers are put back, and the
    attached program carries on cleanly once released.  (Where realm-gdb
    holds Ctrl-C through a call made from Python, gdb 8.2 and 9.2, it lands
    once the window has closed, and the program carries on all the same.)"""
    main = sys.modules["__main__"]
    before = t.regs()
    cmd = "realm-thread apply {:#x} python rgt._interrupted_call()".format(t.ut("chain0"))
    if main._SIGINT_IN_PYTHON_SAFE:
        out = t.apply("chain0", "python rgt._interrupted_call()")
    else:
        # (Held through the call, the Ctrl-C lands once the window has
        # closed: a Quit of the command, or on gdb 8.2 its error.)
        try:
            error, out = t.run_capturing(cmd)
        except KeyboardInterrupt as e:
            error, out = "KeyboardInterrupt", ""
        t.check("interrupted:" not in out, "the call, interrupted:\n" + out)
        out = "not interrupted ({})".format(error)
    yield
    t.check_regs(before)
    t.check_host_stack()
    t.release()
    if "not interrupted" in out:
        # (Where Ctrl-C would kill gdb if it landed as Python runs, gdb 8.2
        # and 9.2, realm-gdb holds it during a window's call made from
        # Python: see _SIGINT_IN_PYTHON_SAFE.)
        raise TestSkip("realm-gdb holds Ctrl-C during a call made from Python here")
    t.check_in("interrupted:", out, "the interrupted call")


# ---- inferior calls -----------------------------------------------------

@test(scenarios=("live", "attach", "freeze"))
def test_infcall_simple(t):
    """Inferior calls from a parked frame see that frame's variables, and
    leave the parked stack (and the selected frame) as they were."""
    before = t.regs()

    def inside():
        level = t.select_frame("rgt_chain_level3")
        m = t.chain_marker(0)
        t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"), m + 1, "p add()")
        t.check_eq(t.selected_level(), level, "selected frame level after call")
        t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level3",
                   "selected frame after call")
        t.check_chain_stack(0, "stack after call")
        t.check_in("= 5", t.run("call realm_gdb_test_add(2, 3)"), "call")
        t.check_eq(t.eval_int("chain_marker"), m, "chain_marker after calls")
    t.within("chain0", inside)
    yield
    t.check_regs(before)
    t.check_host_stack()
    yield
    _finish(t)


@test(scenarios=("live", "attach"))
def test_infcall_member_function(t):
    """Member-function calls on a parked frame's objects."""
    expected = t.eval_int("realm_gdb_test_state.widget_sums[1]")

    def inside():
        t.select_frame("rgt_chain_level3")
        t.check_eq(t.eval_int("widget.sum()"), expected, "widget.sum()")
        t.check_eq(t.eval_int("realm_gdb_test_widget_sum(&widget)"), expected,
                   "realm_gdb_test_widget_sum(&widget)")
        t.select_frame("rgt_chain_level2")
        t.check_eq(t.eval_int("widget.sum()"), expected, "widget.sum() one frame up")
    t.within("chain1", inside)
    yield
    _finish(t)


@test(scenarios=("live", "attach"))
def test_infcall_libc(t):
    """Inferior calls into malloc and stdio, and gdb's printf/output."""
    m = t.chain_marker(2)

    def inside():
        t.select_frame("rgt_chain_level3")
        out = t.run("print realm_gdb_test_describe(chain_marker)")
        t.check_in('"value={}"'.format(m), out, "describe()")
        out = t.run('printf "%s|%d\\n", realm_gdb_test_describe(7), chain_marker')
        t.check_in("value=7|{}".format(m), out, "printf")
        out = t.run("output realm_gdb_test_add(chain_marker, 2)")
        t.check_in(str(m + 2), out, "output")
    t.within("chain2", inside)
    yield
    _finish(t)


@test(scenarios=("live", "attach"), variants=["64", "1024"])
def test_infcall_stack_hungry(t):
    """An inferior call that needs a lot of stack (KiB in the variant name)
    works from a parked frame (it runs on the parked thread's stack, below
    its frames), and nothing on the parked stack or the host is damaged."""
    kb = int(t.variant)
    expected = sum((((i >> 9) & 0xff) ^ 0x80) - 0x80 for i in range(0, kb * 1024, 512))

    def inside():
        t.select_frame("rgt_chain_level3")
        t.check_eq(t.eval_int("realm_gdb_test_stack_hog({})".format(kb)), expected,
                   "stack_hog({})".format(kb))
        t.check_chain_stack(0, "stack after call")
    t.within("chain0", inside)
    yield
    _finish(t)


@test(scenarios=("live",))
def test_infcall_slow(t):
    """A slow inferior call from a parked frame."""
    def inside():
        t.select_frame("rgt_chain_level3")
        t.check_eq(t.eval_int("realm_gdb_test_nap(200000)"), 200000, "nap()")
    t.within("chain0", inside)
    yield
    _check_parked_section(t, t.run("thread apply all bt"))
    t.check_host_stack()
    t.continue_to_exit()


@test(scenarios=("live",), variants=["on", "off", "step"])
def test_infcall_scheduler_locking(t):
    """Inferior calls from a parked frame, whatever the user's
    scheduler-locking, which is the same afterwards."""
    t.run("set scheduler-locking " + t.variant)

    def inside():
        t.select_frame("rgt_chain_level3")
        t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 5)"),
                   t.chain_marker(0) + 5, "add()")
    t.within("chain0", inside)
    yield
    t.check_in('"{}"'.format(t.variant), t.run("show scheduler-locking"),
               "scheduler-locking afterwards")
    t.run("set scheduler-locking off")
    t.continue_to_exit()


@test(scenarios=("live",), variants=["unwindonsignal_off", "unwindonsignal_on"])
def test_infcall_signal_unwound(t):
    """An inferior call from a parked frame that faults is unwound, whatever
    the user's unwindonsignal (gdb reads it only once the call is over, so
    it is on until the command is), so the program never sees the signal,
    and runs to a clean exit."""
    t.run("set unwindonsignal " + t.variant.rpartition("_")[2])

    def inside():
        t.select_frame("rgt_chain_level3")
        t.expect_error("print realm_gdb_test_fault()", "signal")
        t.check_chain_stack(0, "stack after the faulting call")
        t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level3",
                   "selected frame after the faulting call")
    t.within("chain0", inside)
    yield
    t.check_in("is " + t.variant.rpartition("_")[2], t.run("show unwindonsignal"),
               "unwindonsignal afterwards")
    t.check_host_stack()
    t.continue_to_exit()


@test(scenarios=("live",), variants=["stopped thread", "other thread"])
def test_infcall_from_pretty_printer(t):
    """A pretty-printer that makes inferior calls, used on parked stacks
    (`info locals`, `bt full`, `thread apply all bt full`), prints every
    value correctly, even with scheduler-locking off (the other threads are
    held during the calls anyway, so the parked threads stay put)."""
    install_widget_printer()
    locking = t.run("show scheduler-locking")
    if t.variant == "other thread":
        other = min(th.num for th in gdb.selected_inferior().threads()
                    if th.num != t.stop_thread)
        t.run("thread {}".format(other))
    expected = [t.eval_int("realm_gdb_test_state.widget_sums[{}]".format(i))
                for i in range(NUM_CHAIN)]

    def inside():
        t.select_frame("rgt_chain_level2")
        t.check_in("sum={}".format(expected[0]), t.run("info locals"),
                   "pretty-printed widget")
        out = t.run("bt full")
        t.check_no_python_error("bt full", out)
        t.check_in("widget = widget sum={}".format(expected[0]), out, "bt full")
        t.check_chain_stack(0, "stack after bt full")
        t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2",
                   "selected frame")
    t.within("chain0", inside)
    yield
    for _ in range(3):
        out = t.run("thread apply all bt full")
        _check_parked_section(t, out, full=True)
        for i in range(NUM_CHAIN):
            t.check_in("widget = widget sum={}".format(expected[i]), out,
                       "chain{}'s pretty-printed widget".format(i))
        yield
    t.check_eq(t.run("show scheduler-locking"), locking, "scheduler-locking afterwards")
    if t.variant == "other thread":
        t.run("thread {}".format(t.stop_thread))
    t.continue_to_exit()


def install_widget_printer():
    class WidgetPrinter(object):
        def __init__(self, val):
            self.val = val

        def to_string(self):
            s = gdb.parse_and_eval("realm_gdb_test_widget_sum")(self.val.address)
            return "widget sum={}".format(int(s))

    def lookup(val):
        t = val.type.strip_typedefs()
        if t.code == gdb.TYPE_CODE_STRUCT and t.tag == "RealmGdbTestWidget":
            return WidgetPrinter(val)
        return None

    gdb.pretty_printers.append(lookup)


def install_calling_printer(where):
    """A pretty-printer for RealmGdbTestWidget that calls a function in the
    program in its to_string (or "to_string_opaque": one that calls
    realm_gdb_test_opaque on the way), its constructor (like libstdc++'s for
    std::stringstream) or while listing the widget's children; or ("error")
    one that fails without trying to."""
    def widget_sum(val):
        s = int(gdb.parse_and_eval("realm_gdb_test_widget_sum")(val.address))
        if where == "to_string_opaque":
            gdb.parse_and_eval("realm_gdb_test_opaque")(s)
        return s

    class Printer(object):
        def __init__(self, val):
            self.val = val
            if where == "constructor":
                self.sum = widget_sum(val)

        def to_string(self):
            if where == "error":
                raise gdb.GdbError("rgt-printer-error")
            if where == "constructor":
                return "widget sum={}".format(self.sum)
            if where in ("to_string", "to_string_opaque"):
                return "widget sum={}".format(widget_sum(self.val))
            return "widget"

        if where == "children":
            def children(self):
                yield "id", self.val["id"]
                yield "sum", gdb.Value(widget_sum(self.val))

    def lookup(val):
        t = val.type.strip_typedefs()
        if t.code == gdb.TYPE_CODE_STRUCT and t.tag == "RealmGdbTestWidget":
            return Printer(val)
        return None

    gdb.pretty_printers.append(lookup)


def install_pair_printer():
    """A pretty-printer for RgtPair that, like libstdc++'s, fails on a value
    that is entirely optimized out (which gdb's own printer, but for gdb 10's
    of locals, doesn't give it)."""
    class PairPrinter(object):
        def __init__(self, val):
            self.val = val

        def to_string(self):
            first, second = self.val["first"], self.val["second"]
            if first.is_optimized_out and second.is_optimized_out:
                int(first)  # (raises)
            return "pair({}, {})".format(first, second)

    def lookup(val):
        t = val.type.strip_typedefs()
        if t.code == gdb.TYPE_CODE_STRUCT and t.tag == "RgtPair":
            return PairPrinter(val)
        return None

    gdb.pretty_printers.append(lookup)


def _bt_frames(out):
    """The frames in `bt` output: [(level, line, [(local, value), ...])]."""
    frames = []
    for line in out.splitlines():
        m = re.match(r"#(\d+)\s+(.*)$", line)
        if m:
            frames.append((int(m.group(1)), m.group(2), []))
        elif frames and line.startswith("        ") and " = " in line:
            name, _, value = line.strip().partition(" = ")
            frames[-1][2].append((name, value))
    return frames


@test(scenarios=("live",), variants=[
    "to_string", "constructor", "children", "sliced", "outermost", "dump", "user_off",
    "error"])
def test_printer_calls(t):
    """Pretty-printers that call functions in the program, on a parked
    stack, print as they would on any thread: whether the call is in the
    printer's to_string, constructor or the children it lists; when the
    backtrace stops early (`bt N`) or starts from the outermost frames
    (`bt -N`); in the parked-thread dump.  Every frame is numbered.  If the
    user turned calls off, they stay off; a printer's other errors are shown
    as gdb shows them."""
    if t.variant in ("to_string", "constructor", "children", "error"):
        install_calling_printer(t.variant)
    else:
        install_calling_printer("to_string")
    if t.variant == "user_off":
        if not t.supports("show may-call-functions"):
            raise TestSkip("this gdb has no may-call-functions")
        t.run("set may-call-functions off")
    names = t.within("chain0", t.frame_names)
    level = names.index("rgt_chain_level2")
    if t.variant == "sliced":
        cmd = "bt full {}".format(level + 1)
    elif t.variant == "outermost":
        cmd = "bt full -{}".format(len(names) - level)
    else:
        cmd = "bt full"
    if t.variant == "dump":
        out = t.run("thread apply all bt full")
        sections = [_split_parked(_parked_section(t, out))[t.ut("chain{}".format(i))]
                    for i in range(NUM_CHAIN)]
    else:
        # (A printer's errors are expected here.)
        out = t.run("realm-thread apply {:#x} {}".format(t.ut("chain0"), cmd))
        sections = [out]
    for i, section in enumerate(sections):
        expected = t.eval_int("realm_gdb_test_state.widget_sums[{}]".format(i))
        frames = _bt_frames(section)
        levels = [f[0] for f in frames]
        t.check(levels and levels == list(range(levels[0], levels[0] + len(levels))),
                "frame numbers:\n" + section)
        if t.variant != "outermost":
            t.check_eq(levels[0], 0, "first frame number:\n" + section)
        widget = [val for f in frames for n, val in f[2] if n == "widget"]
        if t.variant == "user_off":
            t.check_not_in("widget sum=", section, "bt with calls turned off by the user")
        elif t.variant == "error":
            t.check_in("rgt-printer-error", section, "the printer's error")
        elif not widget:
            t.fail("no widget in:\n" + section)
        elif t.variant == "children":
            t.check(re.match(r"widget(?: =)? \{id = \d+, sum = " + str(expected) + r"\}$",
                             widget[0]),
                    "widget = " + widget[0])
        else:
            t.check_eq(widget[0], "widget sum={}".format(expected), "widget")
    yield
    if t.variant == "user_off":
        t.check_in("is off", t.run("show may-call-functions"), "may-call-functions")
        t.run("set may-call-functions on")
    t.continue_to_exit()


@test(scenarios=("live",), variants=["bt full", "bt", "thread apply all bt full"])
def test_optimized_out(t):
    """In a live process, values entirely optimized out are printed by gdb's
    own printer, as `<optimized out>` (or, for an argument, in its frame's
    line, `...`), even with a pretty-printer that, like libstdc++'s, can't print
    them.  (gdb 10's own printer gives them to the pretty-printer for
    locals: not checked there.)"""
    install_pair_printer()

    def raw():
        t.select_frame("rgt_opt_wait")
        return {name: t.run("print/r " + name).partition(" = ")[2].strip()
                for name in ("gone", "half", "given")}
    values = t.within("opt", raw)
    gone = sorted(n for n, v in values.items() if v == "<optimized out>")
    if not gone:
        raise TestSkip("the compiler kept every RgtPair value in rgt_opt_wait: {}"
                       .format(values))
    if t.variant.startswith("thread apply"):
        out = _split_parked(_parked_section(t, t.run(t.variant)))[t.ut("opt")]
    else:
        # (Not t.apply: gdb 10's printer raises a Python error here.)
        out = t.run("realm-thread apply {:#x} {}".format(t.ut("opt"), t.variant))
    frames = [f for f in _bt_frames(out) if " rgt_opt_wait" in " " + f[1]]
    t.check(frames, "no rgt_opt_wait frame in:\n" + out)
    _, line, local_values = frames[0]
    shown = dict(local_values)
    shown.update(_frame_line_parts(line)[1])
    for name in gone:
        if name == "given":
            # (An argument is shown in its frame's line, where gdb shows a
            # structure as `...`: `print frame-arguments scalars`.)
            t.check(shown.get(name) in ("...", "<optimized out>"),
                    "argument {}={} in:\n{}".format(name, shown.get(name), out))
        elif t.variant != "bt" and _native_checks():
            t.check_eq(shown.get(name), "<optimized out>", "{} in:\n{}".format(name, out))
    if _native_checks():
        t.check_not_in("Python Exception", out, "Python errors")
    yield
    t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_no_stray_events(t):
    """Looking around parked threads (every frame, its locals, bt, function
    calls, the parked-thread dump, the list) posts no gdb events: gdb 14 and
    older crash on exit if one is still pending."""
    posted = []
    real_post = gdb.post_event

    def counting_post(func):
        posted.append(func)
        return real_post(func)

    gdb.post_event = counting_post
    try:
        for role in ("chain0", "recurse", "hot", "opt"):
            def inside(role=role):
                for level in range(len(t.frames(limit=60))):
                    t.run("frame {}".format(level))
                    t.run_clean("info locals")
                    t.run_clean("info frame")
                t.run_clean("bt full")
                if role == "chain0" and not t.is_core:
                    t.select_frame("rgt_chain_level3")
                    t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
                               t.chain_marker(0) + 1, "add()")
            t.within(role, inside)
            yield
        t.run_clean("thread apply all bt full")
        t.run("realm-thread")
        if t.is_core:
            t.show("chain0")
            t.run_clean("bt full")
            t.back()
        yield
        t.check_eq(len(posted), 0, "events posted")
    finally:
        gdb.post_event = real_post
    if not t.is_core:
        t.continue_to_exit()


@test(scenarios=("core",))
def test_stream_refused_core(t):
    """In a core file, where `realm-thread apply` collects each command's
    output too, a command that would switch gdb's output to another stream
    (here logging's) is refused, as in a window, from Python too."""
    ptr = t.ut("chain0")
    msg = t.expect_error("realm-thread apply {:#x} set logging redirect off".format(ptr),
                         "refusing")
    t.check_in("output gdb collects", msg, "the refusal")
    out = t.run("realm-thread apply {:#x} python exec(\"try: gdb.execute('set logging "
                "redirect off')\\nexcept gdb.error as e: print(e)\")".format(ptr))
    t.check_in("refusing", out, "the refusal, from Python")
    t.run("set logging redirect off")  # (Not refused on its own.)
    t.check_host_stack()
    yield


@test(scenarios=("core",))
def test_infcall_core(t):
    """Inferior calls are refused cleanly in a core file."""
    def inside():
        t.select_frame("rgt_chain_level3")
        t.expect_error("print realm_gdb_test_add(chain_marker, 1)")
        t.check_chain_stack(0)
    t.within("chain0", inside)
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    t.expect_error("print realm_gdb_test_add(chain_marker, 1)")
    t.back()
    t.check_host_stack()
    yield


# ---- refused commands ---------------------------------------------------

@test(scenarios=("live",), variants=[
    "step", "s", "next", "n", "stepi", "si", "nexti", "ni", "finish", "fin",
    "until", "u", "advance realm_gdb_test_add", "jump realm_gdb_test_add",
    "return", ("jump_line", "jump +1"), "continue", "c", "signal 0",
    ("python_continue", "python exec(\"try: gdb.execute('continue')\\nexcept gdb.error "
                        "as e: print(e)\")"),
    ("mi_continue", "interpreter-exec mi \"-exec-continue\""),
    ("mi_next", "interpreter-exec mi \"-exec-next\""),
    ("mi_return", "interpreter-exec mi \"-exec-return\""),
    "kill", "run", "start", "detach", "quit", ("gcore", "gcore /dev/null"),
    "checkpoint", "record", ("target", "target native"),
    ("frame_apply_return", "frame apply 1 return"), ("queue_signal", "queue-signal SIGUSR1"),
    ("execute_mi_continue", "python exec(\"try: gdb.execute_mi('-exec-continue')\\nexcept "
                            "gdb.error as e: print(e)\")"),
    ("execute_mi_trace_find", "python exec(\"try: gdb.execute_mi('-trace-find', 'none')"
                              "\\nexcept gdb.error as e: print(e)\")"),
    ("record_full_restore", "record full restore /nonexistent/rgt-record"),
    ("new_ui", "new-ui mi /dev/null"), ("dump_me", "maint dump-me"),
    ("file", "file /bin/true"), ("exec_file", "exec-file /bin/true"),
    ("symbol_file", "symbol-file"), "nosharedlibrary",
    ("delete_checkpoint", "delete checkpoint 1"),
    ("static_markers", "info static-tracepoint-markers"), ("strace_m", "strace -m rgt"),
    ("set_sysroot", "set sysroot /"), ("set_endian", "set endian big"),
    ("set_tdesc", "set tdesc filename /dev/null"),
    ("_gdb_execute_mi", "python exec(\"import _gdb\\ntry: _gdb.execute_mi('-exec-continue')"
                        "\\nexcept gdb.error as e: print(e)\")"),
    ("set_architecture", "set architecture auto"), ("set_processor", "set processor auto"),
    ("set_osabi", "set osabi auto"), ("info_threads", "info threads"),
    ("symbols_from_memory", "add-symbol-file-from-memory 0"), ("tfind", "tfind none"),
    ("flash_erase", "flash-erase"), ("logging_redirect", "set logging redirect off"),
    ("python_logging", "python exec(\"try: gdb.execute('set logging redirect off')\\nexcept "
                       "gdb.error as e: print(e)\")"),
    ("tui_enable", "tui enable"), ("layout_src", "layout src"), ("tui_scroll", "+"),
    ("set_host_charset", "set host-charset ASCII"), ("set_charset", "set charset ASCII"),
    ("with_architecture", "with architecture i386:x64-32 -- print $pc"),
    ("with_arch", "with arch i386 -- print 1"), ("with_endian", "with endian big -- print 1"),
    ("with_osabi", "with osabi none -- print 1"),
    ("with_host_charset", "with host-charset ASCII -- print 1"),
    ("eval_with_architecture", 'eval "with architecture auto -- print 1"'),
    # (Deprecated: kept out of `complete` and `help target`, see
    # _DEPRECATED_GUARDED.)
    ("target_record", "target record"),
    ("reverse_step", "reverse-step"), ("detach_inferiors", "detach inferiors 1"),
    ("kill_inferiors", "kill inferiors 1"), ("record_full", "record full"),
    ("record_btrace", "record btrace"), "starti", ("attach", "attach 1"),
    ("core_file", "core-file"), "disconnect", "load", ("monitor", "monitor help"),
    ("maint_packet", "maint packet qC"), ("remove_symbol_file", "remove-symbol-file -a 0"),
    ("set_solib_search_path", "set solib-search-path /"),
    ("unset_tdesc", "unset tdesc filename"),
    ("execute_mi_detach", "python exec(\"try: gdb.execute_mi('-target-detach')\\nexcept "
                          "gdb.error as e: print(e)\")"),
    ("execute_mi_exec_file", "python exec(\"try: gdb.execute_mi('-file-exec-and-symbols', "
                             "'/bin/true')\\nexcept gdb.error as e: print(e)\")"),
    ("execute_mi_interpreter", "python exec(\"try: gdb.execute_mi('-interpreter-exec', "
                               "'console', 'continue')\\nexcept gdb.error as e: print(e)\")")])
def test_refused_in_window(t):
    """Commands that would let the program run or let go of it (or act on
    the thread showing the parked one, which has its registers) are refused
    while a parked thread is shown, so the program is never run with them;
    afterwards they work as usual, and the program runs to a clean exit.
    So are those that would keep that thread's own registers from being put
    back (`set architecture`, `set host-charset`, ...), set by `with` too,
    which runs no `set` hook, and before gdb 12 didn't put the architecture
    back."""
    v = t.variant
    if v.startswith("frame apply") and not t.supports("frame apply 1 print 1"):
        raise TestSkip("this gdb has no 'frame apply'")
    if "execute_mi" in v and not hasattr(gdb, "execute_mi"):
        raise TestSkip("gdb.execute_mi is gdb >= 14")
    if v in ("checkpoint", "delete checkpoint 1") and not t.supports("help checkpoint"):
        raise TestSkip("this gdb has no checkpoints")
    if v in ("tui enable", "layout src", "+") and not t.supports("help tui enable"):
        raise TestSkip("this gdb has no TUI")
    if v.startswith(("add-symbol-file-from", "flash", "tfind", "set osabi")) and \
            not t.supports("help " + " ".join(v.split()[:2 if v.startswith("set") else 1])):
        raise TestSkip("this gdb has no '{}'".format(v.split()[0]))
    if "with " in v and not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb has no 'with'")
    if "osabi" in v and not t.supports("help set osabi"):
        raise TestSkip("this gdb has no 'set osabi'")
    command = {"detach inferiors 1": "detach inferiors", "kill inferiors 1": "kill inferiors",
               "remove-symbol-file -a 0": "remove-symbol-file", "starti": "starti",
               "target record": "target record", "maint packet qC": "maint packet"}.get(v)
    if command is not None and not t.supports("help " + command):
        raise TestSkip("this gdb has no '{}'".format(command))
    before = t.regs()
    yield
    error, out = t.run_capturing("realm-thread apply {:#x} {}".format(t.ut("chain0"), v))
    # (From Python, the refusal is an exception, printed by the test's code.)
    t.check(error is not None or v.startswith("python"),
            "{!r} wasn't refused:\n{}".format(v, out))
    t.check(re.search(r"refusing '", (error or "") + out),
            "{!r} failed, but not with a refusal: {}\n{}".format(v, error, out))
    t.check(not t.exited, "the program exited")
    yield
    t.check_regs(before, what="registers after '{}'".format(v))
    t.check_host_stack()
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_annotate(t):
    """At annotation level 2 (as some front ends set it), which splits `info
    breakpoints` up among annotations, a window's call still runs with the
    user's breakpoints disabled: a dprintf in the called function doesn't
    print, nor count a hit."""
    t.run('dprintf realm_gdb_test_add,"RGT-DPRINTF\\n"')
    t.run("set annotate 2")
    try:
        out = t.apply("chain0", "print realm_gdb_test_add(1, 2)")
    finally:
        t.run("set annotate 0")
    t.check_not_in("RGT-DPRINTF", out, "the dprintf, in the window's call")
    table = t.run("info breakpoints")
    t.check_not_in("already hit", table[table.find("dprintf"):], "the dprintf's hits")
    t.run("delete")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_max_value_size(t):
    """A `max-value-size` set by a command in a window below a register's
    size (an AVX-512 register is 64 bytes) doesn't keep the window from
    putting the host's registers back, and is the user's afterwards."""
    main = sys.modules["__main__"]
    frame = gdb.newest_frame()
    sizes = [v.type.sizeof for v in main._read_regs(frame, main._snapshot_names(frame)).values()]
    if not sizes or max(sizes) <= 16:
        raise TestSkip("no register over 16 bytes here")
    before = t.regs()
    error, out = t.run_capturing("realm-thread apply {:#x} set max-value-size 16".format(
        t.ut("chain0")))
    t.check(error is None and "could not put back" not in out,
            "the window: {}\n{}".format(error, out))
    t.check(not main._unrestored, "threads still to put back")
    t.check_in("16 bytes", t.run("show max-value-size"), "max-value-size afterwards")
    t.run("set max-value-size unlimited")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_refusal_messages(t):
    """Refusals say what the command would have done."""
    for cmd in ("return", "step", "continue", "detach"):
        msg = t.expect_error("realm-thread apply {:#x} {}".format(t.ut("chain0"), cmd))
        t.check(re.search(r"refusing '{}' in 'realm-thread apply', where thread {} has user "
                          r"thread \d+ \({:#x}\)'s registers until the command is over: "
                          r"the program mustn't run".format(cmd, t.stop_thread,
                                                             t.ut("chain0")),
                          msg), "refusal of {}: {}".format(cmd, msg))
    # (Once, not for each parked thread.)
    error, out = t.run_capturing("realm-thread apply all continue")
    t.check(error is not None and "refusing 'continue'" in str(error) and
            "refusing" not in out, "realm-thread apply all continue: {}\n{}".format(
                error, out))
    msg = t.expect_error("realm-thread apply {:#x} generate-core-file /dev/null".format(
        t.ut("chain0")))
    t.check_in("the core file would have them as thread {}'s own".format(t.stop_thread),
               msg, "generate-core-file's reason")
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["step", "next", "finish", "stepi", "tbreak", "signal"])
def test_commands_after(t):
    """After parked threads have been shown, commands work as usual on the
    program as it really is (nothing is left shown)."""
    _check_parked_section(t, t.run("thread apply all bt"))
    t.within("chain0", lambda: t.eval_int("realm_gdb_test_add(1, 2)"))
    yield
    v = t.variant
    if v == "finish":
        t.run("finish")
        t.check_eq(t.frame_name(gdb.newest_frame()), "realm_gdb_test_wait_for_debugger",
                   "frame #0 after finish")
    elif v == "tbreak":
        t.run("tbreak realm_gdb_test_opaque")
        t.run("continue")
        t.check_eq(t.frame_name(gdb.newest_frame()), "realm_gdb_test_opaque",
                   "stopped in")
        t.check_not_in("rgt_chain_level3", t.frame_names()[:3], "stack at the new stop")
    elif v == "signal":
        t.run("delete")
        t.continue_to_exit("signal 0")
        return
    else:
        t.run(v)
        names = t.frame_names()
        t.check(names[0] in ("realm_gdb_test_stop", "realm_gdb_test_wait_for_debugger"),
                "frame #0 after {}: {}".format(v, names[:3]))
        t.check_not_in("rgt_chain_level3", names, "stack after " + v)
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_gcore_after(t):
    """A core file written after showing parked threads has the threads' real
    registers."""
    core = os.path.join(t.config["tmp"], "after.core")
    _check_parked_section(t, t.run("thread apply all bt"))
    t.run("gcore " + core)
    yield
    t.continue_to_exit()
    yield
    t.run("core-file " + core)
    t.scenario = "core"
    num = t.find_thread_with_frame("rgt_hot_probe")
    t.check(num is not None, "no thread in the core is in rgt_hot_probe")
    t.run("thread {}".format(num))
    t.check_host_stack("stack in the core")


@test(scenarios=("live",))
def test_kill_and_rerun(t):
    """`kill` after showing parked threads, then `run` again: everything
    works on the new process."""
    _check_parked_section(t, t.run("thread apply all bt"))
    t.select("chain0")
    yield
    t.run("kill")
    t.check(gdb.prompt_hook is None, "the prompt hook after kill")
    yield
    t.run_program()
    t.expect_error("rt bt", "no parked user thread is selected")
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop", "stopped in")
    t.check_host_stack("stack after re-run")
    t.stop_thread = gdb.selected_thread().num
    t.find_roles()
    yield
    t.within("chain2", lambda: t.check_chain_stack(2))
    _check_parked_section(t, t.run("thread apply all bt"))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_unrestored(t):
    """If a window can't put the host's registers back (simulated here: it
    has never been seen, but a gdb that can't write some register would do
    it), realm-gdb says so, and refuses to let the program run, or to let go
    of it, until they are back; `kill`, the other way out it suggests, still
    works."""
    main = sys.modules["__main__"]
    real = main._put_back

    def put_back(saved, *args):  # (Puts them back, but says it couldn't.)
        return real(saved, *args) or [sorted(saved)[0]]
    main._put_back = put_back
    try:
        out = t.run("realm-thread apply {:#x} bt 1".format(t.ut("chain0")))
        t.check_in("could not put back", out, "the warning")
        # (Not `quit`, refused only at a terminal: see window_unrestored_quit.)
        for cmd in ("continue", "detach"):
            t.expect_error(cmd, "aren't all back")
        t.run("kill")
    finally:
        main._put_back = real
    t.check(not main._unrestored, "a killed program's threads still to put back")
    if main._can_refuse_calls:
        t.check_in(" on", t.run("show may-call-functions"), "may-call-functions afterwards")
    yield
    t.run_program()
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop", "stopped in")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_close_fails(t):
    """If putting a host's registers back fails unexpectedly (a bug in
    realm-gdb, simulated here), the window isn't forgotten: realm-gdb says
    so, and refuses to let the program run until 'realm-thread back' has put
    them back."""
    before = t.regs()
    main = sys.modules["__main__"]
    real = main._put_back

    def put_back(saved, *args):
        raise RuntimeError("simulated")
    main._put_back = put_back
    try:
        error, out = t.run_capturing("realm-thread apply {:#x} bt 1".format(t.ut("chain0")))
        out += error or ""
        t.check_in("could not put back thread", out, "the warning")
        t.check_in("realm-gdb failed: simulated", out, "why")
        t.expect_error("continue", "aren't all back")  # (It tries again first.)
        # Function calls, which let the program run, are refused meanwhile
        # (gdb >= 9), and so is `delete checkpoint`, which may make one.
        if main._can_refuse_calls:
            t.check_in("function calls are refused", out, "the note on calls")
            t.expect_error("print realm_gdb_test_add(1, 2)", "may-call-functions is off")
        else:
            t.check_in("don't make function calls", out, "the warning on calls")
        if "delete checkpoint" in main._hooked.get("guarded", ()):
            t.expect_error("delete checkpoint 1", "aren't all back")
    finally:
        main._put_back = real
    out = t.run("realm-thread back")
    t.check_in("put back thread", out, "'realm-thread back'")
    t.check(not main._unrestored, "threads still to put back")
    if main._can_refuse_calls:
        t.check_in("function calls are allowed again", out, "the note on calls")
        t.check_in(" on", t.run("show may-call-functions"), "may-call-functions afterwards")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live", "attach"))
def test_window_unwritable_register(t):
    """A register a window can't put back (simulated here; gdb 13 and older
    can't write the SSE registers where the kernel's XSAVE area is bigger
    than gdb's buffer, as with AMX) doesn't keep it from putting back the
    registers after it: $mxcsr, and $orig_rax, which says whether the system
    call the thread stopped in is restarted.  `realm-thread back` puts the
    rest back once it can."""
    if not _is_x86_64():
        raise TestSkip("x86-64 registers")
    regs = ["mxcsr"] + (["orig_rax"] if not _is_freebsd() else [])
    before = {r: t.eval_int("$" + r) for r in regs + ["xmm0.v2_int64[0]"]}
    changes = ["set var $xmm0.v2_int64[0] = {:#x}".format(before["xmm0.v2_int64[0]"] ^ 0x5a5a)]
    changes += ["set var ${} = {:#x}".format(r, before[r] ^ 0x6000) for r in regs]
    main = sys.modules["__main__"]
    real = main._write_reg

    def write_reg(name, value, *args):
        if name.startswith("xmm"):
            raise gdb.error("Couldn't write extended state status: Bad address.")
        real(name, value, *args)
    main._write_reg = write_reg
    try:
        out = t.run("realm-thread apply {:#x} python {}".format(
            t.ut("chain0"), "; ".join("gdb.execute({!r})".format(c) for c in changes)))
        why = "own registers ($xmm0: Couldn't write extended state status: Bad address)"
        t.check_in(why, out, "the warning")
        for r in regs:
            t.check_eq(t.eval_int("$" + r), before[r], "$" + r + ", put back all the same")
        msg = t.expect_error("continue", "aren't all back")
        t.check_in("own registers aren't all back ($xmm0: Couldn't write extended state "
                   "status: Bad address)", msg, "the refusal")
        out = t.run("realm-thread back")
        t.check_in("still could not put back thread", out, "'realm-thread back', failing")
    finally:
        main._write_reg = real
    t.check_in("put back thread", t.run("realm-thread back"), "'realm-thread back'")
    t.check(not main._unrestored, "threads still to put back")
    for r, v in sorted(before.items()):
        t.check_eq(t.eval_int("$" + r), v, "$" + r)
    yield
    t.check_host_stack()
    _finish(t)


@test(scenarios=("live", "attach"), variants=[None, "failing"])
def test_window_calls_unwritable(t):
    """Where gdb can't write the extended registers (simulated here; gdb 13
    and older can't with AMX on), function calls are refused while a parked
    thread is shown, as neither gdb nor the window could put back what a call
    changes.  realm-gdb says why once (also if the command that found it
    failed, in the "failing" variant), and with gdb's error when a call is
    refused; everything else works, and the program carries on cleanly."""
    if not _is_x86_64():
        raise TestSkip("x86-64 registers")
    main = sys.modules["__main__"]
    real = main._probe_extended
    main._probe_extended = lambda: "Couldn't write extended state status: Bad address."
    main._extended_unwritable.clear()
    note = ("function calls are refused" if main._can_refuse_calls else
            "can't refuse function calls")
    try:
        if t.variant == "failing":
            error, out = t.run_capturing("realm-thread apply {:#x} print rgt_nosuchvar"
                                         .format(t.ut("chain0")))
            t.check(error is not None, "print rgt_nosuchvar:\n" + out)
            t.check_in(note, out, "the note, after a failure")
        out = t.apply("chain0", "bt 3")
        t.check_eq(_parked_frame_count(out), 3, "bt 3")
        if t.variant == "failing":
            t.check_not_in("Couldn't write", out, "the note, again")
        else:
            t.check_in("(\"Couldn't write extended state status: Bad address\")", out,
                       "the note")
            t.check_in(note, out, "the note")
        if main._can_refuse_calls and t.variant is None:
            t.check_not_in("refused", t.apply("chain0", "bt 1"), "the note, again")
            why = "can't write the x87, SSE and AVX registers"
            msg = t.expect_error("realm-thread apply {:#x} print realm_gdb_test_add(1, 2)"
                                 .format(t.ut("chain0")), "may-call-functions is off")
            t.check_in(why, msg, "the error")
            t.select("chain0")
            t.check_in(why, t.expect_error("rt print realm_gdb_test_add(1, 2)"), "rt's error")
            if t.plain:
                t.expect_error("print realm_gdb_test_add(1, 2)", "may-call-functions is off")
                # (Said as its window closes, here at the next command: once.)
                note = "function calls are refused while a parked Realm user thread is shown"
                t.check_in(note, t.run("print 1"), "the reason, after a plain call failed")
                t.expect_error("print realm_gdb_test_add(1, 2)", "may-call-functions is off")
                t.check_not_in(note, t.run("print 1"), "the reason, again")
            t.run("realm-thread back")
            t.check_in("is on", t.run("show may-call-functions"), "may-call-functions after")
    finally:
        main._probe_extended = real
    t.check(not main._unrestored, "threads to put back")
    t.check_host_stack()
    yield
    _finish(t)


@test(scenarios=("live",))
def test_window_extended_check_once(t):
    """Whether gdb can write the extended registers is checked once per
    process, by the first window that would allow function calls, and the
    check leaves them as they were."""
    if not _is_x86_64():
        raise TestSkip("x86-64 registers")
    main = sys.modules["__main__"]
    before = t.regs()
    t.check_eq(main._probe_extended(), None, "the check, on its own")
    t.check_regs(before, what="registers after the check")
    real = main._probe_extended
    made = []

    def probe():
        made.append(1)
        return real()
    main._probe_extended = probe
    main._extended_unwritable.clear()
    try:
        for _ in range(3):
            t.apply("chain0", "bt 1")
        t.run("thread apply all bt")
        t.within("chain0", lambda: t.check_eq(t.eval_int("realm_gdb_test_add(1, 2)"), 3,
                                              "add()"))
    finally:
        main._probe_extended = real
    t.check_eq(len(made), 1, "checks made")
    t.check_regs(before)
    yield
    t.continue_to_exit()


@test(scenarios=("live", "attach"), variants=["thread", "python", "inferior"])
def test_window_apply_switches_thread(t):
    """A command run on each parked thread in turn that selects another
    thread (`thread N`, Python's InferiorThread.switch(), `inferior N`)
    doesn't take the window with it: each parked thread is still shown on
    the thread that was selected, whose own registers are put back, and the
    other thread's are never written."""
    host = gdb.selected_thread().num
    other = next(th.num for th in gdb.selected_inferior().threads() if th.num != host)
    before, before_other = t.regs(), t.regs(other)
    switch = {"thread": "gdb.execute('thread {}', to_string=True)".format(other),
              "python": "[th.switch() for th in gdb.selected_inferior().threads() "
                        "if th.num == {}]".format(other),
              "inferior": "gdb.execute('inferior 2', to_string=True)"}[t.variant]
    if t.variant == "inferior":
        t.run("add-inferior")
    out = t.run("realm-thread apply all python print('RGT-SHOWN', gdb.selected_thread().num, "
                "hex(int(gdb.newest_frame().read_register('{}')))); {}".format(
                    "sp" if _is_aarch64() else "rsp", switch))
    if t.variant == "inferior":
        t.check_eq(gdb.selected_inferior().num, 1, "the selected inferior afterwards")
        t.run("remove-inferiors 2")
    shown = re.findall(r"RGT-SHOWN (\d+) (0x[0-9a-f]+)", out)
    t.check(len(shown) > 1, "parked threads shown:\n" + out)
    t.check_eq(set(n for n, _ in shown), {str(host)}, "the threads they were shown on")
    t.check_eq(len(set(sp for _, sp in shown)), len(shown), "their stack pointers")
    t.check_eq(gdb.selected_thread().num, host, "the selected thread afterwards")
    t.check_regs(before_other, other, "thread {}'s registers".format(other))
    t.check_regs(before)
    t.check_host_stack()
    yield
    _finish(t)


@test(scenarios=("live",), variants=["refused", "stopped", "refused limit", "stopped limit"])
def test_window_abandoned_calls(t):
    """A function call left stopped while one of several parked threads is
    shown (one refused, so stopped at its first instruction; or, in the
    "stopped" variants, one stopped at a breakpoint made during the call,
    passed for realm-gdb's own, as nothing else stops one now: see
    test_window_call_new_breakpoint) is popped before the next thread is shown
    over it: gdb keeps no dummy frame, whose breakpoint it would put into
    that thread's stack whenever the program ran.  realm-gdb counts the
    calls abandoned, and the program runs on to a clean exit.  So too in the
    "limit" variants, with `set backtrace limit 1`, which would hide the
    call's dummy frame (frame #1) from a walk of the stack."""
    main = sys.modules["__main__"]
    if not hasattr(gdb, "InferiorCallPreEvent"):
        raise TestSkip("this gdb has no inferior call events")
    kind, _, limit = t.variant.partition(" ")
    made = []

    def stop_it(event):
        if isinstance(event, gdb.InferiorCallPreEvent) and main._window is not None:
            main._making_stop = True
            try:
                made.append(gdb.Breakpoint("realm_gdb_test_add", internal=True))
            finally:
                main._making_stop = False
    real_room = main._call_room
    if kind == "refused":
        if not t.supports("show may-call-functions") or \
                not t.supports("with print pretty -- echo"):
            raise TestSkip("this gdb can't refuse calls, or has no 'with'")
        main._call_room = lambda ut: False
        cmd = "with may-call-functions on -- print realm_gdb_test_add(1, 2)"
    else:
        gdb.events.inferior_call.connect(stop_it)
        cmd = "print realm_gdb_test_add(1, 2)"
    if limit:
        t.run("set backtrace limit 1")
    try:
        error, out = t.run_capturing("realm-thread apply all " + cmd)
        if limit:
            t.check_eq(gdb.parameter("backtrace limit"), 1, "the backtrace limit after")
    finally:
        if limit:
            t.run("set backtrace limit unlimited")
        main._call_room = real_room
        if kind == "stopped":
            gdb.events.inferior_call.disconnect(stop_it)
        for bp in made:
            if bp.is_valid():
                bp.delete()
    failed = len(re.findall(r"<\w+ failed:", out))
    t.check(error is not None and failed > 1, "calls left stopped:\n" + out)
    if kind == "refused":
        t.check_in("refused all the same", out, "the note")
        t.check_not_in("WARNING", out, "a warning")
    else:
        # (Or, where the breakpoint is at the function's first instruction, as
        # it can be in an optimized build, a note that none of them ran.)
        t.check("{} function calls made on thread {}".format(failed, t.stop_thread) in out or
                "{} function calls were stopped at their first instruction, and "
                "abandoned before any of them ran".format(failed) in out,
                "the warning:\n" + out)
    t.check_not_in("id={", t.run("maint print dummy-frames"), "dummy frames left")
    t.check_not_in("call dummy", t.run("maint info breakpoints"),
                   "call-dummy breakpoints left")
    t.check(not main._unrestored, "threads still to put back")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_apply_recording_began(t):
    """`realm-thread apply all` stops showing parked threads if the command
    run on one of them began recording (simulated here: every command that
    can is refused), which would log the next one's registers as they were
    put in place."""
    main = sys.modules["__main__"]
    real, began = main._recording, []

    def recording():
        # (Once it is asked between parked threads: from then on.)
        if sys._getframe(1).f_code.co_name == "_apply_each":
            began.append(1)
        return bool(began)
    main._recording = recording
    try:
        error, out = t.run_capturing("realm-thread apply all print 1")
    finally:
        main._recording = real
    t.check(error is not None and "can't show the other parked user threads" in error and
            "being recorded" in error, "the error:\n{}\n{}".format(error, out))
    t.check_eq(out.count("User thread "), 1, "parked threads shown")
    t.check(main._window is None, "the window, afterwards")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_inline_entered(t):
    """The host stopped at the start of an inlined call, which gdb shows as
    not entered yet (so that `step` steps into it): gdb forgets that once a
    window has written the host's registers, and shows the inlined function
    as frame #0 from then on.  realm-gdb says so, and the frame the user had
    selected stays selected."""
    main = sys.modules["__main__"]
    t.run("break realm_gdb_test_inline_start")
    t.run("set scheduler-locking on")  # (The other threads held.)
    try:
        t.run_capturing("print realm_gdb_test_inline_start(1)")  # (Stops at the breakpoint.)
        if not main._skipped_inline(gdb.newest_frame()):
            t.run("step")  # (Onto the inlined call, at -O0, past what sets it up.)
        skipped = main._skipped_inline(gdb.newest_frame())
        if skipped:
            out = t.apply("chain0", "bt 1")
            t.check_in("now shows inlined rgt_inline_first as frame #0", out, "the note")
            t.check_in("#1  ", t.run("frame"), "the frame selected, which is #1 now")
            t.check_in("realm_gdb_test_inline_start", t.run("frame"),
                       "the frame selected")
            # (Not twice: the window was the one to lose it.)
            t.check_not_in("now shows inlined", t.apply("chain0", "bt 1"), "the note again")
        t.run("frame 0")
        for _ in range(3):  # (Out of the inlined call: before gdb 17, `return` can't.)
            if gdb.newest_frame().type() != gdb.INLINE_FRAME:
                break
            t.run("finish")
        t.run("return")  # (The user's call, popped: confirm is off.)
    finally:
        t.run("set scheduler-locking replay")
        t.run("delete")
    t.check_eq(t.run("maint print dummy-frames").count("id={"), 0, "calls left")
    if not skipped:
        raise TestSkip("gdb shows this inlined call as entered already")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["internal", "user"])
def test_window_call_new_breakpoint(t):
    """A breakpoint made during a function call made while a parked thread
    is shown, by Python code that runs meanwhile (an extension's handler of
    inferior_call), is disabled for the call too, as the others are: the
    call returns, and the breakpoint is enabled again afterwards.  (Popping
    a call stopped in an inlined function fails before gdb 17.)"""
    main = sys.modules["__main__"]
    if not hasattr(gdb, "InferiorCallPreEvent"):
        raise TestSkip("this gdb has no inferior call events")
    made = []

    def make(event):
        if isinstance(event, gdb.InferiorCallPreEvent) and main._window is not None \
                and not made:
            made.append(gdb.Breakpoint("realm_gdb_test_add",
                                       internal=t.variant == "internal"))
    gdb.events.inferior_call.connect(make)
    try:
        out = t.apply("chain0", "print realm_gdb_test_add(1, 2)")
    finally:
        gdb.events.inferior_call.disconnect(make)
    try:
        t.check(len(made) == 1, "a breakpoint made during the call")
        t.check_in("= 3", out, "the call's value")
        t.check_not_in("WARNING", out, "a warning")
        t.check(made[0].is_valid() and made[0].enabled, "the breakpoint, enabled again")
        t.check(not main._disabled_breakpoints, "breakpoints left disabled")
    finally:
        for bp in made:
            if bp.is_valid():
                bp.delete()
    t.check_not_in("id={", t.run("maint print dummy-frames"), "dummy frames left")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_syscall_stop(t):
    """Windows on a thread stopped at a syscall catchpoint, on the way into
    a system call and out of it (usleep's, the stopped thread made to loop
    in it): `realm-thread apply` shows a parked thread there, and its
    function call is refused, saying why (made at the entry, it would make
    the system call fail, on x86 Linux, or be made with the call's
    arguments: on aarch64 the call returned 2, and on FreeBSD the program
    died); the thread's own registers are as they were, and the program
    finishes cleanly."""
    if sys.platform.startswith("freebsd") and not hasattr(gdb, "BP_CATCHPOINT"):
        raise TestSkip("this gdb gives Python no catchpoints, to tell such a stop by")
    # (By its address: at the system call, in libc's assembly stub, gdb takes
    # the program's variables for symbols without debug information.)
    hold = "set var *(int *)&realm_gdb_test_hold = {}"
    t.run(hold.format(1))
    caught = []
    for name in ("clock_nanosleep", "nanosleep"):
        ok, out = t.try_run("catch syscall " + name)
        m = re.search(r"Catchpoint (\d+)", out) if ok else None
        if m:
            caught.append(m.group(1))
    if not caught:
        t.run(hold.format(0))
        raise TestSkip("no syscall catchpoints for usleep's system call here")
    stop = t.stop_thread
    try:
        for when in ("entry", "return"):
            t.run("continue")
            t.check_eq(gdb.selected_thread().num, stop, "the thread stopped at the " + when)
            # (libc's frames may have no names.)
            t.check(any("sleep" in name for name in t.frame_names()[:3]),
                    "the {} stop: {}".format(when, t.frame_names()[:3]))
            before = t.regs()
            if t.supports("show may-call-functions"):
                t.expect_error("realm-thread apply {:#x} print realm_gdb_test_add(1, 2)".format(
                    t.ut("chain0")), "a syscall catchpoint stopped it at a system call")
                t.check_in(PARKED_FRAME0, t.apply("chain0", "bt 1"), "bt, at the " + when)
            else:
                # (gdb 8 can't refuse calls: the thread isn't shown there.)
                error, out = t.run_capturing("realm-thread apply {:#x} bt 1".format(
                    t.ut("chain0")))
                t.check_in("a syscall catchpoint stopped thread", (error or "") + out,
                           "not shown, at the " + when)
            t.check_regs(before, what="registers at the " + when)
    finally:
        t.try_run("delete " + " ".join(caught))
        t.try_run(hold.format(0))
    yield
    t.continue_to_exit()


@test(scenarios=("attach",), variants=["exit", "source"])
def test_window_python_exit(t):
    """Python's SystemExit while a parked thread is shown (`python
    sys.exit()`, or a sourced script's), which ends gdb 14 and older at once
    without gdb's exit event: the thread's registers are put back first, so
    the attached program carries on cleanly."""
    t.release()
    # (gdb 15 reports the exception and carries on, so the window closes as
    # usual; gdb 16 and later quit.)
    t.allow_output("Error occurred in Python: 0")
    t.allow_output("Python Exception <class 'SystemExit'>: 0")
    if t.variant == "exit":
        cmd = "python import sys; sys.exit(0)"
    else:
        path = os.path.join(t.config["tmp"], "exits.py")
        with open(path, "w") as f:
            f.write("import sys\nsys.exit(0)\n")
        cmd = "source " + path
    yield typed_final("realm-thread apply {:#x} {}".format(t.ut("hot"), cmd))


@test(scenarios=("live",))
def test_window_unrestored_selected(t):
    """With a parked thread selected, `realm-thread back` puts back a
    register a window couldn't (simulated here, as in
    test_window_unwritable_register): realm-gdb's own register writes don't
    show the selection."""
    if not _is_x86_64():
        raise TestSkip("x86-64 registers")
    before = t.eval_int("$xmm0.v2_int64[0]")
    main = sys.modules["__main__"]
    real = main._write_reg

    def write_reg(name, value, *args):
        if name.startswith("xmm"):
            raise gdb.error("Couldn't write extended state status: Bad address.")
        real(name, value, *args)
    t.select("chain0")
    main._write_reg = write_reg
    try:
        out = t.rt("python gdb.execute('set var $xmm0.v2_int64[0] = {:#x}')".format(
            before ^ 0x1234))
        t.check_in("could not put back thread {}".format(t.stop_thread), out, "the warning")
    finally:
        main._write_reg = real
    out = t.run("realm-thread back")
    t.check_in("put back thread {}'s own registers".format(t.stop_thread), out,
               "'realm-thread back'")
    t.check_not_in("still could not", out, "'realm-thread back'")
    t.check(not main._unrestored, "threads still to put back")
    t.run("realm-thread back")
    t.check_eq(t.eval_int("$xmm0.v2_int64[0]"), before, "$xmm0")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_breakpoint_kinds(t):
    """Breakpoints of every kind are disabled for a function call made while
    a parked thread is shown (hardware ones and dprintfs too, which Python
    doesn't see before gdb 11, or at all), and are as they were afterwards,
    hit counts and enable counts included."""
    t.try_run("hbreak realm_gdb_test_add")
    t.run('dprintf realm_gdb_test_add,"RGT-DPRINTF\\n"')
    num = re.search(r"Breakpoint (\d+)", t.run("break realm_gdb_test_add")).group(1)
    t.run("enable count 3 " + num)
    before = t.run("info breakpoints")
    t.check_in("disable after next 3 hits", before, "the enable count")
    out = t.apply("chain0", "print realm_gdb_test_add(1, 2)")
    t.check_in("= 3", out, "add()")
    t.check_not_in("RGT-DPRINTF", out, "the dprintf's output")
    t.check_in("breakpoints are disabled while a function call runs", out, "the note")
    t.check_eq(t.run("info breakpoints"), before, "info breakpoints afterwards")
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_guard_error(t):
    """If realm-gdb's check of a command fails unexpectedly (simulated
    here), a command that may have to be refused is refused rather than let
    through; elsewhere it runs as usual."""
    main = sys.modules["__main__"]
    real = main._guard

    def guard(cmd):
        raise TypeError("rgt-simulated")
    before = t.regs()
    main._guard = guard
    try:
        msg = t.expect_error("realm-thread apply {:#x} continue".format(t.ut("chain0")),
                             "check of it failed")
        t.check_in("rgt-simulated", msg, "the refusal")
        t.check(not t.exited, "the program exited")
        t.check_regs(before)
        t.check_host_stack()
        t.run("info threads")  # (Not one to refuse: runs as usual.)
    finally:
        main._guard = real
    yield
    t.continue_to_exit()


def _blocked_signals():
    """The signals gdb's main thread (this one) has blocked, from
    /proc/self/task/TID/status's SigBlk, as a bit mask."""
    with open("/proc/self/task/{}/status".format(_gettid())) as f:
        return int(re.search(r"^SigBlk:\s*([0-9a-f]+)", f.read(), re.M).group(1), 16)


def _gettid():
    import ctypes
    return ctypes.CDLL(None).syscall(186)  # (SYS_gettid, x86-64)


def _caught_signals():
    """The signals gdb has a handler for (Linux)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("SigCgt:"):
                return int(line.split()[1], 16)
    raise TestSkip("no SigCgt in /proc/self/status")


@test(scenarios=("live",))
def test_window_holds_fatal_signals(t):
    """Signals whose default action would end gdb (SIGUSR1, SIGALRM,
    SIGSTKFLT, the real-time ones, ...) or stop it (SIGTSTP), as well as
    SIGTERM and SIGHUP, are held while a parked thread is shown, and have
    their own actions again afterwards."""
    import signal
    if not os.path.exists("/proc/self/status"):
        raise TestSkip("no /proc/self/status")
    bits = 0
    for name in ("SIGUSR1", "SIGALRM", "SIGXCPU", "SIGTERM", "SIGHUP"):
        bits |= 1 << (getattr(signal, name) - 1)
    if signal.getsignal(signal.SIGTSTP) == signal.SIG_DFL:  # (Held if so: Ctrl-Z.)
        bits |= 1 << (signal.SIGTSTP - 1)
    bits |= 1 << (16 - 1)  # (SIGSTKFLT, which Python names only from 3.11)
    bits |= 1 << (int(signal.SIGRTMIN) + 3 - 1)
    before = _caught_signals() & bits
    inside = t.within("chain0", _caught_signals) & bits
    t.check_eq(hex(inside), hex(bits), "held inside")
    t.check_eq(hex(_caught_signals() & bits), hex(before), "afterwards")
    # Ctrl-C (SIGINT) is blocked inside, but for the command itself, where
    # Ctrl-C can land while Python runs a command (see _interruptibly), and
    # for the function calls made there; and not afterwards.
    if _is_x86_64():
        sigint = 1 << (signal.SIGINT - 1)
        main = sys.modules["__main__"]
        t.check_eq(bool(t.within("chain0", _blocked_signals) & sigint),
                   not main._SIGINT_IN_PYTHON_SAFE, "SIGINT blocked inside")
        t.check(not _blocked_signals() & sigint, "SIGINT blocked afterwards")
        _check_sigint_in_calls(t, sigint)
    yield
    t.continue_to_exit()


def _check_sigint_in_calls(t, sigint):
    """window_holds_fatal_signals: SIGINT is let through while a window's
    call runs, also one whose string argument makes gdb call malloc first
    (which ends before the window's own call runs), and blocked again once
    it is over; and after a call that fails before it runs (gdb sends no
    inferior_call_post), blocked again at the next parked thread's
    command."""
    seen = []

    def called(event):  # (After realm-gdb's handler.)
        seen.append((type(event).__name__, bool(_blocked_signals() & sigint)))

    def inside():
        try:
            gdb.execute('print ((int(*)(const char *, int))realm_gdb_test_add)("rgt", 1)',
                        to_string=True)
        except gdb.error as e:
            if "may-call-functions is off" not in str(e):
                raise
            return None  # (Calls refused here: gdb can't write the extended registers.)
        return _blocked_signals() & sigint
    gdb.events.inferior_call.connect(called)
    try:
        after = t.within("chain0", inside)
    finally:
        gdb.events.inferior_call.disconnect(called)
    if after is None:
        return
    # (gdb calls malloc for the string once the window's call has begun: as
    # malloc's ends, the window's has yet to run.  Where Ctrl-C kills gdb
    # if it lands as Python runs, it is held throughout a call made from
    # Python, as this one is.)
    main = sys.modules["__main__"]
    ends = [blocked for name, blocked in seen if name == "InferiorCallPostEvent"]
    want = [False, True] if main._SIGINT_IN_PYTHON_SAFE else [True, True]
    t.check(ends == want,
            "SIGINT blocked as malloc's call, then the window's, ended: {}".format(seen))
    t.check(after, "SIGINT blocked after the call, inside")
    if not t.supports("with print pretty -- echo"):
        return
    del _probe_states[:]
    t.run_capturing("realm-thread apply all python rgt._failed_call_probe()")
    main = sys.modules["__main__"]
    t.check(len(_probe_states) >= 2, "the probe on two parked threads: {}".format(
        _probe_states))
    if len(_probe_states) >= 2:
        # (Let through for its command alone, as for the first's, where it
        # is: not left let through for a call.)
        t.check_eq(bool(_probe_states[1][1] & sigint), not main._SIGINT_IN_PYTHON_SAFE,
                   "SIGINT blocked at the second parked thread")
        t.check_eq(_probe_states[1][0], 0, "calls counted at the second parked thread")
    t.check_eq(main._infcall_depth, 0, "calls counted afterwards")


# What _failed_call_probe saw at each parked thread: (calls counted in
# progress, the signals blocked).
_probe_states = []


def _failed_call_probe():
    """Run inside a register window (see _check_sigint_in_calls): note
    what it sees, then make a call that fails before it runs."""
    main = sys.modules["__main__"]
    _probe_states.append((main._infcall_depth, _blocked_signals()))
    try:
        # (A struct passed by value: see window_call_failed_early.)
        gdb.execute("with may-write-memory off -- print ((int(*)(RealmGdbTestState))"
                    "realm_gdb_test_add)(realm_gdb_test_state)", to_string=True)
    except gdb.error:
        pass


@test(scenarios=("live",))
def test_window_call_failed_outside(t):
    """A function call that fails before it runs, made outside any window
    with no prompt after it (gdb sends no inferior_call_post then), doesn't
    leave later windows' calls unprotected: their breakpoints are disabled,
    and nothing of theirs is left on the parked thread's stack."""
    main = sys.modules["__main__"]
    if not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb has no 'with'")
    t.run("break realm_gdb_test_add")
    own = t.regs()
    # (A struct passed by value: see window_call_failed_early.)
    t.expect_error("with may-write-memory off -- print ((int(*)(RealmGdbTestState))"
                   "realm_gdb_test_add)(realm_gdb_test_state)")
    # (On aarch64 that call leaves the thread's link register changed, as
    # gdb writes it before the struct, and doesn't put it back on the error:
    # gdb's own doing, outside any window, which would crash the program
    # later.  Put back what it changed.)
    for name, value in sorted(t.regs().items()):
        if value != own[name] and isinstance(own[name], int):
            t.run("set var ${} = {:#x}".format(name, own[name]))
    before = t.regs()
    out = t.apply("chain0", "print realm_gdb_test_add(1, 2)")
    t.check_in("= 3", out, "the window's call")
    t.check_not_in("abandoned", out, "a call stopped at the breakpoint")
    t.check_eq(main._infcall_depth, 0, "calls counted in progress")
    t.check_regs(before)
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_unrestored_host(t):
    """A thread whose own registers a window couldn't put back (simulated)
    shows no parked thread again until they are back: a second window would
    save the parked thread's registers as the host's own, and put those back
    later."""
    main = sys.modules["__main__"]
    before = t.regs()
    real = main._put_back

    def put_back(saved, *args):
        raise RuntimeError("simulated")
    main._put_back = put_back
    try:
        t.run_capturing("realm-thread apply {:#x} bt 1".format(t.ut("chain0")))
        # (Another parked thread: chain0's own is refused anyway, as its
        # stack pointer is the host's now.)
        msg = t.expect_error("realm-thread apply {:#x} bt 1".format(t.ut("chain1")))
        t.check_in("aren't all back", msg, "the refusal of a second window")
        t.check_eq(len(main._unrestored), 1, "windows left to put back")
    finally:
        main._put_back = real
    t.check_in("put back thread", t.run("realm-thread back"), "'realm-thread back'")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["other", "host"])
def test_window_pop_baseline(t):
    """A function call of the user's, stopped at a breakpoint before a
    window opens, is left alone when the window pops a call it refused
    (simulated) on the same thread: only the window's own is popped.  On
    the host itself ("host"), the window's own is still popped: the parked
    thread's frames, shown over the user's call, don't reach it."""
    main = sys.modules["__main__"]
    if not main._can_refuse_calls or not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb can't refuse calls, or has no 'with'")
    return (_pop_baseline_host if t.variant == "host" else _pop_baseline_other)(t)


def _pop_baseline_other(t):
    main = sys.modules["__main__"]
    other = [th.num for th in gdb.selected_inferior().threads()
             if th.num != t.stop_thread][0]
    t.run("break realm_gdb_test_add")
    t.run("thread {}".format(other))
    # (Stops at the breakpoint; the other threads held, the host among them.)
    t.run("set scheduler-locking on")
    try:
        t.run_capturing("print realm_gdb_test_add(5, 6)")
    finally:
        t.run("set scheduler-locking replay")
    t.run("thread {}".format(t.stop_thread))
    t.run("delete")
    count = lambda: t.run("thread apply {} bt".format(other)).count(
        "<function called from gdb>")
    t.check_eq(count(), 1, "the user's call on thread {}".format(other))
    with t.patched('_calls_refused', lambda window: "simulated"):
        t.run_capturing("realm-thread apply {:#x} thread apply {} with may-call-functions "
                        "on -- print realm_gdb_test_add(1, 2)".format(t.ut("chain0"), other))
    t.check_eq(count(), 1, "the user's call on thread {}, afterwards".format(other))
    t.run("thread {}".format(other))
    t.run("return")  # (The user's call, popped: confirm is off.)
    t.run("thread {}".format(t.stop_thread))
    t.check_host_stack()
    yield
    t.continue_to_exit()


def _pop_baseline_host(t):
    main = sys.modules["__main__"]
    t.run("break realm_gdb_test_add")
    # (Stops at the breakpoint; the other threads held.)
    t.run("set scheduler-locking on")
    try:
        t.run_capturing("print realm_gdb_test_add(5, 6)")
    finally:
        t.run("set scheduler-locking replay")
    t.run("delete")
    dummies = lambda: t.run("maint print dummy-frames").count("id={")
    t.check_eq(dummies(), 1, "the user's call")
    with t.patched('_calls_refused', lambda window: "simulated"):
        error, out = t.run_capturing("realm-thread apply {:#x} with may-call-functions on "
                                     "-- print realm_gdb_test_add(1, 2)".format(
                                         t.ut("chain0")))
    t.check_in("refused all the same", out + (error or ""), "the note on the refused call")
    t.check_eq(dummies(), 1, "calls left, afterwards (the window's popped)")
    t.check(not main._unrestored, "threads still to put back")
    t.check_in("<function called from gdb>", t.run("bt"), "the user's call, afterwards")
    t.run("return")  # (The user's call, popped: confirm is off.)
    t.check_eq(dummies(), 0, "calls left")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_nested_external(t):
    """A command that starts another program, run in a window by something
    other than realm-gdb while realm-gdb runs a command of its own (here
    another extension's register_changed handler), is refused as it is in
    the window itself."""
    if not hasattr(gdb.events, "register_changed"):
        raise TestSkip("no gdb.events.register_changed")
    errors = []

    def handler(event):
        try:
            gdb.execute("shell true", to_string=True)
            errors.append(None)
        except gdb.error as e:
            errors.append(str(e))
    gdb.events.register_changed.connect(handler)
    try:
        t.apply("chain0", "bt 1")
    finally:
        gdb.events.register_changed.disconnect(handler)
    t.check(errors, "the handler ran")
    t.check(all(e is not None and "refusing 'shell'" in e for e in errors),
            "shell in the window: {}".format(errors[:3]))
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_reload_in_thread_apply(t):
    """realm-gdb.py loaded again in the middle of realm-gdb's own `thread
    apply all` (by an embedded copy, say) leaves plain commands showing a
    selected parked thread afterwards."""
    main = sys.modules["__main__"]
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    t.run("thread apply all source {}".format(t.config["script"]))
    t.check_eq(main._plain_suppressed, 0, "plain commands suppressed")
    t.check_in(PARKED_FRAME0, t.run("bt 1"), "plain bt 1 afterwards")
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_left_open(t):
    """A window left open (simulated here: a Ctrl-C just as the host's
    registers were to be put back) is closed by the next prompt, which is
    never inside one."""
    return _window_left_open(t)


@test(scenarios=("live",), interaction="mi")
def test_window_left_open_mi(t):
    """In MI, which has no prompt event, a window left open (simulated) is
    closed as soon as gdb is back in its event loop, before the next
    command: else the front end's -exec-continue would run the program with
    the parked thread's registers.  (Not before gdb 15: see
    plain_failed_posted.)"""
    if not hasattr(gdb, "interrupt"):
        raise TestSkip("gdb 9 to 14 die exiting with a posted event left: none is posted")
    return _window_left_open(t)


def _window_left_open(t):
    main = sys.modules["__main__"]
    real = main._Window.close
    raised = []

    def close(self):
        if not raised:
            raised.append(True)
            raise KeyboardInterrupt
        return real(self)
    before = t.regs()
    main._Window.close = close
    try:
        try:
            gdb.execute("realm-thread apply {:#x} bt 1".format(t.ut("chain0")),
                        to_string=True)
        except (KeyboardInterrupt, gdb.error):  # (gdb 8: "Error occurred in ...")
            pass
    finally:
        main._Window.close = real
    t.check(raised, "the interruption")
    t.check(main._window is not None, "the window, left open")
    yield
    t.check(main._window is None, "the window, after the prompt")
    t.check_regs(before)
    t.check_host_stack()
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_failed_top_level(t):
    """A plain command run at the top level (typed, or from -ex), whose
    failure leaves its window open, can't be told from Python code run
    after it (simulated here by running its hook as gdb would): that code
    can't resume the program with gdb.execute_mi, and a `source` run after
    it closes the window first, as other hooked commands do."""
    main = sys.modules["__main__"]
    before = t.regs()
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    main._hook("print", False, ())  # (a typed `print`, which then fails)
    t.check(main._window is not None, "the failed command's window")
    if hasattr(gdb, "execute_mi"):
        try:
            gdb.execute_mi("-exec-continue")
            t.fail("-exec-continue wasn't refused")
        except gdb.error as e:
            t.check_in("refusing '-exec-continue'", str(e), "the refusal")
        t.check(not t.exited, "the program exited")
        t.check(main._window is not None, "the window, after the refusal")
    main._hook("source", False, ())  # (a typed `source` after it)
    t.check(main._window is None, "the window, after 'source'")
    main._hook("print", False, ())  # (again)
    main._hook("inferior", False, ())  # (`inferior N` after it, which would be undone)
    t.check(main._window is None, "the window, after 'inferior'")
    t.check_regs(before)
    t.back()
    yield
    _finish(t)


@test(scenarios=("live", "attach"), variants=["warning", "error"])
def test_window_internal_problem(t):
    """A gdb internal warning or error while a parked thread is shown
    (`maint internal-warning` stands in for a bug in gdb) ends the command
    rather than gdb, which `confirm off` would otherwise make it do at once;
    the settings are the user's again afterwards, and the program carries
    on cleanly.  (An internal error is shown to a plain command here, whose
    output isn't collected: gdb 16 and older die of one under
    gdb.execute(..., to_string=True), with "Recursive internal problem",
    with or without realm-gdb.)"""
    show = "maint show internal-{} quit".format(t.variant)
    setting = t.run(show)
    before = t.regs()
    if t.variant == "warning":
        error, out = t.run_capturing("realm-thread apply {:#x} maint internal-warning "
                                     "rgt-simulated".format(t.ut("chain0")))
        t.check_in("rgt-simulated", (error or "") + out, "gdb's report")
    else:
        t.select("chain0")
        if not t.plain:
            raise TestSkip("plain commands don't show parked threads here")
        try:
            # (Into a file: the driver fails a test whose gdb prints one.)
            error, out = t.run_capturing('eval "maint internal-error rgt-simulated"')
            t.check(error is not None, "the internal error didn't end the command")
        except KeyboardInterrupt:  # (The quit it ends a command with.)
            pass
        t.back()
    t.check_eq(t.run(show), setting, show + " afterwards")
    t.check_regs(before)
    t.check_host_stack()
    yield
    _finish(t)


@test(scenarios=("live",))
def test_window_solib_events(t):
    """With `stop-on-solib-events` on, a function call made while a parked
    thread is shown that loads a library (dlopen) doesn't stop at the event
    (it would be abandoned holding the dynamic loader's lock, and the
    program would hang at exit), and the program runs on to a clean exit."""
    # (A library the program hasn't loaded, which this system has.)
    names = ("libresolv.so.2", "libz.so.6", "libutil.so.9", "libz.so.1")
    t.run("set stop-on-solib-events 1")
    try:
        for name in names:
            error, out = t.run_capturing("realm-thread apply {:#x} print (long)dlopen("
                                         "\"{}\", 2)".format(t.ut("chain0"), name))
            if error is not None or not re.search(r"= 0$", out, re.M):
                break
    finally:
        t.run("set stop-on-solib-events 0")
    if error is not None and "No symbol" in error:
        raise TestSkip("no dlopen here: " + error)
    if error is None and re.search(r"= 0$", out, re.M):
        raise TestSkip("none of {} here".format(", ".join(names)))
    t.check(error is None and re.search(r"= [1-9]\d*", out), "dlopen({}):\n{}\n{}".format(
        name, error, out))
    t.check_not_in("abandoned", out, "a warning")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_reload_in_call(t):
    """realm-gdb.py loaded again during a function call made in a window (as
    the embedded copy in a library the call loads would be) keeps what the
    window disabled for the call: the user's breakpoints are enabled again
    afterwards, with the note that they were ignored."""
    names = ("libresolv.so.2", "libz.so.6", "libutil.so.9", "libz.so.1")
    main = sys.modules["__main__"]
    t.run("break realm_gdb_test_add")
    loads = []

    def on_new_objfile(event):
        if not loads and any(n in event.new_objfile.filename for n in names):
            loads.append(event.new_objfile.filename)
            gdb.execute("source {}".format(t.config["script"]), to_string=True)
    gdb.events.new_objfile.connect(on_new_objfile)
    outs = []
    try:
        for name in names:
            error, out = t.run_capturing("realm-thread apply {:#x} print (long)dlopen("
                                         "\"{}\", 2)".format(t.ut("chain0"), name))
            outs.append(out)
            if error is not None or not re.search(r"= 0$", out, re.M):
                break
    finally:
        gdb.events.new_objfile.disconnect(on_new_objfile)
    if error is not None and "No symbol" in error:
        raise TestSkip("no dlopen here: " + error)
    if not loads:
        raise TestSkip("none of {} loaded here".format(", ".join(names)))
    t.check(error is None, "dlopen({}): {}".format(name, error))
    # (Once for the same breakpoints: after the first call, which may have
    # been one that loaded nothing.)
    t.check_in("breakpoints are disabled while a function call runs", "".join(outs), "the note")
    t.check_eq(main._infcall_depth, 0, "calls in progress afterwards")
    t.check_eq(main._disabled_breakpoints, [], "breakpoints still disabled")
    t.check(re.search(r"^\d+\s+breakpoint\s+keep\s+y", t.run("info breakpoints"), re.M),
            "the breakpoint, enabled again")
    t.run("delete")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("attach",), ending="none", variants=[None, "batch_terminal"])
def test_window_unrestored_quit(t):
    """If a host's registers couldn't be put back (simulated here), a `quit`
    from a script, rather than a terminal (it would come again at every end
    of its input), kills the attached program, saying so, rather than let it
    run on corrupted; as does one in batch mode, from a terminal too (gdb
    then lets go of the program as the script ends)."""
    main = sys.modules["__main__"]
    real = main._put_back
    real_terminal, real_batch = main._stdin_is_terminal, main._batch_session

    def put_back(saved, *args):  # (Puts them back, but says it couldn't.)
        return real(saved, *args) or [sorted(saved)[0]]
    main._put_back = put_back
    if t.variant == "batch_terminal":
        main._stdin_is_terminal = lambda: True
        main._batch_session = lambda: True
    try:
        t.check_in("could not put back", t.apply("chain0", "bt 1"), "the warning")
        out = t.run("python import __main__; __main__._guard('quit')")
    finally:
        main._put_back = real
        main._stdin_is_terminal, main._batch_session = real_terminal, real_batch
    t.check_in("killing process", out, "the warning")
    t.check_eq(gdb.selected_inferior().pid, 0, "the program's process afterwards")
    t.check(not main._unrestored, "threads still to put back")
    yield


@test(scenarios=("attach",), ending="killed", variants=["TERM", "HUP"])
def test_window_unrestored_signal(t):
    """SIGTERM or SIGHUP to gdb at its prompt, after a host's registers
    couldn't be put back (simulated here), kills the attached program,
    saying so, rather than let gdb detach and leave it to run on corrupted:
    also on gdb 12 and 13, whose quit raises KeyboardInterrupt as the first
    gdb_exiting handler starts."""
    if not hasattr(gdb.events, "gdb_exiting"):
        raise TestSkip("gdb.events.gdb_exiting (gdb >= 12)")
    main = sys.modules["__main__"]
    real = main._put_back

    def put_back(saved, *args):  # (Puts them back, but says it couldn't.)
        return real(saved, *args) or [sorted(saved)[0]]
    main._put_back = put_back  # (Kept: realm-gdb tries again as gdb exits.)
    t.check_in("could not put back", t.apply("chain0", "bt 1"), "the warning")
    t.release()  # (So that it would end soon, if it were let go.)
    t.allow_output("Python Exception <class 'KeyboardInterrupt'>")
    t.await_exit()
    yield typed_final("python rgt._signal_gdb_later('{}', 0.5)".format(t.variant))


@test(scenarios=("live",))
def test_window_may_write_registers(t):
    """A command in a window that turns may-write-registers off (gdb >= 14
    lets it, with a live process) doesn't keep the window from putting the
    thread's own registers back; the setting stays as the command left it."""
    before = t.regs()
    error, out = t.run_capturing("realm-thread apply {:#x} set may-write-registers off"
                                 .format(t.ut("chain0")))
    if error is not None:
        raise TestSkip("set may-write-registers off: " + error)
    try:
        t.check_not_in("could not put back", out, "a warning")
        t.check_in("is off", t.run("show may-write-registers"), "the setting afterwards")
    finally:
        t.run("set may-write-registers on")
    t.check(not sys.modules["__main__"]._unrestored, "threads still to put back")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_may_write_registers_off(t):
    """With may-write-registers off (gdb >= 14 lets it, with a live
    process), parked threads aren't shown, saying why; and realm-gdb doesn't
    take the user's setting for gdb being unable to write the extended
    registers: calls aren't refused once it is on."""
    main = sys.modules["__main__"]
    ok, why = t.try_run("set may-write-registers off")
    if not ok:
        raise TestSkip("set may-write-registers off: " + why)
    try:
        msg = t.expect_error("realm-thread apply {:#x} bt 1".format(t.ut("chain0")))
        t.check_in("'may-write-registers' is off", msg, "why it isn't shown")
        t.check_in("set may-write-registers on", msg, "the advice")
    finally:
        t.run("set may-write-registers on")
    out = t.apply("chain0", "print realm_gdb_test_add(1, 2)")
    t.check_in("= 3", out, "a call, with may-write-registers on again")
    t.check_not_in("can't write", out, "a note")
    t.check(not any(main._extended_unwritable.values()), "extended registers taken for "
            "unwritable: {}".format(main._extended_unwritable))
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_source_blocks(t):
    """A script run by `rt source FILE` or `realm-thread apply PTR source
    FILE` may have blocks of its own (`if`, `define`): the file gives their
    lines.  Afterwards, a command that reads the lines after it is refused
    in such windows again."""
    path = os.path.join(t.config["tmp"], "blocks.gdb")
    with open(path, "w") as f:
        f.write("if 1\n  echo RGT-IF\\n\nend\ndefine rgt-sourced\n  echo RGT-SOURCED\\n\n"
                "end\nbt 1\n")
    before = t.regs()
    out = t.apply("chain0", "source " + path)
    t.check_in("RGT-IF", out, "the script's if")
    t.check_eq(_parked_frame_count(out), 1, "its bt 1")
    t.check_in("RGT-SOURCED", t.run("rgt-sourced"), "the command it defined")
    t.expect_error("realm-thread apply {:#x} define rgt-other".format(t.ut("chain0")),
                   "reads the lines after it")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("core",))
def test_core_body_commands(t):
    """In a core file, `rt COMMAND` refuses a COMMAND that reads the lines
    after it, as in a live process: run with gdb.execute it would have none
    (`rt define foo` would empty foo)."""
    t.run("define rgt-defined\necho RGT-DEFINED\\n\nend")
    t.show("chain0")
    for cmd in ("define rgt-defined", 'eval "define rgt-defined"'):
        t.expect_error("rt " + cmd, "reads the lines after it")
    t.back()
    t.check_in("RGT-DEFINED", t.run("rgt-defined"), "rgt-defined afterwards")
    yield


@test(scenarios=("core",))
def test_core_input_radix(t):
    """In a core file, with `set input-radix 16`, the backtrace limit that
    realm-gdb raises around a parked thread's `bt N`, and puts back, stays
    the user's, and `bt 3` shows three parked frames."""
    t.run("set backtrace limit 20")
    t.run("set input-radix 16")
    try:
        for _ in range(2):
            out = t.run("realm-thread apply {:#x} bt 3".format(t.ut("chain0")))
            t.check_eq(_parked_frame_count(out), 3, "bt 3")
    finally:
        t.run("set input-radix 0xa")
    t.check_in("20", t.run("show backtrace limit"), "backtrace limit afterwards")
    t.run("set backtrace limit unlimited")
    yield


@test(scenarios=("live",))
def test_load_repeated(t):
    """Loading the script again and again (as each re-read of librealm's
    symbols does: `file PROG`, a rebuilt program run again) defines the same
    hooks each time, quickly: not hooks of its own hooks, whose number
    would double at each load."""
    def hooks():
        out = t.run("show user")
        return len(re.findall(r'^User command "(?:\S+ )*hook(?:post)?-', out, re.M))
    first = hooks()
    import time
    for i in range(3):
        start = time.time()
        t.run("source {}".format(t.config["script"]))
        took = time.time() - start
        t.check_eq(hooks(), first, "hooks after load {}".format(i + 2))
        t.check(took < 5, "load {} took {:.1f}s".format(i + 2, took))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_call_timeout(t):
    """A function call made in a window that runs too long is stopped, and
    unwound: by gdb's own timeout (direct-call-timeout, gdb >= 15), or, in a
    gdb without one, or where it doesn't stop the call (FreeBSD's gdb 15.1),
    by realm-gdb, with a SIGINT to the call's thread, which it says.  (The
    timeout is shortened here.)"""
    import time
    main = sys.modules["__main__"]
    real = main._CALL_TIMEOUT
    main._CALL_TIMEOUT = 1
    try:
        start = time.time()
        error, out = t.run_capturing("realm-thread apply {:#x} print realm_gdb_test_nap("
                                     "6000000)".format(t.ut("chain0")))
        took = time.time() - start
    finally:
        main._CALL_TIMEOUT = real
    out += error or ""
    ours = main._no_call_timeouts()
    if took >= 6 and ours:
        # (gdb 8.2 keeps Python's lock in a call made from Python.)
        raise TestSkip("realm-gdb's timeout thread couldn't run during the call")
    t.check(took < 5, "the call ran {:.1f} seconds".format(took))
    t.check(error is not None, "the call didn't fail: " + out)
    if ours:
        t.check_in("stopped after 1 seconds, with a SIGINT from realm-gdb, as " + ours, out,
                   "the note")
    elif "SIGINT from realm-gdb" in out:  # (Where gdb's own timeout doesn't stop it.)
        t.check_in("stopped after 3 seconds, with a SIGINT from realm-gdb, as gdb's own "
                   "call timeout didn't stop it", out, "the note")
    t.check_eq(t.run("maint print dummy-frames").count("id="), 0, "calls left")
    t.check_host_stack()
    # (The next call works, and isn't stopped by a SIGINT left over.)
    t.check_in("= 3", t.apply("chain0", "print realm_gdb_test_add(1, 2)"), "the next call")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_input_radix(t):
    """With `set input-radix 16`, the numbers realm-gdb writes itself (the
    settings a window puts back, the check of the extended registers) are
    read as it means them: a window's call works, the user's call timeout
    is theirs again afterwards, and the extended registers aren't taken for
    unwritable."""
    main = sys.modules["__main__"]
    timeout = t.try_run("set direct-call-timeout 30")[0]  # (gdb >= 15)
    t.run("set input-radix 16")
    try:
        out = t.apply("chain0", "print realm_gdb_test_add(1, 2)")
    finally:
        t.run("set input-radix 0xa")
    t.check_in("= 3", out, "the call")
    t.check(not any(main._extended_unwritable.values()), "extended registers taken for "
            "unwritable: {}".format(main._extended_unwritable))
    if timeout:
        t.check_in("30", t.run("show direct-call-timeout"), "direct-call-timeout afterwards")
        t.run("set direct-call-timeout unlimited")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_body_commands(t):
    """Commands that read the lines after them (`define`, `commands`,
    `document`, `while`, bare `python`) are refused in `rt` and
    `realm-thread apply`, which can't pass those on (gdb would make them
    empty), and leave what they would have changed alone; so is a `watch`
    without -l, whose watchpoint gdb would delete at once.  `rt pi EXPR` and
    `rt watch -l` work."""
    t.run("define rgt-defined\necho RGT-DEFINED\\n\nend")
    num = re.search(r"Breakpoint (\d+)", t.run("break realm_gdb_test_opaque")).group(1)
    t.run("commands {}\nsilent\nend".format(num))
    commands = t.run("info breakpoints " + num)
    ptr = t.ut("chain0")
    for cmd in ("define rgt-defined", "commands " + num, "document rgt-defined",
                "while 0", "python"):
        msg = t.expect_error("realm-thread apply {:#x} {}".format(ptr, cmd), "refusing")
        t.check_in("reads the lines after it", msg, cmd)
    t.run("alias rgt-mydefine = define")
    t.check_in("lines after it", t.expect_error("realm-thread apply {:#x} rgt-mydefine "
                                                "rgt-defined".format(ptr)), "an alias")
    t.check_in("RGT-DEFINED", t.run("rgt-defined"), "rgt-defined afterwards")
    t.check_in("$_shell", t.expect_error('realm-thread apply {:#x} print $_shell("true")'
                                         .format(ptr)), "$_shell")
    # However they come: through `eval`, `with`, a user-defined command.
    tries = ['eval "define rgt-defined"', 'eval "commands {}"'.format(num), 'eval "if 1"',
             'eval "python"', 'eval "compile code"', "compile print", "compile print /x",
             "compile code -raw", "expression print"]
    if t.supports("with print pretty -- echo"):
        tries += ["with confirm off -- define rgt-defined", "with confirm off -- python"]
    if t.supports("frame apply 1 print 1"):
        # (Wrapped in the commands that run others, abbreviated too.)
        tries += ["frame apply level 0 python", "frame apply all -q -past-main on python",
                  "fr app 1 -- python", "faas python", "thread apply 1 python",
                  "thread apply all -ascending -s compile code", "taas python",
                  "tfaas python", "t a 1 frame apply 1 python", 'eval "faas python"']
        # (Aliases of their subcommands, and given arguments too, gdb >= 10.)
        for alias, cmd in (("rgt-fa = frame apply all", "rgt-fa python"),
                           ("rgt-fal = frame apply level", "rgt-fal 0 python"),
                           ("rgt-taa = thread apply all", "rgt-taa python"),
                           ("rgt-tap = thread apply", "rgt-tap 1 python"),
                           ("rgt-faq = frame apply all -q", "rgt-faq python")):
            if t.supports("alias " + alias):
                tries.append(cmd)
    for cmd in tries:
        t.check_in("reads the lines after it", t.expect_error(
            "realm-thread apply {:#x} {}".format(ptr, cmd), "refusing"), cmd)
    t.check_in("RGT-DEFINED", t.run("rgt-defined"), "rgt-defined, afterwards")
    t.check_eq(t.run("info breakpoints " + num), commands, "the breakpoint's commands, "
               "afterwards")
    t.check_eq(t.run("info breakpoints " + num), commands, "the breakpoint's commands")
    t.select("chain0")
    t.check_in("lines after it", t.expect_error("rt define rgt-defined"), "rt define")
    t.check_in("'rt watch -l chain_marker'", t.expect_error("rt watch chain_marker"), "rt watch")
    if t.supports("frame apply 1 print 1"):
        t.check_in("-l EXPR", t.expect_error("rt frame apply 1 watch chain_marker"),
                   "rt frame apply 1 watch")
    names = re.findall(_BT_LINE + r"([\w:~<>]+)", t.rt("bt"), re.M)
    t.rt("frame {}".format(names.index("rgt_chain_level3")))
    out = t.rt("watch -l chain_marker")
    t.check(re.search(r"[Ww]atchpoint \d+: -location chain_marker", out), "rt watch -l:\n" + out)
    t.check(re.search(r"^2$", t.rt("pi 1 + 1"), re.M), "rt pi EXPR")
    # (A refused `rt pi EXPR` leaves no leave for a later bare `pi`.)
    t.expect_error('rt pi $_shell("true")', "its shell command would run")
    t.expect_error('rt eval "pi"', "refusing")
    t.back()
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",), variants=["settings", "signals"])
def test_window_setup_interrupted(t):
    """A Ctrl-C while a window is being set up (simulated here) leaves
    neither the user's settings changed nor any signal held."""
    import signal
    main = sys.modules["__main__"]
    before = _settings(t)
    caught = _caught_signals() if os.path.exists("/proc/self/status") else None
    raised = []
    if t.variant == "settings":
        real, owner, name = main._set, main, "_set"

        def interrupted(*args):
            if args[0] == "unwindonsignal" and not raised:
                raised.append(True)
                raise KeyboardInterrupt
            return real(*args)
    else:
        real, owner, name = signal.signal, signal, "signal"

        def interrupted(sig, *args):
            if sig == signal.SIGHUP and not raised:
                raised.append(True)
                raise KeyboardInterrupt
            return real(sig, *args)
    setattr(owner, name, interrupted)
    try:
        try:
            gdb.execute("realm-thread apply {:#x} bt 1".format(t.ut("chain0")),
                        to_string=True)
        except (KeyboardInterrupt, gdb.error):  # (gdb 8: "Error occurred in ...")
            pass
    finally:
        setattr(owner, name, real)
    t.check(raised, "the interruption")
    t.check(main._window is None, "a window left open")
    t.check_eq(_settings(t), before, "settings")
    if caught is not None:
        t.check_eq(hex(_caught_signals()), hex(caught), "the signals caught")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_dprintf_in_call(t):
    """A dprintf hit in a function call made outside any window (by an `if`
    condition, which isn't hooked) while a parked thread is selected prints
    as it does without one: its `printf` doesn't try to show the parked
    thread while threads are running."""
    t.run("set scheduler-locking on")
    # (`a` is realm_gdb_test_add's own: not in the parked thread's frame.)
    t.run('dprintf realm_gdb_test_add,"RGT-DPRINTF a=%d\\n", a')
    t.select("chain0")
    try:
        error, out = t.run_capturing("if realm_gdb_test_add(1, 2) == 3\necho RGT-OK\\n\nend")
    finally:
        t.run("realm-thread back")
        t.run("delete")
        t.run("set scheduler-locking off")
    t.check(error is None, "the if: {}\n{}".format(error, out))
    # (Not always 1: gdb 8.2 and 9.2 stop before `a` is stored.)
    t.check(re.search(r"RGT-DPRINTF a=-?\d+", out),
            "the dprintf's output, from the call's frame:\n" + out)
    t.check_in("RGT-OK", out, "the if's body")
    t.check_not_in("can't show", out, "an error")
    yield
    t.continue_to_exit()


@test(scenarios=("live",), ending="none")
def test_window_program_died(t):
    """After the program dies in a function call made while a parked thread
    is shown (gdb then has no selected thread), realm-gdb's commands and
    hooks say so, rather than raise Python exceptions, and the selection
    goes."""
    t.select("chain0")
    t.run_capturing("rt print (void)_exit(3)")
    for cmd in ("realm-thread", "thread apply all bt", "rt bt", "bt 1", "python print(1)",
                "realm-thread apply all bt"):
        error, out = t.run_capturing(cmd)
        both = (error or "") + out
        t.check_not_in("Python Exception", both, cmd)
        t.check_not_in("SystemError", both, cmd)
    main = sys.modules["__main__"]
    if main._selected_thread() is None:  # (FreeBSD's gdb keeps a thread.)
        t.check(not main._selections, "the selection afterwards")
    t.run("kill")
    yield


@test(scenarios=("live",))
def test_window_second_ui_calls(t):
    """With a second UI open (simulated here), whose commands gdb runs even
    while a function call runs, calls are refused while a parked thread is
    shown, with a note once (gdb 8, which can't refuse them, warns); the
    rest works, and may-call-functions is the user's again afterwards."""
    main = sys.modules["__main__"]
    if main._arch.probe_reg is not None and main._probe_extended() is not None:
        raise TestSkip("calls are refused here anyway: gdb can't write the extended "
                       "registers")
    saved = main._second_ui, main._second_ui_noted
    main._second_ui, main._second_ui_noted = "rgt-simulated second UI", False
    try:
        out = t.apply("chain0", "bt 1")
        t.check_eq(_parked_frame_count(out), 1, "bt 1")
        t.check_in("gt-simulated second UI", out, "the note")
        if main._can_refuse_calls:
            msg = t.expect_error("realm-thread apply {:#x} print realm_gdb_test_add(1, 2)"
                                 .format(t.ut("chain0")), "may-call-functions is off")
            t.check_in("second UI", msg, "the error")
            t.check_not_in("rgt-simulated", t.apply("chain0", "bt 1"), "the note, again")
        else:
            t.check_in("WARNING", out, "the warning")
    finally:
        main._second_ui, main._second_ui_noted = saved
    if main._can_refuse_calls:
        t.check_in("is on", t.run("show may-call-functions"), "may-call-functions after")
    t.check_eq(t.within("chain0", lambda: t.eval_int("realm_gdb_test_add(1, 2)")), 3,
               "a call, with no second UI")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_other_terminal_fifo(t):
    """A named pipe gdb has open besides its standard streams, as `new-ui
    mi FIFO` run before realm-gdb loaded leaves one, counts as a second
    UI's terminal (see _other_terminal), named in its messages; an unnamed
    pipe doesn't."""
    main = sys.modules["__main__"]
    if not os.path.isdir("/proc/self/fd"):
        raise TestSkip("no /proc/self/fd")
    t.check(not main._other_terminal(), "before")
    path = os.path.abspath("rgt-fifo")  # (In the session's own directory.)
    os.mkfifo(path)
    fifo = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        t.check_eq(main._other_terminal(), "{} (file descriptor {})".format(path, fifo),
                   "with a named pipe open: what it says")
    finally:
        os.close(fifo)
        os.unlink(path)
    r, w = os.pipe()
    try:
        t.check(not main._other_terminal(), "with an unnamed pipe open")
    finally:
        os.close(r)
        os.close(w)
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_call_failed_early(t):
    """A function call in a window that fails after gdb has said it is
    starting but before it runs (it can't write its stack, or has too few
    arguments: gdb sends no inferior_call_post then) leaves the user's
    breakpoints as they were, and nothing counting a call in progress."""
    main = sys.modules["__main__"]
    t.run("break realm_gdb_test_opaque")
    before = t.run("info breakpoints")
    if t.supports("with print pretty -- echo"):
        # (A struct passed by value, which every ABI here copies to the
        # stack: aarch64's calls write nothing else there, and one that got
        # as far as inserting breakpoints would fail there instead, which
        # crashes Ubuntu 24.04's gdb 15.1 on aarch64, without realm-gdb too.)
        cmd = ("with may-write-memory off -- print ((int(*)(RealmGdbTestState))"
               "realm_gdb_test_add)(realm_gdb_test_state)")
    else:
        cmd = "print ((int(*)(int, int))realm_gdb_test_add)(1)"
    error, out = t.run_capturing("realm-thread apply all " + cmd)
    t.check(error is not None, "the call failed:\n" + out)
    t.check_eq(main._infcall_depth, 0, "calls counted in progress")
    t.check_eq(t.run("info breakpoints"), before, "info breakpoints afterwards")
    t.check_not_in("abandoned", out, "a call that ran")
    t.check_not_in("breakpoints are disabled while", out, "the note, for calls that didn't run")
    t.check_host_stack()
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_interrupt_command(t):
    """A Ctrl-C (here a SIGINT from another thread, as the terminal's would
    come) during a window's command, where Ctrl-C can land while Python
    runs a command (gdb >= 10), interrupts it, as it would without a
    window: here a Python loop of `realm-thread apply PTR python`, which
    gets KeyboardInterrupt, as a looping pretty-printer would.  The window
    is closed as usual."""
    import signal
    main = sys.modules["__main__"]
    if not main._SIGINT_IN_PYTHON_SAFE:
        raise TestSkip("Ctrl-C is held through a window's command on this gdb")
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        raise TestSkip("this gdb ignores SIGINT")
    before = t.regs()
    error, out = t.run_capturing("realm-thread apply {:#x} python "
                                 "rgt._loop_until_interrupted()".format(t.ut("chain0")),
                                 quits=True)
    t.check_in("RGT-INTERRUPTED", out, "the loop, interrupted ({})".format(error))
    t.check(main._window is None and not main._sigint_is_blocked(), "the window, afterwards")
    if hasattr(signal, "sigpending"):
        t.check(signal.SIGINT not in signal.sigpending(), "a Ctrl-C still pending")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


def _loop_until_interrupted():
    """Run inside a register window (see test_window_interrupt_command)."""
    import signal
    import threading
    import time
    start = time.time()
    # (A thread of Python's blocks SIGINT, as realm-gdb has its threads do:
    # gdb's main thread gets it, as from the terminal.)
    timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        while time.time() - start < 20:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("RGT-INTERRUPTED after {:.1f}s".format(time.time() - start))
        return
    finally:
        timer.cancel()
    print("RGT-NOT-INTERRUPTED")


@test(scenarios=("live",))
def test_window_apply_call_interrupted(t):
    """A function call of `realm-thread apply all`'s command that a SIGINT
    stopped (a Ctrl-C, which goes to the program: here the call raises it
    itself) stops it at the next parked thread, as a Ctrl-C held does,
    rather than run the command on the rest."""
    error, out = t.run_capturing("realm-thread apply all print (int)raise(2)")
    t.check(error is not None, "the command didn't fail:\n" + out)
    t.check_eq(len(re.findall(r"^User thread ", out, re.M)), 1, "parked threads shown:\n" + out)
    t.check_eq(t.run("maint print dummy-frames").count("id="), 0, "calls left")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["apply_all", "notes"])
def test_window_interrupt_held(t):
    """A Ctrl-C (here a SIGINT gdb sends itself) during `realm-thread
    apply`, where its window holds it (in realm-gdb's own code, and after a
    function call's end), stops it at the next parked thread ("apply_all"),
    and lands once what was collected, and the notes, are printed
    ("notes"), rather than losing them."""
    import signal
    main = sys.modules["__main__"]
    if not hasattr(signal, "sigpending"):
        raise TestSkip("Python has no signal.sigpending")
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        # (gdb 10 and older leave it so when started with it ignored, as by
        # a shell's background job.)
        raise TestSkip("this gdb ignores SIGINT")
    total = t.run("realm-thread apply all print 1").count("User thread ")
    written, printed = [], []
    real_write, real_run, real_stdout = gdb.write, main._run_in_window, sys.stdout

    def write(text, *args, **kwargs):
        written.append(text)
        return real_write(text, *args, **kwargs)

    class Stdout(object):  # (print()'s, which gdb >= 13 writes with _gdb.write.)
        def write(self, text):
            printed.append(text)
            return real_stdout.write(text)

        def flush(self):
            return real_stdout.flush()

    def run(*args, **kwargs):
        out = real_run(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        return out
    if t.variant == "apply_all":
        main._run_in_window = run
        cmd = "realm-thread apply all bt 1"
    else:
        # (After a function call's end, which holds Ctrl-C for the rest of
        # the command: see _interruptibly.)
        cmd = ("realm-thread apply {:#x} python gdb.execute('print realm_gdb_test_add(1, 2)'); "
               "gdb.execute('set var $sp = $sp - 16'); "
               "import os, signal; os.kill(os.getpid(), signal.SIGINT)".format(t.ut("chain0")))
    before = t.regs()
    interrupted = False
    gdb.write, sys.stdout = write, Stdout()
    try:
        gdb.execute(cmd, to_string=True)
    except KeyboardInterrupt:
        interrupted = True
    except gdb.error as e:  # (gdb 8.2's, for a Python command's KeyboardInterrupt.)
        interrupted = str(e) == "Quit" or "Error occurred in Python command" in str(e)
        if not interrupted:
            raise
    finally:
        gdb.write, main._run_in_window, sys.stdout = real_write, real_run, real_stdout
    out = "".join(written)
    t.check(interrupted, "the Ctrl-C:\n" + out)
    t.check(signal.SIGINT not in signal.sigpending(), "a Ctrl-C still pending")
    t.check(not main._sigint_is_blocked() and main._window is None, "the window, afterwards")
    if _is_x86_64() and os.path.exists("/proc/self/status"):
        t.check_eq(bool(t.within("chain0", _blocked_signals) & (1 << (signal.SIGINT - 1))),
                   not main._SIGINT_IN_PYTHON_SAFE, "SIGINT blocked in the next window's "
                   "command (let through for it alone, where it can be)")
    if t.variant == "apply_all":
        shown = out.count("User thread ")
        t.check(1 <= shown < total, "parked threads shown, of {}: {}\n{}".format(
            total, shown, out))
        t.check_in(PARKED_FRAME0, out, "the first one's output")
    else:
        t.check_in("which doesn't last", out + "".join(printed), "the note")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["unblocked", "blocked", "default"])
def test_window_interrupt_other_thread(t):
    """A Python thread that doesn't block SIGINT, which the kernel gives a
    Ctrl-C to while a window has it blocked on gdb's main thread: a Ctrl-C
    taken there kills gdb 8.2 and 9.2 (as Python runs) and 17 and later (on
    any thread but the main one), and quits gdb 10 to 16 wherever they are,
    in the middle of popping a function call, say.  So no parked thread is
    shown while it runs, saying why and what to do, and once it ends they
    are again.  (Where realm-gdb can't tell, off Linux, a Ctrl-C, here a
    SIGINT gdb sends itself, may interrupt the window's command, which puts
    the host's registers back all the same.)  A thread that blocks SIGINT,
    as the message says to, is no matter ("blocked"), nor one that does
    nothing about it ("default"): realm-gdb has Python's threads started
    once it has loaded block it (see _install_thread_sigmask)."""
    import signal
    import threading
    main = sys.modules["__main__"]
    if not os.path.isdir("/proc/self/task"):
        raise TestSkip("no /proc/self/task, to tell which threads would take a Ctrl-C")
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        raise TestSkip("this gdb ignores SIGINT")
    done, ready = threading.Event(), threading.Event()

    def run():
        if t.variant == "blocked":
            signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
        elif t.variant == "unblocked":
            signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGINT])
        ready.set()
        done.wait()
    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()
    before = t.regs()
    try:
        ready.wait(10)
        fatal = bool(main._sigint_takers())
        if t.variant == "unblocked":
            # (Linux, which shows each thread's blocked signals: on every gdb.)
            t.check(fatal, "the thread that doesn't block SIGINT, found")
        if t.variant != "unblocked":
            t.check(not fatal, "the thread taken for one that doesn't block SIGINT")
            t.check_in("= 1", t.apply("chain0", "print 1"), "'print 1' on a parked thread")
        elif fatal:
            for cmd in ("realm-thread apply {:#x} print 1", "realm-thread {:#x}"):
                t.expect_error(cmd.format(t.ut("chain0")), "that Python started")
            t.check_in("can't show parked user threads now", t.run("realm-thread"), "the list's note")
        else:
            try:
                t.run_capturing(
                    "realm-thread apply {:#x} python import os, signal, time; "
                    "os.kill(os.getpid(), signal.SIGINT); time.sleep(0.2); "
                    "gdb.execute('output 1')".format(t.ut("chain0")))
            except KeyboardInterrupt:
                pass
            t.check(main._window is None and not main._sigint_is_blocked(), "the window, afterwards")
    finally:
        done.set()
        thread.join(5)
    t.check(not thread.is_alive(), "the thread, at its end")
    t.check_in("= 1", t.apply("chain0", "print 1"), "'print 1' once the thread ended")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_logging_with(t):
    """`with logging enabled off -- COMMAND` in `rt` or `realm-thread
    apply`, with gdb's logging on (gdb >= 13), is refused, as `set logging
    enabled` is there: `with` switches logging itself, inside the collected
    output, where no `set` hook sees it, and gdb 13 to 17 crashed as the
    output was printed.  However it comes (abbreviated, `eval`, wrapped in
    `frame apply`); logging is left on, and its file has what followed."""
    if not t.supports("show logging enabled"):
        raise TestSkip("this gdb has no 'logging enabled' setting")
    path = os.path.join(t.config["tmp"], "rgt-logging.txt")
    # (Not collected, as t.run's are: switching logging there is what crashes.)
    gdb.execute("set logging file " + path)
    gdb.execute("set logging overwrite on")
    gdb.execute("set logging enabled on")
    ptr = t.ut("chain0")
    try:
        gdb.execute("echo RGT-BEFORE\\n")  # (Not collected, as t.run's would be: logged.)
        tries = ["with logging enabled off -- print 101", "with logging ena off -- print 101",
                 'eval "with logging enabled off -- print 101"']
        if t.supports("frame apply 1 print 1"):
            tries.append("frame apply 1 with logging enabled off -- print 101")
        for cmd in tries:
            t.check_in("logging", t.expect_error(
                "realm-thread apply {:#x} {}".format(ptr, cmd), "refusing"), cmd)
        t.expect_error("realm-thread apply all with logging enabled off -- print 101",
                       "refusing")
        t.select("chain0")
        t.expect_error("rt with logging enabled off -- print 101", "refusing")
        t.back()
        t.check(gdb.parameter("logging enabled") is True, "logging, afterwards")
        gdb.execute("realm-thread apply {:#x} print 102".format(ptr))
    finally:
        gdb.execute("set logging enabled off")
        gdb.execute("set logging overwrite off")
    with open(path) as f:
        logged = f.read()
    t.check("RGT-BEFORE" in logged and "= 102" in logged, "the log:\n" + logged)
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_alias_arguments(t):
    """A user alias that gives its command arguments (`alias u2 = up 2`), in
    `rt`: it moves as many frames as it says, as typed it would.  One of
    `python` given code (`alias pyx = python print(...)`) is refused as
    reading the lines after it, which it does: gdb drops its arguments where
    it reads the block itself, as from gdb.execute (and before gdb 13 reads
    the block from the terminal there)."""
    if not t.supports("alias rgt-u2 = up 2"):
        raise TestSkip("this gdb's aliases take no arguments")
    t.run("alias rgt-d1 = down 1")
    t.run("alias rgt-pyx = python print('RGT-' + 'PYX')")
    t.select("chain0")
    level = lambda: int(re.search(r"^#(\d+)", t.rt("frame"), re.M).group(1))
    t.rt("frame 0")
    t.rt("rgt-u2")
    t.check_eq(level(), 2, "the frame after 'rt rgt-u2'")
    t.rt("rgt-d1")
    t.check_eq(level(), 1, "the frame after 'rt rgt-d1'")
    t.check_in("reads the lines after it", t.expect_error("rt rgt-pyx", "refusing"),
               "rt rgt-pyx")
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_registry_unreadable(t):
    """With the registry's head unreadable (librealm without its debug
    information, gdb's --readnever: simulated here), `realm-thread` and
    `realm-thread apply all` say they can't read it, and why, rather than
    that it is empty."""
    main = sys.modules["__main__"]
    real = main._registry_head

    def unreadable():
        main._registry_error = "No symbol table is loaded.  Use the \"file\" command."
        return None
    main._registry_head = unreadable
    try:
        listed = t.run("realm-thread")
        applied = t.run("realm-thread apply all bt 1")
    finally:
        main._registry_head = real
    for what, out in (("realm-thread", listed), ("realm-thread apply all", applied)):
        t.check_in("can't read the Realm user thread registry", out, what)
        t.check_in("No symbol table is loaded", out, what + ": gdb's error")
    t.check_in("parked", t.run("realm-thread"), "realm-thread, afterwards")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_list_other_inferior(t):
    """In an inferior with no process (or no Realm), `realm-thread` says so,
    and which inferior has user threads, and neither it nor `thread apply
    all bt` leaves that inferior selected."""
    t.run("add-inferior")
    num = max(inf.num for inf in gdb.inferiors())
    t.run("inferior {}".format(num))
    try:
        out = t.run("realm-thread")
        t.check_in("no process or core file", out, "the list")
        t.check(re.search(r"inferior 1 has \d+ Realm user threads: select it", out),
                "the other inferior's:\n" + out)
        t.check_eq(gdb.selected_inferior().num, num, "the inferior, after realm-thread")
        t.run("thread apply all bt 1")
        t.check_eq(gdb.selected_inferior().num, num, "the inferior, after thread apply all bt")
    finally:
        t.run("inferior 1")
        t.run("remove-inferiors {}".format(num))
    t.switch_thread(t.stop_thread)
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_registry_dropped(t):
    """With librealm's symbols dropped from a live process
    (`nosharedlibrary`), `realm-thread` and `realm-thread apply all` say so,
    rather than that the registry is empty, and a plain command with a
    parked thread selected says why it can't find it, dropping the
    selection; `sharedlibrary` brings them back."""
    t.select("chain0")
    t.run("nosharedlibrary")
    try:
        gdb.parse_and_eval("&Realm::user_thread_registry_head")  # (Runs no hooks.)
        linked = True
    except gdb.error:
        linked = False
    if linked:
        t.run("sharedlibrary")
        raise TestSkip("librealm is linked into the program here")
    try:
        listed = t.run("realm-thread")
        applied = t.run("realm-thread apply all bt 1")
        dumped = t.run("thread apply all bt 1")
        error = t.expect_error("bt 1") if t.plain else None
    finally:
        t.run("sharedlibrary")
    t.check(re.search(r"=== Parked Realm user threads: not shown ===\n[^\n]*no longer has "
                      r"librealm's symbols", dumped), "thread apply all bt:\n" + dumped)
    for what, out in (("realm-thread", listed), ("realm-thread apply all", applied)):
        t.check_in("no longer has librealm's symbols", out, what)
    if error is not None:
        t.check_in("can't be found: gdb no longer has librealm's symbols", error, "bt")
        t.check_in("no longer selecting it", error, "bt")
        # (Not to run 'realm-thread back' first: nothing is selected now.)
        t.check_in("run it again to run it on thread {} as it really is".format(
            t.stop_thread), error, "bt: the advice")
    t.check_in("parked", t.run("realm-thread"), "realm-thread, after sharedlibrary")
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["plain", "rt"])
def test_window_quit_at_call(t):
    """A quit that gdb hands Python while a parked thread is shown, as the
    command evaluates an expression before a function call (here gdb's quit
    flag, set by gdb.interrupt in a convenience function, as a Ctrl-C
    landing in gdb sets it): it is held, rather than raised as a Python
    exception, and passed to the call, which still runs as a call made in a
    window must (the user's breakpoint in it isn't hit, scheduler-locking is
    on), and nothing is left behind (scheduler-locking the user's,
    breakpoints enabled, no call's record).  (One that gdb hands realm-gdb's
    handler of the call itself, before its first line, is
    terminal.plain_ctrl_c_calls's.)"""
    import signal
    main = sys.modules["__main__"]
    if not hasattr(gdb, "interrupt"):
        raise TestSkip("this gdb has no gdb.interrupt, to set its quit flag")
    _define_quit_function()
    before = t.regs()
    t.run("set scheduler-locking replay")
    t.run("break realm_gdb_test_add")
    expr = "print $_rgt_quit() + realm_gdb_test_add(1, 2)"
    if t.variant == "plain":
        t.select("chain0")
        if not t.plain:
            raise TestSkip("plain commands don't show parked threads here")
        cmd = expr
    else:
        cmd = "realm-thread apply {:#x} {}".format(t.ut("chain0"), expr)
    error, out = t.run_capturing(cmd, quits=True)
    if t.variant == "plain":
        t.back()
    t.check(not re.search(r"hit Breakpoint|WARNING|Python Exception", out),
            "the call, run as one made in a window ({}):\n{}".format(error, out))
    t.check_in("replay", t.run("show scheduler-locking"), "scheduler-locking afterwards")
    t.check(all(b.enabled for b in gdb.breakpoints()), "breakpoints enabled afterwards")
    t.check_eq(t.run("maint print dummy-frames").count("id="), 0, "calls left")
    t.check(main._window is None and not main._sigint_is_blocked(), "the window, afterwards")
    if hasattr(signal, "sigpending"):
        t.check(signal.SIGINT not in signal.sigpending(), "a Ctrl-C still pending")
    t.run("delete 2")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["default", "failing", "unstoppable"])
def test_window_ctrl_c_before_call(t):
    """A Ctrl-C (here a SIGINT gdb sends itself) that comes while a window
    holds it, before a function call made in the window (after an earlier
    call's end, which holds it for the rest of the command, see
    _interruptibly; and throughout, where Ctrl-C is held through a call
    made from Python, gdb 8.2 and 9.2): the call is stopped as it begins,
    and abandoned, rather than the Ctrl-C being lost (gdb printed it as a
    Python exception there); and the window is closed as usual.  In the
    "failing" variant, the call fails before it runs (it can't write its
    argument): no SIGINT is left for the program, which a Ctrl-C passed to
    the call would have been (the program then stopped with it, or died,
    with `handle SIGINT pass`).  In the "unstoppable" variant, the call
    can't be stopped as it begins (simulated): it runs, and the Ctrl-C,
    held rather than lost, quits the command once the window is closed."""
    import signal
    main = sys.modules["__main__"]
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        raise TestSkip("this gdb ignores SIGINT")
    failing = t.variant == "failing"
    if failing and not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb has no 'with'")
    before = t.regs()
    with t.patched("_stop_at_entry", (lambda *args, **kwargs: False)
                   if t.variant == "unstoppable" else main._stop_at_entry):
        error, out = t.run_capturing(
            "realm-thread apply {:#x} python rgt._ctrl_c_then_call({})".format(
                t.ut("chain0"), failing), quits=True)
    m = re.search(r"RGT-CALL (.*) after ([\d.]+)s", out)
    if t.variant == "unstoppable":
        t.check(m is not None and m.group(1) == "not interrupted",
                "the call, run ({}):\n{}".format(error, out))
        t.check(error is not None and "Quit" in error, "the Ctrl-C, held: {}".format(error))
    else:
        t.check(m is not None and m.group(1).startswith("interrupted") and
                float(m.group(2)) < 2.5, "the call, interrupted ({}):\n{}".format(error, out))
    if failing:
        t.check("emory" in m.group(1), "the call's own error:\n" + out)
        t.check_not_in("first instruction", out, "the note of a call stopped")
        pending = _pending_signals(gdb.selected_inferior().pid)
        if pending is not None:
            t.check(not pending & (1 << (signal.SIGINT - 1)),
                    "a SIGINT pending for the program ({:#x})".format(pending))
    elif t.variant == "default":
        t.check_in("first instruction", out, "the note of the call stopped")
    t.check(main._window is None and not main._sigint_is_blocked(), "the window, afterwards")
    if hasattr(signal, "sigpending"):
        t.check(signal.SIGINT not in signal.sigpending(), "a Ctrl-C still pending")
    t.check_eq(t.run("maint print dummy-frames").count("id="), 0, "calls left")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_call_record_left(t):
    """A function call made while a parked thread is shown whose record gdb
    keeps once its frame is gone (as a Ctrl-C quitting gdb in the middle of
    popping it leaves it: simulated here), with the call's breakpoint, which
    gdb would insert into the parked thread's stack whenever the program
    ran: realm-gdb says so, and refuses to let the program run, as for a
    call it couldn't pop."""
    main = sys.modules["__main__"]
    real, looks = main._dummy_records, []

    def records():
        found = real()
        looks.append(found)
        if found is not None and len(looks) > 1:
            found = found | set([(0x1000, 0x1007)])
        return found
    before = t.regs()
    main._dummy_records = records
    try:
        out = t.run("realm-thread apply {:#x} print realm_gdb_test_add(1, 2)".format(
            t.ut("chain0")))
    finally:
        main._dummy_records = real
    t.check(len(looks) == 2, "gdb's records looked at {} times".format(len(looks)))
    t.check_in("gdb lost its frame but kept its breakpoint", out, "the warning")
    t.check_in("0x1007", out, "  the breakpoint's address")
    t.check_in("'kill' it", out, "  what to do")
    t.check(main._unrestored, "the window, kept to refuse resuming")
    t.expect_error("continue", "realm-gdb")
    # (Simulated: let the program go.)
    del main._unrestored[:]
    main._unrestored_calls()
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["plain", "rt"])
def test_window_refused_call_note(t):
    """A command whose function call the window refuses, but which turned
    may-call-functions back on for it (a convenience function), with a
    parked thread selected ("plain") or by `rt`: the call is stopped at its
    first instruction and abandoned, saying so, and there is no note that
    the command changed the parked thread's registers (those were the
    stopped call's)."""
    main = sys.modules["__main__"]
    if not main._can_refuse_calls:
        raise TestSkip("this gdb can't refuse function calls")
    _define_calls_on_function()
    before = t.regs()
    t.select("chain0")
    if t.variant == "plain" and not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    with t.patched('_calls_refused', lambda window: "for a test"):
        cmd = "print $_rgt_calls_on() + realm_gdb_test_add(1, 2)"
        error, out = t.run_capturing(cmd if t.variant == "plain" else "rt " + cmd)
        out += t.run("output 1")  # (Closes a failed plain command's window.)
    t.back()
    t.check(error is not None, "the call, refused: " + out)
    t.check_in("refused all the same", out, "the note that it was refused")
    t.check("doesn't last" not in out, "a note of registers changed:\n" + out)
    t.check_eq(t.run("maint print dummy-frames").count("id="), 0, "calls left")
    t.check_in("is on", t.run("show may-call-functions"), "may-call-functions afterwards")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_gdb_setting(t):
    """With a parked thread selected, the gdb manual's idiom that saves a
    setting with $_gdb_setting_str, changes it, and sets it back leaves
    scheduler-locking and follow-fork-mode the user's: a plain command's
    window sets those only while its calls run."""
    if not t.supports('output $_gdb_setting_str("confirm")'):
        raise TestSkip("this gdb has no $_gdb_setting_str")
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    before = {}
    for name in ("scheduler-locking", "follow-fork-mode"):
        before[name] = t.run("show " + name)
        var = "$rgt_" + name.replace("-", "_")
        t.run('set {} = $_gdb_setting_str("{}")'.format(var, name))
        t.run("set {} {}".format(name, "step" if name == "scheduler-locking" else "child"))
        t.run('eval "set {} %s", {}'.format(name, var))
    t.back()
    for name, shown in sorted(before.items()):
        t.check_eq(t.run("show " + name), shown, name + " afterwards")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_call_failed_early(t):
    """A plain command with a parked thread selected whose function call
    fails before it runs (gdb sends no inferior_call_post then) leaves its
    window to be closed by the next hooked command, as any failed command's
    is: the call it counted in progress doesn't keep it open."""
    main = sys.modules["__main__"]
    if not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb has no 'with'")
    before = t.regs()
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    # (A struct passed by value: see window_call_failed_early.)
    t.expect_error("with may-write-memory off -- print ((int(*)(RealmGdbTestState))"
                   "realm_gdb_test_add)(realm_gdb_test_state)")
    t.run("output 1")
    t.check(main._plain is None and main._window is None, "the window, after 'output 1'")
    t.check_eq(main._infcall_depth, 0, "calls counted in progress")
    t.back()
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["stopped_first", "failed_first"])
def test_plain_call_left_stopped(t):
    """With a function call of the user's left stopped on the host, a call
    that fails before it runs (gdb sends no inferior_call_post then), made
    after it ("stopped_first") or before ("failed_first"), doesn't keep a
    parked thread from being shown: the stopped call's dummy frame is not
    that of a call in progress.  Plain `bt` and `print $sp` show the
    selected parked thread, not the host."""
    main = sys.modules["__main__"]
    dummies = lambda: t.run("maint print dummy-frames").count("id={")
    # (First: a window drops a count of calls left up, see _call_running.)
    parked_sp = t.within("chain0", lambda: int(gdb.parse_and_eval("$sp")) & _U64)

    # (An argument gdb can't read, which it finds it can't as it pushes it.)
    failing = "realm_gdb_test_add(1, *(int *)0)"
    t.run("break realm_gdb_test_add")
    # (Stops at the breakpoint; the other threads held.)
    t.run("set scheduler-locking on")
    if t.variant == "stopped_first":
        t.run_capturing("print realm_gdb_test_add(5, 6)")
        t.run("set scheduler-locking replay")
        t.run("delete")
        t.expect_error("print " + failing, "Cannot access memory")
    else:
        # (Both without a hooked command between them, which would drop the
        # count the first left up: see _call_running.)
        for expr in (failing, "realm_gdb_test_add(5, 6)"):
            try:
                gdb.parse_and_eval(expr)
            except gdb.error:
                pass
        depth = main._infcall_depth
        t.run("set scheduler-locking replay")
        t.run("delete")
        if not depth:
            raise TestSkip("the call failed before gdb counted it")
    t.check_eq(dummies(), 1, "the user's call")
    if t.variant == "stopped_first" and not main._infcall_depth:
        raise TestSkip("the call failed before gdb counted it")
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    t.check_in(PARKED_FRAME0, t.run("bt 1"), "plain 'bt 1'")
    t.check_eq(int(t.run("output/x $sp").strip(), 16), parked_sp, "plain 'output/x $sp'")
    t.check_eq(main._infcall_depth, 0, "calls counted in progress")
    t.back()
    t.check_in("<function called from gdb>", t.run("bt"), "the user's call, afterwards")
    t.run("return")  # (The user's call, popped: confirm is off.)
    t.check_eq(dummies(), 0, "calls left")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), init="set max-completions 0")
def test_max_completions_zero(t):
    """With completion turned off before librealm loads (`set
    max-completions 0`, which leaves `complete` listing nothing), realm-gdb
    still finds the commands it guards, and the setting is the user's
    afterwards."""
    t.check_in("is 0", t.run("show max-completions"), "max-completions")
    t.expect_error("realm-thread apply {:#x} continue".format(t.ut("chain0")),
                   "refusing 'continue'")
    t.check_eq(_parked_frame_count(t.apply("chain0", "bt 1")), 1, "bt 1")
    t.check(not t.exited, "the program exited")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["prompt_hook", "before_prompt", "prompt_hook_after",
                                     "raising_hook"])
def test_plain_failed_at_prompt(t):
    """A prompt hook (set before a parked thread is selected, or after), or
    a before_prompt handler that runs after realm-gdb's, running a hooked
    command that fails while a parked thread is selected, leaves no window
    open at the prompt: the thread has its own registers there.  So does a
    before_prompt handler's when the user's prompt hook raises."""
    main = sys.modules["__main__"]
    before = t.regs()

    def fails(*args):
        try:
            gdb.execute("print rgt_nosuchvar", to_string=True)
        except gdb.error:
            pass
    def raises(current):
        raise RuntimeError("the user's prompt hook fails")
    saved = gdb.prompt_hook
    if t.variant == "prompt_hook":
        gdb.prompt_hook = fails
    elif t.variant in ("before_prompt", "raising_hook"):
        gdb.events.before_prompt.connect(fails)
        if t.variant == "raising_hook":
            gdb.prompt_hook = raises
    try:
        t.select("chain0")
        if not t.plain:
            raise TestSkip("plain commands don't show parked threads here")
        if t.variant == "prompt_hook_after":
            gdb.prompt_hook = fails
        yield  # (the prompt)
        t.check(main._window is None and main._plain is None, "a window open at the prompt")
        t.check_regs(before, what="registers at the prompt")
        t.back()
    finally:
        if t.variant in ("before_prompt", "raising_hook"):
            gdb.events.before_prompt.disconnect(fails)
        gdb.prompt_hook = saved
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_failed_with(t):
    """`with SETTING V -- COMMAND`, where COMMAND is a plain command that
    fails while a parked thread is selected, leaves SETTING as it was: the
    window COMMAND opened doesn't put back V, which `with` had set."""
    if not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb has no 'with'")
    before = t.run("show scheduler-locking")
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    t.expect_error("with scheduler-locking step -- print rgt_nosuchvar", "No symbol")
    yield  # (the prompt closes its window)
    t.check_eq(t.run("show scheduler-locking"), before, "scheduler-locking afterwards")
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["plain", "rt", "apply"])
def test_with_guarded_around(t):
    """`with SETTING VALUE -- COMMAND`, of a setting that a window can't be
    opened with changed (architecture, endian, osabi, host-charset), around
    a command that would open one (a plain command with a parked thread
    selected, `rt`, `realm-thread apply`): refused before it reads the
    program, saying why; nothing in the program changes, and the next such
    command (after `with` has put the setting back, or the user, before gdb
    12) works."""
    if not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb has no 'with'")
    before = t.regs()
    t.select("chain0")
    if t.variant == "plain" and not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    inner = {"plain": "bt 1", "rt": "rt bt 1",
             "apply": "realm-thread apply {:#x} bt 1".format(t.ut("chain0"))}[t.variant]
    settings = [("endian", "big"), ("osabi", "none"), ("host-charset", "ASCII")]
    if not _is_aarch64():
        settings.append(("architecture", "i386:x64-32"))
    for name, value in settings:
        if not t.supports("help set " + name):
            continue
        try:
            error = t.expect_error("with {} {} -- {}".format(name, value, inner))
        finally:
            t.run("set {} auto".format(name))  # (gdb < 12's `with` leaves some.)
        t.check_in("inside 'with {}'".format(name), error, "with " + name)
        t.check_in(PARKED_FRAME0, t.run(inner), "{}, afterwards".format(inner))
    # With no prompt since those failed `with`s (which run no hookpost), a
    # `set` of one is the user's own, and a `with` of it again is seen.
    osabi = re.search(r'currently "([^"]+)"', t.run("show osabi"))
    if osabi:
        t.run('set osabi {}'.format(osabi.group(1)))
        try:
            t.check_in(PARKED_FRAME0, t.run(inner), "{}, after 'set osabi {}'".format(
                inner, osabi.group(1)))
            t.expect_error("with osabi none -- " + inner, "inside 'with osabi'")
        finally:
            t.run("set osabi auto")
    t.check_regs(before)
    t.check_host_stack()
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_failed_input(t):
    """Python's input() after a plain command that failed while a parked
    thread is selected (a `code.interact()` console's next line; here a
    stand-in for the terminal's) closes the failed command's window first,
    and reads the line, rather than refusing it for the failed command;
    input() in a plain command's window (a convenience function's, as
    `print` evaluates it) is still refused."""
    import builtins
    main = sys.modules["__main__"]

    class Ask(gdb.Function):
        def __init__(self):
            super(Ask, self).__init__("rgt_ask")

        def invoke(self):
            return input("? ")
    Ask()
    before = t.regs()
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    real = builtins.input
    builtins.input = lambda *args: "RGT-LINE"
    try:
        t.expect_error("print rgt_nosuchvar", "No symbol")
        t.check(main._plain is not None, "the failed command's window, left open")
        try:
            line = input("> ")
        except gdb.GdbError as e:
            line = "refused: {}".format(e)
        t.check_eq(line, "RGT-LINE", "input() after the failed command")
        t.check(main._plain is None and main._window is None, "the window, after input()")
        t.check_regs(before, what="registers after input()")
        t.expect_error("print $rgt_ask()", r"'print' reads the terminal, which can't be done")
        t.check(sys.stdin.readline is not None and input("> ") == "RGT-LINE",
                "input() after that")
    finally:
        builtins.input = real
    t.check_regs(before)
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_failed_posted(t):
    """A plain command that fails while a parked thread is selected where no
    prompt event follows (breakpoint commands at a stop in the background,
    `continue &`; a posted event; MI) has its window closed as soon as gdb
    is back in its event loop: realm-gdb posts a closer as the window opens,
    which closes it if it is still open then (here run by hand)."""
    main = sys.modules["__main__"]
    if not hasattr(gdb, "interrupt"):
        raise TestSkip("gdb 9 to 14 die exiting with a posted event left: none is posted")
    before = t.regs()
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    posted = []
    real = gdb.post_event
    gdb.post_event = posted.append
    try:
        t.expect_error("print rgt_nosuchvar", "No symbol")
    finally:
        gdb.post_event = real
    t.check(main._plain is not None, "the failed command's window, left open")
    t.check(posted, "a closer posted")
    for event in posted:
        event()
    t.check(main._window is None and main._plain is None, "the window, after the closer")
    t.check_regs(before, what="registers afterwards")
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["with", "set", "eval_with", "python_with_parameter"])
def test_plain_failed_settings(t):
    """The settings a window changes are the user's afterwards: after a
    plain command that failed while a parked thread was selected (with no
    prompt between, as in a script), `with SETTING V -- COMMAND` doesn't
    keep the failed command's window's value of SETTING, nor does Python's
    gdb.with_parameter, nor is a `set` of one undone as that window closes;
    and `eval "with ..."` doesn't keep its own window's."""
    if not t.supports("with print pretty -- echo"):
        raise TestSkip("this gdb has no 'with'")
    if t.variant == "python_with_parameter" and not hasattr(gdb, "with_parameter"):
        raise TestSkip("this gdb has no gdb.with_parameter")
    t.run("set scheduler-locking replay")
    t.run("set confirm on")
    t.run("set multiple-symbols ask")
    want = {"confirm": "on", "locking": "replay", "symbols": "ask"}
    try:
        t.select("chain0")
        if not t.plain:
            raise TestSkip("plain commands don't show parked threads here")
        if t.variant == "eval_with":
            t.check_in("= 1", t.run('eval "with confirm on -- print %d", 1'), "eval")
        else:
            t.expect_error("print rgt_nosuchvar", "No symbol")
            if t.variant == "with":
                t.check_in("= 2", t.run("with scheduler-locking off -- print 2"), "with")
            elif t.variant == "python_with_parameter":
                # (Not a boolean, which gdb 12's set_parameter can't put
                # back.)
                with gdb.with_parameter("multiple-symbols", "cancel"):
                    pass
                want["symbols"] = "ask"
            else:
                t.run("set confirm off")
                t.run("set scheduler-locking on")
                want.update(confirm="off", locking="on")
        yield  # (A prompt, which closes any window left.)
        t.back()
        t.check_in("is {}.".format(want["confirm"]), t.run("show confirm"), "confirm")
        t.check_in('"{}"'.format(want["locking"]), t.run("show scheduler-locking"),
                   "scheduler-locking")
        t.check_in('"{}"'.format(want["symbols"]), t.run("show multiple-symbols"),
                   "multiple-symbols")
    finally:
        t.run("set confirm off")
        t.run("set scheduler-locking replay")
        t.run("set multiple-symbols all")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_python_noted(t):
    """While a parked thread is selected, `python` runs on the thread as it
    really is (refused, a bare `python` typed at a terminal would leave its
    lines to run as gdb commands: terminal.body_lines), saying so once a
    selection; the gdb commands it runs show the selection, which stays."""
    before = t.regs()
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    out = t.run("python print('AT', gdb.selected_frame().name())")
    t.check_in("'python' runs on thread {} as it really is".format(t.stop_thread), out,
               "the note")
    t.check_in("AT realm_gdb_test_stop", out, "Python's selected frame")
    out = t.run("python gdb.execute('bt 1')")
    t.check_not_in("as it really is", out, "the note, again")
    t.check_in(PARKED_FRAME0, out, "bt, run from Python")
    t.check(re.search(r"^2$", t.run("pi 1 + 1"), re.M), "pi EXPR")
    t.check(sys.modules["__main__"]._selection() is not None, "the selection afterwards")
    t.back()
    t.check_regs(before)
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_break_pending(t):
    """With a parked thread selected, `break` of a function that isn't
    loaded makes no breakpoint (gdb's question whether to make it pending
    takes its default answer in the command's window), and says so; with
    `set breakpoint pending on` it makes one, as without realm-gdb."""
    t.select("chain0")
    if not t.plain:
        t.back()
        raise TestSkip("plain commands don't show parked threads here")
    t.check_in("no breakpoint was made", t.run("break rgt_no_such_function"), "the note")
    t.check_not_in("no breakpoint was made", t.run("break rgt_chain_level3"),
                   "a breakpoint made")
    t.run("set breakpoint pending on")
    try:
        t.check_not_in("no breakpoint was made", t.run("break rgt_no_such_function"),
                       "with breakpoint pending on")
    finally:
        t.run("set breakpoint pending auto")
    t.check_in("<PENDING>", t.run("info breakpoints"), "the pending breakpoint")
    t.back()
    t.run("delete")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_plain_refused_advice(t):
    """A plain command that can't show a selected parked thread (the program
    is being recorded, here) says it wasn't run, and that 'realm-thread
    back' would run it on the thread as it really is."""
    t.select("chain0")
    if not t.plain:
        raise TestSkip("plain commands don't show parked threads here")
    ok, out = t.try_run("record")
    if not ok:
        t.back()
        raise TestSkip("record: " + out)
    try:
        msg = t.expect_error("print 1", "'print' was not run")
        t.check_in("run 'realm-thread back' first to run it on thread {} as it really "
                   "is".format(t.stop_thread), msg, "the advice")
    finally:
        t.run("record stop")
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_refused_call_other_thread(t):
    """A function call the window refuses (simulated here), let through by
    may-call-functions turned back on, but made on another thread that a
    command in the window selected (`thread apply N print f()`), is stopped
    at its first instruction and popped there too, rather than left in that
    thread to run once the program does."""
    main = sys.modules["__main__"]
    if not main._can_refuse_calls:
        raise TestSkip("this gdb can't refuse function calls (no may-call-functions)")
    other = [th.num for th in gdb.selected_inferior().threads()
             if th.num != t.stop_thread][0]
    before = t.regs()
    with t.patched('_calls_refused', lambda window: "simulated"):
        error, out = t.run_capturing(
            "realm-thread apply {:#x} python gdb.execute('set may-call-functions on'); "
            "gdb.execute('thread apply {} print realm_gdb_test_add(1, 2)')".format(
                t.ut("chain0"), other))
    out += error or ""
    t.check_in("refused all the same", out, "the note")
    t.check_not_in("<function called from gdb>", t.run("thread apply {} bt".format(other)),
                   "thread {}'s stack afterwards".format(other))
    t.check(not main._unrestored, "threads left to put back")
    t.check_in("is on", t.run("show may-call-functions"), "may-call-functions afterwards")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_hook_stop(t):
    """A user's hook-stop, which gdb 13 runs when a function call made in a
    window stops (here unwound on a signal), runs inside the window: its
    `continue` is refused there.  The host's registers are put back, and the
    program runs to a clean exit."""
    hook = os.path.join(t.config["tmp"], "hook-stop.gdb")
    with open(hook, "w") as f:
        f.write("define hook-stop\necho RGT-HOOK-STOP\\n\ncontinue\nend\n")
    unhook = os.path.join(t.config["tmp"], "no-hook-stop.gdb")
    with open(unhook, "w") as f:
        f.write("define hook-stop\nend\n")
    before = t.regs()
    t.run("source " + hook)
    try:
        error, out = t.run_capturing("realm-thread apply {:#x} print realm_gdb_test_fault()"
                                     .format(t.ut("chain0")))
    finally:
        t.run("source " + unhook)
    out += error or ""
    t.check_not_in("Continuing", out, "hook-stop's continue")
    if "RGT-HOOK-STOP" in out:
        t.check_in("refusing 'continue'", out, "hook-stop's continue, refused")
    t.check(not t.exited, "the program exited")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["pascal", "in_window", "fortran", "modula_2_refused"])
def test_window_language(t):
    """realm-gdb parses its own expressions and register writes as C++,
    whatever the language: with `set language pascal` (which takes `set var
    $r12 = 0x10` for a comparison, and writes another value) parked threads
    are found and shown; and a command in a window that sets it doesn't
    keep the window from putting the host's registers back exactly."""
    before = t.regs()
    try:
        main = sys.modules["__main__"]
        if t.variant == "modula_2_refused":
            # A call the window refuses (simulated), let through by
            # may-call-functions turned on, is still stopped (its breakpoint
            # is set as C++: Modula-2 can't parse `*ADDR`).
            if not t.supports("with print pretty -- echo"):
                raise TestSkip("this gdb has no 'with'")
            t.run("set language modula-2")
            with t.patched('_calls_refused', lambda window: "simulated"):
                error, out = t.run_capturing(
                    "realm-thread apply {:#x} with may-call-functions on -- print "
                    "realm_gdb_test_add(1, 2)".format(t.ut("chain0")))
            t.check(error is not None, "the call ran:\n" + out)
            t.check_in("refused all the same", out, "the note")
        elif t.variant in ("pascal", "fortran"):
            t.run("set language " + t.variant)
            t.check_eq(_parked_frame_count(t.apply("chain0", "bt 2")), 2, "bt 2")
            # (The check of whether gdb can write the extended registers
            # reads one, as C++: Fortran can't parse it.)
            if main._arch.probe_reg is not None:
                t.check(main._probe_extended() is not main._NOT_CHECKED,
                        "the extended registers, checked")
        else:
            t.apply("chain0", "set language pascal")
        t.check_in('"{}"'.format({"fortran": "fortran", "modula_2_refused": "modula-2"}.get(
            t.variant, "pascal")), t.run("show language"), "the language afterwards")
        t.check(not sys.modules["__main__"]._unrestored, "threads left to put back")
    finally:
        t.run("set language auto")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_watchpoint_lost(t):
    """A function call made while a parked thread is shown makes gdb mark a
    watchpoint on the host thread's local variable for deletion (gdb checks
    its frame among those the host has then, the parked thread's, as the
    call resumes it, whether it is enabled or not): realm-gdb says so, and
    how to set it again.  `watch -l`'s isn't."""
    t.select_frame("probe_task")
    local = re.search(r"atchpoint (\d+)", t.run("watch probe_marker")).group(1)
    location = re.search(r"atchpoint (\d+)", t.run("watch -l probe_marker")).group(1)
    t.run("frame 0")
    out = t.apply("chain0", "print realm_gdb_test_add(2, 3)")
    t.check_in("= 5", out, "add()")
    t.check_in("deletes watchpoint {} at the next stop".format(local), out, "the warning")
    t.check_not_in("watchpoint {} at".format(location), out, "watch -l's")
    t.check_in("keep y", t.run("info breakpoints " + location), "watch -l's afterwards")
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",), variants=["call", "stopped_call", "internal_stopped_call"])
def test_window_finish_breakpoint(t):
    """A Python FinishBreakpoint on the host's own frame outlasts a function
    call made while a parked thread is shown; but gdb deletes it if the call
    stops (one refused, simulated here, but let through by `with
    may-call-functions on`), looking for its frame among the parked
    thread's, and realm-gdb says so, also of one made with internal=True,
    which gdb.breakpoints() lists too ("internal_stopped_call")."""
    if not hasattr(gdb, "FinishBreakpoint"):
        raise TestSkip("no gdb.FinishBreakpoint")
    if t.variant != "call" and not t.supports("with print pretty -- echo"):
        raise TestSkip("`with` (gdb >= 9)")
    t.select_frame("probe_task")
    finish = gdb.FinishBreakpoint(gdb.selected_frame(),
                                  internal=t.variant == "internal_stopped_call")
    num = finish.number
    t.run("frame 0")
    call = "print realm_gdb_test_add(2, 3)"
    if t.variant == "call":
        out = t.apply("chain0", call)
        t.check_in("= 5", out, "add()")
        t.check(finish.is_valid(), "the FinishBreakpoint, afterwards")
        t.check_not_in("deleted breakpoint", out, "a warning")
    else:
        main = sys.modules["__main__"]
        with t.patched('_calls_refused', lambda window: "simulated"):
            error, out = t.run_capturing("realm-thread apply {:#x} with may-call-functions "
                                         "on -- {}".format(t.ut("chain0"), call))
        t.check_in("refused all the same", out, "the call's note")
        t.check(not finish.is_valid(), "the FinishBreakpoint, afterwards (gdb kept it)")
        t.check_in("deleted breakpoint {}{} (a Python FinishBreakpoint, say) as a function "
                   "call".format(num, " (internal)" if num < 0 else ""), out, "the warning")
    t.check_host_stack()
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_session_options(t):
    """What kind of session this is, judged from gdb's command line, as gdb
    takes its options: abbreviations included (`--interp=mi`, `-interp mi2`,
    `--batch-s`, `--eval=new-ui ...`), and nothing after `--args`."""
    main = sys.modules["__main__"]
    real = main._gdb_args
    cases = [(["--interp=mi"], "MI"), (["-interp", "mi2"], "MI"), (["--interpreter=dap"], "DAP"),
             (["--ui=mi3"], "MI"), (["-i", "mi"], "MI"), (["-q", "-nx", "prog"], None),
             (["-ex", "print 1", "--args", "prog", "-i=mi"], None),
             (["--eval=new-ui mi /dev/pts/9"], "second UI"),
             (["-ex", "new-ui mi /dev/pts/9"], "second UI"),
             (["-ex", "new mi /dev/pts/9"], "second UI"),
             (["-co", "core"], "can't tell"),
             (["--no-escape-args", "prog", "-i=mi"], None)]
    if not hasattr(gdb.events, "gdb_exiting"):
        cases += [(["--batch-s"], "batch mode"), (["-batch"], "batch mode")]
    try:
        for args, want in cases:
            main._gdb_args = lambda args=args: ["gdb"] + args
            got = main._session_problem()
            if want is None:
                t.check(got is None, "{}: {}".format(args, got))
            else:
                t.check(got is not None and want in got, "{}: {}".format(args, got))
        # From `ps` (FreeBSD), which splits arguments with spaces: `-ex 'foo
        # -s' --interpreter mi`, where -s takes --interpreter for its value.
        for args, want in ((["-ex", "foo", "-s", "--interpreter", "mi", "prog"],
                            "can't tell from gdb's command line"),
                           (["-ex", "echo", "new-ui", "mi", "/dev/pts/9"], "can't tell"),
                           (["-q", "-ex", "echo", "hi", "prog"], None)):
            main._gdb_args = lambda args=args: main._SplitArgs(["gdb"] + args)
            got = main._session_problem()
            t.check(got is None if want is None else got is not None and want in got,
                    "{} from ps: {}".format(args, got))
    finally:
        main._gdb_args = real
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_internal_bypass(t):
    """A command realm-gdb runs for itself (`_internal`) leaves only that
    command unguarded: another that runs meanwhile (here from Python, as
    another extension's event handler might) is still refused in a
    window."""
    main = sys.modules["__main__"]

    def inside():
        return main._internal("python exec(\"try: gdb.execute('continue')\\n"
                              "except gdb.error as e: print(e)\")")
    t.check_in("refusing 'continue'", t.within("chain0", inside), "continue")
    t.check(not t.exited, "the program exited")
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["apply", "background", "unrestored"])
def test_window_bypass_stopped(t):
    """Something that resumes the program past realm-gdb's refusals while a
    host has a parked thread's registers (here a `continue` after a `define`
    in a block, for which gdb runs no hook, replaced the guard's
    hook-continue): realm-gdb stops the host before it runs, with a SIGINT
    of its own, and puts its registers back; the program runs on to the
    end.  In a window ("apply"), in one resumed in the background
    ("background"), whose host is stopped once gdb sees the SIGINT, and
    outside one, for a host whose registers couldn't be put back
    ("unrestored", simulated)."""
    main = sys.modules["__main__"]
    before = t.regs()
    path = os.path.join(t.config["tmp"], "bypass.gdb")
    with open(path, "w") as f:
        f.write("if 1\n  define hook-continue\n  end\nend\ncontinue{}\n".format(
            " &" if t.variant == "background" else ""))
    if t.variant == "unrestored":
        real = main._put_back

        def put_back(saved, *args):  # (Puts them back, but says it couldn't.)
            return real(saved, *args) or [sorted(saved)[0]]
        main._put_back = put_back
        try:
            t.check_in("could not put back", t.apply("chain0", "bt 1"), "the warning")
        finally:
            main._put_back = real
        out = t.run("source " + path)
        t.check_in("stopping thread {}".format(t.stop_thread), out, "the warning")
        out = t.run("realm-thread back")
        t.check_in("stopped at that SIGINT before it ran", out, "the note")
        t.check_in("realm-gdb: put back thread", out, "  and the registers")
    else:
        out = t.run("realm-thread apply {:#x} source {}".format(t.ut("chain0"), path))
        t.check_in("stopping thread {} before it runs".format(t.stop_thread), out,
                   "the warning")
        if t.variant == "background":
            t.check_in("gets its own registers back once it has stopped", out, "the note")
            for _ in t.wait_for_stop():
                yield
            out = t.run("realm-thread back")
        t.check_in("stopped at that SIGINT before it ran", out, "the note")
    t.check(not main._unrestored, "registers still to put back: {}\n{}".format(
        [main._unrestored_words(w) for w in main._unrestored], out))
    t.check_regs(before)
    t.check_host_stack()
    main._check_guard_hooks()  # (Put back realm-gdb's hook-continue.)
    t.check_in("realm-gdb-guard", t.run("show user hook-continue"), "the guard")
    yield
    out = t.run("continue")  # (It may stop at its breakpoint once more.)
    if not t.exited:
        t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop", "stopped in")
        t.continue_to_exit()


@test(scenarios=("live",))
def test_execute_mi_taken_early(t):
    """References to gdb.execute_mi that Python code took for itself before
    realm-gdb loaded (`from gdb import execute_mi` in a module, and a closure
    over one: simulated here, as realm-gdb wraps it again) are pointed at
    realm-gdb's wrapper too, so that -exec-continue from them is refused in
    a window."""
    if not hasattr(gdb, "execute_mi"):
        raise TestSkip("gdb.execute_mi is gdb >= 14")
    import types
    main = sys.modules["__main__"]
    real = gdb.execute_mi._realm_gdb_real
    module = types.ModuleType("rgt_early")
    module.execute_mi = real
    sys.modules["rgt_early"] = module

    def closure_over(f):
        return lambda *args: f(*args)
    closure = closure_over(real)
    try:
        main._install_execute_mi()
        hows = [("module", module.execute_mi)]
        if sys.version_info >= (3, 7):  # (Before, a closure's cells can't be changed.)
            hows.append(("closure", closure))
        for how, f in hows:
            def inside(f=f):
                try:
                    f("-exec-continue")
                except gdb.error as e:
                    return str(e)
                return "not refused"
            t.check_in("refusing", t.within("chain0", inside), how)
            t.check(not t.exited, "the program exited")
    finally:
        del sys.modules["rgt_early"]
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_load_interrupted(t):
    """A Ctrl-C while realm-gdb installs its hooks (simulated here, as on a
    first load, when no guard is in place yet) leaves windows refused, with
    the reason, rather than open without their guards; once it is installed,
    they open."""
    main = sys.modules["__main__"]
    real = main._define_hook
    made = []

    def define_hook(*args):
        made.append(args)
        if len(made) == 20:
            raise KeyboardInterrupt
        return real(*args)
    main._unguarded[:] = list(main._REQUIRED_GUARDS)  # (As a first load starts.)
    main._define_hook = define_hook
    try:
        try:
            main._install_hooks()
        except KeyboardInterrupt:
            pass
    finally:
        main._define_hook = real
    t.check(len(made) == 20, "the interruption")
    msg = t.expect_error("realm-thread apply {:#x} bt 1".format(t.ut("chain0")),
                         "couldn't install its guards")
    t.check_in("'continue'", msg, "the reason")
    main._install_hooks()
    t.check_eq(_parked_frame_count(t.apply("chain0", "bt 1")), 1, "bt 1, installed")
    t.expect_error("realm-thread apply {:#x} continue".format(t.ut("chain0")), "refusing")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_multiple_symbols(t):
    """With `set multiple-symbols ask`, a `break` on an overloaded function
    in `realm-thread apply` doesn't wait at gdb's menu (with its question
    collected, unseen, and the window open): windows answer it with all."""
    t.run("set multiple-symbols ask")
    try:
        out = t.apply("chain0", "break Realm::Processor::spawn")
        t.check(re.search(r"Breakpoint \d+ at .*\(\d+ locations\)", out),
                "a breakpoint at each:\n" + out)
        t.check(re.search(r'is "?ask', t.run("show multiple-symbols")), "multiple-symbols afterwards")
    finally:
        t.run("set multiple-symbols all")
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_window_other_inferior_running(t):
    """While another inferior's program runs in the background, parked
    threads aren't shown (a window's calls disable breakpoints, which are
    every inferior's), and realm-gdb says why."""
    if not os.path.exists("/bin/sleep"):
        raise TestSkip("no /bin/sleep")
    host = gdb.selected_thread().num
    t.run("add-inferior -exec /bin/sleep")
    t.run("inferior 2")
    t.run("run 60 &")
    t.run("inferior 1")
    try:
        msg = t.expect_error("realm-thread apply {:#x} bt 1".format(t.ut("chain0")),
                             "[Ii]nferior 2's program is running")
        t.check_in("interrupt -a", msg, "the advice")
    finally:
        t.run("kill inferiors 2")
        t.run("inferior 1")
        t.run("remove-inferiors 2")
        t.run("thread {}".format(host))
        t.exited = False  # (inferior 2's)
    yield
    t.continue_to_exit()


# ---- leaving ------------------------------------------------------------

@test(scenarios=("attach",), ending="release", variants=[
    "detach", "quit", "eof", ("python", "python gdb.execute('detach')"),
    ("mi", "interpreter-exec mi \"-target-detach\""),
    ("mi_exit", "interpreter-exec mi \"-gdb-exit\"")])
def test_leave(t):
    """Leaving an attached process in any way after showing parked threads
    (with function calls) lets it carry on cleanly."""
    _check_parked_section(t, t.run("thread apply all bt full"))

    def inside():
        t.select_frame("rgt_hot_wait")
        t.check_eq(t.eval_int("realm_gdb_test_add(2, 3)"), 5, "add()")
    t.within("hot", inside)
    t.select("hot")
    t.rt("frame 1")
    t.release()
    if t.plain and t.variant.startswith("interpreter-exec"):
        # (Refused while plain commands show the selection: test_plain_refused.)
        t.expect_error(t.variant, "refusing")
        t.run("realm-thread back")
    yield
    if t.variant == "eof":
        return  # the driver closes gdb's stdin
    yield typed_final(t.variant)


@test(scenarios=("attach",), interaction="batch")
def test_leave_batch(t):
    """`gdb -batch` ending (which runs no `quit` hook) right after showing
    parked threads, with function calls, lets the attached process carry on
    cleanly, with every Python exit handler run."""
    _check_parked_section(t, t.run("thread apply all bt"))

    def inside():
        t.select_frame("rgt_chain_level3")
        t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
                   t.chain_marker(0) + 1, "add()")
    t.within("chain0", inside)
    t.release()
    yield


@test(scenarios=("attach",), interaction="mi", variants=["-target-detach", "-gdb-exit"])
def test_mi_leave(t):
    """MI detach/exit after showing parked threads, as an IDE's 'stop
    debugging' does: the attached program carries on cleanly."""
    yield typed('-interpreter-exec console "thread apply all bt full"')
    t.check_mi_result("done")
    t.within("hot", lambda: t.check_eq(t.eval_int("realm_gdb_test_add(2, 3)"), 5, "add()"))
    t.release()
    yield
    yield typed_final(t.variant)


@test(scenarios=("live",), ending="none", cpus=2, variants=["dump", "call"])
def test_quit_in_user_command(t):
    """A user-defined command shows parked threads, then quits: gdb exits
    cleanly (gdb 14 and older would crash as they exit if an event posted
    for later were still waiting)."""
    if t.variant == "dump":
        body = "thread apply all bt full"
    else:
        body = "realm-thread apply {:#x} print realm_gdb_test_nap(1000)".format(
            t.ut("chain0"))
    t.run("define rgt-then-quit\n{}\nquit\nend".format(body))
    if t.info.interaction == "mi":
        yield typed_final('-interpreter-exec console "rgt-then-quit"')
    else:
        yield typed_final("rgt-then-quit")


@test(scenarios=("live",), interaction="mi", ending="none", cpus=2,
      variants=["dump", "call"])
def test_mi_quit_in_user_command(t):
    """test_quit_in_user_command through MI (as from an IDE's console)."""
    return test_quit_in_user_command(t)


@test(scenarios=("freeze",))
def test_frozen(t):
    """Attach to a process frozen by REALM_FREEZE_ON_ERROR: look around,
    make calls, and detach without disturbing it."""
    out = t.run("thread apply all bt")
    _check_parked_section(t, out)
    t.check_in("realm_freeze", out, "frozen thread's stack")
    yield

    def inside():
        t.select_frame("rgt_chain_level3")
        t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
                   t.chain_marker(0) + 1, "add()")
    t.within("chain0", inside)
    yield
    yield typed_final("detach")


# ---- MI, as under an IDE -------------------------------------------------

@test(scenarios=("live",), interaction="mi", variants=["dump", "apply"])
def test_mi_console(t):
    """MI's -interpreter-exec console runs `thread apply all bt` and
    `realm-thread apply` (no prompts, so nothing is left to one), and the
    thread's registers are as they were afterwards."""
    before = t.regs()
    if t.variant == "dump":
        yield typed('-interpreter-exec console "thread apply all bt full"')
    else:
        yield typed('-interpreter-exec console "realm-thread apply {:#x} print '
                    'realm_gdb_test_add(2, 3)"'.format(t.ut("chain0")))
    t.check_mi_result("done")
    t.check_regs(before)
    t.check_host_stack()
    yield typed("-stack-list-frames 0 3")
    t.check_mi_result("done", r'func="realm_gdb_test_stop"')
    yield
    t.continue_to_exit()


@test(scenarios=("live",), interaction="mi")
def test_mi_resume(t):
    """MI -exec-continue after showing parked threads lets the program run
    to a clean exit."""
    _check_parked_section(t, t.run("thread apply all bt"))
    yield
    yield typed("-exec-continue")
    t.check_mi_result("running")
    for _ in t.wait_for_exit():
        yield
    t.check_exited("-exec-continue")


@test(scenarios=("live",), interaction="mi")
def test_mi_breakpoint_in_call(t):
    """MI: a call made in `realm-thread apply` with a breakpoint in the way
    completes; the breakpoint is as it was afterwards."""
    yield typed("-break-insert realm_gdb_test_opaque")
    t.check_mi_result("done")
    yield typed('-interpreter-exec console "realm-thread apply {:#x} print '
                'realm_gdb_test_add(2, 3)"'.format(t.ut("chain0")))
    t.check_mi_result("done")
    bp = gdb.breakpoints()[-1]
    t.check(bp.enabled and bp.hit_count == 0, "breakpoint afterwards: enabled={} hits={}"
            .format(bp.enabled, bp.hit_count))
    t.run("delete")
    yield
    t.continue_to_exit()


@test(scenarios=("live",), interaction="mi_new_ui", variants=["dump", "apply"])
def test_mi_new_ui(t):
    """MI on a second UI of a console gdb (`new-ui`), whose console sits at
    its prompt: test_mi_console there, except that function calls are
    refused while a parked thread is shown (gdb would run the other UI's
    commands during one)."""
    if t.variant == "dump":
        return test_mi_console(t)
    return _mi_new_ui_apply(t)


def _mi_new_ui_apply(t):
    before = t.regs()
    yield typed('-interpreter-exec console "realm-thread apply {:#x} print '
                'realm_gdb_test_add(2, 3)"'.format(t.ut("chain0")))
    if sys.modules["__main__"]._can_refuse_calls:
        t.check_mi_result("error", r"may-call-functions is off")
    else:  # (gdb 8 warns instead.)
        t.check_mi_result("done")
    yield typed('-interpreter-exec console "realm-thread apply {:#x} bt 1"'.format(
        t.ut("chain0")))
    t.check_mi_result("done")
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


# ---- the context-switch window ------------------------------------------

class _SwitchingBreakpoint(gdb.Breakpoint if gdb else object):
    """Stops when a Realm user thread (`only` that one, if given) is
    executing although ut->running is false: at `spec`, with the stack
    pointer inside the user thread's stack."""

    def __init__(self, ctx, spec, only=None):
        super(_SwitchingBreakpoint, self).__init__(spec, internal=True)
        self.ctx = ctx
        self.only = only
        self.switching = None

    def stop(self):
        try:
            sp = int(gdb.newest_frame().read_register("sp" if "aarch64" in
                     gdb.newest_frame().architecture().name() else "rsp"))
            for ut in self.ctx.user_threads():
                if self.only is not None and ut.ptr != self.only:
                    continue
                if not ut.running and ut.contains(sp & _U64):
                    self.switching = ut.ptr
                    return True
        except gdb.error:
            pass
        return False


@test(scenarios=("live",), variants=["out", "in"])
def test_switching_user_thread(t):
    """A user thread caught in the middle of a context switch, while
    ut->running is false but it is executing, isn't taken for a parked
    one, whichever thread is selected: "out" stops it in swapcontext() as
    it switches out, "in" just after swapcontext() returns into it (where
    its saved context is the one it has just resumed from)."""
    if t.variant == "out":
        bps = [_SwitchingBreakpoint(t, "swapcontext")]
    else:
        # A parked thread resumes where its saved context says (its frame
        # #0 when shown), just after its own swapcontext() call.
        pcs = set()
        for role in ("chain0", "chain1", "chain2", "recurse", "hot", "opt"):
            pcs.add(t.within(role, lambda: gdb.newest_frame().pc()))
        bps = [_SwitchingBreakpoint(t, "*{:#x}".format(pc)) for pc in sorted(pcs)]
    t.run("continue")  # (the program releases the parked tasks)
    switching = [bp.switching for bp in bps if bp.switching is not None]
    t.check(switching, "did not stop in a context switch (last stop: {})"
            .format(t.last_stop))
    ptr = switching[0]
    host = gdb.selected_thread().num
    other = min(th.num for th in gdb.selected_inferior().threads() if th.num != host)
    t.run("thread {}".format(other))
    yield
    t.expect_error("realm-thread apply {:#x} bt".format(ptr),
                   "switching contexts on thread {}".format(host))
    out = t.run("realm-thread")
    t.check(re.search(r"{:#x}\s+RUNNING\s+\(on thread {}, switching".format(
        ptr, host), out), "realm-thread list:\n" + out)
    out = t.run("thread apply all bt")
    t.check(re.search(r"\({:#x}\):\n\s*<not shown: it is switching contexts on thread {}".format(
        ptr, host), out), "parked section:\n" + out)
    yield
    for bp in bps:
        bp.delete()
    t.run("delete")
    t.continue_to_exit()


def _main_thread(t):
    """The program's main thread, waiting for Realm to shut down."""
    num = t.find_thread_with_frame("main")
    t.check(num is not None and num != t.stop_thread, "no main thread to use")
    return num


# ---- non-stop mode and a running program ---------------------------------

def _stop_all(t):
    """Generator: stop every thread (non-stop mode)."""
    t.run("interrupt -a")
    import time
    deadline = time.time() + 30
    while (any(th.is_running() for th in gdb.selected_inferior().threads())
           and time.time() < deadline):
        time.sleep(0.1)
        yield
    t.check(not any(th.is_running() for th in gdb.selected_inferior().threads()),
            "threads still running after interrupt -a")


def _restart_non_stop(t):
    """_setup ran the program in all-stop mode; start over in non-stop
    mode, stopped at the same place."""
    t.run("kill")
    t.run("set non-stop on")
    ok, out = t.try_run("run {} < /dev/null".format(t.program_args()))
    if not ok and "does not support running in non-stop mode" in out:
        raise TestSkip(out)  # (FreeBSD's native target)
    t.check(ok, "run in non-stop mode: " + out)
    # In non-stop mode gdb doesn't switch to the thread that stopped.
    stopped = [th for th in gdb.selected_inferior().threads() if th.is_stopped()]
    t.check_eq(len(stopped), 1, "stopped threads")
    t.run("thread {}".format(stopped[0].num))
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop",
               "stopped in")
    t.stop_thread = gdb.selected_thread().num
    t.find_roles()


def _check_says_why(t, text, what, why="non-stop mode"):
    """A message about parked threads not being shown says why, and how to
    see them."""
    t.check_in(why, text.lower(), what)
    advice = {"non-stop mode": "gcore", "the program is running": "('interrupt')",
              "being recorded": "record stop"}[why]
    t.check_in(advice, text, what)


def _check_not_shown(t, out, num_parked, why="non-stop mode"):
    """`thread apply all bt` ends with a section saying how many parked
    threads it couldn't show (if it can tell: gdb 9 can't read memory
    through a running thread), why, and how to see them."""
    section = _parked_section(t, out)
    t.check(re.match(r"Parked Realm user threads(?: \({}\))?: not shown ===\n".format(
        num_parked), section), "parked section:\n" + section)
    _check_says_why(t, section, "parked section", why)
    t.check(out.rstrip().endswith(PARKED_END), "parked section isn't last:\n" + out)


@test(scenarios=("live",))
def test_non_stop(t):
    """In non-stop mode, parked threads are listed, but not shown, even once
    every thread is stopped: that would put a parked thread's registers in a
    stopped thread while others may run.  realm-gdb says so, and how to see
    them."""
    _restart_non_stop(t)
    yield
    running = [th.num for th in gdb.selected_inferior().threads() if th.is_running()]
    t.check(running, "no other thread is running in non-stop mode")
    num_parked = len([u for u in t.user_threads() if not u.running])
    for num in (running[0], t.stop_thread):
        t.run("thread {}".format(num))
        _check_says_why(t, t.expect_error("realm-thread apply {:#x} bt".format(
            t.ut("chain0"))), "realm-thread apply")
        _check_says_why(t, t.expect_error("realm-thread {:#x}".format(
            t.ut("chain0"))), "realm-thread PTR")
    out = t.run("realm-thread")
    t.check(re.search(r"{:#x}\s+parked".format(t.ut("chain0")), out),
            "realm-thread list:\n" + out)
    _check_says_why(t, out, "realm-thread list")
    # Like gdb's own, plain `thread apply all bt` stops at the first running
    # thread; -c (gdb >= 9) carries on past it.
    if t.supports("thread apply {} -c print 1".format(t.stop_thread)):
        ok, out = t.run_clean("thread apply all -c bt full")
        t.check(ok, "thread apply all -c bt full: " + out)
        _check_not_shown(t, out, num_parked)
        t.check_not_in("rgt_chain_level3", out, "thread apply all -c bt full")
    error, out = t.run_capturing("thread apply all bt")
    t.check(error is not None, "thread apply all bt didn't fail:\n" + out)
    t.check_clean_error("thread apply all bt", error)
    _check_not_shown(t, out, num_parked)
    yield
    for step in _stop_all(t):
        yield step
    t.run("thread {}".format(t.stop_thread))
    _check_not_shown(t, t.run("thread apply all bt"), num_parked)
    _check_says_why(t, t.expect_error("realm-thread apply {:#x} bt".format(
        t.ut("chain0"))), "realm-thread apply with every thread stopped")
    yield
    t.continue_to_exit("continue -a")


@test(scenarios=("live",))
def test_running_in_background(t):
    """While the program runs in the background (all-stop mode, `continue
    &`), parked threads can't be shown; realm-gdb says so, and how to see
    them, rather than leaving them out."""
    num_parked = len([u for u in t.user_threads() if not u.running])
    dash_c = t.supports("thread apply all -c print 1")
    # (Keep the stopped thread, and so the parked ones, where they are.)
    t.run("set var realm_gdb_test_hold = 1")
    t.run("continue &")
    yield
    why = "the program is running"
    _check_says_why(t, t.expect_error("realm-thread apply {:#x} bt".format(
        t.ut("chain0"))), "realm-thread apply", why)
    _check_says_why(t, t.expect_error("realm-thread {:#x}".format(
        t.ut("chain0"))), "realm-thread PTR", why)
    out = t.run("realm-thread")
    _check_says_why(t, out, "realm-thread list", why)
    if dash_c:
        ok, out = t.run_clean("thread apply all -c bt")
        t.check(ok, "thread apply all -c bt: " + out)
        _check_not_shown(t, out, num_parked, why)
    error, out = t.run_capturing("thread apply all bt")
    t.check(error is not None, "thread apply all bt didn't fail:\n" + out)
    t.check_clean_error("thread apply all bt", error)
    _check_not_shown(t, out, num_parked, why)
    yield
    t.run("interrupt")
    import time
    deadline = time.time() + 30
    while (any(th.is_running() for th in gdb.selected_inferior().threads())
           and time.time() < deadline):
        time.sleep(0.1)
        yield
    t.run("set var realm_gdb_test_hold = 0")
    t.continue_to_exit()


# ---- core files: views ---------------------------------------------------

@test(scenarios=("core", "kcore"))
def test_core_show(t):
    """In a core file `realm-thread <ptr>` keeps a parked thread's stack
    shown, with working frame selection, locals, arguments and registers;
    `realm-thread back` restores the thread exactly."""
    thread = gdb.selected_thread().num
    before = t.regs()
    for i in range(NUM_CHAIN):
        out = t.show("chain{}".format(i))
        t.check_in("#0 is a marker for thread", out, "realm-thread's description of #0")
        yield
        t.check_chain_stack(i)
        sel = t.frame_name(gdb.selected_frame())
        t.check(sel not in ("realm_gdb_test_stop", "rgt_hot_probe"),
                "selected frame is still the host's: " + sel)
        t.select_frame("rgt_chain_level3")
        yield
        t.check_eq(t.eval_int("chain_marker"), t.chain_marker(i), "chain_marker")
        t.check_in("chain_marker = {}".format(t.chain_marker(i)),
                   t.run("info locals"), "info locals")
        for cmd in ("info frame", "x/2i $pc", "info registers", "list", "frame", "bt full"):
            t.check_no_python_error(cmd, t.run(cmd))
        t.run("up")
        t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2", "after up")
        t.back()
        yield
        t.check_eq(gdb.selected_thread().num, thread, "selected thread")
        t.check_regs(before)
        t.check_host_stack()


@test(scenarios=("core",))
def test_core_show_nested(t):
    """Views nest: `realm-thread back` pops one level at a time."""
    before = t.regs()
    t.show("chain0")
    yield
    t.show("chain1")
    t.check_chain_stack(1)
    yield
    t.back()
    t.check_chain_stack(0)
    t.select_frame("rgt_chain_level3")
    t.check_eq(t.eval_int("chain_marker"), t.chain_marker(0), "chain_marker")
    out = t.apply("chain2", "bt")
    t.check_in("rgt_chain_level3", out, "realm-thread apply on top of a view")
    t.check_chain_stack(0, "the view after realm-thread apply")
    yield
    t.back()
    t.check_regs(before)
    t.check_host_stack()
    t.check_in("no parked user thread is selected", t.run("realm-thread back"))


@test(scenarios=("core",))
def test_core_rt_error(t):
    """In a core file, `rt COMMAND`'s error is gdb's own, not a Python
    exception."""
    t.show("chain0")
    msg = t.expect_error("rt print rgt_nosuchvar", "No symbol")
    t.check_not_in("Python", msg, "the error")
    t.back()
    yield


@test(scenarios=("core",))
def test_core_thread_apply_all_shown(t):
    """In a core file, `thread apply all bt` shows gdb's threads as they are
    even with a parked thread shown on one (as in a live process), and
    leaves it shown afterwards, at the frame selected."""
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield
    out = t.run("thread apply all bt")
    host = out[:out.index("Parked Realm user threads")]
    t.check_in("realm_gdb_test_stop", host, "the threads' own stacks")
    t.check_not_in("<parked Realm user thread", host, "the threads' own stacks")
    t.check_not_in("rgt_chain_level3", host, "the threads' own stacks")
    t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level3",
               "the frame selected afterwards")
    t.check_chain_stack(0, "the stack shown afterwards")
    t.back()
    t.check_host_stack()


@test(scenarios=("core",))
def test_core_thread_switch(t):
    """In a core file, a parked thread shown on one thread stays with
    that thread."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.show("chain0")
    t.check(re.search(r"{:#x}\s+parked\s+\(shown on thread {}\)".format(
        t.ut("chain0"), t.stop_thread), t.run("realm-thread")), "realm-thread list")
    yield
    t.run("thread {}".format(other))
    names = t.frame_names()
    t.check_not_in("rgt_chain_level3", names, "other thread's stack")
    yield
    t.run("thread {}".format(t.stop_thread))
    yield
    t.check_chain_stack(0, "stack after switching back")
    t.back()
    t.check_host_stack()


@test(scenarios=("core",))
def test_core_frame_cache_flush(t):
    """The parked stack survives gdb flushing its frame cache."""
    t.show("chain0")
    t.select_frame("rgt_chain_level2")
    yield
    t.run_clean("maintenance flush register-cache")
    yield
    t.check_chain_stack(0, "stack after flushing the register cache")
    t.run("set backtrace limit 100")
    yield
    t.check_chain_stack(0, "stack after changing the backtrace limit")
    t.run("set backtrace limit unlimited")
    t.back()
    t.check_host_stack()


@test(scenarios=("core",), variants=["bt -no-filters", "bt full", "bt 2",
                                     ("bt_outermost_3", "bt -3")])
def test_core_bt_options(t):
    """bt options while showing a parked thread in a core file."""
    if not t.supports(t.variant):
        raise TestSkip("this gdb does not support '{}'".format(t.variant))
    t.show("chain0")
    yield
    ok, out = t.run_clean(t.variant)
    t.check(ok, "{}: {}".format(t.variant, out))
    yield
    t.back()
    t.check_host_stack()


@test(scenarios=("core",))
def test_core_reload_while_showing(t):
    """Loading the script again (another copy of librealm, or by hand)
    while a parked thread is shown keeps it shown, and the selected frame."""
    t.show("chain0")
    t.select_frame("rgt_chain_level2")
    yield
    t.run("source {}".format(t.config["script"]))
    yield
    t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2", "selected frame")
    t.check_chain_stack(0)
    t.back()
    t.check_host_stack()


@test(scenarios=("core",), variants=["bt", "thread apply all bt", "realm-thread apply"])
def test_marker_line(t):
    """In a core file, frame #0 of a thread showing a parked thread (the
    thread's own) is drawn as a marker line, which `bt N` counts as a frame,
    and `thread apply all bt N` and `realm-thread apply PTR bt N` don't."""
    marker = "<parked Realm user thread {:#x}>".format(t.ut("chain0"))
    if t.variant == "bt":
        t.check_in("#0 is a marker for thread", t.show("chain0"),
                   "realm-thread's description of frame #0")
        out = t.run("bt 3")
        expected = 3
        t.check_no_python_error("bt full", t.run("bt full"))
        t.check_not_in("<parked", t.run("bt no-filters 1"), "'bt no-filters 1'")
    elif t.variant == "thread apply all bt":
        out = _split_parked(_parked_section(t, t.run("thread apply all bt 3")))[
            t.ut("chain0")]
        expected = 4
    else:
        out = t.apply("chain0", "bt 3")
        expected = 4
    frames = re.findall(r"^#\d+\s+(.*)$", out, re.M)
    t.check(frames and frames[0].startswith(marker),
            "frame #0 isn't {}:\n{}".format(marker, out))
    t.check_eq(len(frames), expected, "frames printed:\n" + out)
    yield
    if t.variant == "bt":
        t.back()


@test(scenarios=("core",))
def test_marker_line_printers(t):
    """In a core file the marker line is drawn whatever pretty-printers are
    in use (libstdc++'s, and a test one): nothing in a core file can call
    functions, which is what made the other branch leave it out on gdb 8."""
    install_pair_printer()
    out = t.show("chain0")
    t.check_in("#0 is a marker for thread", out, "realm-thread's description of frame #0")
    t.check_in("<parked Realm user thread", t.run("bt 1"), "bt 1")
    yield
    t.back()


@test(scenarios=("core",), interaction="mi")
def test_mi_marker_frame_filters(t):
    """MI with frame filters enabled (as Emacs does), on a core file: frame
    #0 is the marker; an entirely optimized-out value is `<optimized out>`,
    with its own type; and nothing realm-gdb puts in place to print it is
    left afterwards."""
    install_pair_printer()
    yield typed("-enable-frame-filters")
    t.check_mi_result("done")
    t.show("chain0")
    yield typed("-stack-list-frames 0 1")
    t.check_mi_result("done", r'level="0",func="<parked Realm user thread')
    level = t.select_frame("rgt_chain_level3")
    yield typed("-stack-list-variables --all-values")
    t.check_mi_result("done", r'name="chain_marker",value="{}"'.format(t.chain_marker(0)))
    yield typed("-stack-list-arguments --all-values {0} {0}".format(level))
    t.check_mi_result("done")
    t.check_eq(_printing_lookups(), [], "realm-gdb's pretty-printer lookup, left in place")
    t.back()
    t.show("opt")
    raw = _opt_values(t)
    gone = sorted(n for n, v in raw.items() if v == "<optimized out>" and n != "given")
    if gone:
        yield typed("-stack-list-variables --all-values")
        t.check_mi_result("done", r'name="{}",value="<optimized out>"'.format(gone[0]))
        yield typed("-stack-list-variables --simple-values")
        t.check_mi_result("done", r'name="{}",type="RgtPair"'.format(gone[0]))
    t.check_eq(_printing_lookups(), [], "realm-gdb's pretty-printer lookup, left in place")
    t.back()
    yield


def _split_top(text, sep):
    """Split `text` at each `sep` outside brackets and quotes."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i])
            start = i + len(sep)
            i = start
            continue
        i += 1
    parts.append(text[start:])
    return parts


def _frame_line_parts(line):
    """(address or None, [(argument, value)], location or None) of a `bt`
    frame line, e.g. `0x1234 in f (a=1, b=...) at f.c:3`."""
    address = None
    m = re.match(r"(0x[0-9a-f]+)(?: \[PAC\])? in (.*)$", line)
    if m:
        address, line = m.group(1), m.group(2)
    # The argument list is the last bracketed group at the top level; what
    # follows it is the location.
    depth, quote, open_at, groups, i = 0, None, None, [], 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "([{":
            if depth == 0 and c == "(":
                open_at = i
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0 and c == ")" and open_at is not None:
                groups.append((open_at, i))
                open_at = None
        i += 1
    if not groups:
        return address, [], None
    start, end = groups[-1]
    inner = line[start + 1:end]
    args = [a.partition("=")[::2] for a in _split_top(inner, ", ")] if inner else []
    rest = line[end + 1:].strip()
    return address, args, rest or None


def _gdb_major():
    m = re.match(r"\s*(\d+)\.", gdb.VERSION)
    return int(m.group(1)) if m else None


def _native_checks():
    """Whether gdb's own printer prints values that are entirely optimized
    out without asking pretty-printers (gdb 10 does ask for locals)."""
    return _gdb_major() != 10


def _same_value(marked, native):
    # (gdb 10 to 12 don't tell realm-gdb when they print in summary mode,
    # so it may show `<optimized out>` for an argument gdb shows as `...`;
    # and gdb 10's own printer gives pretty-printers locals that are
    # entirely optimized out, or shows them in pieces.)
    if marked == native or (native == "..." and marked == "<optimized out>"):
        return True
    return (not _native_checks() and marked == "<optimized out>"
            and ("<optimized out>" in native or native.startswith("Python Exception")))


def _check_like_native(t, args, what):
    """`bt ARGS`, with the marker line, shows the parked frames as gdb's own
    printer does (`bt no-filters ARGS`), but for how it looks: the same
    frames, numbered the same, with the same arguments and locals, and no
    more Python errors.  Returns the output."""
    marked = t.run("bt " + args)
    native = t.run("bt no-filters " + args)
    m_frames, n_frames = _bt_frames(marked), _bt_frames(native)
    t.check(m_frames and m_frames[0][1].startswith("<parked Realm user thread"),
            "{}: no marker line:\n{}".format(what, marked))
    if len(m_frames) < 2:
        t.fail("{}: no parked frames:\n{}".format(what, marked))
    host_level = m_frames[1][0] - 1
    n_parked = [f for f in n_frames if f[0] > host_level]
    t.check_eq([f[0] for f in m_frames[1:]], [f[0] for f in n_parked],
               "{}: frame numbers (bt {}):\n{}\n----- bt no-filters {}:\n{}".format(
                   what, args, marked, args, native))
    for (level, m_line, m_locals), (_, n_line, n_locals) in zip(m_frames[1:], n_parked):
        where = "{}: frame #{}".format(what, level)
        m_addr, m_args, m_loc = _frame_line_parts(m_line)
        n_addr, n_args, n_loc = _frame_line_parts(n_line)
        if m_addr and n_addr:
            t.check_eq(m_addr, n_addr, where + ": address")
        if m_loc and n_loc and m_loc.startswith("at ") and n_loc.startswith("at "):
            # (gdb 9's adds " from LIBRARY" to a frame in one.)
            t.check_eq(m_loc.split(" from ")[0], n_loc.split(" from ")[0],
                       where + ": location")
        t.check_eq([a[0] for a in m_args], [a[0] for a in n_args],
                   "{}: arguments ({} / {})".format(where, m_line, n_line))
        for (name, m_value), (_, n_value) in zip(m_args, n_args):
            if not _same_value(m_value, n_value):
                t.fail("{}: argument {}={} (as gdb shows it: {})".format(
                    where, name, m_value, n_value))
        # (realm-gdb lists the locals as gdb does: gdb's FrameDecorator
        # would leave out some and, before gdb 13, add others.)
        t.check_eq(len(m_locals), len(n_locals), "{}: locals ({} / {})".format(
            where, m_locals, n_locals))
        for (m_name, m_value), (n_name, n_value) in zip(m_locals, n_locals):
            t.check_eq(m_name, n_name, "{}: locals ({} / {})".format(
                where, m_locals, n_locals))
            if not _same_value(m_value, n_value):
                t.fail("{}: {} = {} (as gdb shows it: {})".format(
                    where, m_name, m_value, n_value))
    t.check(marked.count("Python Exception") <= native.count("Python Exception"),
            "{}: Python errors in bt {}:\n{}\n----- bt no-filters {}:\n{}".format(
                what, args, marked, args, native))
    return marked


_PARKED_ROLES = (["chain{}".format(i) for i in range(NUM_CHAIN)]
                 + ["recurse", "hot", "fhot", "opt"])


@test(scenarios=("core",), variants=["full", "plain", "all args", "entry values"])
def test_marker_like_native(t):
    """In a core file, backtraces with the marker line (printed by gdb's
    Python frame printer) show every parked thread's frames as gdb's own
    printer does, with libstdc++'s pretty-printers and one that, like them,
    can't print a value that is optimized out: in particular, those entirely
    optimized out as `<optimized out>`."""
    if _native_checks():  # (For gdb 10's own printer it fails: see above.)
        install_pair_printer()
    if t.variant == "all args":
        t.run("set print frame-arguments all")
    elif t.variant == "entry values":
        t.run("set print entry-values both")
    args = "" if t.variant == "plain" else "full"
    for role in _PARKED_ROLES:
        t.show(role)
        _check_like_native(t, args, role)
        t.back()
        yield


def _opt_values(t):
    """{name: value} of rgt_opt_wait's RgtPair locals and argument, as gdb
    itself prints them without pretty-printers."""
    t.select_frame("rgt_opt_wait")
    return {name: t.run("print/r " + name).partition(" = ")[2].strip()
            for name in ("gone", "half", "given")}


@test(scenarios=("core",), variants=["bt full", "bt", "thread apply all bt full"])
def test_marker_optimized_out(t):
    """In a core file, values entirely optimized out are printed as
    `<optimized out>` (or, for an argument, in its frame's line, `...`) by
    backtraces with the marker line, as gdb itself prints them, rather than
    by a pretty-printer that, like libstdc++'s, can't print them; those
    partly optimized out are left to it, as gdb leaves them."""
    install_pair_printer()
    t.show("opt")
    raw = _opt_values(t)
    gone = sorted(n for n, v in raw.items() if v == "<optimized out>")
    if not gone:
        raise TestSkip("the compiler kept every RgtPair value in rgt_opt_wait: {}"
                       .format(raw))
    out = t.run(t.variant)
    native = ""
    if _native_checks():  # (For gdb 10's own printer the printer fails.)
        native = t.run(t.variant.replace("bt", "bt no-filters", 1))
    if t.variant.startswith("thread apply"):
        out = _split_parked(_parked_section(t, out))[t.ut("opt")]
        if native:
            native = _split_parked(_parked_section(t, native))[t.ut("opt")]
    frames = [f for f in _bt_frames(out) if " rgt_opt_wait" in " " + f[1]]
    t.check(frames, "no rgt_opt_wait frame in:\n" + out)
    _, line, local_values = frames[0]
    shown = dict(local_values)
    shown.update(_frame_line_parts(line)[1])
    for name in gone:
        if name == "given":
            # (In its frame's line: see test_optimized_out.)
            t.check(shown.get(name) in ("...", "<optimized out>"),
                    "argument {}={} in:\n{}".format(name, shown.get(name), out))
        elif t.variant != "bt":
            t.check_eq(shown.get(name), "<optimized out>", "{} in:\n{}".format(name, out))
    t.check_eq(out.count("Python Exception"), native.count("Python Exception"),
               "Python errors:\n" + out)
    yield
    t.back()


def _printing_lookups():
    """The pretty-printer lookup functions realm-gdb puts in place while a
    backtrace with the marker line is printed, wherever they are."""
    lists = [gdb.pretty_printers, gdb.current_progspace().pretty_printers]
    lists += [o.pretty_printers for o in gdb.objfiles()]
    return [p for l in lists for p in l if getattr(p, "name", None) == "realm-gdb"]


_left_open = []


@test(scenarios=("core",))
def test_marker_printing_left_open(t):
    """Should something leave an iteration of the frame filter's frames
    for gdb's frame printer unfinished in the middle of printing a frame's
    values, what realm-gdb put in place for it is gone once gdb has
    finished the command."""
    import gdb.frames
    t.show("chain0")
    frames = gdb.frames.execute_frame_filters(gdb.newest_frame(), 0, -1)
    next(frames)  # (the marker line)
    for dec in frames:  # (the first parked frame with arguments: see PARKED_FRAME0)
        args = iter(dec.frame_args() or ())
        if next(args, None) is not None:
            break
    _left_open[:] = [frames, args]
    t.check(_printing_lookups(), "nothing in place while printing a frame's values")
    yield
    t.check_eq(_printing_lookups(), [], "realm-gdb's pretty-printer lookup, left in place")
    del _left_open[:]
    t.check_eq(_printing_lookups(), [], "realm-gdb's pretty-printer lookup, left in place")
    t.back()
