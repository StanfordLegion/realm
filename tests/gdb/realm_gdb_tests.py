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
interactively.  The driver feeds `python rgt.step()` lines to gdb's stdin, one
per step.

Tests describe what a user should see, not how realm-gdb.py achieves it:
frames are found by function name rather than by number, and the program
itself checks, once resumed, that nothing the debugger did corrupted it.

In a live process a parked user thread is shown for one command at a time
(`realm-thread apply PTR COMMAND`), so a test looks at it from inside such a
command with Context.within, which runs a Python function there.  In a core
file `realm-thread PTR` also keeps it shown (Context.show and back).

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

    # ---- running commands ----

    def run(self, cmd):
        """Run a gdb command, returning its output.  Any error fails the
        test."""
        try:
            return gdb.execute(cmd, to_string=True)
        except gdb.error as e:
            raise TestFailure("command {!r} failed: {}".format(cmd, e))

    def try_run(self, cmd):
        try:
            return True, gdb.execute(cmd, to_string=True)
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

    def run_capturing(self, cmd):
        """Run a command, returning (its error message, or None if it
        succeeded, and what it printed), which gdb.execute(to_string=True)
        doesn't return for a command that fails."""
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
        and vector ones included), or with gdb < 10 the general ones."""
        arch = gdb.selected_frame().architecture()
        try:
            names = [r.name for r in arch.registers("save")]
            if names:
                return names
        except AttributeError:
            pass
        if "x86-64" in arch.name() or "x86_64" in arch.name():
            return (["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"]
                    + ["r{}".format(i) for i in range(8, 16)]
                    + ["rip", "eflags"])
        if "aarch64" in arch.name():
            return ["x{}".format(i) for i in range(31)] + ["sp", "pc", "cpsr"]
        raise TestSkip("unsupported architecture " + arch.name())

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
        diff = ["{}: {} -> {}".format(r, _hex(before[r]), _hex(after[r]))
                for r in before if before[r] != after.get(r)]
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
        """Let an attached program carry on once gdb lets go of it."""
        self.run("set var realm_gdb_test_release = 1")


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


def step():
    if _current is not None:
        _current.step()


def run_all():
    """Batch mode: run every step back to back, without a prompt."""
    while _current is not None and not _current.done:
        _current.step()


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


def _starts(name, prefix):
    return name.startswith(prefix)


USER_SWITCH = "Realm::UserThread::user_switch"


def _check_parked_frame0(t, what):
    """Inside a register window, frame #0 is the parked thread's own
    (user_switch, where it called swapcontext); in a core-file view it is
    the host's."""
    if not t.is_core:
        t.check(_starts(t.frame_name(gdb.newest_frame()), USER_SWITCH),
                "{}: frame #0 is {}, not {}".format(
                    what, t.frame_name(gdb.newest_frame()), USER_SWITCH))


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
                "help thread apply all", "help realm-thread-guard"):
        t.check_no_python_error(cmd, t.run(cmd))
    t.check_in("realm-thread apply", t.run("realm-help"), "realm-help")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_loaded_twice(t):
    """Sourcing the script a second time (e.g. an installed copy on top of
    the embedded one) leaves everything working, with no duplicated output
    or notices about its own hooks."""
    out = t.run("source {}".format(t.config["script"]))
    t.check_not_in("you defined", out, "notices on reloading")
    t.check_not_in("could not define", out, "notices on reloading")
    yield
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
                                "define hook-continue\necho user-hook-continue\\n\nend\n")
def test_user_hook_kept(t):
    """The script doesn't replace hooks the user defined first, and says
    what that means; it hooks the rest."""
    t.check_in("user-hook-step", t.run("show user hook-step"), "the user's hook-step")
    t.check_in("user-hook-continue", t.run("show user hook-continue"),
               "the user's hook-continue")
    t.check_in("realm-thread-guard", t.run("show user hook-next"), "realm-gdb's hook-next")
    # What it said when it loaded (load it again to see).
    out = t.run("source {}".format(t.config["script"]))
    for cmd in ("step", "continue"):
        t.check(re.search(r"you defined hook-{0}, so '{0}' is not refused in "
                          r"'realm-thread apply'.*may corrupt or crash the "
                          r"program".format(cmd), out) is not None,
                "notice about hook-{}:\n{}".format(cmd, out))
    t.check_not_in("you defined hook-next", out)
    yield
    # The rest are refused as usual.
    t.expect_error("realm-thread apply {:#x} next".format(t.ut("chain0")), "refusing")
    _check_parked_section(t, t.run("thread apply all bt"))
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
        t.check_eq(m.group(1).lower(), "running" if ut.running else "parked",
                   "state of {:#x}".format(ut.ptr))
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
    parts = re.split(r"^User thread (0x[0-9a-f]+):?.*$", section, flags=re.M)
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
    if v == "bt 3":
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
                        "realm-thread apply 0x1234 print realm_gdb_test_add")):
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
    everything from its innermost frame (user_switch) on, or every frame if
    that isn't shown (`bt -N`)."""
    names = re.findall(_BT_LINE + r"(\S+)", section, re.M)
    starts = [i for i, n in enumerate(names) if "user_switch" in n]
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
            t.check_in("User thread {:#x}:".format(ptr), m.group(0),
                       "inferior {}'s parked section".format(num))
            t.check_in("rgt_chain_level3", m.group(0),
                       "inferior {}'s parked section".format(num))
        t.check(re.search(r"inferior 2 has \d+ Realm user threads too",
                          t.run("realm-thread")), "realm-thread list in inferior 1")
        only2 = addrs2 - set(u.ptr for u in t.user_threads())
        if only2:
            t.expect_error("realm-thread apply {:#x} bt".format(min(only2)),
                           "is a user thread of inferior 2")
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
    t.check(re.search(r"^User thread 0x10:\s+<not shown: cannot read it: ", out, re.M),
            "the broken entry:\n" + out)
    t.check_in("cannot read the rest of the Realm user thread registry", out,
               "thread apply all bt")
    t.expect_error("realm-thread", "cannot read the Realm user thread registry")
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
        t.check(re.match(r"#0\s+.*" + re.escape(USER_SWITCH), out),
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
    def inside():
        t.select_frame("rgt_fhot_wait")
        shown = 0
        for k in range(NUM_FHOT):
            expected = float(gdb.parse_and_eval(
                "realm_gdb_test_state.fhot_wait_expected[{}]".format(k)))
            val = gdb.parse_and_eval("fhot{}".format(k))
            if val.is_optimized_out:
                continue
            t.check_eq(float(val), expected, "fhot{}".format(k))
            shown += 1
        return shown
    shown = t.within("fhot", inside)
    yield
    _finish(t)
    if not shown:
        raise TestSkip("every fhot value is <optimized out> where rgt_fhot_wait waits")


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
    changing anything (and in a core file, by `realm-thread PTR` too)."""
    before = t.regs()
    t.expect_error("realm-thread apply {} bt".format(t.variant))
    if t.is_core:
        t.expect_error("realm-thread " + t.variant)
        t.check_in("nothing to restore", t.run("realm-thread back"))
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
    """`realm-thread apply` without a command says how to use it; plain
    `realm-thread PTR` in a live process says to use `realm-thread apply`."""
    for cmd in ("realm-thread apply", "realm-thread apply all",
                "realm-thread apply {:#x}".format(t.ut("chain0"))):
        t.expect_error(cmd, "usage: realm-thread apply")
    msg = t.expect_error("realm-thread {:#x}".format(t.ut("chain0")))
    t.check_in("realm-thread apply {:#x}".format(t.ut("chain0")), msg,
               "realm-thread PTR in a live process")
    t.check_in("nothing to restore", t.run("realm-thread back"))
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
    t.check_in(USER_SWITCH.split("::")[-1], out, "thread apply N bt inside")
    out = t.apply("chain0", "thread apply all print $_thread")
    t.check_not_in("Parked Realm user threads", out, "thread apply all print inside")
    yield
    t.check_regs(before)
    t.check_host_stack()
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


def _callee_saved_reg():
    """A callee-saved register of this architecture."""
    if _is_aarch64():
        return "x20"
    if _is_x86_64():
        return "r12"
    raise TestSkip("unsupported architecture " + _arch_name())


@test(scenarios=("live", "attach"), variants=[
    "dump", "call", "fault", "set_reg", "set_pc", "set_vector", "set_flags",
    "execute_mi_return", "error"])
def test_window_registers_exact(t):
    """Whatever happens while a parked thread is shown, the thread showing
    it gets back every one of its registers (floating-point and vector ones
    included): after a dump, a function call, one that faults, the user
    changing a register (the PC, a vector register, the flags), MI's
    -exec-return from Python (gdb >= 14), or a command that fails.  (In the
    attach scenario the thread is in the middle of a system call.)"""
    if t.variant == "execute_mi_return" and not hasattr(gdb, "execute_mi"):
        raise TestSkip("gdb.execute_mi is gdb >= 14")
    if _is_x86_64():
        # (Something no call would leave there.  xmm0 is dead here.)
        t.run("set var $xmm0.v2_int64[0] = 0x1122334455667788")
    before = t.regs()
    v = t.variant
    if v == "dump":
        t.run("thread apply all bt full")
    elif v == "call":
        t.within("chain0", lambda: t.check_eq(
            t.eval_int("realm_gdb_test_add(2, 3)"), 5, "add()"))
    elif v == "fault":
        t.within("chain0", lambda: t.expect_error("print realm_gdb_test_fault()",
                                                  "signal"))
    elif v == "set_reg":
        t.apply("chain0", "set var ${} = 0x1234".format(_callee_saved_reg()))
    elif v == "set_pc":
        t.apply("chain0", "set var $pc = realm_gdb_test_add")
    elif v == "set_vector":
        t.apply("chain0", "set var $xmm1.v2_int64[0] = 0x5" if _is_x86_64()
                else "set var $v1.d.u[0] = 0x5")
    elif v == "set_flags":
        t.apply("chain0", "set var $eflags = $eflags ^ 0x1" if _is_x86_64()
                else "set var $cpsr = $cpsr ^ 0x20000000")
    elif v == "execute_mi_return":
        t.apply("chain0", "python gdb.execute_mi('-exec-return')")
    else:
        t.expect_error("realm-thread apply {:#x} print no_such_variable_xyz".format(
            t.ut("chain0")), "no_such_variable_xyz")
    yield
    t.check_regs(before, what="registers after " + v)
    t.check_host_stack()
    _finish(t)


_WINDOW_SETTINGS = ("scheduler-locking", "unwindonsignal", "unwind-on-terminating-exception",
                    "confirm", "may-call-functions", "direct-call-timeout",
                    "indirect-call-timeout", "unwind-on-timeout")


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
                    "set unwind-on-timeout off"):
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
    t.check_in("breakpoints were ignored", out, "the note")
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


@test(scenarios=("live",))
def test_window_bad_context(t):
    """A parked thread whose saved stack pointer isn't on its stack (a
    corrupt UserThread) isn't shown: a function call would run there."""
    reg = "gregs[15]" if _is_x86_64() else "sp"
    field = "((Realm::UserThread *){:#x})->ctx.uc_mcontext.{}".format(t.ut("chain0"), reg)
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
        t.check_in("function calls were refused", out, "the note")
        error, out = t.run_capturing("realm-thread apply {:#x} print realm_gdb_test_add"
                                     "(1, 2)".format(t.ut("chain0")))
        t.check(error is not None, "a call was made:\n" + out)
        t.check_in("is on", t.run("show may-call-functions"), "may-call-functions after")
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
    gdb doesn't release Python's lock while it waits for a call)."""
    import subprocess
    subprocess.Popen(["sh", "-c", "sleep {}; kill -{} {}".format(delay, sig, os.getpid())])


def _interrupted_call():
    """Run inside a register window (see test_window_interrupted_call)."""
    _signal_gdb_later("INT", 0.5)
    try:
        gdb.execute("print realm_gdb_test_nap(5000000)", to_string=True)
        print("not interrupted")
    except (gdb.error, KeyboardInterrupt) as e:
        print("interrupted: {}".format(e))


def _signal_gdb(sig):
    """Run inside a register window (see test_window_signals)."""
    import signal
    import time
    os.kill(os.getpid(), getattr(signal, "SIG" + sig))
    time.sleep(0.2)
    gdb.execute("bt full", to_string=True)


@test(scenarios=("attach",), variants=["TERM", "HUP", "TERM_in_call", "HUP_in_call"])
def test_window_signals(t):
    """SIGTERM or SIGHUP to gdb while a parked thread is shown (as an IDE or
    a closed terminal may send), even in the middle of a function call: gdb
    first puts the thread's registers back, then exits, so the attached
    program carries on cleanly."""
    sig = t.variant.split("_")[0]
    t.release()
    if t.variant.endswith("in_call"):
        cmd = ("python rgt._signal_gdb_later('{}', 0.5); "
               "gdb.execute('print realm_gdb_test_nap(2000000)')".format(sig))
    else:
        cmd = "python rgt._signal_gdb('{}')".format(sig)
    yield typed_final("realm-thread apply {:#x} {}".format(t.ut("hot"), cmd))


@test(scenarios=("attach",))
def test_window_interrupted_call(t):
    """Ctrl-C in the middle of a function call made while a parked thread is
    shown: the call is unwound, the thread's registers are put back, and the
    attached program carries on cleanly once released."""
    before = t.regs()
    out = t.apply("chain0", "python rgt._interrupted_call()")
    yield
    t.check_regs(before)
    t.check_host_stack()
    t.release()
    if "not interrupted" in out:
        # (gdb 10 and older hand Ctrl-C to Python while Python code runs, so
        # a call made from Python isn't interrupted.)
        raise TestSkip("this gdb doesn't interrupt a call made from Python")
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
    own printer, as `<optimized out>` (or, for an argument in plain `bt`,
    `...`), even with a pretty-printer that, like libstdc++'s, can't print
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
        if name == "given" and t.variant == "bt":
            t.check(shown.get(name) in ("...", "<optimized out>"),
                    "argument {}={} in:\n{}".format(name, shown.get(name), out))
        elif (name == "given" or t.variant != "bt") and _native_checks():
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
    ("frame_apply_return", "frame apply 1 return")])
def test_refused_in_window(t):
    """Commands that would let the program run or let go of it (or act on
    the thread showing the parked one, which has its registers) are refused
    while a parked thread is shown, so the program is never run with them;
    afterwards they work as usual, and the program runs to a clean exit."""
    v = t.variant
    if v.startswith("frame apply") and not t.supports("frame apply 1 print 1"):
        raise TestSkip("this gdb has no 'frame apply'")
    if v == "checkpoint" and not t.supports("help checkpoint"):
        raise TestSkip("this gdb has no checkpoints")
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
def test_refusal_messages(t):
    """Refusals say what the command would have done."""
    for cmd in ("return", "step", "continue", "detach"):
        msg = t.expect_error("realm-thread apply {:#x} {}".format(t.ut("chain0"), cmd))
        t.check(re.search(r"refusing '{}' while thread {} shows a parked Realm user "
                          r"thread.*corrupt".format(cmd, t.stop_thread), msg),
                "refusal of {}: {}".format(cmd, msg))
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
    yield
    t.run("kill")
    yield
    t.run_program()
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop", "stopped in")
    t.check_host_stack("stack after re-run")
    t.stop_thread = gdb.selected_thread().num
    t.find_roles()
    yield
    t.within("chain2", lambda: t.check_chain_stack(2))
    _check_parked_section(t, t.run("thread apply all bt"))
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
    t.release()
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
    its prompt: test_mi_console there."""
    return test_mi_console(t)


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
                   "running on thread {}".format(host))
    out = t.run("realm-thread")
    t.check(re.search(r"{:#x}\s+RUNNING\s+\(on thread {}, switching".format(
        ptr, host), out), "realm-thread list:\n" + out)
    out = t.run("thread apply all bt")
    t.check(re.search(r"User thread {:#x}:\n\s*<running on thread {}".format(
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
    t.run_program()
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
    t.check_in("nothing to restore", t.run("realm-thread back"))


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
    `<optimized out>` (or, for an argument in plain `bt`, `...`) by
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
        if name == "given" and t.variant == "bt":
            t.check(shown.get(name) in ("...", "<optimized out>"),
                    "argument {}={} in:\n{}".format(name, shown.get(name), out))
        elif name == "given" or t.variant != "bt":
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
    args = iter(next(frames).frame_args())
    next(args)
    _left_open[:] = [frames, args]
    t.check(_printing_lookups(), "nothing in place while printing a frame's values")
    yield
    t.check_eq(_printing_lookups(), [], "realm-gdb's pretty-printer lookup, left in place")
    del _left_open[:]
    t.check_eq(_printing_lookups(), [], "realm-gdb's pretty-printer lookup, left in place")
    t.back()
