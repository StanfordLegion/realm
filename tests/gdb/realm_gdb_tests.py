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

    def show(self, role_or_ptr):
        ptr = self.ut(role_or_ptr) if isinstance(role_or_ptr, str) else role_or_ptr
        out = self.run("realm-thread {:#x}".format(ptr))
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


def _gdb_11():
    """gdb 11: PendingFrame.level() but no gdb.events.gdb_exiting."""
    return _have_pending_level() and not hasattr(gdb.events, "gdb_exiting")


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

# ---- loading ------------------------------------------------------------

@test(scenarios=("live",))
def test_loaded(t):
    """The script is auto-loaded from librealm's .debug_gdb_scripts and
    registers its commands."""
    out = t.run("info auto-load python-scripts")
    if t.config.get("embedded", True):
        t.check(re.search(r"Yes\s+realm-gdb\.py", out) is not None,
                "realm-gdb.py not auto-loaded:\n" + out)
    for cmd in ("realm-help", "help realm-thread", "help thread apply"):
        t.check_no_python_error(cmd, t.run(cmd))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_loaded_twice(t):
    """Sourcing the script a second time (e.g. an installed copy on top of
    the embedded one) leaves everything working, with no duplicated output."""
    t.run("source {}".format(t.config["script"]))
    yield
    out = t.run("thread apply all bt")
    t.check_eq(out.count("Parked Realm user threads"), 1,
               "parked-thread sections in 'thread apply all bt'")
    yield
    before = t.regs()
    t.show("chain0")
    t.check_chain_stack(0)
    yield
    t.select_frame("rgt_chain_level3")
    t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
               t.chain_marker(0) + 1, "inferior call")
    yield
    t.back()
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), init="define hook-step\necho user-hook-step\\n\nend\n"
                                "define hook-return\necho user-hook-return\\n\nend\n")
def test_user_hook_kept(t):
    """The script doesn't replace hooks the user defined first, and says
    what that means."""
    out = t.run("show user hook-step")
    t.check_in("user-hook-step", out, "the user's hook-step")
    out = t.run("show user hook-next")
    t.check_in("realm-thread-guard", out, "realm-gdb.py's hook-next")
    # What it said when it loaded (load it again to see).
    out = t.run("source {}".format(t.config["script"]))
    t.check(re.search(r"you defined hook-step, so 'step' is not refused.*not on the "
                      r"parked user thread", out) is not None,
            "notice about hook-step:\n" + out)
    t.check(re.search(r"you defined hook-return, so 'return' is not refused.*may "
                      r"corrupt or crash the program", out) is not None,
            "notice about hook-return:\n" + out)
    t.check_not_in("you defined hook-next", out)
    yield
    # With the user's own hook-return, `return` isn't refused: realm-gdb.py
    # puts the thread back where it was at the next prompt.
    before = t.regs()
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield typed("return")
    t.check_regs(before, what="registers after the forced return")
    t.check_chain_stack(0, "stack after the forced return")
    t.back()
    t.show("chain0")
    t.back()
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
                "thread apply all bt"):
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
    t.expect_error("realm-thread 0x1234")
    yield
    t.run("continue")
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop",
               "stopped in")
    t.scenario = "live"
    t.find_roles()
    t.show("chain1")
    t.check_chain_stack(1)
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
    yield
    if t.scenario == "live":
        t.continue_to_exit()
    elif t.scenario == "attach":
        t.release()


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
    if t.scenario == "live":
        t.continue_to_exit()
    elif t.scenario == "attach":
        t.release()


@test(scenarios=("live", "core"), variants=[
    "backtrace", "where", "bt full", ("bt_dash_full", "bt -full"), "-q bt", "bt 3",
    "bt -frame-arguments all"])
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
        # three of the parked thread's frames, after the host's frame (or
        # the marker line)
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


@test(scenarios=("live", "core"))
def test_thread_apply_all_bt_while_showing(t):
    """`thread apply all bt` while a parked thread is shown lists the
    parked threads and leaves that thread shown as it was."""
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
    yield
    if t.scenario == "live":
        t.continue_to_exit()


# ---- realm-thread <ptr> -------------------------------------------------

@test(scenarios=("live", "attach", "core", "kcore", "freeze"))
def test_show(t):
    """`realm-thread <ptr>` shows a parked thread's stack, with working
    frame selection, locals, arguments and registers; `realm-thread back`
    restores the thread exactly."""
    thread = gdb.selected_thread().num
    before = t.regs()
    for i in range(NUM_CHAIN):
        out = t.show("chain{}".format(i))
        yield
        t.check_chain_stack(i)
        sel = t.frame_name(gdb.selected_frame())
        t.check(sel not in ("realm_gdb_test_stop", "rgt_hot_probe"),
                "selected frame is still the host's: " + sel)
        t.select_frame("rgt_chain_level3")
        yield
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
        yield
        t.run("up")
        t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2", "after up")
        t.run("down")
        t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level3", "after down")
        t.back()
        yield
        t.check_eq(gdb.selected_thread().num, thread, "selected thread")
        t.check_regs(before)
        t.check_host_stack()
    yield
    if t.scenario == "live":
        t.continue_to_exit()
    elif t.scenario == "attach":
        t.release()


@test(scenarios=("live", "attach", "core", "kcore"))
def test_show_callee_saved(t):
    """Variables held in callee-saved registers in a parked frame show the
    parked thread's values, not the host thread's."""
    t.show("hot")
    yield
    t.select_frame("rgt_hot_wait")
    for k in range(NUM_HOT - 1):  # hot4 is spilled to the stack
        expected = t.eval_int("realm_gdb_test_state.hot_wait_expected[{}]".format(k))
        val = gdb.parse_and_eval("hot{}".format(k))
        if val.is_optimized_out:
            t.fail("hot{} is <optimized out>".format(k))
        t.check_eq(int(val), expected, "hot{}".format(k))
    yield
    t.back()
    if t.scenario == "live":
        t.continue_to_exit()
    elif t.scenario == "attach":
        t.release()


@test(scenarios=("live", "core"))
def test_show_recursion(t):
    """A deep parked stack unwinds completely, with the right locals at
    every level."""
    t.show("recurse")
    yield
    names = t.frame_names(400)
    t.check_eq(names.count("rgt_recurse"), RECURSE_DEPTH, "rgt_recurse frames")
    t.check_in("recurse_task", names, "recursive stack")
    level, _ = t.find_frame("rgt_recurse")
    for depth in (RECURSE_DEPTH, RECURSE_DEPTH - 7, 1):
        t.run("frame {}".format(level + (RECURSE_DEPTH - depth)))
        t.check_eq(t.eval_int("recurse_depth"), depth, "recurse_depth")
    yield
    t.back()
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_show_every_thread(t):
    """Every registered user thread can be shown, including Realm's own
    idle workers."""
    before = t.regs()
    for ut in t.user_threads():
        if ut.running:
            continue
        t.show(ut.ptr)
        ok, out = t.run_clean("bt")
        t.check(ok, "bt of {:#x} failed: {}".format(ut.ptr, out))
        t.back()
        yield
    t.check_regs(before)
    t.check_host_stack()
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_show_nested(t):
    """Impersonations nest: `realm-thread back` pops one level at a time."""
    before = t.regs()
    t.show("chain0")
    yield
    t.show("chain1")
    t.check_chain_stack(1)
    t.select_frame("rgt_chain_level3")
    t.check_eq(t.eval_int("chain_marker"), t.chain_marker(1), "chain_marker")
    yield
    t.back()
    t.check_chain_stack(0)
    t.select_frame("rgt_chain_level3")
    t.check_eq(t.eval_int("chain_marker"), t.chain_marker(0), "chain_marker")
    yield
    t.back()
    t.check_regs(before)
    t.check_host_stack()
    out = t.run("realm-thread back")
    t.check_in("nothing to restore", out)
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"), variants=[
    "0", "0x10", "no_such_symbol_xyz", ("address_of_state", "&realm_gdb_test_state"),
    ("function", "realm_gdb_test_add"), ("void_convenience", "$_nonexistent_convenience"),
    ("syntax_error", "1 +")])
def test_show_bad_pointer(t):
    """Anything that isn't a parked user thread is refused cleanly, without
    changing anything."""
    before = t.regs()
    t.expect_error("realm-thread " + t.variant)
    yield
    t.check_regs(before)
    t.check_host_stack()
    out = t.run("realm-thread back")
    t.check_in("nothing to restore", out)
    yield
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live",), stop_at="rgt_inline_stop")
def test_inline_host_frame(t):
    """The stopped thread's innermost frame is an inline frame (common in
    optimized code): showing a parked thread, inferior calls and going back
    all still work."""
    host = t.frame_names()[:2]
    t.check_eq(host, ["rgt_inline_stop", "rgt_hot_probe"], "host stack")
    before = t.regs()
    t.show("chain0")
    t.check_chain_stack(0)
    level = t.select_frame("rgt_chain_level3")
    yield
    t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
               t.chain_marker(0) + 1, "add()")
    t.check_eq(t.selected_level(), level, "selected frame level after call")
    yield
    _check_parked_section(t, t.run("thread apply all bt"))
    t.check_chain_stack(0, "stack after 'thread apply all bt'")
    t.back()
    t.check_eq(t.frame_names()[:2], host, "host stack after going back")
    t.check_regs(before)
    yield
    t.run("delete")
    t.continue_to_exit()


@test(scenarios=("live", "attach"))
def test_show_write_memory(t):
    """Assigning to a parked frame's variables keeps that frame selected and
    the parked stack shown (gdb rebuilds its frames after a write)."""
    t.show("chain0")
    level = t.select_frame("rgt_chain_level3")
    m = t.chain_marker(0)
    yield
    t.run("set var chain_marker = {}".format(m))
    t.check_eq(t.selected_level(), level, "selected frame level after the write")
    t.check_eq(t.eval_int("chain_marker"), m, "chain_marker")
    # The same through the Python API, which gdb doesn't follow with a
    # rebuild of its frames.
    inf = gdb.selected_inferior()
    addr = t.eval_int("&chain_marker")
    inf.write_memory(addr, inf.read_memory(addr, 4))
    yield
    t.check_chain_stack(0, "stack after writing memory")
    t.back()
    t.check_host_stack()
    if t.scenario == "live":
        t.continue_to_exit()
    else:
        t.release()


@test(scenarios=("live", "core"))
def test_show_running_thread(t):
    """A running user thread's saved context is stale, so showing it
    is refused (its stack is already the host thread's)."""
    before = t.regs()
    t.expect_error("realm-thread {:#x}".format(t.ut("probe")), "running")
    yield
    t.check_regs(before)
    t.check_host_stack()
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live", "core"))
def test_show_convenience_variable(t):
    """The pointer can be any expression, e.g. a value history entry."""
    t.run("print (void *){:#x}".format(t.ut("chain2")))
    t.check_no_python_error("realm-thread $", t.run("realm-thread $"))
    t.check_chain_stack(2)
    yield
    t.back()
    t.check_host_stack()
    if t.scenario == "live":
        t.continue_to_exit()


@test(scenarios=("live",))
def test_show_from_other_thread(t):
    """Impersonating from a thread other than the one that stopped, and
    leaving it with `thread`, restores that thread exactly."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.run("thread {}".format(other))
    before_other = t.regs()
    before_stop = t.regs(t.stop_thread)
    yield
    t.show("chain0")
    t.check_chain_stack(0)
    yield
    t.run("thread {}".format(t.stop_thread))
    t.check_host_stack()
    t.check_regs(before_other, other, "other thread's registers")
    t.check_regs(before_stop, t.stop_thread, "stopped thread's registers")
    yield
    t.continue_to_exit()


# ---- inferior calls -----------------------------------------------------

@test(scenarios=("live", "attach", "freeze"))
def test_infcall_simple(t):
    """Inferior calls from a parked frame see that frame's variables, and
    leave the parked thread shown (and the selected frame) as they were."""
    before = t.regs()
    t.show("chain0")
    level = t.select_frame("rgt_chain_level3")
    yield
    m = t.chain_marker(0)
    t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"), m + 1, "p add()")
    yield
    t.check_eq(t.selected_level(), level, "selected frame level after call")
    t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level3",
               "selected frame after call")
    t.check_chain_stack(0, "stack after call")
    out = t.run("call realm_gdb_test_add(2, 3)")
    t.check_in("= 5", out, "call")
    t.check_eq(t.eval_int("chain_marker"), m, "chain_marker after calls")
    yield
    t.back()
    t.check_regs(before)
    t.check_host_stack()
    yield
    if t.scenario == "live":
        t.continue_to_exit()
    elif t.scenario == "attach":
        t.release()


@test(scenarios=("live", "attach"))
def test_infcall_member_function(t):
    """Member-function calls on a parked frame's objects."""
    t.show("chain1")
    t.select_frame("rgt_chain_level3")
    yield
    expected = t.eval_int("realm_gdb_test_state.widget_sums[1]")
    t.check_eq(t.eval_int("widget.sum()"), expected, "widget.sum()")
    t.check_eq(t.eval_int("realm_gdb_test_widget_sum(&widget)"), expected,
               "realm_gdb_test_widget_sum(&widget)")
    yield
    t.select_frame("rgt_chain_level2")
    t.check_eq(t.eval_int("widget.sum()"), expected, "widget.sum() one frame up")
    yield
    t.back()
    if t.scenario == "live":
        t.continue_to_exit()
    else:
        t.release()


@test(scenarios=("live", "attach"))
def test_infcall_libc(t):
    """Inferior calls into malloc and stdio, and gdb's printf/output."""
    t.show("chain2")
    t.select_frame("rgt_chain_level3")
    yield
    out = t.run("print realm_gdb_test_describe(chain_marker)")
    t.check_in('"value={}"'.format(t.chain_marker(2)), out, "describe()")
    out = t.run('printf "%s|%d\\n", realm_gdb_test_describe(7), chain_marker')
    t.check_in("value=7|{}".format(t.chain_marker(2)), out, "printf")
    out = t.run("output realm_gdb_test_add(chain_marker, 2)")
    t.check_in(str(t.chain_marker(2) + 2), out, "output")
    yield
    t.back()
    if t.scenario == "live":
        t.continue_to_exit()
    else:
        t.release()


@test(scenarios=("live", "attach"), variants=["64", "1024"])
def test_infcall_stack_hungry(t):
    """An inferior call that needs a lot of stack (KiB in the variant name)
    works from a parked frame, and nothing on the parked stack or the host
    is damaged."""
    kb = int(t.variant)
    expected = sum((((i >> 9) & 0xff) ^ 0x80) - 0x80 for i in range(0, kb * 1024, 512))
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield
    t.check_eq(t.eval_int("realm_gdb_test_stack_hog({})".format(kb)), expected,
               "stack_hog({})".format(kb))
    yield
    t.check_chain_stack(0, "stack after call")
    t.back()
    if t.scenario == "live":
        t.continue_to_exit()
    else:
        t.release()


@test(scenarios=("live",))
def test_infcall_slow(t):
    """A slow inferior call (other threads run meanwhile) from a parked
    frame."""
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield
    t.check_eq(t.eval_int("realm_gdb_test_nap(200000)"), 200000, "nap()")
    yield
    out = t.run("thread apply all bt")
    _check_parked_section(t, out)
    t.check_chain_stack(0, "stack after call")
    yield
    t.back()
    t.check_host_stack()
    t.continue_to_exit()


@test(scenarios=("live",))
def test_infcall_scheduler_locking(t):
    """An inferior call from a parked frame with scheduler-locking on."""
    t.run("set scheduler-locking on")
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield
    t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 5)"),
               t.chain_marker(0) + 5, "add()")
    yield
    t.back()
    t.run("set scheduler-locking off")
    t.continue_to_exit()


@test(scenarios=("live",))
def test_infcall_hits_breakpoint(t):
    """An inferior call that stops at a breakpoint: the stop shows the
    call's real frames and arguments, and the program can still be finished
    safely."""
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    t.run("break realm_gdb_test_opaque")
    yield
    t.expect_error("print realm_gdb_test_add(chain_marker, 1)",
                   "stopped while in a function called from GDB")
    yield
    names = t.frame_names()
    t.check_eq(names[:2], ["realm_gdb_test_opaque", "realm_gdb_test_add"],
               "innermost frames at the breakpoint")
    t.check_not_in("rgt_chain_level3", names, "stack at the breakpoint")
    t.select_frame("realm_gdb_test_add")
    t.check_eq(t.eval_int("a"), t.chain_marker(0), "argument a")
    t.check_eq(t.eval_int("b"), 1, "argument b")
    yield
    t.run("delete")
    # The first continue finishes the call ("GDB will silently stop").
    for _ in range(3):
        if t.exited:
            break
        t.run("continue")
    t.check_eq(t.exit_code, 0, "program exit code")


def _check_stopped_in_call(t, what):
    """While stopped inside a call made from a parked frame, the real frames
    show: the called function and the call itself, not parked frames."""
    names = t.frame_names()
    t.check_eq(names[:2], ["realm_gdb_test_opaque", "realm_gdb_test_add"],
               what + ": innermost frames")
    dummy = [f.type() == gdb.DUMMY_FRAME for f in t.frames()[:4]]
    t.check(any(dummy), what + ": no dummy frame in " + str(names[:4]))
    t.check_not_in("rgt_chain_level3", names, what)
    t.check_not_in("Realm::UserThread::user_switch", names[:4], what)


@test(scenarios=("live",), variants=["thread apply all bt", "nested", "second call",
                                     "display"])
def test_infcall_stopped_then(t):
    """After an inferior call from a parked frame stops at a breakpoint,
    other commands (which rebuild frames, or make calls of their own) must
    not show parked frames under the stopped call, and `return` must act on
    the call's real frames."""
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    t.run("break realm_gdb_test_opaque")
    yield
    t.expect_error("print realm_gdb_test_add(chain_marker, 1)",
                   "stopped while in a function called from GDB")
    yield
    _check_stopped_in_call(t, "at the breakpoint")
    t.run("delete")  # so that the calls below don't stop too
    v = t.variant
    if v == "thread apply all bt":
        t.run("thread apply all bt")
    elif v == "nested":
        t.show("chain1")
        t.back()
    elif v == "second call":
        t.check_eq(t.eval_int("realm_gdb_test_add(2, 3)"), 5, "second call")
    else:
        t.run("display realm_gdb_test_add(4, 5)")
    yield
    _check_stopped_in_call(t, "after " + v)
    t.run("delete")
    t.run("undisplay")
    yield typed("return")
    t.check_eq(t.frame_name(gdb.newest_frame()), "realm_gdb_test_add",
               "frame #0 after 'return'")
    yield
    # The first continue finishes the call ("GDB will silently stop").
    for _ in range(3):
        if t.exited:
            break
        t.run("continue")
    t.check_eq(t.exit_code, 0, "program exit code")


@test(scenarios=("live",))
def test_infcall_on_other_thread(t):
    """Every thread runs during an inferior call, so one made on another
    thread lets the stopped thread carry on, release the parked tasks, and
    so invalidate what was being shown: stop showing it."""
    t.show("chain0")
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.switch_thread(other)
    yield
    for _ in range(50):
        t.check_eq(t.eval_int("realm_gdb_test_nap(100000)"), 100000, "nap()")
        yield
        if t.shown_on("chain0") is None:
            break
    t.switch_thread(t.stop_thread)
    t.check_not_in("rgt_chain_level3", t.frame_names(), "stack after the call")
    t.check_in("nothing to restore", t.run("realm-thread back"))
    yield
    for _ in range(3):
        if t.exited:
            break
        t.run("continue")
    t.check_eq(t.exit_code, 0, "program exit code")


@test(scenarios=("live",))
def test_gcore_while_showing(t):
    """A core file written while a parked thread is shown has the threads'
    real registers."""
    core = os.path.join(t.config["tmp"], "showing.core")
    t.show("chain0")
    t.run("gcore " + core)
    t.back()
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
def test_infcall_signal_unwound(t):
    """An inferior call that faults, with `set unwindonsignal on`, returns
    to the parked thread it was made from."""
    t.run("set unwindonsignal on")
    t.show("chain0")
    level = t.select_frame("rgt_chain_level3")
    yield
    t.expect_error("print realm_gdb_test_fault()", "signal")
    yield
    t.check_chain_stack(0, "stack after the faulting call")
    # Without PendingFrame.level() (gdb < 11), realm-gdb.py can't tell that
    # gdb is about to restore the thread, so gdb can't re-select the frame,
    # and the view comes back at the next prompt with its innermost frame
    # selected.
    if _have_pending_level():
        t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level3",
                   "selected frame after the faulting call")
    t.back()
    t.check_host_stack()
    t.continue_to_exit()


@test(scenarios=("live",), variants=["stopped thread", "other thread"])
def test_infcall_from_pretty_printer(t):
    """A pretty-printer that makes inferior calls, used while showing parked
    stacks (`info locals`, `bt no-filters full`, `thread apply all bt
    no-filters full`: with the marker line, calls are refused, as
    test_marker_printer_calls checks; with gdb 8, which can't refuse them,
    such a printer leaves the marker line out).  Since every thread runs
    during a call, the parked-thread dump stops after one unless
    scheduler-locking is on."""
    gdb.execute("python import realm_gdb_tests; realm_gdb_tests.install_widget_printer()")
    if t.variant == "other thread":
        other = min(th.num for th in gdb.selected_inferior().threads()
                    if th.num != t.stop_thread)
        t.run("thread {}".format(other))
    t.run("set scheduler-locking on")
    out = t.show("chain0")
    if _refuses_calls(t):
        t.check_in("#0 is a marker for thread", out, "realm-thread's description of frame #0")
        t.check_in("<parked", t.run("bt 1"), "marker line with a calling printer")
    else:
        t.check_in("#0 is thread", out, "realm-thread's description of frame #0")
        t.check_not_in("<parked", t.run("bt 1"), "marker line with a calling printer")
    t.select_frame("rgt_chain_level2")
    yield
    expected = [t.eval_int("realm_gdb_test_state.widget_sums[{}]".format(i))
                for i in range(NUM_CHAIN)]
    t.check_in("sum={}".format(expected[0]), t.run("info locals"), "pretty-printed widget")
    out = t.run("bt no-filters full")
    t.check_no_python_error("bt no-filters full", out)
    t.check_in("widget = widget sum={}".format(expected[0]), out, "bt no-filters full")
    t.check_chain_stack(0, "stack after bt full")
    yield
    out = t.run("thread apply all bt no-filters full")
    _check_parked_section(t, out, full=True)
    for i in range(NUM_CHAIN):
        t.check_in("widget = widget sum={}".format(expected[i]), out,
                   "chain{}'s pretty-printed widget".format(i))
    t.check_chain_stack(0, "stack after thread apply all bt full")
    t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2", "selected frame")
    yield
    # With scheduler-locking off, realm-gdb.py still holds the other threads
    # during calls made on a thread showing a parked one (realm-lock-calls),
    # so the parked threads stay put and the whole dump is printed.  (Without
    # that, a call here would let the stopped thread carry on and wake the
    # parked threads, and gdb >= 13 aborts when the frames it is printing go
    # away.  Repeat, since that failed only some of the time.)
    t.run("set scheduler-locking off")
    for _ in range(3 if t.variant == "other thread" else 1):
        out = t.run("thread apply all bt no-filters full")
        _check_parked_section(t, out, full=True)
        t.check_not_in("let the program run", out, "the dump")
        yield
    if t.variant == "stopped thread":
        # Without holding them, the dump stops after a call.  (Safe here:
        # the parked tasks' only CPU thread is the one making the calls.)
        t.run("set realm-lock-calls off")
        out = t.run("thread apply all bt no-filters full")
        t.check(re.search(r"let the program run, so the other \d+ parked user "
                          r"threads? may have moved: not showing (them|it)\.  To "
                          r"see (them|it), run 'set realm-lock-calls on'", out),
                "the dump stopping after a call:\n" + out)
        t.run("set realm-lock-calls on")
        yield
    t.back()
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
    program in its constructor (like libstdc++'s for std::stringstream) or
    while listing the widget's children; or ("error") one that fails
    without trying to."""
    def widget_sum(val):
        return int(gdb.parse_and_eval("realm_gdb_test_widget_sum")(val.address))

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


@test(scenarios=("live", "core"))
def test_no_stray_events(t):
    """Looking around parked threads (every frame, its locals, bt, function
    calls, the parked-thread dump, the list) posts no gdb events: gdb 14 and
    older crash on exit if one is still pending (see test_quit_in_user_command
    and test_mi_return_at_exit), so realm-gdb.py posts one only when
    something has to wait until a command is over."""
    posted = []
    real_post = gdb.post_event

    def counting_post(func):
        posted.append(func)
        return real_post(func)

    gdb.post_event = counting_post
    try:
        for role in ("chain0", "recurse", "hot", "opt"):
            t.show(role)
            for level in range(len(t.frames(limit=60))):
                t.run("frame {}".format(level))
                t.run_clean("info locals")
                t.run_clean("info frame")
            t.run_clean("bt full")
            if role == "chain0" and not t.is_core:
                t.select_frame("rgt_chain_level3")
                t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
                           t.chain_marker(0) + 1, "add()")
            yield
        t.run_clean("thread apply all bt full")
        t.run("realm-thread")
        for _ in range(3):
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
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield
    t.expect_error("print realm_gdb_test_add(chain_marker, 1)")
    t.check_chain_stack(0)
    t.back()
    t.check_host_stack()


# ---- refused commands ---------------------------------------------------

# Not MI's -exec-return: it bypasses the CLI hook that refuses `return`, and
# forcing a return from a parked frame writes the parked thread's registers
# into the selected thread.  (Forcing a return is a deliberate change to the
# program's state anyway.)
@test(scenarios=("live",), variants=[
    "step", "s", "next", "n", "stepi", "si", "nexti", "ni", "finish", "fin",
    "until", "u", "advance realm_gdb_test_add", "jump realm_gdb_test_add",
    "return", ("return_frame0", "return"), ("jump_into_parked", "jump +1"),
    ("mi_next", "interpreter-exec mi \"-exec-next\""),
    ("mi_finish", "interpreter-exec mi \"-exec-finish\"")])
def test_resume_one_thread(t):
    """Commands that step or resume just the selected thread must not run
    the host thread with a parked thread's stack.  Either they are refused,
    or they act on the host thread as it really is; either way the program
    must be able to run to a clean exit afterwards."""
    before = t.regs()
    t.show("chain0")
    if t.info.name.endswith("return_frame0"):
        t.run("frame 0")
    else:
        t.select_frame("rgt_chain_level3")
    yield
    yield typed(t.variant)
    # All refused (MI's -exec-next and -exec-finish run the CLI commands,
    # hooks included).
    t.check_regs(before, what="registers after '{}'".format(t.variant))
    t.check_chain_stack(0, "stack after '{}'".format(t.variant))
    t.run("delete")
    # Undo whatever is still shown, then finish.
    for _ in range(4):
        if "nothing to restore" in t.run("realm-thread back"):
            break
    yield
    if not t.exited:
        t.continue_to_exit()
    else:
        t.check_eq(t.exit_code, 0, "program exit code")


# ---- resuming and leaving -----------------------------------------------

@test(scenarios=("live",))
def test_refusal_messages(t):
    """Refusals say what the command would have done."""
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield
    for cmd, pattern in (("return", "may corrupt or crash the program"),
                         ("jump +1", "may corrupt or crash the program"),
                         ("step", r"act on thread \d+ itself, not on the parked"),
                         ("finish", "until the parked frame returns")):
        t.expect_error(cmd, r"refusing to '{}'.*{}".format(cmd.split()[0], pattern))
    t.check_chain_stack(0, "stack after the refusals")
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=[
    "continue", "c", "signal 0", ("python", "python gdb.execute('continue')"),
    ("mi", "interpreter-exec mi \"-exec-continue\"")])
def test_resume_all(t):
    """Resuming the program in any way while a parked thread is shown (two levels
    deep) lets it run to a clean exit."""
    t.show("chain0")
    t.show("chain1")
    t.select_frame("rgt_chain_level3")
    yield
    yield typed(t.variant)
    # MI's -exec-continue returns before the program stops.
    for _ in t.wait_for_exit():
        yield
    t.check_exited(repr(t.variant))


@test(scenarios=("live",), variants=["thread", "python", "mi", "python_no_unwind"])
def test_switch_thread(t):
    """A parked thread shown on one thread stays with that thread: other
    threads show their own stacks, and switching back (by command, Python
    or MI) shows it again."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    before = t.regs()
    t.show("chain0")
    yield

    def switch(num):
        if t.variant == "thread":
            return typed("thread {}".format(num))
        if t.variant == "python":
            t.switch_thread(num)
            return None
        return typed('interpreter-exec mi "-thread-select {}"'.format(num))

    if t.variant == "python_no_unwind":
        # Without looking at the other thread's frames.
        t.switch_thread(other)
        t.switch_thread(t.stop_thread)
        names = t.frame_names()
        if "rgt_chain_level3" in names or _have_pending_level():
            t.check_chain_stack(0, "stack after switching back")
        else:
            # gdb < 11 can't tell that the frames were rebuilt: realm-gdb.py
            # shows the parked thread again at the next prompt or command.
            t.check_host_stack("stack after switching back")
        t.back()
        t.check_host_stack()
        t.check_regs(before)
        yield
        t.continue_to_exit()
        return
    yield switch(other)
    t.check_eq(gdb.selected_thread().num, other, "selected thread")
    names = t.frame_names()
    t.check_not_in("rgt_chain_level3", names, "other thread's stack")
    yield switch(t.stop_thread)
    t.check_eq(gdb.selected_thread().num, t.stop_thread, "selected thread")
    t.check_chain_stack(0, "stack after switching back")
    t.back()
    t.check_host_stack()
    t.check_regs(before)
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_kill_and_rerun(t):
    """`kill` while a parked thread is shown, then `run` again: the new process starts
    clean and everything works on it."""
    t.show("chain0")
    t.show("chain1")
    yield
    t.run("kill")
    yield
    t.run_program()
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop",
               "stopped in")
    t.check_host_stack("stack after re-run")
    t.stop_thread = gdb.selected_thread().num
    t.find_roles()
    yield
    t.show("chain2")
    t.check_chain_stack(2)
    t.back()
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_rerun_while_showing(t):
    """`run` (restart) while a parked thread is shown."""
    t.show("chain0")
    yield
    t.run_program()
    t.check_eq(t.frame_name(gdb.selected_frame()), "realm_gdb_test_stop",
               "stopped in")
    t.check_host_stack("stack after re-run")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_tbreak_continue(t):
    """Stopping again at a breakpoint after resuming from showing a parked thread
    shows the real stack, with no parked thread left shown."""
    t.show("chain0")
    t.run("tbreak realm_gdb_test_opaque")
    yield
    t.run("continue")
    t.check_eq(t.frame_name(gdb.newest_frame()), "realm_gdb_test_opaque",
               "stopped in")
    names = t.frame_names()
    t.check_not_in("rgt_chain_level3", names[:3], "stack at the new stop")
    out = t.run("realm-thread back")
    t.check_in("nothing to restore", out)
    yield
    t.continue_to_exit()


@test(scenarios=("attach",), ending="release", variants=[
    "detach", "quit", "eof", ("python", "python gdb.execute('detach')"),
    ("mi", "interpreter-exec mi \"-target-detach\"")])
def test_leave(t):
    """Leaving an attached process in any way while a parked thread is shown (two
    levels deep, with a parked frame selected) lets it carry on cleanly."""
    t.release()
    t.show("chain0")
    t.show("hot")
    t.select_frame("rgt_hot_wait")
    yield
    if t.variant == "eof":
        return  # the driver closes gdb's stdin
    yield typed_final(t.variant)


@test(scenarios=("attach",), interaction="batch")
def test_leave_batch(t):
    """`gdb -batch` exiting while a parked thread is shown (gdb runs no `quit` hook)
    lets the attached process carry on cleanly."""
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
               t.chain_marker(0) + 1, "add()")
    # (Only now: every thread runs during an inferior call, so a released
    # program would carry on and wake chain0 up.)
    t.release()
    yield


@test(scenarios=("attach",), interaction="batch")
def test_mi_return_at_exit(t):
    """The last command of `gdb -batch` is MI's -exec-return from a parked
    frame: the return is undone as gdb exits, so the program carries on
    cleanly, and gdb exits cleanly (gdb 14 and older crash as they exit if
    an event posted for the undoing hasn't run)."""
    if not _have_pending_level():
        raise TestSkip("undoing it needs PendingFrame.level() (gdb >= 11)")
    if not hasattr(gdb.events, "gdb_exiting"):
        raise TestSkip("undoing it as gdb exits needs gdb.events.gdb_exiting (gdb >= 12)")
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    t.release()
    yield typed_final('interpreter-exec mi "-exec-return 42"')


def _return_then(cmd, final=False):
    """A command to type that forces a return from the selected parked frame
    with MI's -exec-return, then runs `cmd`, so before gdb has finished the
    command.  (From Python: `interpreter-exec mi` in a user-defined command
    crashes gdb 12, and from Python aborts gdb 10 and older.)"""
    return Type("python gdb.execute('interpreter-exec mi \"-exec-return 42\"', "
                "to_string=True); gdb.execute({!r}, to_string=True)".format(cmd), final)


@test(scenarios=("attach",), variants=[
    "detach", "nested_detach", ("mi_detach", 'interpreter-exec mi "-target-detach"'),
    "detach_inferiors", "quit", "nested_quit"])
def test_return_then_leave(t):
    """MI's -exec-return from a parked frame, then, in the same command, so
    before gdb has finished the command, gdb lets go of the program: the
    return is undone first, so the program carries on cleanly once
    detached.  ("nested": with another parked thread shown under the one
    returned from; "mi_detach": MI's -target-detach, through
    `interpreter-exec`.)"""
    if not _have_pending_level():
        raise TestSkip("forcing a return from Python aborts gdb 10 and older")
    if t.variant.startswith("nested"):
        t.show("chain1")
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    t.release()
    if t.variant == "detach_inferiors":
        leave = "detach inferiors {}".format(gdb.selected_inferior().num)
    else:
        leave = t.variant.rpartition("_")[2]
    yield _return_then(leave, final=True)


@test(scenarios=("live",), variants=["continue", ("signal", "signal 0")])
def test_return_then_resume(t):
    """MI's -exec-return from a parked frame, then, in the same command, the
    program is resumed: the return is undone first, so the program runs to
    a clean exit.  (Not with MI's -exec-continue through `interpreter-exec`
    from Python, after which gdb 16 stops reading commands.)"""
    if not _have_pending_level():
        raise TestSkip("forcing a return from Python aborts gdb 10 and older")
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield _return_then(t.variant)
    for step in t.wait_for_exit():
        yield step
    t.check_exited("resuming right after -exec-return")


@test(scenarios=("live",), ending="none", variants=["host", "other"])
def test_return_then_call(t):
    """MI's -exec-return from a parked frame, then, in the same command (as
    a pretty-printer may as MI prints the frame), a function call on the
    thread it moved ("host"), or one that lets that thread run ("other",
    on another thread): realm-gdb.py can't undo the return first, so it
    says that the program may crash.  (The program is left as it is.)"""
    if not _have_pending_level():
        raise TestSkip("forcing a return from Python aborts gdb 10 and older")
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    t.run('interpreter-exec mi "-exec-return 42"')
    if t.variant == "other":
        t.switch_thread(_main_thread(t))
    t.run_clean("print realm_gdb_test_add(1, 2)")
    out = t.run("realm-thread")
    t.check_in({"host": "a function call is running on thread {}".format(t.stop_thread),
                "other": "a function call let thread {} run".format(t.stop_thread)}[
                    t.variant], out, "warning after the call")
    t.check_in("the program may crash", out, "warning after the call")


@test(scenarios=("attach",), interaction="mi", variants=[
    ("separate", "-exec-return 42|-target-detach"),
    ("pipelined", "-exec-return 42\n-target-detach"),
    ("pipelined_exit", "-exec-return 42\n-gdb-exit")])
def test_mi_return_then_leave(t):
    """MI's -exec-return from a parked frame, then MI's -target-detach or
    -gdb-exit, typed separately or (as an IDE may) sent together: the
    return is undone first, so the program carries on cleanly."""
    if not _have_pending_level():
        raise TestSkip("undoing it needs PendingFrame.level() (gdb >= 11)")
    t.show("chain0")
    level = t.select_frame("rgt_chain_level3")
    t.release()
    yield typed("-stack-select-frame {}".format(level))
    t.check_mi_result("done")
    commands = t.variant.split("|")
    for cmd in commands[:-1]:
        yield typed(cmd)
        t.check_mi_result("done")
    yield typed_final(commands[-1])


@test(scenarios=("attach",), ending="none")
def test_return_then_execute_mi_detach(t):
    """gdb.execute_mi() detaching right after MI's -exec-return from a
    parked frame, in the same command: no hook runs before it, so the
    return can't be undone first, and realm-gdb.py says the program may
    crash."""
    if not hasattr(gdb, "execute_mi"):
        raise TestSkip("gdb.execute_mi is gdb >= 14")
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    out = t.run('python gdb.execute_mi("-exec-return", "42"); '
                'gdb.execute_mi("-target-detach")')
    t.check_in("before realm-gdb could undo a forced return", out,
               "warning after detaching")
    t.check_in("it may crash", out, "warning after detaching")


def _track_posts():
    """Count the events posted with gdb.post_event that haven't run yet."""
    state = {"waiting": 0}
    real = gdb.post_event

    def post(func):
        state["waiting"] += 1

        def run():
            state["waiting"] -= 1
            func()
        real(run)
    gdb.post_event = post
    return state


@test(scenarios=("attach",), interaction="batch")
def test_exit_with_event_waiting(t):
    """`gdb -batch` ends right after a function call that let every thread
    run while a parked thread was shown on another thread, so that an event
    realm-gdb.py posted for once the command is over (checking the view)
    hasn't run: gdb exits cleanly, with every other Python exit handler run
    (gdb 14 and older would crash as they exit, and realm-gdb.py stops
    them)."""
    posts = _track_posts()
    t.run("thread {}".format(_main_thread(t)))
    t.show("chain0")
    t.run("thread {}".format(t.stop_thread))
    t.check_eq(t.eval_int("realm_gdb_test_add(2, 3)"), 5, "add()")
    if hasattr(gdb.events, "gdb_exiting"):
        t.check(posts["waiting"] > 0, "realm-gdb.py posted no event to check the "
                "view after the call (so this tests nothing)")
    t.release()
    yield


@test(scenarios=("live",), ending="none", cpus=2, variants=["resume_other", "call"])
def test_quit_in_user_command(t):
    """A user-defined command does something after which realm-gdb.py has
    work to do once the command is over, then quits: gdb exits cleanly (gdb
    14 and older crash as they exit if an event posted for that work hasn't
    run).  "resume_other" resumes another thread in non-stop mode while a
    parked thread is shown (gdb 11 to 13 drop the view only then);
    "call" makes a function call that lets every thread run (the views are
    then checked against where every thread is)."""
    mi = t.info.interaction == "mi"
    if mi and t.variant == "resume_other" and _gdb_11():
        raise TestSkip("gdb 11 crashes as it exits if an event is still waiting "
                       "to run, which realm-gdb.py can't prevent there")
    if t.variant == "resume_other":
        _restart_non_stop(t)
        for step in _stop_all(t):
            yield step
        t.run("thread {}".format(t.stop_thread))
        t.show("chain0")
        t.select_frame("rgt_chain_level2")
        other = min(th.num for th in gdb.selected_inferior().threads()
                    if th.num != t.stop_thread)
        body = "thread apply {} continue &".format(other)
    else:
        t.run("thread {}".format(_main_thread(t)))
        t.show("chain0")
        t.run("thread {}".format(t.stop_thread))
        body = "call realm_gdb_test_nap(1000)"
    t.run("define rgt-then-quit\n{}\nquit\nend".format(body))
    if mi:
        yield typed_final('-interpreter-exec console "rgt-then-quit"')
    else:
        yield typed_final("rgt-then-quit")


@test(scenarios=("live",), interaction="mi", ending="none", cpus=2,
      variants=["resume_other", "call"])
def test_mi_quit_in_user_command(t):
    """test_quit_in_user_command with the user-defined command run through
    MI (as from an IDE's console), which shows no prompt, so that the work
    is left to an event that hasn't run when gdb exits."""
    return test_quit_in_user_command(t)


@test(scenarios=("live",), variants=["one", "nested", "execute_mi", "nested_execute_mi"])
def test_mi_return_then_resume_other(t):
    """MI's -exec-return from a parked frame, then, in the same command, so
    before gdb has finished it, another thread is resumed in non-stop mode,
    after which parked threads aren't shown any more (with gdb 11 to 13,
    once the command is over): the return is undone even so.  `continue`
    undoes it first; MI's -exec-continue from gdb.execute_mi ("execute_mi",
    gdb >= 14) can't, so the parked thread stops being shown first, and
    the return is undone once the command is over.  ("nested": with another
    parked thread shown under the one returned from.)"""
    if not _have_pending_level():
        raise TestSkip("undoing it needs PendingFrame.level() (gdb >= 11)")
    if t.variant.endswith("execute_mi") and not hasattr(gdb, "execute_mi"):
        raise TestSkip("gdb.execute_mi is gdb >= 14")
    # (So that MI's -exec-continue doesn't wait for the thread to stop.)
    _restart_non_stop(t, mi_async=t.variant.endswith("execute_mi"))
    for step in _stop_all(t):
        yield step
    t.run("thread {}".format(t.stop_thread))
    before = t.regs()
    if t.variant.startswith("nested"):
        t.show("chain1")
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    if t.variant.endswith("execute_mi"):
        yield typed("python gdb.execute('interpreter-exec mi \"-exec-return 42\"', "
                    "to_string=True); gdb.execute_mi('-exec-continue', '--thread', "
                    "'{}')".format(other))
    else:
        yield _return_then("thread apply {} continue &".format(other))
    yield
    t.check_regs(before, what="the thread's registers after -exec-return")
    t.check_host_stack()
    for role in ("chain0", "chain1"):
        t.check_eq(t.shown_on(role), None, role + " shown on")
    yield
    t.continue_to_exit("continue -a")


@test(scenarios=("live",), interaction="mi_new_ui", cpus=2,
      variants=["return", "resume_other", "switching"])
def test_mi_new_ui(t):
    """MI on a second UI of a console gdb (`new-ui`), whose console shows
    its prompt but never again: what realm-gdb.py does once a command is
    over still gets done.  "return" is test_mi_return's int_value,
    "resume_other" test_mi_non_stop_resume_other's exec_continue, and
    "switching" test_mi_switching_shown_thread's in."""
    if _gdb_11():
        raise TestSkip("gdb 11 posts events only before a prompt has been shown "
                       "(see realm-gdb.py)")
    test, t.variant = {
        "return": (test_mi_return, "int_value"),
        "resume_other": (test_mi_non_stop_resume_other, "exec_continue"),
        "switching": (test_mi_switching_shown_thread, "in"),
    }[t.variant]
    return test(t)


@test(scenarios=("attach",))
def test_leave_mi_exit(t):
    """MI's -gdb-exit while a parked thread is shown (what IDEs send on 'stop
    debugging') lets the attached process carry on cleanly."""
    t.release()
    t.show("chain0")
    yield
    yield typed_final('interpreter-exec mi "-gdb-exit"')


@test(scenarios=("freeze",))
def test_frozen(t):
    """Attach to a process frozen by REALM_FREEZE_ON_ERROR: look around,
    impersonate, make calls, and detach without disturbing it."""
    out = t.run("thread apply all bt")
    _check_parked_section(t, out)
    t.check_in("realm_freeze", out, "frozen thread's stack")
    yield
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    t.check_eq(t.eval_int("realm_gdb_test_add(chain_marker, 1)"),
               t.chain_marker(0) + 1, "add()")
    yield
    yield typed_final("detach")


@test(scenarios=("live",), variants=[("all", "thread apply all print 1"),
                                     ("list", "thread apply LIST print 1")])
def test_thread_apply_keeps_frame(t):
    """A `thread apply` that doesn't unwind the other threads leaves the
    parked thread shown and the selected frame where it was."""
    t.show("chain0")
    t.select_frame("rgt_chain_level2")
    yield
    ids = " ".join(str(n) for n in sorted((th.num for th in
                                           gdb.selected_inferior().threads()),
                                          reverse=True))
    yield typed(t.variant.replace("LIST", ids))
    t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2", "selected frame")
    t.check_chain_stack(0)
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_multiple_inferiors(t):
    """A second inferior disturbs neither what is shown in the first nor
    `thread apply all bt`."""
    t.show("chain0")
    t.select_frame("rgt_chain_level2")
    t.run("add-inferior")
    t.run("inferior 2")
    yield
    t.run("thread 1.{}".format(t.stop_thread))
    yield
    t.check_chain_stack(0, "stack after visiting inferior 2")
    t.check_eq(t.shown_on("chain0"), t.stop_thread, "chain0 shown on")
    out = t.run("thread apply all bt")
    t.check_in("of inferior 1", out, "parked section header")
    _check_parked_section(t, out)
    t.back()
    t.run("remove-inferiors 2")
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_completion(t):
    """`thread apply all` completes like gdb's own."""
    for text, want in (("thread apply all bac", "thread apply all backtrace"),
                       ("thread apply all -as", "thread apply all -ascending"),
                       ("thread apply all print realm_gdb_test_ad",
                        "thread apply all print realm_gdb_test_add"),
                       ("thread apply all -q fr", "thread apply all -q frame")):
        out = t.run("complete " + text)
        t.check_in(want, out, "completions of {!r}".format(text))
    yield
    t.continue_to_exit()


@test(scenarios=("live",))
def test_reload_while_showing(t):
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


def _gdb_major():
    m = re.match(r"\s*(\d+)\.", gdb.VERSION)
    return int(m.group(1)) if m else None


def _native_checks():
    """Whether gdb's own printer prints values that are entirely optimized
    out without asking pretty-printers (gdb 10 does ask for locals)."""
    return _gdb_major() != 10


def _refuses_calls(t):
    """Whether this gdb can be told to refuse function calls (gdb >= 9), as
    realm-gdb.py does while it prints a backtrace with the marker line, and
    so draws the marker line whatever pretty-printers are in use."""
    return t.supports("show may-call-functions")


@test(scenarios=("live", "core"), variants=["bt", "thread apply all bt"])
def test_marker_line(t):
    """Frame #0 of a thread showing a parked thread (the thread's own) is
    drawn as a marker line, which `bt N` counts as a frame, and `thread
    apply all bt N` doesn't.  (With gdb 8, only while no pretty-printers
    are in use but gdb's own: see test_marker_line_printers.)"""
    if not _refuses_calls(t):
        t.run("disable pretty-printer")
    marker = "<parked Realm user thread {:#x}>".format(t.ut("chain0"))
    if t.variant == "bt":
        t.check_in("#0 is a marker for thread", t.show("chain0"),
                   "realm-thread's description of frame #0")
        out = t.run("bt 3")
        expected = 3
        t.check_no_python_error("bt full", t.run("bt full"))
        t.check_not_in("<parked", t.run("bt no-filters 1"), "'bt no-filters 1'")
    else:
        out = _split_parked(_parked_section(t, t.run("thread apply all bt 3")))[
            t.ut("chain0")]
        expected = 4
    frames = re.findall(r"^#\d+\s+(.*)$", out, re.M)
    t.check(frames and frames[0].startswith(marker),
            "frame #0 isn't {}:\n{}".format(marker, out))
    t.check_eq(len(frames), expected, "frames printed:\n" + out)
    yield
    if t.scenario == "live":
        t.back()
        t.continue_to_exit()


@test(scenarios=("live",))
def test_marker_line_printers(t):
    """With pretty-printers other than gdb's own in use (libstdc++'s, and a
    test one), the marker line is drawn, but with gdb 8 (which can't refuse
    the calls they might make: see test_marker_printer_calls) it isn't, and
    realm-gdb says what frame #0 is either way."""
    install_pair_printer()
    out = t.show("chain0")
    bt = t.run("bt 1")
    if _refuses_calls(t):
        t.check_in("#0 is a marker for thread", out, "realm-thread's description of frame #0")
        t.check_in("<parked Realm user thread", bt, "bt 1")
    else:
        t.check_in("#0 is thread", out, "realm-thread's description of frame #0")
        t.check_not_in("<parked", bt, "bt 1")
    yield
    t.back()
    t.continue_to_exit()


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


_PARKED_ROLES = ["chain{}".format(i) for i in range(NUM_CHAIN)] + ["recurse", "hot", "opt"]


@test(scenarios=("live", "core"), variants=["full", "plain", "all args", "entry values"])
def test_marker_like_native(t):
    """Backtraces with the marker line (printed by gdb's Python frame
    printer) show every parked thread's frames as gdb's own printer does,
    with libstdc++'s pretty-printers and one that, like them, can't print
    a value that is optimized out: in particular, those entirely
    optimized out as `<optimized out>`."""
    if not _refuses_calls(t):
        raise TestSkip("this gdb draws no marker line with pretty-printers loaded")
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
    if t.scenario == "live":
        t.continue_to_exit()


def _opt_values(t):
    """{name: value} of rgt_opt_wait's RgtPair locals and argument, as gdb
    itself prints them without pretty-printers."""
    t.select_frame("rgt_opt_wait")
    return {name: t.run("print/r " + name).partition(" = ")[2].strip()
            for name in ("gone", "half", "given")}


@test(scenarios=("live", "core"), variants=["bt full", "bt", "thread apply all bt full"])
def test_marker_optimized_out(t):
    """Values entirely optimized out are printed as `<optimized out>` (or,
    for an argument in plain `bt`, `...`) by backtraces with the marker
    line, as gdb itself prints them, rather than by a pretty-printer that,
    like libstdc++'s, can't print them; those partly optimized out are left
    to it, as gdb leaves them."""
    if not _refuses_calls(t):
        raise TestSkip("this gdb draws no marker line with pretty-printers loaded")
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
    if t.scenario == "live":
        t.back()
        t.continue_to_exit()


def _printing_lookups():
    """The pretty-printer lookup functions realm-gdb puts in place while a
    backtrace with the marker line is printed, wherever they are."""
    lists = [gdb.pretty_printers, gdb.current_progspace().pretty_printers]
    lists += [o.pretty_printers for o in gdb.objfiles()]
    return [p for l in lists for p in l if getattr(p, "name", None) == "realm-gdb"]


def _check_printing_over(t, calls=True):
    """What realm-gdb puts in place to print a backtrace with the marker
    line is gone again, and function calls work (unless `calls` is
    False: the user turned them off)."""
    t.check_eq(_printing_lookups(), [], "realm-gdb's pretty-printer lookup, left in place")
    setting = t.run("show may-call-functions")
    t.check_in("is on" if calls else "is off", setting, "show may-call-functions")
    if calls and not t.is_core:
        t.check_eq(t.eval_int("realm_gdb_test_add(2, 3)"), 5, "a call after the backtrace")


_REFUSED_NOTE = "realm-gdb: a pretty-printer above tried to call a function in the program"


@test(scenarios=("live",), variants=[
    "to_string", "constructor", "children", "sliced", "outermost", "dump", "user_off",
    "error"])
def test_marker_printer_calls(t):
    """While a backtrace with the marker line is printed, pretty-printers
    can't call functions in the program (gdb's Python frame printer loses
    frame numbers after a call, and before gdb 13 goes on using freed
    memory): a value whose printer tries is printed without it, and a note
    says so; every frame is numbered; and afterwards calls work again.
    Whether the call is in the printer's to_string, constructor or the
    children it lists; when the backtrace stops early (`bt N`) or is printed
    after gdb has gone through every frame (`bt -N`); in the parked-thread
    dump.  If the user turned calls off, they stay off; a printer's other
    errors are shown as gdb shows them."""
    if not _refuses_calls(t):
        raise TestSkip("this gdb draws no marker line with pretty-printers loaded")
    if t.variant in ("constructor", "children", "error"):
        install_calling_printer(t.variant)
    else:
        install_widget_printer()
    if t.variant == "user_off":
        t.run("set may-call-functions off")
    t.show("chain0")
    names = t.frame_names()
    level = names.index("rgt_chain_level2")
    if t.variant == "sliced":
        cmd = "bt full {}".format(level + 1)
    elif t.variant == "outermost":
        cmd = "bt full -{}".format(len(names) - level)
    elif t.variant == "dump":
        cmd = "thread apply all bt full"
    else:
        cmd = "bt full"
    out = t.run(cmd)
    sections = [out]
    if t.variant == "dump":
        sections = [_split_parked(_parked_section(t, out))[t.ut("chain{}".format(i))]
                    for i in range(NUM_CHAIN)]
    for section in sections:
        frames = _bt_frames(section)
        levels = [f[0] for f in frames]
        t.check(levels and levels == list(range(levels[0], levels[0] + len(levels))),
                "frame numbers:\n" + section)
        widget = [v for f in frames for n, v in f[2] if n == "widget"]
        t.check(widget, "no widget in:\n" + section)
        if t.variant == "user_off":
            t.check_not_in(_REFUSED_NOTE, out, "bt with calls turned off by the user")
        elif t.variant == "error":
            t.check_in("rgt-printer-error", out, "the printer's error")
            t.check_not_in(_REFUSED_NOTE, out, "bt with a failing printer")
        elif t.variant == "children":
            t.check(re.match(r"widget = \{id = \d+\}$", widget[0]), "widget = " + widget[0])
        else:
            t.check(re.match(r"\{id = \d+, values = \{", widget[0]), "widget = " + widget[0])
    if t.variant not in ("user_off", "error"):
        t.check_in(_REFUSED_NOTE, out, cmd)
        t.check_not_in("Python Exception", out, cmd)
    _check_printing_over(t, calls=t.variant != "user_off")
    if t.variant in ("to_string", "constructor"):
        t.check_in("widget sum=", t.run("bt no-filters full"), "bt no-filters full")
    yield
    t.run("set may-call-functions on")
    t.back()
    t.continue_to_exit()


_left_open = []


@test(scenarios=("live",))
def test_marker_printing_left_open(t):
    """Should something leave an iteration of the frame filter's frames
    for gdb's frame printer unfinished in the middle of printing a frame's
    values (when calls are refused), calls are allowed again once gdb has
    finished the command."""
    if not _refuses_calls(t):
        raise TestSkip("this gdb can't refuse function calls")
    import gdb.frames
    t.show("chain0")
    frames = gdb.frames.execute_frame_filters(gdb.newest_frame(), 0, -1)
    next(frames)  # (the marker line)
    args = iter(next(frames).frame_args())
    next(args)
    _left_open[:] = [frames, args]
    t.check_in("is off", t.run("show may-call-functions"), "while printing a frame's values")
    yield
    _check_printing_over(t)
    del _left_open[:]
    _check_printing_over(t)
    t.back()
    t.continue_to_exit()


@test(scenarios=("live",), interaction="mi")
def test_mi_marker_frame_filters(t):
    """MI with frame filters enabled (as Emacs does): frame #0 is the
    marker; a pretty-printer can't make calls while the frames' values are
    listed (its value is listed without it), and can again afterwards; an
    entirely optimized-out value is `<optimized out>`, with its own type."""
    if not _refuses_calls(t):
        raise TestSkip("this gdb draws no marker line with pretty-printers loaded")
    install_widget_printer()
    install_pair_printer()
    yield typed("-enable-frame-filters")
    t.check_mi_result("done")
    t.show("chain0")
    yield typed("-stack-list-frames 0 1")
    t.check_mi_result("done", r'level="0",func="<parked Realm user thread')
    level = t.select_frame("rgt_chain_level2")
    yield typed("-stack-list-variables --all-values")
    t.check_mi_result("done", r'name="widget",value="\{id = \d+, values = \{')
    yield typed("-stack-list-arguments --all-values {0} {0}".format(level))
    t.check_mi_result("done")
    _check_printing_over(t)
    yield typed('-data-evaluate-expression "realm_gdb_test_add(2, 3)"')
    t.check_mi_result("done", r'value="5"')
    t.back()
    t.show("opt")
    raw = _opt_values(t)
    gone = sorted(n for n, v in raw.items() if v == "<optimized out>" and n != "given")
    if gone:
        yield typed("-stack-list-variables --all-values")
        t.check_mi_result("done", r'name="{}",value="<optimized out>"'.format(gone[0]))
        yield typed("-stack-list-variables --simple-values")
        t.check_mi_result("done", r'name="{}",type="RgtPair"'.format(gone[0]))
    _check_printing_over(t)
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=[
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
    t.continue_to_exit()


@test(scenarios=("live",), stop_at="rgt_inline_stop", variants=["3", "no-filters 3"])
def test_thread_apply_all_bt_count_inline_host(t):
    """`thread apply all bt N` when the thread showing the parked threads has
    inline frames inside its innermost frame (more frames before the parked
    stack, unless the marker line replaces them)."""
    _check_bt_count(t, t.variant)
    yield
    t.run("delete")
    t.continue_to_exit()


def _check_bt_limit(t, args):
    """`set backtrace limit 3` limits each parked thread to 3 frames (or
    fewer, with `bt 2`), as for any thread, and `realm-thread` says when the
    limit leaves none of a parked thread's frames."""
    if "no-filters" in args and not t.supports("bt -no-filters 1"):
        raise TestSkip("this gdb does not support 'bt -no-filters'")
    t.run("set backtrace limit 3")
    try:
        _check_bt_count(t, args, expected=2 if args.endswith("2") else 3)
        yield
        t.run("set backtrace limit 1")
        t.expect_error("realm-thread {:#x}".format(t.ut("chain0")),
                       r"backtrace limit 1")
    finally:
        t.run("set backtrace limit unlimited")
    t.check_not_in("rgt_chain_level3", t.frame_names(), "stack")


@test(scenarios=("live",), variants=[("plain", ""), ("count", "2"),
                                     ("no_filters", "-no-filters")])
def test_thread_apply_all_bt_limit(t):
    """`thread apply all bt` with a backtrace limit (`set backtrace limit`):
    the frames before each parked stack don't count."""
    for step in _check_bt_limit(t, t.variant):
        yield step
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
        # A parked thread resumes where its saved context says, just after
        # its own swapcontext() call.  (Find those places by showing the
        # parked threads: their frame #1 is the saved context.)
        pcs = set()
        for role in ("chain0", "chain1", "chain2", "recurse", "hot", "opt"):
            t.show(role)
            pcs.add(gdb.newest_frame().older().pc())
            t.back()
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
    t.expect_error("realm-thread {:#x}".format(ptr),
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
    """The program's main thread, waiting for Realm to shut down: it stays
    where it is when it runs during a function call made on another
    thread, so tests can show parked threads on it meanwhile."""
    num = t.find_thread_with_frame("main")
    t.check(num is not None and num != t.stop_thread, "no main thread to use")
    return num


def _calls_until(t, done, what, mi=False):
    """Generator: make function calls on the selected thread, which let
    every thread run (all-stop mode, scheduler-locking off), until `done()`.
    The first call wakes the parked tasks.  A call may be cut short by a
    breakpoint another thread hits."""
    import time
    deadline = time.time() + 60
    call = "realm_gdb_test_release_parked()"
    while not done() and time.time() < deadline:
        if mi:
            yield typed('-data-evaluate-expression "{}"'.format(call))
        else:
            t.run_clean("call " + call)
            yield
        call = "realm_gdb_test_nap(100000)"
    t.check(done(), "timed out waiting until " + what)


def _parked_tasks_finished(t):
    return t.eval_int("realm_gdb_test_state.num_finished") == NUM_CHAIN + 3


def _resume_pc(t, role):
    """Where a parked user thread shown on the selected thread resumes:
    the PC of its innermost frame."""
    for f in t.frames():
        if t.roles[role].contains(int(f.read_register("sp")) & _U64):
            return f.pc()
    t.fail("no frame of {} on the stack:\n{}".format(role, "\n".join(t.frame_names())))


@test(scenarios=("live",), cpus=2, variants=["one", "nested"])
def test_switching_shown_thread(t):
    """A parked thread shown on one thread is woken up during a function
    call made on another (which lets every thread run), and caught just as
    it resumes, while ut->running is still false and its saved context is
    still the one shown.  Its stack is now being executed by another
    thread, so at the next prompt it is no longer shown.  ("nested": the
    parked thread shown under it is shown again, except with gdb < 11,
    which drops every view on other threads after such a call.)"""
    host = _main_thread(t)
    t.run("thread {}".format(host))
    if t.variant == "nested":
        t.show("chain1")
    t.show("chain0")
    bp = _SwitchingBreakpoint(t, "*{:#x}".format(_resume_pc(t, "chain0")),
                              only=t.ut("chain0"))
    t.run("thread {}".format(t.stop_thread))
    for step in _calls_until(t, lambda: bp.switching is not None,
                             "chain0 was caught resuming"):
        yield step  # (the views are checked at the prompt)
    switching = gdb.selected_thread().num
    # (Before anything else that checks the views.)
    t.switch_thread(host)
    if t.variant == "nested" and _have_pending_level():
        t.check_chain_stack(1, "stack after chain0 resumed")
    else:
        t.check_not_in("rgt_chain_level3", t.frame_names(), "stack after chain0 resumed")
    t.check_eq(t.shown_on("chain0"), None, "chain0 shown on")
    out = t.run("realm-thread")
    t.check(re.search(r"{:#x}\s+RUNNING\s+\(on thread {}, switching".format(
        t.ut("chain0"), switching), out), "realm-thread list:\n" + out)
    if t.variant == "nested" and _have_pending_level():
        t.check_eq(t.shown_on("chain1"), host, "chain1 shown on")
        t.check_no_python_error("info locals", t.run("info locals"))
        t.back()
    t.check_not_in("rgt_chain_level3", t.frame_names(), "stack after going back")
    t.check_in("nothing to restore", t.run("realm-thread back"))
    bp.delete()
    yield
    # (The call that was cut short finishes, and stops, first.)
    t.run("thread {}".format(t.stop_thread))
    for _ in range(4):
        if t.exited:
            break
        t.run("continue")
    t.check_eq(t.exit_code, 0, "program exit code")


@test(scenarios=("live",), interaction="mi", cpus=2, variants=["in", "out", "nested"])
def test_mi_switching_shown_thread(t):
    """test_switching_shown_thread under MI, which shows no prompt at which
    to check: once the call's command is over, a parked thread that another
    thread is switching into ("in", "nested") or out of ("out") is no
    longer shown, without anything but MI commands having run.  ("nested":
    the parked thread shown under it is shown again, except with gdb < 11,
    which drops every view on other threads after such a call.)"""
    if _gdb_11():
        raise TestSkip("gdb 11 can't post an event for this safely, so under MI "
                       "it is done at the next realm-gdb command (see realm-gdb.py)")
    host = _main_thread(t)
    t.run("thread {}".format(host))
    if t.variant == "nested":
        t.show("chain1")
    t.show("chain0")
    spec = ("swapcontext" if t.variant == "out"
            else "*{:#x}".format(_resume_pc(t, "chain0")))
    bp = _SwitchingBreakpoint(t, spec, only=t.ut("chain0"))
    yield typed("-thread-select {}".format(t.stop_thread))
    t.check_mi_result("done")
    for step in _calls_until(t, lambda: bp.switching is not None,
                             "chain0 was caught switching", mi=True):
        yield step
    bp.delete()
    yield typed("-stack-list-frames --thread {}".format(host))
    t.check_mi_result("done")
    frames = t.mi_result()[1]
    t.switch_thread(host)
    names = t.frame_names()
    if t.variant == "nested" and _have_pending_level():
        t.check_in('func="rgt_chain_level3', frames, "MI stack")
        t.check_chain_stack(1, "stack after chain0 resumed")
    else:
        for name in ("rgt_chain_level3", "Realm::UserThread::user_switch"):
            t.check_not_in(name, frames, "MI stack")
            t.check_not_in(name, names, "stack")
        t.check_in("main", names, "main thread's stack")
    t.check_no_python_error("info locals", t.run("info locals"))
    t.check_eq(t.shown_on("chain0"), None, "chain0 shown on")
    if t.variant == "nested" and _have_pending_level():
        t.back()
    t.check_in("nothing to restore", t.run("realm-thread back"))
    yield
    # (The call that was cut short finishes, and stops, first.)
    t.run("thread {}".format(t.stop_thread))
    for _ in range(4):
        if t.exited:
            break
        t.run("continue")
    t.check_eq(t.exit_code, 0, "program exit code")


@test(scenarios=("live",), cpus=2, variants=["one", "nested"])
def test_parked_thread_runs(t):
    """While a parked thread is shown on one thread, a function called on
    another lets every thread run, and the parked thread gets woken up and
    runs.  It stops being shown, its frames go with it, and nothing is
    refused on their account."""
    host = _main_thread(t)
    t.run("thread {}".format(host))
    t.show("chain0")
    if t.variant == "nested":
        t.show("chain1")
    t.select_frame("rgt_chain_level2")
    yield
    t.run("thread {}".format(t.stop_thread))
    for step in _calls_until(t, lambda: _parked_tasks_finished(t),
                             "the parked tasks finished"):
        yield step
    t.run("thread {}".format(host))
    t.check_eq(t.shown_on("chain0"), None, "chain0 shown on")
    t.check_eq(t.shown_on("chain1"), None, "chain1 shown on")
    names = t.frame_names()
    t.check_not_in("rgt_chain_level3", names, "stack after chain0 ran")
    t.check_in("main", names, "main thread's stack")
    t.check_in("nothing to restore", t.run("realm-thread back"))
    t.run("realm-thread-guard return")  # nothing parked is on show
    yield
    t.run("thread {}".format(t.stop_thread))
    t.continue_to_exit()


@test(scenarios=("live",), cpus=2, interaction="mi")
def test_mi_parked_thread_runs(t):
    """The same under MI, which shows no prompt at which to notice: the
    parked thread's frames are gone from the thread it was shown on."""
    host = _main_thread(t)
    t.run("thread {}".format(host))
    t.show("chain0")
    t.select_frame("rgt_chain_level3")
    yield typed("-thread-select {}".format(t.stop_thread))
    t.check_mi_result("done")
    for step in _calls_until(t, lambda: _parked_tasks_finished(t),
                             "the parked tasks finished", mi=True):
        yield step
    yield typed("-stack-list-frames --thread {}".format(host))
    t.check_mi_result("done")
    t.check_not_in("rgt_chain_level", t.mi_result()[1], "MI stack after chain0 ran")
    yield typed("-thread-select {}".format(host))
    t.check_mi_result("done")
    t.check_not_in("rgt_chain_level3", t.frame_names(), "the thread's stack")
    t.check_eq(t.shown_on("chain0"), None, "chain0 shown on")
    yield typed("-thread-select {}".format(t.stop_thread))
    t.continue_to_exit()


def _callee_saved_reg(t):
    """A callee-saved register of this architecture."""
    arch = gdb.selected_frame().architecture().name()
    if "aarch64" in arch:
        return "x20"
    if "x86-64" in arch or "x86_64" in arch:
        return "r12"
    raise TestSkip("unsupported architecture " + arch)


@test(scenarios=("live",), interaction="mi", variants=[
    "innermost", "outer", "into_inline", "int_value", "double", "long_double",
    "limited", "recursion", "edited", "edited_mi", "written_mi",
    "written_mi_rebuilt"])
def test_mi_return(t):
    """MI's -exec-return from a parked frame can't be refused (CLI hooks
    don't apply), so its effect is undone: the thread showing the parked
    thread keeps all its registers, floating-point and vector ones
    included.  "into_inline" returns into the middle of an inlined
    function, so that the thread's innermost frame becomes an inline one
    (which gdb doesn't ask unwinders about).  "int_value", "double" and
    "long_double" return a value, which gdb puts in a general, an SSE or an
    x87 register on x86-64.  "limited" shows the parked thread with a small
    backtrace limit and returns from beyond it; "recursion" returns from
    far down a deep stack.  "edited" and "edited_mi" first change one of
    the thread's registers (with the CLI or MI's -data-evaluate-expression),
    which must stay changed.  gdb doesn't report a change made with MI's
    -data-write-register-values, so undoing the return undoes it too
    ("written_mi", as realm-gdb.py says it may), unless the thread's frames
    have been rebuilt since ("written_mi_rebuilt")."""
    if not _have_pending_level():
        raise TestSkip("undoing it needs PendingFrame.level() (gdb >= 11)")
    arch = gdb.selected_frame().architecture().name()
    if t.variant in ("double", "long_double") and "x86" not in arch:
        raise TestSkip("x86-64 only")
    if t.variant == "double":
        # (Something the returned 3.5 isn't.)
        t.run("set var $xmm0.v2_int64[0] = 0x1122334455667788")
    value = ""
    if t.variant == "into_inline":
        t.show("hot")
        inline_level, inline_frame = t.find_frame("rgt_inline_hot_wait")
        t.check(inline_frame is not None and inline_frame.type() == gdb.INLINE_FRAME,
                "no inline frame for rgt_inline_hot_wait in:\n" + "\n".join(t.frame_names()))
        function = gdb.block_for_pc(inline_frame.pc()).function
        name = function.name.split("(")[0] if function is not None else None
        t.check(name == "rgt_inline_hot_wait", "rgt_inline_hot_wait's return "
                "address {:#x} isn't inside it (in {})".format(inline_frame.pc(), name))
        level = inline_level - 1
    elif t.variant == "recursion":
        t.show("recurse")
        names = t.frame_names()
        # Return from the second outermost rgt_recurse into the outermost.
        level = max(i for i, n in enumerate(names) if n == "rgt_recurse") - 1
        t.check(level > 2 * RECURSE_DEPTH // 3, "recursion only {} deep".format(level))
    else:
        if t.variant == "limited":
            t.run("set backtrace limit 3")
        t.show("chain0")
        if t.variant == "limited":
            t.run("set backtrace limit unlimited")
        level = t.select_frame({"innermost": "Realm::UserThread::user_switch",
                                "double": "rgt_chain_level1",
                                "long_double": "rgt_chain_level2",
                                "limited": "rgt_chain_level1"}.get(t.variant,
                                                                   "rgt_chain_level3"))
        value = {"int_value": " 42", "double": " 3.5", "long_double": " 2.5"}.get(
            t.variant, "")
    edited = t.variant.startswith(("edited", "written"))
    if edited:
        reg = _callee_saved_reg(t)
        t.run("frame 0")
        original = t.eval_int("${}".format(reg))
        if t.variant == "edited":
            t.run("set var ${} = 0x123456789".format(reg))
        elif t.variant == "edited_mi":
            yield typed('-data-evaluate-expression "${} = 0x123456789"'.format(reg))
            t.check_mi_result("done")
        else:
            yield typed("-data-list-register-names")
            t.check_mi_result("done")
            names = json.loads(t.mi_result()[1].partition("register-names=")[2])
            yield typed("-data-write-register-values x {} 0x123456789".format(
                names.index(reg)))
            t.check_mi_result("done")
            if t.variant == "written_mi_rebuilt":
                t.back()
                t.show("chain0")
                t.run("frame 0")
        t.check_eq(t.eval_int("${}".format(reg)), 0x123456789, "$" + reg)
    t.run("frame {}".format(level))
    before = t.regs()
    if t.variant == "written_mi":
        before[reg] = original
    yield typed("-stack-select-frame {}".format(level))
    t.check_mi_result("done")
    yield typed("-exec-return" + value)
    t.check_mi_result("done")
    yield  # (the undo happens once gdb has finished the command)
    t.check_regs(before, what="the thread's registers after -exec-return")
    if t.variant == "into_inline":
        t.check_names_in_order(["rgt_inline_hot_wait", "hot_task"],
                               t.frame_names(), "stack after -exec-return")
    elif t.variant == "recursion":
        t.check_in("rgt_recurse", t.frame_names(), "stack after -exec-return")
    else:
        t.check_chain_stack(0, "stack after -exec-return")
    t.back()
    if edited:
        t.run("set var ${} = {:#x}".format(reg, original))
    yield
    t.continue_to_exit()


@test(scenarios=("live",), interaction="mi", variants=["one", "both", "nested"])
def test_mi_call_other_thread(t):
    """MI: a call made on another thread (scheduler-locking on, so nothing
    else runs) leaves what is shown on this thread alone."""
    t.run("set scheduler-locking on")
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.show("chain0")
    if t.variant == "nested":
        t.show("chain2")
    yield typed("-thread-select {}".format(other))
    t.check_mi_result("done")
    if t.variant == "both":
        t.show("chain1")
    yield typed('-data-evaluate-expression "realm_gdb_test_add(2, 3)"')
    t.check_mi_result("done", r'value="5"')
    if t.variant == "both":
        t.check_chain_stack(1, "this thread's stack after the call")
        t.back()
    yield typed("-thread-select {}".format(t.stop_thread))
    t.check_mi_result("done")
    t.check_chain_stack(2 if t.variant == "nested" else 0,
                        "the other thread's stack after the call")
    if t.variant == "nested":
        t.back()
        t.check_chain_stack(0, "after going back")
    t.back()
    t.run("set scheduler-locking off")
    yield
    t.continue_to_exit()


@test(scenarios=("live",), variants=["one", "both", "call", "call_locked",
                                     "call_every_inferior"])
def test_two_inferiors(t):
    """Running one inferior leaves what is shown in another alone; with
    schedule-multiple both run, and nothing stays shown.  The same for a
    function call made in one inferior, which lets its threads run (unless
    scheduler-locking is on), and with schedule-multiple every inferior's."""
    t.show("chain0")
    t.select_frame("rgt_chain_level2")
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
    t.show(chain0_2)
    yield
    t.run("thread 1.{}".format(stop1))
    t.check_chain_stack(0, "inferior 1's stack, with inferior 2 started")
    yield
    if t.variant.startswith("call"):
        every = t.variant == "call_every_inferior"
        if every:
            # (Keep inferior 1's stopped thread from finishing the program.)
            t.run("set var realm_gdb_test_hold = 1")
        t.run("thread 2.{}".format(stop2))
        t.back()  # (so that realm-lock-calls doesn't hold the other threads)
        if t.variant == "call_locked":
            t.run("set scheduler-locking on")
        if every:
            t.run("set schedule-multiple on")
        t.check_eq(t.eval_int("realm_gdb_test_add(2, 3)"), 5, "call in inferior 2")
        t.run("set scheduler-locking off")
        t.run("set schedule-multiple off")
        yield
        t.run("thread 1.{}".format(stop1))
        if every:
            t.check_eq(t.shown_on(chain0_1), None, "inferior 1's chain0 shown on")
            t.check_not_in("rgt_chain_level3", t.frame_names(), "inferior 1's stack")
            t.run("set var realm_gdb_test_hold = 0")
        else:
            t.check_eq(t.shown_on(chain0_1), stop1, "inferior 1's chain0 shown on")
            t.check_chain_stack(0, "inferior 1's stack, after a call in inferior 2")
            t.back()
        t.run("thread 2.{}".format(stop2))
        t.run("continue")
        t.run("thread 1.{}".format(stop1))
        t.continue_to_exit()
        t.check_eq(t.exit_codes, [0, 0], "exit codes")
        return
    if t.variant == "one":
        # `thread apply all bt` shows every inferior's parked threads, and
        # realm-thread points to the other inferior's.
        t.select_frame("rgt_chain_level2")
        out = t.run("thread apply all bt")
        for num, ptr in ((1, chain0_1), (2, chain0_2)):
            m = re.search(r"^=== Parked Realm user threads of inferior {0} \(\d+\), "
                          r"shown on thread .*?^=== End of parked Realm user threads "
                          r"of inferior {0} ===$".format(num), out, re.M | re.S)
            t.check(m, "inferior {}'s parked section:\n{}".format(num, out))
            t.check_in("User thread {:#x}:".format(ptr), m.group(0),
                       "inferior {}'s parked section".format(num))
            t.check_in("rgt_chain_level3", m.group(0),
                       "inferior {}'s parked section".format(num))
        t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2",
                   "selected frame after thread apply all bt")
        t.check_chain_stack(0, "inferior 1's stack after thread apply all bt")
        t.check(re.search(r"inferior 2 has \d+ Realm user threads too",
                          t.run("realm-thread")), "realm-thread list in inferior 1")
        only2 = addrs2 - set(u.ptr for u in t.user_threads())
        if only2:
            t.expect_error("realm-thread {:#x}".format(min(only2)),
                           "is a user thread of inferior 2")
        yield
        # Only inferior 2 runs (to the end).
        t.run("thread 2.{}".format(stop2))
        t.run("continue")
        t.run("thread 1.{}".format(stop1))
        t.check_chain_stack(0, "inferior 1's stack, after inferior 2 ran")
        t.back()
        t.continue_to_exit()
        t.check_eq(t.exit_codes, [0, 0], "exit codes")
        return
    # Both run (until one of them reaches a breakpoint the woken-up tasks
    # pass), so nothing stays shown in either.
    t.run("set schedule-multiple on")
    t.run("break realm_gdb_test_opaque")
    t.run("continue")
    t.run("delete")
    t.run("set schedule-multiple off")
    for num in (1, 2):
        t.run("inferior {}".format(num))
        out = t.run("realm-thread")
        t.check_not_in("shown on", out, "inferior {}'s user threads".format(num))
    yield
    for num in (1, 2):
        t.run("inferior {}".format(num))
        t.run("continue")
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
    t.check(re.search(r"^User thread 0x10:\s+<bt failed: ", out, re.M),
            "the broken entry:\n" + out)
    t.check_in("cannot read the rest of the Realm user thread registry", out,
               "thread apply all bt")
    t.expect_error("realm-thread", "cannot read the Realm user thread registry")
    t.run("set var {} = (Realm::UserThread *){:#x}".format(link, saved))
    yield
    t.continue_to_exit()


# ---- MI, as under an IDE -------------------------------------------------

@test(scenarios=("live",), interaction="mi")
def test_mi_resume(t):
    """MI -exec-continue while a parked thread is shown (two levels deep)
    lets the program run to a clean exit."""
    t.show("chain0")
    t.show("chain1")
    yield
    yield typed("-exec-continue")
    t.check_mi_result("running")
    for _ in t.wait_for_exit():
        yield
    t.check_exited("-exec-continue")


@test(scenarios=("live",), interaction="mi")
def test_mi_infcall(t):
    """MI expression evaluation with a function call from a parked frame
    (no prompts, so nothing gets repaired at one)."""
    before = t.regs()
    t.show("chain0")
    level = t.select_frame("rgt_chain_level3")
    yield typed("-stack-select-frame {}".format(level))
    t.check_mi_result("done")
    yield typed('-data-evaluate-expression "realm_gdb_test_add(chain_marker, 1)"')
    t.check_mi_result("done", r'value="{}"'.format(t.chain_marker(0) + 1))
    yield typed('-data-evaluate-expression "widget.sum()"')
    t.check_mi_result("done", r'value="{}"'.format(
        t.eval_int("realm_gdb_test_state.widget_sums[0]")))
    t.check_eq(t.selected_level(), level, "selected frame level after the calls")
    t.check_chain_stack(0, "stack after the calls")
    yield typed("-enable-frame-filters")
    t.check_mi_result("done")
    yield typed("-stack-list-frames")
    t.check_mi_result("done", r'func="rgt_chain_level3\b.*func="rgt_chain_level2\b')
    yield typed("-stack-list-variables --all-values")
    t.check_mi_result("done", r'name="chain_marker",value="{}"'.format(
        t.chain_marker(0)))
    yield typed("-thread-info")
    t.check_mi_result("done")
    t.back()
    t.check_regs(before)
    t.check_host_stack()
    yield
    t.continue_to_exit()


@test(scenarios=("live",), interaction="mi")
def test_mi_stopped_call(t):
    """MI: a call from a parked frame stops at a breakpoint, another call
    stops there too, then -exec-return; nothing must claim the wrong frames
    or damage the program."""
    t.show("chain0")
    level = t.select_frame("rgt_chain_level3")
    yield typed("-break-insert realm_gdb_test_opaque")
    t.check_mi_result("done")
    yield typed('-data-evaluate-expression "realm_gdb_test_add(chain_marker, 1)"')
    t.check_mi_result("error", "stopped while in a function called from GDB")
    _check_stopped_in_call(t, "at the breakpoint")
    yield typed('-data-evaluate-expression "realm_gdb_test_add(2, 3)"')
    t.check_mi_result("error", "stopped while in a function called from GDB")
    t.check_eq(t.frame_name(gdb.newest_frame()), "realm_gdb_test_opaque",
               "frame #0 after the second call")
    t.check_not_in("Realm::UserThread::user_switch", t.frame_names()[:4],
                   "stack after the second call")
    yield typed("-exec-return")
    t.check_mi_result("done")
    t.check_not_in("Realm::UserThread::user_switch", t.frame_names()[:4],
                   "stack after -exec-return")
    t.run("delete")
    yield
    for _ in range(4):
        if t.exited:
            break
        t.run("continue")
    t.check_eq(t.exit_code, 0, "program exit code")


@test(scenarios=("live",), interaction="mi")
def test_mi_thread_select(t):
    """MI -thread-select away and back keeps the parked thread shown."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.show("chain0")
    yield typed("-thread-select {}".format(other))
    t.check_mi_result("done")
    t.check_not_in("rgt_chain_level3", t.frame_names(), "other thread's stack")
    yield typed("-thread-select {}".format(t.stop_thread))
    t.check_mi_result("done", r'new-thread-id="{}"'.format(t.stop_thread))
    t.check_chain_stack(0, "stack after switching back")
    t.back()
    yield
    t.continue_to_exit()


@test(scenarios=("attach",), interaction="mi", variants=["-target-detach", "-gdb-exit"])
def test_mi_leave(t):
    """MI detach/exit while a parked thread is shown, as an IDE's 'stop
    debugging' does: the attached program carries on cleanly."""
    t.show("chain0")
    t.show("hot")
    t.select_frame("rgt_hot_wait")
    t.release()
    yield
    yield typed_final(t.variant)


# ---- non-stop mode ------------------------------------------------------

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


def _restart_non_stop(t, mi_async=False):
    """_setup ran the program in all-stop mode; start over in non-stop
    mode, stopped at the same place.  With mi_async, MI's execution commands
    don't wait for the program to stop (as IDEs have them)."""
    t.run("kill")
    if mi_async:
        t.run("set mi-async on")
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


@test(scenarios=("live",))
def test_non_stop(t):
    """In non-stop mode (only the thread that hit the breakpoint stops),
    parked threads are listed, but not shown while other threads run: Realm
    may resume them at any moment, and gdb may abort if their frames go
    while it uses them (as with a pretty-printer that makes calls, loaded
    here).  Once every thread is stopped they are, until a thread runs
    again."""
    _restart_non_stop(t)
    gdb.execute("python import realm_gdb_tests; realm_gdb_tests.install_widget_printer()")
    yield
    running = [th.num for th in gdb.selected_inferior().threads() if th.is_running()]
    t.check(running, "no other thread is running in non-stop mode")
    num_parked = len([u for u in t.user_threads() if not u.running])
    t.run("thread {}".format(running[0]))
    _check_says_why(t, t.expect_error("realm-thread {:#x}".format(t.ut("chain0"))),
                    "realm-thread on a running thread")
    t.run("thread {}".format(t.stop_thread))
    _check_says_why(t, t.expect_error("realm-thread {:#x}".format(t.ut("chain0"))),
                    "realm-thread on the stopped thread")
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
    # Without -c, the command fails with gdb's error at the running
    # thread, but only after the parked threads' section.
    error, out = t.run_capturing("thread apply all bt")
    t.check(error is not None, "thread apply all bt didn't fail:\n" + out)
    t.check_clean_error("thread apply all bt", error)
    _check_not_shown(t, out, num_parked)
    yield
    for step in _stop_all(t):
        yield step
    t.run("thread {}".format(t.stop_thread))
    t.show("chain0")
    t.check_chain_stack(0)
    t.select_frame("rgt_chain_level2")
    widget = "widget = widget sum={}".format(
        t.eval_int("realm_gdb_test_state.widget_sums[0]"))
    t.check_in(widget, t.run("bt no-filters full"), "bt no-filters full")
    _check_parked_section(t, t.run("thread apply all bt full"), full=True)
    t.check_eq(t.frame_name(gdb.selected_frame()), "rgt_chain_level2", "selected frame")
    yield
    # Show another on another thread, and resume that thread: every parked
    # thread may be resumed now, so none is shown any more (with gdb 11 to
    # 13, only once the command is over).
    other = running[0]
    t.run("thread {}".format(other))
    t.show("chain1")
    t.check_chain_stack(1)
    t.check_eq(t.shown_on("chain1"), other, "chain1 shown on")
    out = t.run("continue &")
    # (gdb 11 to 13 say so only once the command is over.)
    if "no longer showing user thread" in out:
        t.check_in("{:#x} (on thread {})".format(t.ut("chain0"), t.stop_thread), out,
                   "what is no longer shown")
        _check_says_why(t, out, "continue &")
    yield
    # (gdb 9 can't read memory through a running thread.)
    t.run("thread {}".format(t.stop_thread))
    t.check_eq(t.shown_on("chain1"), None, "chain1 shown on, after resuming")
    t.check_eq(t.shown_on("chain0"), None, "chain0 shown on, after resuming another thread")
    t.check_host_stack()
    _check_says_why(t, t.expect_error("realm-thread {:#x}".format(t.ut("chain0"))),
                    "realm-thread after resuming another thread")
    yield
    t.continue_to_exit("continue -a")


def _check_says_why(t, text, what, non_stop=True):
    """A message about parked threads not being shown says why, and how to
    see them."""
    if non_stop:
        t.check_in("non-stop mode", text, what)
        t.check_in("('interrupt -a')", text, what)
    else:
        t.check_in("the program is running", text.lower(), what)
        t.check_in("('interrupt')", text, what)


def _check_not_shown(t, out, num_parked, non_stop=True):
    """`thread apply all bt` ends with a section saying how many parked
    threads it couldn't show (if it can tell: gdb 9 can't read memory
    through a running thread), why, and how to see them."""
    section = _parked_section(t, out)
    t.check(re.match(r"Parked Realm user threads(?: \({}\))?: not shown ===\n".format(
        num_parked), section), "parked section:\n" + section)
    _check_says_why(t, section, "parked section", non_stop)
    t.check(section.endswith("rerun this command.\n" + PARKED_END),
            "parked section:\n" + section)
    t.check(out.rstrip().endswith(PARKED_END), "parked section isn't last:\n" + out)


@test(scenarios=("live",))
def test_running_in_background(t):
    """While the program runs in the background (all-stop mode, `continue
    &`), parked threads can't be shown; realm-gdb.py says so, and how to see
    them, rather than leaving them out."""
    num_parked = len([u for u in t.user_threads() if not u.running])
    dash_c = t.supports("thread apply all -c print 1")
    # (Keep the stopped thread, and so the parked ones, where they are.)
    t.run("set var realm_gdb_test_hold = 1")
    t.run("continue &")
    yield
    _check_says_why(t, t.expect_error("realm-thread {:#x}".format(t.ut("chain0"))),
                    "realm-thread", non_stop=False)
    out = t.run("realm-thread")
    t.check_not_in("shown on", out, "realm-thread list")
    _check_says_why(t, out, "realm-thread list", non_stop=False)
    if dash_c:
        ok, out = t.run_clean("thread apply all -c bt")
        t.check(ok, "thread apply all -c bt: " + out)
        _check_not_shown(t, out, num_parked, non_stop=False)
    error, out = t.run_capturing("thread apply all bt")
    t.check(error is not None, "thread apply all bt didn't fail:\n" + out)
    t.check_clean_error("thread apply all bt", error)
    _check_not_shown(t, out, num_parked, non_stop=False)
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


@test(scenarios=("live",))
def test_non_stop_resume_other(t):
    """Non-stop mode, every thread stopped, a parked thread shown with an
    outer frame of it selected: resuming another thread from `thread apply`,
    after which gdb looks for that frame again, is fine, and the parked
    thread is no longer shown.  (gdb 11 to 13, which abort if the frame is
    gone, keep it until then.)"""
    _restart_non_stop(t)
    for step in _stop_all(t):
        yield step
    t.run("thread {}".format(t.stop_thread))
    t.show("chain0")
    t.select_frame("rgt_chain_level2")
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    yield
    t.run_clean("thread apply {} continue &".format(other))
    yield
    t.check_eq(t.shown_on("chain0"), None, "chain0 shown on")
    t.check_host_stack()
    t.check_eq(t.selected_level(), 0, "selected frame")
    yield
    t.continue_to_exit("continue -a")


@test(scenarios=("live",), interaction="mi", variants=["exec_continue", "thread_apply"])
def test_mi_non_stop_resume_other(t):
    """test_non_stop_resume_other under MI, which shows no prompt: the
    parked thread is no longer shown once the command is over, without a
    realm-gdb command having run.  "exec_continue" resumes the other thread
    with -exec-continue --thread, "thread_apply" with the CLI's `thread apply
    N continue &`."""
    _restart_non_stop(t, mi_async=True)
    for step in _stop_all(t):
        yield step
    t.run("thread {}".format(t.stop_thread))
    t.show("chain0")
    level = t.select_frame("rgt_chain_level2")
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    yield typed("-stack-select-frame {}".format(level))
    t.check_mi_result("done")
    if t.variant == "exec_continue":
        yield typed("-exec-continue --thread {}".format(other))
        t.check_mi_result("running")
    else:
        yield typed('-interpreter-exec console "thread apply {} continue &"'.format(other))
        t.check_mi_result("running")
    yield
    t.switch_thread(t.stop_thread)
    t.check_host_stack()
    t.check_eq(t.shown_on("chain0"), None, "chain0 shown on")
    yield
    t.continue_to_exit("continue -a")


# ---- core files only ----------------------------------------------------

@test(scenarios=("core",))
def test_core_thread_switch(t):
    """In a core file too, a parked thread shown on one thread stays with
    that thread."""
    other = min(th.num for th in gdb.selected_inferior().threads()
                if th.num != t.stop_thread)
    t.show("chain0")
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
