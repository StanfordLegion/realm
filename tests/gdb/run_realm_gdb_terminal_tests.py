#!/usr/bin/env python3
# Copyright 2026 Stanford University, NVIDIA Corporation
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

"""Tests of realm-gdb.py with gdb on a terminal (a pseudo-terminal this
driver types at), where gdb shows its prompt and repeats a command for an
empty line.

  run_realm_gdb_terminal_tests.py --list
  run_realm_gdb_terminal_tests.py --gdb gdb --program ./realm_gdb_test [-j N] [TEST|GLOB ...]

Each test runs its own sessions, from a fresh temporary directory (also
HOME).

Exit status: 0 if every selected test passed (or skipped), 1 if any failed,
77 if every selected test was skipped (CTest's SKIP_RETURN_CODE).
"""

import argparse
import fcntl
import fnmatch
import os
import pty
import random
import re
import select
import shutil
import signal
import struct
import sys
import tempfile
import termios
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True  # keep the source tree clean
import run_realm_gdb_tests as base  # noqa: E402

# A parked thread's innermost frame in a backtrace: user_switch, where it
# called swapcontext; on FreeBSD, whose swapcontext is a system call, libc's
# stub making it.
FRAME0 = "_swapcontext" if sys.platform.startswith("freebsd") else "user_switch"

SOURCE = os.path.join(HERE, "..", "..", "src", "realm")
PROMPT = rb"\(gdb\) $"
# Terminal control sequences, taken out of what tests look at.
CONTROL = re.compile(rb"\x1b(?:\[[0-9;?<=>]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|"
                     rb"[()][0-9A-Za-z]|[=>78DEHMc])|\r")

# Prints CHAIN0=<the UserThread* of the program's first chain task>.
FIND_CHAIN = """\
import gdb
addr = int(gdb.parse_and_eval("realm_gdb_test_state.chain_locals[0]"))
h = gdb.parse_and_eval("Realm::user_thread_registry_head")
while int(h):
    base, size = int(h["stack_base"]), int(h["stack_size"])
    if base <= addr < base + size:
        print("CHAIN0=%#x" % int(h))
        break
    h = h["registry_next"]
"""

# Sets $rgt_chain0 to it instead.
FIND_CHAIN_VAR = FIND_CHAIN.replace(
    'print("CHAIN0=%#x" % int(h))',
    'gdb.execute("set $rgt_chain0 = %#x" % int(h), to_string=True)')

# The SP of `maint print raw-registers` (realm-gdb hooks no maint command):
# what the selected thread's register cache has.
RAW_SP = re.compile(r"^ ?(?:rsp|sp) .*?(0x[0-9a-f]+)\s*$", re.M)


def clean(data):
    return CONTROL.sub(b"", data).decode("utf-8", "replace")


class Skip(Exception):
    pass


TESTS = {}


def test(name):
    def register(func):
        TESTS[name] = func
        return func
    return register


class Term(object):
    """A session of gdb's on a pseudo-terminal."""

    def __init__(self, ctx, argv, env, size=(50, 200)):
        self.ctx = ctx
        self.out = b""
        # What gdb may print, which finish() takes for an error otherwise.
        self.allowed = []
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            try:
                fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", size[0], size[1], 0, 0))
                # (Not ignored, as a shell's background job has it, which gdb
                # 10 and older keep: Ctrl-C would do nothing.)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                os.chdir(ctx.tmp)
                os.execve(argv[0], argv, env)
            finally:
                os._exit(99)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, fcntl.fcntl(self.fd, fcntl.F_GETFL) | os.O_NONBLOCK)

    def read(self, until=PROMPT, timeout=None):
        """What the session shows, until `until` matches it (cleaned) or
        `timeout` seconds pass."""
        timeout = self.ctx.args.step_timeout if timeout is None else timeout
        end = time.time() + timeout
        mark = len(self.out)
        while time.time() < end and self.fd is not None:
            r, _, _ = select.select([self.fd], [], [], 0.1)
            if r:
                try:
                    data = os.read(self.fd, 65536)
                except BlockingIOError:
                    continue
                except OSError:
                    break
                if not data:
                    break
                self.out += data
                continue
            if until is not None and re.search(until, CONTROL.sub(b"", self.out[mark:])):
                break
        return clean(self.out[mark:])

    def send(self, keys):
        """Type `keys`, reading the session's output meanwhile (it may be
        waiting to write it); failing the test if it stops taking keys."""
        data = keys.encode() if isinstance(keys, str) else keys
        end = time.time() + self.ctx.args.step_timeout
        try:
            while data:
                try:
                    data = data[os.write(self.fd, data):]
                except BlockingIOError:
                    if time.time() > end:
                        raise OSError("it takes no more keys")
                    self.read(None, 0.05)
        except (OSError, TypeError) as e:
            self.ctx.check("the session takes keys ({})".format(e), False, clean(self.out))
            raise Skip("(failed)")

    def step(self, label, keys, want=None, until=PROMPT, timeout=None, unwanted=None):
        """Type `keys`, then read; `want` (a regular expression) must be in
        what shows, and `unwanted` not."""
        self.send(keys)
        shown = self.read(until, timeout)
        if want is not None or unwanted is not None:
            self.ctx.check(label, (want is None or re.search(want, shown) is not None) and (
                unwanted is None or re.search(unwanted, shown) is None), shown)
        return shown

    def finish(self, quit="quit\r"):
        """Quit (killing the program if it's there), and check the session
        ended cleanly."""
        if quit:
            self.send(quit)
            self.read(rb"\(y or n\) $", 10)
            if re.search(rb"\(y or n\) $", CONTROL.sub(b"", self.out)):
                self.send("y\r")
        self.read(None, 3)
        status = self.wait(20)
        text = clean(self.out)
        self.ctx.check("the session exits with status 0", status == 0, "status {}".format(status))
        for allowed in self.allowed:
            text = text.replace(allowed, "")
        bad = [b for b in base.BAD_OUTPUT if b in text]
        # (Around the first; check() shows 3000 characters.)
        at = min(text.index(b) for b in bad) if bad else len(text)
        self.ctx.check("no errors of gdb's or Python's ({})".format(bad), not bad,
                       text[max(0, at - 2000):at + 900])

    def wait(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            p, st = os.waitpid(self.pid, os.WNOHANG)
            if p:
                self.pid = None
                return os.WEXITSTATUS(st) if os.WIFEXITED(st) else -os.WTERMSIG(st)
            self.read(None, 0.2)
        return None

    def close(self):
        if self.pid is not None:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.waitpid(self.pid, 0)
            except OSError:
                pass
            self.pid = None
        if self.fd is not None:
            # (Once: the number may be another session's next, in another
            # thread.)
            fd, self.fd = self.fd, None
            try:
                os.close(fd)
            except OSError:
                pass


class Context(object):
    def __init__(self, args, name, tmp):
        self.args, self.name, self.tmp = args, name, tmp
        self.failures = []
        self.log = []
        self.sessions = []
        os.mkdir(os.path.join(tmp, "home"))

    def check(self, label, ok, shown=""):
        self.log.append(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            self.failures.append(label)
            self.log.append("      " + shown.strip().replace("\n", "\n      ")[-3000:])
        return ok

    def file(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def env(self, extra=None):
        env = base.clean_env()
        env.pop("INPUTRC", None)
        env.update({"TERM": "xterm", "HOME": os.path.join(self.tmp, "home")})
        env.update(extra or {})
        return env

    def gdb(self, args, env=None, size=(50, 200)):
        """gdb ARGS, at a terminal (`size`: rows, columns), with realm-gdb.py
        sourced first."""
        s = Term(self, [shutil.which(self.args.gdb), "-iex", "set auto-load python-scripts off",
                        "-iex", "source " + self.args.script] + args, self.env(env), size)
        self.sessions.append(s)
        return s

    def realm_args(self, extra=()):
        """gdb's arguments to stop the test program in realm_gdb_test_stop."""
        return (["-nx", "-q", "-iex", "set height 0", "-iex", "set width 0",
                 "-ex", "break realm_gdb_test_stop", "-ex", "run"] + list(extra) +
                ["--args", self.args.program] + base.program_args() + ["-mode", "stop"])

    def select_chain0(self, s, rt="", shown=None):
        """From the stop in realm_gdb_test_stop (`shown`, unless still to be
        read): select the first chain task's user thread, and its frame in
        rgt_chain_level3 (with commands after `rt`, plain gdb).  Returns (its
        UserThread*, that frame's level)."""
        if shown is None:
            shown = s.read(timeout=self.args.timeout)
        if not self.check("the program stops in realm_gdb_test_stop",
                          "realm_gdb_test_stop" in shown, shown):
            raise Skip("(failed)")
        shown = s.step("find chain0's user thread", "source {}\r".format(
            self.file("find_chain.py", FIND_CHAIN)))
        m = re.search(r"CHAIN0=(0x[0-9a-f]+)", shown)
        if not self.check("chain0's user thread found", m is not None, shown):
            raise Skip("(failed)")
        ptr = m.group(1)
        s.step("realm-thread PTR selects it", "realm-thread {}\r".format(ptr),
               r"selected user thread \d+ \(" + ptr)
        shown = s.step("bt: the parked thread's stack", rt + "bt\r",
                       r"#0 .*" + FRAME0 + r"[\s\S]*rgt_chain_level3")
        m = re.search(r"#(\d+) .* rgt_chain_level3 ", shown)
        level = m.group(1) if m else "0"
        s.step("frame N", rt + "frame {}\r".format(level), r"rgt_chain_level3")
        return ptr, level

    def close(self):
        for s in self.sessions:
            s.close()


# ---- realm-gdb.py on a terminal ----------------------------------------

@test("terminal.prompt")
def test_terminal_prompt(t):
    """realm-gdb.py with gdb on a terminal: the prompt shows the selection
    once however many prompts go by, and is gdb's own again afterwards,
    also when the program runs.  (gdb makes the prompt a prompt hook returns
    its `prompt` setting.)"""
    s = t.gdb(t.realm_args())
    ptr, _ = t.select_chain0(s, rt="rt ")
    for i in range(3):
        s.step("prompt {}: the selection, once".format(i + 1), "echo\r",
               r"\n\[realm-thread (?:0x[0-9a-f]+|\d+)\] \(gdb\) $", unwanted=r"\] \[realm-thread")
    s.step("realm-thread back", "realm-thread back\r", r"no parked user thread is selected now")
    s.step("  then gdb's prompt", "echo\r", r"(?:^|\n)echo\n\(gdb\) $")
    s.step("  and setting", "show prompt\r", r"Gdb's prompt is \"\(gdb\) \"\.")
    s.step("select again", "realm-thread {}\r".format(ptr), r"\[Selecting user thread")
    s.step("continue: the program finishes", "continue\r",
           r"REALM_GDB_TEST: PASS[\s\S]*exited normally", timeout=t.args.timeout)
    s.step("  then gdb's prompt", "show prompt\r", r"Gdb's prompt is \"\(gdb\) \"\.\n\(gdb\) $")
    s.finish()


@test("terminal.output")
def test_terminal_output(t):
    """realm-gdb.py with gdb on a terminal: output of `rt` and `realm-thread
    apply` that doesn't end a line (`output`'s) comes before the prompt, as
    without them; a backtrace they run that stops short says so, as gdb's
    own does at a terminal."""
    s = t.gdb(t.realm_args())
    ptr, _ = t.select_chain0(s, rt="rt ")
    early = rb"\(gdb\) $|\(gdb\) 1234$"
    s.step("rt output: before the prompt", "rt output 1234\r",
           r"rt output 1234\n1234\[realm-thread (?:0x[0-9a-f]+|\d+)\] \(gdb\) $", until=early)
    s.step("  not after the next command's output", "print 77\r", r"\$\d+ = 77\n",
           unwanted="1234")
    s.step("realm-thread apply PTR output: before the prompt",
           "realm-thread apply {} output 4321\r".format(ptr),
           r"\n4321\[realm-thread (?:0x[0-9a-f]+|\d+)\] \(gdb\) $")
    # (As gdb's own backtraces typed at a terminal.)
    s.step("rt bt 1: says more frames follow", "rt bt 1\r",
           r"#0 [^\n]*\n\(More stack frames follow\.\.\.\)")
    s.step("realm-thread apply PTR bt 1: too",
           "realm-thread apply {} bt 1\r".format(ptr),
           r"#0 [^\n]*\n\(More stack frames follow\.\.\.\)")
    s.step("thread apply all bt 1: in the parked section too", "thread apply all bt 1\r",
           r"=== Parked[^\n]*\n\nUser thread[^\n]*\n#0 [^\n]*\n"
           r"\(More stack frames follow\.\.\.\)")
    # (gdb.execute keeps styling from gdb 17.)
    probe = s.step("", "python exec(\"try:\\n gdb.execute('echo', to_string=True, "
                   "styling=True)\\n print('RGT-' + 'STYLES')\\nexcept TypeError:\\n "
                   "print('RGT-' + 'PLAIN')\")\r")
    styles = "RGT-STYLES" in probe
    mark = len(s.out)
    s.step("rt bt 1", "rt bt 1\r", r"#0 ")
    t.check("  styled as gdb's own bt is, where gdb.execute can keep it ({})".format(
        "gdb >= 17" if styles else "not before gdb 17"),
        bool(re.search(rb"\x1b\[[0-9;]*m", s.out[mark:])) == styles, repr(s.out[mark:]))
    if styles:
        s.step("rt pipe bt 1 | cat -v: unstyled, as gdb's pipe gives it", "rt pipe bt 1 | cat -v\r",
               r"#0 ", unwanted=r"\^\[")
    s.finish()


@test("terminal.repeat")
def test_terminal_repeat(t):
    """realm-gdb.py with gdb on a terminal: an empty line after `realm-thread
    PTR` or `realm-thread back` doesn't select, or go back, again.  The
    thread selected again starts over at frame #0, rather than being
    selected twice; `realm-thread back` prints the frame it is back at."""
    s = t.gdb(t.realm_args())
    ptr, _ = t.select_chain0(s, rt="rt ")
    s.step("realm-thread PTR, again", "realm-thread {}\r".format(ptr),
           r"\[Selecting user thread")
    s.step("  Enter", "\r", None, unwanted=r"selected user thread|\[Selecting")
    shown = s.step("realm-thread: one selection", "realm-thread\r", r"\(selected\)")
    t.check("  and none earlier", "selected earlier" not in shown, shown)
    s.step("  at frame #0", "frame\r", r"#0 .*" + FRAME0)
    s.step("realm-thread back: the thread's own frame", "realm-thread back\r",
           r"no parked user thread is selected now\.\n#0 [^\n]*realm_gdb_test_stop")
    s.step("  Enter", "\r", None, unwanted="no parked user thread is selected")
    s.finish()


def _syscall_sp_pc(shown):
    """The SP and PC at the end of a /proc/PID/task/TID/syscall line."""
    m = re.findall(r"(0x[0-9a-f]+) (0x[0-9a-f]+)\s*$", shown, re.M)
    return m[-1] if m else None


@test("terminal.plain")
def test_terminal_plain(t):
    """With gdb on a terminal, after `realm-thread PTR`: plain commands show
    the parked thread, printing as gdb does (`frame N`, `up`, `x` and its
    repeat), while the thread shows its own registers in /proc between
    commands, after a command that fails, and after Ctrl-C in a function
    call or in a long command; `step` is refused, and `continue` ends the
    selection and runs."""
    s = t.gdb(t.realm_args())
    shown = s.read(timeout=t.args.timeout)
    if not t.check("the program stops in realm_gdb_test_stop",
                   "realm_gdb_test_stop" in shown, shown):
        return
    shown = s.step("find chain0's user thread", "source {}\r".format(
        t.file("find_chain.py", FIND_CHAIN)))
    m = re.search(r"CHAIN0=(0x[0-9a-f]+)", shown)
    if not t.check("chain0's user thread found", m is not None, shown):
        return
    ptr = m.group(1)
    pid = re.search(r"process (\d+)", s.step("", "info inferiors\r")).group(1)
    tid = re.search(r"LWP (\d+)", s.step("", "thread\r")).group(1)
    proc = None
    if os.path.isdir("/proc/{}/task".format(pid)):  # (Linux's)
        proc = "shell cat /proc/{}/task/{}/syscall\r".format(pid, tid)
        own = _syscall_sp_pc(s.step("", proc))
        t.check("the thread's own SP and PC, from /proc", own is not None, "")
    s.step("realm-thread PTR", "realm-thread {}\r".format(ptr),
           r"show it, and 'step'[\s\S]*#0 .*" + FRAME0)
    shown = s.step("bt: the parked stack", "bt\r",
                   r"#0 .*" + FRAME0 + r"[\s\S]*rgt_chain_level3")
    m = re.search(r"#(\d+) .* rgt_chain_level3 ", shown)
    level = m.group(1) if m else "0"
    s.step("frame N: prints the frame", "frame {}\r".format(level),
           r"#{} .*rgt_chain_level3".format(level))
    s.step("print a parked local", "print chain_marker\r", r"= 4242")
    s.step("up: prints the frame", "up\r", r"rgt_chain_level2")
    s.step("down: prints the frame", "down\r", r"rgt_chain_level3")
    s.step("a function call", "print realm_gdb_test_add(chain_marker, 1)\r", r"= 4243")
    if proc:
        t.check("/proc: the thread's own SP and PC between commands",
                _syscall_sp_pc(s.step("", proc)) == own, "")
    addr = re.compile(r"^(0x[0-9a-f]+)(?: <[^>]*>)?:", re.M)
    first = addr.findall(s.step("x/2gx $sp", "x/2gx $sp\r", r"0x"))
    again = addr.findall(s.step("  Enter: gdb's repeat, further on", "\r", r"0x"))
    t.check("  16 bytes further on", first and again and
            int(again[0], 16) == int(first[0], 16) + 16, "{} {}".format(first, again))
    s.step("a command that fails", "print rgt_nosuchvar\r", r"No symbol")
    if proc:
        t.check("/proc: the thread's own SP and PC at the prompt",
                _syscall_sp_pc(s.step("", proc)) == own, "")
    s.step("  still parked, in the same frame", "print chain_marker\r", r"= 4242")
    s.send("print realm_gdb_test_nap(3000000)\r")
    time.sleep(1)
    s.step("Ctrl-C in a function call", "\x03", r"[Ii]nterrupt|Quit|signaled", timeout=20)
    if proc:
        t.check("/proc: the thread's own SP and PC after it",
                _syscall_sp_pc(s.step("", proc)) == own, "")
    # A long command (the disassembly of librealm), cut short by Ctrl-C as
    # without a selection: typed, it gets it, rather than having it held.
    m = re.search(r"^(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+Yes.*librealm",
                  s.step("", "info sharedlibrary\r"), re.M)
    # (Not where a Ctrl-C that lands as Python runs kills gdb, gdb 8.2 and
    # 9.2: realm-gdb holds it there.)
    safe = "RGT-SAFE True" in s.step("", 'python print("RGT-SAFE", '
                                     '__import__("__main__")._SIGINT_IN_PYTHON_SAFE)\r')
    if m and safe:
        s.send("disassemble /r {},{}\r".format(m.group(1), m.group(2)))
        busy = s.read(None, 1)
        if not re.search(r"\(gdb\) $", busy):
            start = time.time()
            s.send("\x03")
            end = s.read(PROMPT, 20)
            # (gdb 17 and later put a mark before it: "❌️ Quit".)
            t.check("Ctrl-C in a long command: it quits", re.search(
                r"\n[^\w\n]*Quit\s*\n", end[-2000:]) is not None and time.time() - start < 10,
                end[-1000:])
            if proc:
                t.check("/proc: the thread's own SP and PC after it",
                        _syscall_sp_pc(s.step("", proc)) == own, "")
    s.step("  still parked", "print chain_marker\r", r"= 4242")
    s.step("step: refused", "step\r",
           r"refusing 'step' while user thread \d+ \({}\) is selected[\s\S]*realm-thread "
           r"back".format(ptr))
    s.step("continue: ends the selection, and the program finishes", "continue\r",
           r"continuing; no longer selecting user thread \d+ \({}\)[\s\S]*REALM_GDB_TEST: "
           r"PASS[\s\S]*exited normally".format(ptr), timeout=t.args.timeout)
    s.finish()


@test("terminal.plain_startup")
def test_terminal_plain_startup(t):
    """With gdb on a terminal, commands given with -ex run before the first
    prompt: one that fails while it shows a selected parked thread leaves it
    shown no longer than until the next command (`continue` here, which
    ends the selection and runs), even with realm-gdb.py loaded again in
    between."""
    find = t.file("find_chain.py", FIND_CHAIN.replace(
        'print("CHAIN0=%#x" % int(h))',
        'gdb.execute("set $rgt_chain0 = %#x" % int(h), to_string=True)'))
    s = t.gdb(t.realm_args(["-ex", "source " + find, "-ex", "realm-thread $rgt_chain0",
                            "-ex", "print rgt_nosuchvar", "-ex", "bt 1",
                            "-ex", "print rgt_nosuchvar",
                            "-ex", "source " + os.path.join(SOURCE, "realm-gdb.py"),
                            "-ex", "continue"]))
    shown = s.read(timeout=t.args.timeout)
    t.check("selected, then bt: the parked thread", re.search(
        r"selected user thread[\s\S]*No symbol[\s\S]*#0 .*" + FRAME0 + r"[\s\S]*No symbol"
        r"[\s\S]*realm-gdb.py loaded again", shown), shown)
    t.check("continue: ends the selection, and the program finishes", re.search(
        r"continuing; no longer selecting user thread[\s\S]*REALM_GDB_TEST: PASS[\s\S]*"
        r"exited normally", shown), shown)
    s.finish()


@test("terminal.plain_x_script")
def test_terminal_plain_x_script(t):
    """A Python script given with -x after an -ex command that fails while
    it shows a selected parked thread (gdb runs no `source` hook for -x)
    isn't taken for part of that command: its `continue` ends the selection
    and runs, rather than being refused inside the failed command's
    window."""
    find = t.file("find_chain.py", FIND_CHAIN.replace(
        'print("CHAIN0=%#x" % int(h))',
        'gdb.execute("set $rgt_chain0 = %#x" % int(h), to_string=True)'))
    script = t.file("cont.py", 'gdb.execute("continue")\n')
    s = t.gdb(t.realm_args(["-ex", "source " + find, "-ex", "realm-thread $rgt_chain0",
                            "-ex", "print rgt_nosuchvar", "-x", script]))
    shown = s.read(timeout=t.args.timeout)
    t.check("the failure, then the script's continue runs", re.search(
        r"No symbol[\s\S]*continuing; no longer selecting user thread[\s\S]*"
        r"REALM_GDB_TEST: PASS[\s\S]*exited normally", shown), shown)
    t.check("  not refused", "refusing" not in shown, shown)
    s.finish()


@test("terminal.body_lines")
def test_terminal_body_lines(t):
    """With gdb on a terminal and a parked thread selected, no line of a
    command's body (up to its `end`) runs as a gdb command of its own, a
    `continue` among them: a bare `python` reads its body, and runs it on
    the thread as it really is, with a note; `rt python` and `realm-thread
    apply PTR python`, which are refused, read theirs first, saying so."""
    s = t.gdb(t.realm_args())
    ptr, _ = t.select_chain0(s)
    body = rb"\n>\s*$"
    lines = ["for i in range(1):", "  continue", 'print("PY-AT", gdb.selected_frame().name())']
    s.step("bare python: a note", "python\r", r"'python' runs on thread \d+ as it really "
           r"is, not on user thread \d+ \(" + ptr, until=body)
    for line in lines:
        s.step("", line + "\r", until=body)
    s.step("  its body runs on the thread as it really is", "end\r",
           r"PY-AT realm_gdb_test_stop", unwanted=r"Continuing|refusing")
    for cmd in ("rt python", "realm-thread apply {} python".format(ptr)):
        s.step(cmd + ": says first that it is refused, and its lines dropped", cmd + "\r",
               r"refuses 'python', which reads the lines after it: type them, up to 'end', "
               r"to have them dropped", until=body)
        for line in lines:
            s.step("", line + "\r", until=body)
        s.step(cmd + ": refused, its lines dropped", "end\r",
               r"refusing 'python' in '[^']+': it reads the lines after it[^\n]*\(the "
               r"lines typed were dropped\)", unwanted=r"Continuing|PY-AT")
    s.step("the program never ran", "info program\r", r"stopped|Program stopped|It stopped",
           unwanted=r"is running|not being run")
    s.step("continue: the program finishes", "continue\r",
           r"REALM_GDB_TEST: PASS[\s\S]*exited normally", timeout=t.args.timeout)
    s.finish()


@test("terminal.pi_failed")
def test_terminal_pi_failed(t):
    """With gdb on a terminal and a parked thread selected, a plain command
    that fails in Python's REPL (`pi`) leaves no window open while the REPL
    waits for its next line: the thread has its own registers there."""
    s = t.gdb(t.realm_args())
    ptr, _ = t.select_chain0(s)
    repl = rb">>> $"
    s.step("pi: a note", "pi\r", r"'python-interactive' runs on thread|'pi' runs on thread",
           until=repl)
    s.step("  a plain command that fails",
           r"""exec("try:\n gdb.execute('print rgt_nosuchvar')\nexcept gdb.error as e:\n"""
           r""" print('RGT-ERR', e)")""" + "\r", r"RGT-ERR No symbol", until=repl)
    s.step("  no window left at the next line",
           'm = __import__("__main__"); print("RGT-OPEN", m._plain is not None or '
           'm._window is not None, gdb.selected_frame().name())\r',
           r"RGT-OPEN False realm_gdb_test_stop", until=repl)
    s.step("  out of the REPL, still selected", "\x04", None)
    s.step("  bt: the parked thread", "bt 1\r", r"#0 .*" + FRAME0)
    s.step("continue: the program finishes", "continue\r",
           r"REALM_GDB_TEST: PASS[\s\S]*exited normally", timeout=t.args.timeout)
    s.finish()


@test("terminal.pager_notes")
def test_terminal_pager_notes(t):
    """With gdb's pager on, `realm-thread apply all` printing more than a
    screen of what it collected: `q` at the pager's prompt, and Ctrl-C there
    (but on gdb 8.2 and 9.2, where realm-gdb holds Ctrl-C as Python runs),
    cut the output short, and the notes (here that each thread's register
    change doesn't last) come all the same."""
    s = t.gdb(t.realm_args())
    shown = s.read(timeout=t.args.timeout)
    if not t.check("the program stops in realm_gdb_test_stop",
                   "realm_gdb_test_stop" in shown, shown):
        return
    reg = "$x19" if re.search(r"aarch64", s.step("", "show architecture\r")) else "$rbx"
    safe = "RGT-SAFE True" in s.step("", 'python print("RGT-SAFE", '
                                     '__import__("__main__")._SIGINT_IN_PYTHON_SAFE)\r')
    s.step("", "set height 8\r")
    # (gdb 8's prompt is "---Type <return> to continue, or q <return> to quit---".)
    pager = rb"Type <(?:RET|return)>[^\n]*-+\s*$"
    for label, key in (("q", "q\r"), ("Ctrl-C", "\x03")):
        if key == "\x03" and not safe:
            continue
        s.step("realm-thread apply all: the pager stops it",
               "realm-thread apply all print {} = 7\r".format(reg), r"User thread \d",
               until=pager)
        shown = s.step("", key, until=PROMPT + b"|" + pager)
        for _ in range(20):  # (The notes may fill a screen too.)
            if not re.search(pager, shown.encode()):
                break
            shown += s.step("", "\r", until=PROMPT + b"|" + pager)
        # (gdb says "Quit" once the notes are printed, as realm-gdb's command ends.)
        t.check("{} at the pager: the output stops, and the notes come".format(label),
                re.search(r"\bQuit\b", shown) is not None and re.search(
                    r"'print' changed \{}, which doesn't last".format(reg), shown) is not None,
                shown)
    s.step("", "set height 0\r")
    s.step("continue: the program finishes", "continue\r",
           r"REALM_GDB_TEST: PASS[\s\S]*exited normally", timeout=t.args.timeout)
    s.finish()


@test("terminal.banner_pager")
def test_terminal_banner_pager(t):
    """realm-gdb.py's banner on a 24-line terminal: two lines, which leave
    room for gdb's messages of attaching to a process (`gdb -p PID`, say)
    before gdb's pager stops them; `q` at the pager, should it come as
    realm-gdb.py loads, stops the banner without saying that loading
    realm-gdb.py failed, and its commands work."""
    s = t.gdb(["-nx", "-q", "-ex", "set height 0", "-ex", "set width 0",
               "-ex", "break realm_gdb_test_stop", "-ex", "run",
               "--args", t.args.program] + base.program_args() + ["-mode", "stop"],
              size=(24, 80))
    # (gdb 8's prompt is "---Type <return> to continue, or q <return> to quit---".)
    pager = rb"Type <(?:RET|return)>[^\n]*-+\s*$"
    shown = s.read(rb"(?:" + pager + rb")|(?:" + PROMPT + rb")", timeout=t.args.timeout)
    m = re.search(r"realm-gdb\.py loaded: [\s\S]*see 'realm-help'\.", shown)
    t.check("the banner, two lines", m is not None and m.group(0).count("\n") == 1, shown)
    if re.search(pager, shown.encode()):
        shown = s.step("q at the pager: no message that loading failed", "q\r", None,
                       timeout=t.args.timeout)
        t.check("  none", re.search(r"interrupted|Python Exception|Traceback", shown) is None,
                shown)
    t.select_chain0(s, shown=shown if "realm_gdb_test_stop" in shown else None)
    s.step("continue: the program finishes", "continue\r",
           r"REALM_GDB_TEST: PASS[\s\S]*exited normally", timeout=t.args.timeout)
    s.finish()


@test("terminal.breakpoint_commands")
def test_terminal_breakpoint_commands(t):
    """Breakpoint commands, which gdb runs itself at a stop that a typed
    `continue` comes to: `thread apply all bt` (the parked threads too), a
    function call in `realm-thread apply all`, selecting a parked thread
    and a plain `bt` of it, then `continue`, which ends the selection and
    runs; the program finishes cleanly."""
    find = t.file("find_chain.py", FIND_CHAIN_VAR)
    bp = t.file("bp.gdb", "break realm_gdb_test_release_parked\ncommands\n"
                "thread apply all bt 1\n"
                "realm-thread apply all print realm_gdb_test_add(1, 2)\n"
                "realm-thread $rgt_chain0\n"
                "bt 1\n"
                "continue\n"
                "end\n")
    s = t.gdb(t.realm_args())
    shown = s.read(timeout=t.args.timeout)
    if not t.check("the program stops in realm_gdb_test_stop",
                   "realm_gdb_test_stop" in shown, shown):
        return
    s.step("", "source {}\r".format(find))
    s.step("", "source {}\r".format(bp))
    shown = s.step("continue: the breakpoint's commands run", "continue\r",
                   r"exited normally", timeout=t.args.timeout)
    t.check("  thread apply all bt: the parked threads too",
            re.search(r"Parked Realm user threads[\s\S]*User thread \d+ \(0x[0-9a-f]+\)[^\n]*\n"
                      r"#0 [^\n]*" + FRAME0, shown), shown)
    t.check("  realm-thread apply all: the function calls",
            re.search(r"User thread \d+ \(0x[0-9a-f]+\):?[^\n]*\n\$\d+ = 3\n", shown), shown)
    t.check("  the selection, a plain bt of it, then continue ends it", re.search(
        r"selected user thread[\s\S]*#0 [^\n]*" + FRAME0 + r"[\s\S]*continuing; no "
        r"longer selecting user thread[\s\S]*REALM_GDB_TEST: PASS", shown), shown)
    t.check("  nothing refused", "refusing" not in shown, shown)
    s.finish()


@test("terminal.plain_failed_background")
def test_terminal_plain_failed_background(t):
    """A plain command that fails, with a parked thread selected, in the
    breakpoint commands of a stop in the background (`continue &`), where
    gdb gives no prompt event: the window it opened is closed as soon as gdb
    is back in its event loop (realm-gdb posts a closer, from gdb 15), so
    the thread has its own registers while gdb waits; `continue` ends the
    selection and the program finishes cleanly."""
    find = t.file("find_chain.py", FIND_CHAIN_VAR)
    bp = t.file("bp.gdb", "break realm_gdb_test_release_parked\ncommands\n"
                "printf \"RGT-OWN-SP %#lx\\n\", $sp\n"
                "realm-thread $rgt_chain0\n"
                "print rgt_nosuchvar\n"
                "end\n")
    s = t.gdb(t.realm_args())
    shown = s.read(timeout=t.args.timeout)
    if not t.check("the program stops in realm_gdb_test_stop",
                   "realm_gdb_test_stop" in shown, shown):
        return
    if "RGT-POSTED True" not in s.step("", 'python print("RGT-POSTED", '
                                       'hasattr(gdb, "interrupt"))\r'):
        s.finish()
        raise Skip("gdb 9 to 14 die exiting with a posted event left: none is posted")
    s.step("", "source {}\r".format(find))
    parked = re.search(r"= (0x[0-9a-f]+)", s.step(
        "", "realm-thread apply $rgt_chain0 p/x $sp\r"))
    s.step("", "source {}\r".format(bp))
    shown = s.step("continue &: the breakpoint's commands, and the failure", "continue &\r",
                   r"No symbol", until=rb"No symbol[^\n]*\n", timeout=t.args.timeout)
    own = re.search(r"RGT-OWN-SP (0x[0-9a-f]+)", shown)
    time.sleep(1)
    s.read(None, 0.5)
    raw = RAW_SP.search(s.step("", "maint print raw-registers\r"))
    t.check("the thread's own SP while gdb waits, not the parked thread's",
            own and raw and parked and int(raw.group(1), 16) == int(own.group(1), 16) and
            int(raw.group(1), 16) != int(parked.group(1), 16),
            "own {} raw {} parked {}".format(own and own.group(1), raw and raw.group(1),
                                             parked and parked.group(1)))
    s.step("continue: ends the selection, and the program finishes", "continue\r",
           r"continuing; no longer selecting user thread[\s\S]*REALM_GDB_TEST: PASS"
           r"[\s\S]*exited normally", timeout=t.args.timeout)
    s.finish()


@test("terminal.plain_x_gdb_script")
def test_terminal_plain_x_gdb_script(t):
    """A gdb script given with -x whose plain command fails while it shows a
    selected parked thread (which ends the script there), followed by -ex
    commands: one realm-gdb doesn't hook (`maint print raw-registers`), and
    `continue`, which ends the selection and runs, rather than being refused
    inside the failed command's window; the program finishes cleanly."""
    find = t.file("find_chain.py", FIND_CHAIN_VAR)
    script = t.file("sel_fail.gdb", "source {}\nrealm-thread $rgt_chain0\n"
                    "print rgt_nosuchvar\necho RGT-NOT-REACHED\\n\n".format(find))
    s = t.gdb(t.realm_args(["-x", script, "-ex", "maint print raw-registers",
                            "-ex", "continue"]))
    shown = s.read(timeout=t.args.timeout)
    t.check("the failure ends the script; then the -ex commands", re.search(
        r"No symbol[\s\S]*Raw value[\s\S]*continuing; no longer selecting user thread"
        r"[\s\S]*REALM_GDB_TEST: PASS[\s\S]*exited normally", shown), shown)
    t.check("  the rest of the script doesn't run", "RGT-NOT-REACHED" not in shown, shown)
    t.check("  not refused", "refusing" not in shown, shown)
    s.finish()


@test("terminal.piped_stdin")
def test_terminal_piped_stdin(t):
    """gdb reading its commands from a pipe, not a terminal: a plain command
    that fails while a parked thread is selected leaves no window open at
    the next command, one realm-gdb doesn't hook (`maint print
    raw-registers`: the thread's own SP); `continue` ends the selection and
    the program finishes cleanly."""
    import subprocess
    find = t.file("find_chain.py", FIND_CHAIN_VAR)
    commands = "\n".join([
        "source " + find, 'printf "RGT-OWN-SP %#lx\\n", $sp',
        "realm-thread apply $rgt_chain0 p/x $sp", "realm-thread $rgt_chain0",
        "bt 1", "print rgt_nosuchvar", "maint print raw-registers", "continue", ""])
    argv = [shutil.which(t.args.gdb), "-iex", "set auto-load python-scripts off",
            "-iex", "source " + t.args.script] + t.realm_args()
    try:
        out = subprocess.run(argv, input=commands.encode(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, cwd=t.tmp, env=t.env(),
                             timeout=t.args.timeout + 60).stdout
    except subprocess.TimeoutExpired as e:
        t.check("gdb finishes", False, clean(e.stdout or b""))
        return
    shown = clean(out)
    own = re.search(r"RGT-OWN-SP (0x[0-9a-f]+)", shown)
    parked = re.search(r"\$\d+ = (0x[0-9a-f]+)", shown)
    raw = RAW_SP.search(shown[shown.find("No symbol"):])
    t.check("the selection, a plain bt of it, the failure", re.search(
        r"selected user thread[\s\S]*#0 [^\n]*" + FRAME0 + r"[\s\S]*No symbol", shown), shown)
    t.check("the thread's own SP at the next command, not the parked thread's",
            own and raw and parked and int(raw.group(1), 16) == int(own.group(1), 16) and
            int(raw.group(1), 16) != int(parked.group(1), 16), shown)
    t.check("continue: ends the selection, and the program finishes", re.search(
        r"continuing; no longer selecting user thread[\s\S]*REALM_GDB_TEST: PASS"
        r"[\s\S]*exited normally", shown), shown)
    bad = [b for b in base.BAD_OUTPUT if b in shown]
    t.check("no errors of gdb's or Python's ({})".format(bad), not bad, shown)


def continue_after_ctrl_cs(t, s):
    """Continue the program to its end, after a test's Ctrl-Cs at random
    moments.  One that lands as a call's thread stops, before gdb takes the
    terminal back from the program (gdb's own race, with or without
    realm-gdb), goes to the program, which gdb then stops at that SIGINT as
    it resumes it: continued past, once, if the terminal sent it (not
    realm-gdb, see _our_sigint)."""
    done = r"REALM_GDB_TEST: PASS[\s\S]*exited normally"
    shown = s.step("", "continue\r", timeout=t.args.timeout)
    if re.search(done, shown) is None and "received signal SIGINT" in shown:
        why = s.step("", 'python import __main__ as rgt_m; print("RGT-SIGINT", '
                     'rgt_m._stopped_by_sigint(), rgt_m._our_sigint())\r')
        if "RGT-SIGINT True False" in why:
            shown += why + s.step("", "continue\r", timeout=t.args.timeout)
    t.check("continue: the program finishes", re.search(done, shown) is not None, shown)


@test("terminal.plain_ctrl_c_calls")
def test_terminal_plain_ctrl_c_calls(t):
    """With gdb on a terminal and a parked thread selected, typed plain
    commands that make several function calls, each cut short by a Ctrl-C
    at a random moment (as gdb runs the command itself, between its calls,
    as one is set up, runs, or ends): every call runs as a call made in a
    window must (the user's breakpoint in the called function isn't hit),
    and nothing is left behind (scheduler-locking the user's, breakpoints
    enabled, no call's record in gdb, no warning).  gdb hands a Ctrl-C that
    came as it ran to the next Python it runs, realm-gdb's handler of the
    next call, say, before its first line."""
    s = t.gdb(t.realm_args())
    t.select_chain0(s)
    s.step("", "set scheduler-locking replay\r")
    s.step("a breakpoint in the called function", "break realm_gdb_test_add\r",
           r"Breakpoint 2")
    expr = "+".join(["realm_gdb_test_add(1, 2)"] * 8)
    check = ('python print("RGT-CHK", all(b.enabled for b in gdb.breakpoints()), '
             'gdb.parameter("scheduler-locking"), '
             'gdb.execute("maint print dummy-frames", to_string=True).count("id="), "END")\r')
    # (gdb 8 says so of a Ctrl-C that lands as a Python command starts, a
    # hook of realm-gdb's, which it takes for an error; later gdbs quit.)
    old = re.search(r"RGT-QUITS (\w+)", s.step("", 'python print("RGT-QUITS", '
                                               'hasattr(gdb.Value, "format_string"))\r'))
    gdb8 = old is not None and old.group(1) == "False"
    if gdb8:
        s.allowed += ["Python Exception <class 'KeyboardInterrupt'>",
                      "Error occurred in Python command"]
    rng = random.Random(1)
    bad, quits = None, 0
    for i in range(40):
        s.send("print {}\r".format(expr))
        time.sleep(rng.uniform(0, 0.04))
        s.send("\x03")
        shown = s.read(PROMPT, 30)
        quits += "= 24" not in shown
        # (A Ctrl-C that lands as gdb reads the line drops it: so the check
        # may come before a prompt of that one's.)
        chk, m = "", None
        for _ in range(4):
            chk += s.step("", check, until=rb"RGT-CHK .* END\s*\n[\s\S]*\(gdb\) $", timeout=10)
            m = re.search(r"RGT-CHK (.*) END", chk)
            if m is not None:
                break
        if m is None or m.group(1) != "True replay 0" or re.search(
                r"WARNING|hit Breakpoint|doesn't last", shown) or (
                "Python Exception" in shown.replace(
                    "Python Exception <class 'KeyboardInterrupt'>" if gdb8 else "\0", "")):
            bad = "after Ctrl-C {}:\n{}\n{}".format(i + 1, shown[-2000:], chk[-500:])
            break
    t.check("40 Ctrl-Cs, {} cutting a command short: nothing left behind".format(quits),
            bad is None, bad or "")
    s.step("realm-thread back", "realm-thread back\r", r"no parked user thread is selected")
    s.step("", "delete 2\r")
    continue_after_ctrl_cs(t, s)
    s.finish()


@test("terminal.rt_ctrl_c_calls")
def test_terminal_rt_ctrl_c_calls(t):
    """As plain_ctrl_c_calls, through `rt` and `realm-thread apply`, which
    let Ctrl-C through for their command where Ctrl-C can land while Python
    runs one (gdb >= 10): Ctrl-C at random moments in commands that make
    several calls cuts some short, and leaves nothing behind (breakpoints
    enabled, scheduler-locking as it was, no dummy frames), the host has its
    own registers, and the program passes."""
    s = t.gdb(t.realm_args())
    ptr, level = t.select_chain0(s, rt="rt ")
    s.step("", "set scheduler-locking replay\r")
    s.step("a breakpoint in the called function", "break realm_gdb_test_add\r",
           r"Breakpoint 2")
    expr = "+".join(["realm_gdb_test_add(1, 2)"] * 8)
    check = ('python print("RGT-CHK", all(b.enabled for b in gdb.breakpoints()), '
             'gdb.parameter("scheduler-locking"), '
             'gdb.execute("maint print dummy-frames", to_string=True).count("id="), '
             'gdb.selected_thread().num, hex(int(gdb.parse_and_eval("$sp"))), "END")\r')
    s.step("realm-thread back", "realm-thread back\r")
    base_chk = s.step("", check, until=rb"RGT-CHK .* END\s*\n[\s\S]*\(gdb\) $", timeout=10)
    want = re.search(r"RGT-CHK (.*) END", base_chk).group(1)
    s.step("select again", "realm-thread {}\r".format(ptr))
    rng = random.Random(2)
    bad, quits = None, 0
    for i in range(40):
        cmd = ("rt print {}" if i % 2 == 0 else
               "realm-thread apply " + ptr + " print {}").format(expr)
        s.send(cmd + "\r")
        time.sleep(rng.uniform(0, 0.06))
        s.send("\x03")
        shown = s.read(PROMPT, 30)
        quits += "= 24" not in shown
        s.step("", "realm-thread back\r")
        chk, m = "", None
        for _ in range(4):
            chk += s.step("", check, until=rb"RGT-CHK .* END\s*\n[\s\S]*\(gdb\) $", timeout=10)
            m = re.search(r"RGT-CHK (.*) END", chk)
            if m is not None:
                break
        s.step("", "realm-thread {}\r".format(ptr))
        if m is None or m.group(1) != want or re.search(
                r"WARNING|hit Breakpoint|doesn't last", shown) or (
                "Python Exception" in shown):
            bad = "after Ctrl-C {}:\n{}\n{}\nwant {}".format(i + 1, shown[-2000:], chk[-500:],
                                                               want)
            break
    t.check("40 Ctrl-Cs, {} cutting a command short: nothing left behind".format(quits),
            bad is None, bad or "")
    s.step("realm-thread back", "realm-thread back\r", r"no parked user thread is selected")
    s.step("", "delete 2\r")
    continue_after_ctrl_cs(t, s)
    s.finish()


# Has realm-gdb's handler of Ctrl-C in Python take one landing as gdb starts
# realm-gdb's prompt hook, and one landing as gdb starts what realm-gdb
# posted (gdb >= 15): a Ctrl-C that came as a command ended, which gdb hands
# the next Python it runs.  (Called with the frame, rather than a signal sent:
# Python >= 3.12 runs signal handlers only at some points, a function's first
# line, after a call, and the like, so one sent as the hook begins would land
# past its first line.)
IDLE_SIGINT = """\
import gdb, signal, sys
import __main__ as rgt_m


def rgt_landing(what, frame):
    try:
        rgt_m._on_python_sigint(signal.SIGINT, frame)
        print("RGT-IDLE-SIGINT", what, "handled")
    except KeyboardInterrupt:
        print("RGT-IDLE-SIGINT", what, "raised")


rgt_real_prompt = rgt_m._realm_prompt


def rgt_prompt(current):
    rgt_m._realm_prompt = rgt_real_prompt
    rgt_landing("prompt", sys._getframe(1))
    return rgt_real_prompt(current)


def rgt_posted():
    rgt_landing("posted", sys._getframe())


rgt_m._realm_prompt = rgt_prompt
"""


@test("terminal.idle_ctrl_c")
def test_terminal_idle_ctrl_c(t):
    """With gdb on a terminal and a parked thread selected, a Ctrl-C that
    lands as gdb starts realm-gdb's prompt hook, as one that came as a
    command ended does, or what realm-gdb posted (gdb >= 15): "Quit", as gdb
    says of one at its prompt, rather than a Python exception; the prompt
    still shows the selection."""
    s = t.gdb(t.realm_args())
    t.select_chain0(s)
    shown = s.step("as the prompt hook starts: Quit", "source {}\r".format(
        t.file("idle_sigint.py", IDLE_SIGINT)), r"RGT-IDLE-SIGINT prompt handled",
        unwanted=r"raised|Python Exception")
    t.check("  saying so", "Quit" in shown, shown)
    t.check("  the prompt shows the selection",
            re.search(r"\n\[realm-thread \d+\] \(gdb\) $", shown), shown)
    if "RGT-POSTED True" in s.step("", 'python print("RGT-POSTED", hasattr(gdb, "interrupt"))\r'):
        # (gdb runs it once back in its event loop, after the prompt.)
        s.step("as what realm-gdb posted starts: Quit", "python rgt_m._post(rgt_posted)\r",
               r"Quit\s+RGT-IDLE-SIGINT posted handled",
               until=rb"RGT-IDLE-SIGINT posted (?:handled|raised)\b",
               unwanted=r"raised|Python Exception")
        s.step("", "echo\r")
    s.step("realm-thread back", "realm-thread back\r", r"no parked user thread is selected")
    s.step("continue: the program finishes", "continue\r",
           r"REALM_GDB_TEST: PASS[\s\S]*exited normally", timeout=t.args.timeout)
    s.finish()


@test("terminal.plain_handoff_interrupted")
def test_terminal_plain_handoff_interrupted(t):
    """With gdb on a terminal and a parked thread selected, a Ctrl-C that
    comes while a typed plain command's window opens (here a SIGINT gdb
    sends itself then), which lands as the window lets Ctrl-C through for
    the command: the command quits, its window closes, and the next prompt
    finds nothing left of it (no note that the command changed registers,
    which comparing the host's own with the parked thread's would give);
    the next command shows the parked thread."""
    s = t.gdb(t.realm_args())
    t.select_chain0(s)
    s.step("", "python import __main__ as rgt_m, os, signal\r")
    s.step("a SIGINT as the window opens", "python rgt_real = rgt_m._show_in_window; "
           "rgt_m._show_in_window = lambda *a: (rgt_real(*a), os.kill(os.getpid(), "
           "signal.SIGINT))[0]\r")
    s.step("print 1: quits", "print 1\r", r"Quit", unwanted=r"doesn't last|WARNING")
    s.step("", "python rgt_m._show_in_window = rgt_real\r")
    s.step("the next prompt: nothing left", "echo RGT-NEXT\\n\r", r"RGT-NEXT",
           unwanted=r"doesn't last|WARNING|Python Exception")
    s.step("no window left", "python print('RGT-STATE', rgt_m._plain is None, "
           "rgt_m._window is None)\r", r"RGT-STATE True True")
    s.step("bt 1: the parked thread", "bt 1\r", r"#0 .*" + FRAME0)
    s.step("realm-thread back", "realm-thread back\r", r"no parked user thread is selected")
    s.step("continue: the program finishes", "continue\r",
           r"REALM_GDB_TEST: PASS[\s\S]*exited normally", timeout=t.args.timeout)
    s.finish()


@test("terminal.plain_detach")
def test_terminal_plain_detach(t):
    """With gdb on a terminal, attached: detaching with a parked thread
    selected, right after a command that failed while it showed it, lets
    the program carry on cleanly."""
    prog = base.Proc([t.args.program] + base.program_args() +
                     ["-mode", "attach", "-timeout", "120"], base.clean_env(), t.tmp)
    try:
        m = prog.wait_for(r"REALM_GDB_TEST: READY (\d+)", 60)
        if not t.check("the program is ready to attach to", m is not None, prog.output):
            return
        s = t.gdb(["-nx", "-q", "-iex", "set height 0", "-iex", "set width 0",
                   "-p", m.group(1), t.args.program])
        s.read(timeout=t.args.timeout)
        s.step("", "thread apply all bt\r", r"rgt_hot_probe", timeout=60)
        num = re.search(r"Thread (\d+) [^\n]*\n(?:#[^\n]*\n)*#\d+ [^\n]*rgt_hot_probe",
                        clean(s.out))
        if num:
            s.step("", "thread {}\r".format(num.group(1)))
        shown = s.step("find chain0's user thread", "source {}\r".format(
            t.file("find_chain.py", FIND_CHAIN)))
        m = re.search(r"CHAIN0=(0x[0-9a-f]+)", shown)
        if not t.check("chain0's user thread found", m is not None, shown):
            return
        s.step("realm-thread PTR", "realm-thread {}\r".format(m.group(1)),
               r"show it, and 'step'")
        s.step("bt: the parked thread", "bt 1\r", FRAME0)
        s.step("release the program (a global: rt or not, the same)",
               "rt set var realm_gdb_test_release = 1\r", unwanted="refusing")
        s.step("a command that fails", "print rgt_nosuchvar\r", r"No symbol")
        s.step("detach", "detach\r", r"Detaching|detached", unwanted="refusing")
        s.finish()
        rc = prog.wait(90)
        t.check("the program carries on, and finishes cleanly",
                rc == 0 and "REALM_GDB_TEST: PASS" in prog.output,
                "status {}\n{}".format(rc, prog.output[-2000:]))
    finally:
        prog.kill()


@test("terminal.attach_gdb_killed")
def test_terminal_attach_gdb_killed(t):
    """gdb killed (SIGKILL) while a window has an attached program's thread
    hold a parked thread's registers: the kernel kills the program too
    (PTRACE_O_EXITKILL, see _ExitKill), rather than let it run on
    corrupted.  Killed once the window is over, gdb leaves the program to
    carry on and finish cleanly.  (Linux.)"""
    import subprocess
    if not sys.platform.startswith("linux"):
        raise Skip("PTRACE_O_EXITKILL is Linux's")
    for when, mi in (("in a window", False), ("after one", False), ("in a window", True)):
        marker = os.path.join(t.tmp, "rgt-marker")
        if os.path.exists(marker):
            os.unlink(marker)
        sleep = 'python open("{}", "w").close(); import time; time.sleep(120)'.format(marker)
        cmds = ["set var realm_gdb_test_release = 1"] + (
            ["realm-thread apply all " + sleep] if when == "in a window" else
            ["realm-thread apply all bt 1", sleep])
        prog = base.Proc([t.args.program] + base.program_args() +
                         ["-mode", "attach", "-timeout", "120"], base.clean_env(), t.tmp)
        g = None
        try:
            m = prog.wait_for(r"REALM_GDB_TEST: READY (\d+)", 60)
            if not t.check("the program is ready to attach to", m is not None, prog.output):
                return
            argv = [shutil.which(t.args.gdb), "-nx", "-q", "-batch"] + (
                ["--interpreter=mi"] if mi else []) + [
                    "-iex", "set auto-load python-scripts off",
                    "-iex", "source " + t.args.script]
            for cmd in cmds:
                argv += ["-ex", cmd]
            g = base.Proc(argv + ["-p", m.group(1), t.args.program], t.env(), t.tmp)
            deadline = time.time() + t.args.timeout + 60
            while not os.path.exists(marker) and g.p.poll() is None and \
                    time.time() < deadline:
                time.sleep(0.05)
            when = when + (" (MI)" if mi else "")
            if not t.check("gdb gets to the sleep " + when, os.path.exists(marker),
                           g.output):
                return
            g.p.kill()
            g.p.wait()
            rc = prog.wait(90)
            if when.startswith("in a window"):
                t.check("killed in a window: the program is killed too",
                        rc == -signal.SIGKILL and "REALM_GDB_TEST: PASS" not in prog.output,
                        "status {}\n{}\n{}".format(rc, prog.output[-2000:], g.output))
            else:
                t.check("killed after one: the program carries on, and finishes cleanly",
                        rc == 0 and "REALM_GDB_TEST: PASS" in prog.output,
                        "status {}\n{}\n{}".format(rc, prog.output[-2000:], g.output))
        finally:
            if g is not None:
                g.kill()
            prog.kill()


# ---- the driver ---------------------------------------------------------

def run_test(args, name):
    tmp = tempfile.mkdtemp(prefix="rgt-term-")
    t = Context(args, name, tmp)
    status, detail = "PASS", ""
    try:
        TESTS[name](t)
        if t.failures:
            status, detail = "FAIL", t.failures[0]
    except Skip as e:
        status, detail = ("FAIL", t.failures[0]) if t.failures else ("SKIP", str(e))
    except Exception:
        import traceback
        status, detail = "FAIL", "driver exception: " + traceback.format_exc()
    finally:
        t.close()
        if args.keep:
            sys.stderr.write("kept {}\n".format(tmp))
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    log = "\n".join(t.log)
    if status == "FAIL" and detail.startswith("driver exception"):
        log += "\n" + detail
    if status == "FAIL":
        log += "\n----- the sessions' output -----\n" + "\n-----\n".join(
            clean(s.out)[-6000:] for s in t.sessions)
    return status, detail, log


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*", help="test names or glob patterns")
    ap.add_argument("--list", action="store_true", help="list tests and exit")
    ap.add_argument("--gdb", default="gdb")
    ap.add_argument("--program", help="path to realm_gdb_test")
    ap.add_argument("--script", default=os.path.join(SOURCE, "realm-gdb.py"),
                    help="realm-gdb.py")
    ap.add_argument("--timeout", type=int, default=180,
                    help="seconds allowed for the program to start, or run to its end")
    ap.add_argument("--step-timeout", type=int, default=30,
                    help="seconds allowed for each step")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every test's log, not just failures'")
    ap.add_argument("--keep", action="store_true", help="keep temporary directories")
    args = ap.parse_args()

    if args.list:
        for name in sorted(TESTS):
            print(name)
        return 0
    if not args.program:
        ap.error("--program is required")
    for name in ("program", "script"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    problem = base.environment_problem(args)
    if problem:
        print("SKIP: " + problem)
        return base.SKIP

    names = []
    for pat in args.tests or ["*"]:
        matched = [n for n in sorted(TESTS) if fnmatch.fnmatchcase(n, pat)]
        if not matched:
            sys.exit("no test matches {!r}".format(pat))
        names += [n for n in matched if n not in names]
    results = {}
    lock = threading.Lock()
    queue = list(names)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                name = queue.pop(0)
            t0 = time.time()
            status, detail, log = run_test(args, name)
            with lock:
                results[name] = status
                print("{:4}  {:40} {:5.1f}s  {}".format(status, name, time.time() - t0,
                                                       detail.splitlines()[0] if detail else ""))
                if args.verbose or status == "FAIL":
                    print("----- log of {} -----\n{}\n----- end of {} -----".format(
                        name, log.rstrip(), name))
                sys.stdout.flush()

    threads = [threading.Thread(target=worker) for _ in range(max(1, args.jobs))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    counts = {s: list(results.values()).count(s) for s in ("PASS", "FAIL", "SKIP")}
    print("{} passed, {} failed, {} skipped".format(counts["PASS"], counts["FAIL"],
                                                   counts["SKIP"]))
    if counts["FAIL"] or len(results) != len(names):
        return 1
    return base.SKIP if counts["SKIP"] == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
