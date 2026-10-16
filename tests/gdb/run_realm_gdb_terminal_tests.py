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
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            try:
                fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", size[0], size[1], 0, 0))
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
        bad = [b for b in base.BAD_OUTPUT if b in text]
        self.ctx.check("no errors of gdb's or Python's ({})".format(bad), not bad, text[-3000:])

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

    def gdb(self, args, env=None):
        """gdb ARGS, at a terminal, with realm-gdb.py sourced first."""
        s = Term(self, [shutil.which(self.args.gdb), "-iex", "set auto-load python-scripts off",
                        "-iex", "source " + self.args.script] + args, self.env(env))
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
               r"selected user thread " + ptr)
        shown = s.step("bt: the parked thread's stack", rt + "bt\r",
                       r"#0 .*Realm::UserThread::user_switch[\s\S]*rgt_chain_level3")
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
               r"\n\[realm-thread 0x[0-9a-f]+\] \(gdb\) $", unwanted=r"\] \[realm-thread")
    s.step("realm-thread back", "realm-thread back\r", r"no parked user thread is selected now")
    s.step("  then gdb's prompt", "echo\r", r"(?:^|\n)echo\n\(gdb\) $")
    s.step("  and setting", "show prompt\r", r"Gdb's prompt is \"\(gdb\) \"\.")
    s.step("select again", "realm-thread {}\r".format(ptr), r"selected")
    s.step("continue: the program finishes", "continue\r",
           r"REALM_GDB_TEST: PASS[\s\S]*exited normally", timeout=t.args.timeout)
    s.step("  then gdb's prompt", "show prompt\r", r"Gdb's prompt is \"\(gdb\) \"\.\n\(gdb\) $")
    s.finish()


@test("terminal.output")
def test_terminal_output(t):
    """realm-gdb.py with gdb on a terminal: output of `rt` and `realm-thread
    apply` that doesn't end a line (`output`'s) comes before the prompt, as
    without them."""
    s = t.gdb(t.realm_args())
    ptr, _ = t.select_chain0(s, rt="rt ")
    early = rb"\(gdb\) $|\(gdb\) 1234$"
    s.step("rt output: before the prompt", "rt output 1234\r",
           r"rt output 1234\n1234\[realm-thread 0x[0-9a-f]+\] \(gdb\) $", until=early)
    s.step("  not after the next command's output", "print 77\r", r"\$\d+ = 77\n",
           unwanted="1234")
    s.step("realm-thread apply PTR output: before the prompt",
           "realm-thread apply {} output 4321\r".format(ptr),
           r"\n4321\[realm-thread 0x[0-9a-f]+\] \(gdb\) $")
    s.finish()


@test("terminal.repeat")
def test_terminal_repeat(t):
    """realm-gdb.py with gdb on a terminal: an empty line after `realm-thread
    PTR` or `realm-thread back` doesn't select, or go back, again."""
    s = t.gdb(t.realm_args())
    ptr, _ = t.select_chain0(s, rt="rt ")
    s.step("realm-thread PTR, again", "realm-thread {}\r".format(ptr), r"selected")
    s.step("  Enter", "\r", None)
    shown = s.step("realm-thread: one selection", "realm-thread\r", r"\(selected\)")
    t.check("  and one earlier (from select_chain0's and this)",
            shown.count("selected earlier") <= 1, shown)
    s.step("realm-thread back", "realm-thread back\r", r"back to user thread")
    s.step("  Enter", "\r", None)
    s.step("  one still selected", "realm-thread\r", r"\(selected\)")
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
    call; `step` is refused, and `continue` ends the selection and runs."""
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
    proc = "shell cat /proc/{}/task/{}/syscall\r".format(pid, tid)
    own = _syscall_sp_pc(s.step("", proc))
    t.check("the thread's own SP and PC, from /proc", own is not None, "")
    s.step("realm-thread PTR", "realm-thread {}\r".format(ptr),
           r"commands that work with the selected frame[\s\S]*#0 .*user_switch")
    shown = s.step("bt: the parked stack", "bt\r", r"#0 .*user_switch[\s\S]*rgt_chain_level3")
    m = re.search(r"#(\d+) .* rgt_chain_level3 ", shown)
    level = m.group(1) if m else "0"
    s.step("frame N: prints the frame", "frame {}\r".format(level),
           r"#{} .*rgt_chain_level3".format(level))
    s.step("print a parked local", "print chain_marker\r", r"= 4242")
    s.step("up: prints the frame", "up\r", r"rgt_chain_level2")
    s.step("down: prints the frame", "down\r", r"rgt_chain_level3")
    s.step("a function call", "print realm_gdb_test_add(chain_marker, 1)\r", r"= 4243")
    t.check("/proc: the thread's own SP and PC between commands",
            _syscall_sp_pc(s.step("", proc)) == own, "")
    addr = re.compile(r"^(0x[0-9a-f]+)(?: <[^>]*>)?:", re.M)
    first = addr.findall(s.step("x/2gx $sp", "x/2gx $sp\r", r"0x"))
    again = addr.findall(s.step("  Enter: gdb's repeat, further on", "\r", r"0x"))
    t.check("  16 bytes further on", first and again and
            int(again[0], 16) == int(first[0], 16) + 16, "{} {}".format(first, again))
    s.step("a command that fails", "print rgt_nosuchvar\r", r"No symbol")
    t.check("/proc: the thread's own SP and PC at the prompt",
            _syscall_sp_pc(s.step("", proc)) == own, "")
    s.step("  still parked, in the same frame", "print chain_marker\r", r"= 4242")
    s.send("print realm_gdb_test_nap(3000000)\r")
    time.sleep(1)
    s.step("Ctrl-C in a function call", "\x03", r"[Ii]nterrupt|Quit|signaled", timeout=20)
    t.check("/proc: the thread's own SP and PC after it",
            _syscall_sp_pc(s.step("", proc)) == own, "")
    s.step("  still parked", "print chain_marker\r", r"= 4242")
    s.step("step: refused", "step\r",
           r"refusing 'step' while user thread {} is selected[\s\S]*realm-thread back".format(ptr))
    s.step("continue: ends the selection, and the program finishes", "continue\r",
           r"continuing; no longer selecting user thread {}[\s\S]*REALM_GDB_TEST: PASS"
           r"[\s\S]*exited normally".format(ptr), timeout=t.args.timeout)
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
        r"selected user thread[\s\S]*No symbol[\s\S]*#0 .*user_switch[\s\S]*No symbol"
        r"[\s\S]*realm-gdb.py loaded again", shown), shown)
    t.check("continue: ends the selection, and the program finishes", re.search(
        r"continuing; no longer selecting user thread[\s\S]*REALM_GDB_TEST: PASS[\s\S]*"
        r"exited normally", shown), shown)
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
               r"commands that work with the selected frame")
        s.step("bt: the parked thread", "bt 1\r", r"user_switch")
        s.step("release the program (rt: plain set var is refused)",
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
