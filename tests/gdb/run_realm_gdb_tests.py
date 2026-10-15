#!/usr/bin/env python3
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

"""Driver for the realm-gdb.py tests.

Runs each test (defined in realm_gdb_tests.py) in a fresh gdb session against
realm_gdb_test, sets up the scenario (running, attaching, or loading a core
file), and checks what happened to the program afterwards.

  run_realm_gdb_tests.py --list
  run_realm_gdb_tests.py --gdb gdb --program ./realm_gdb_test [-j N] [TEST|GLOB ...]
  run_realm_gdb_tests.py --gdb gdb --program ./realm_gdb_test --make-core FILE
  run_realm_gdb_tests.py --gdb gdb --program ./realm_gdb_test --self-test

The core.* tests share one core file, written by gcore when first needed,
or beforehand with --make-core and passed in with --core (CTest does that,
as a fixture); likewise the kcore.* tests, with --make-kcore and --kcore.

Exit status: 0 if every selected test passed (or skipped), 1 if any failed
(or none ran), 77 if every selected test was skipped (CTest's
SKIP_RETURN_CODE).
"""

import argparse
import fnmatch
import json
import os
import pty
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tty

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True  # keep the source tree clean
import realm_gdb_tests  # noqa: E402

SKIP = 77
# (-ll:cpu is given per test: see program_args.)
PROGRAM_ARGS = ["-ll:util", "1", "-ll:csize", "16",
                "-ll:networks", "none"]

# Anything in gdb's output that means something went wrong, whatever the test
# itself concluded.
BAD_OUTPUT = [
    "Traceback (most recent call last)",
    "Python Exception",
    "Error occurred in Python",
    "internal-error",
    "A problem internal to GDB has been detected",
    "Fatal signal",
    "realm-gdb: could not define",
]


class Result(object):
    def __init__(self, name, status, detail="", log=""):
        self.name = name
        self.status = status  # PASS, FAIL, SKIP
        self.detail = detail
        self.log = log


def program_args(info=None):
    cpus = info.cpus if info is not None else 1
    return ["-ll:cpu", str(cpus)] + PROGRAM_ARGS


def clean_env():
    env = dict(os.environ)
    for var in ("REALM_DEFAULT_ARGS", "REALM_FREEZE_ON_ERROR", "LEGION_FREEZE_ON_ERROR",
                "REALM_BACKTRACE", "LEGION_BACKTRACE", "GDBHISTFILE"):
        env.pop(var, None)
    # gdb's Python must not pick up a user's site customisations.
    env["PYTHONNOUSERSITE"] = "1"
    env["TERM"] = "dumb"
    # Never go looking for debug info over the network.
    env["DEBUGINFOD_URLS"] = ""
    return env


def gdb_base_cmd(args):
    cmd = [args.gdb, "-nx", "-nh", "-q",
           "-iex", "set auto-load safe-path /",
           "-iex", "set pagination off",
           "-iex", "set height 0",
           "-iex", "set width 0",
           "-iex", "set confirm off",
           "-iex", "set startup-with-shell on"]
    if args.source_script:
        # Use the script from the source tree instead of the embedded copy.
        cmd += ["-iex", "set auto-load python-scripts off",
                "-iex", "source " + args.script]
    return cmd


def exit_handler_cmd(protocol, when):
    """A gdb command registering a Python exit handler that records that it
    ran in the protocol file (see check_gdb_output)."""
    return ('python __import__("atexit").register(lambda p={!r}: (lambda f: '
            '(f.write("RGT-EXIT-HANDLER {}\\n"), f.close()))(open(p, "a")))'.format(
                protocol, when))


def start_harness_cmds(info, config):
    return [
        "-ex", "python import sys; sys.dont_write_bytecode = True; "
               "sys.path.insert(0, {!r})".format(HERE),
        "-ex", "python import realm_gdb_tests as rgt",
        "-ex", "python rgt.start({!r}, {!r})".format(info.name, json.dumps(config)),
    ]


class Proc(object):
    """A subprocess whose combined output is collected in the background.
    With `ui`, the master side of a pseudo-terminal the process runs a
    second user interface on, input goes there, and its output is collected
    too (after the process's own)."""

    def __init__(self, cmd, env, cwd, stdin=subprocess.DEVNULL, preexec_fn=None,
                 ui=None):
        self.cmd = cmd
        self.p = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                  preexec_fn=preexec_fn, start_new_session=True)
        self.chunks = []
        self.ui = ui
        self.ui_chunks = []
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._read)
        self.reader.daemon = True
        self.reader.start()
        if ui is not None:
            self.ui_reader = threading.Thread(target=self._read_ui)
            self.ui_reader.daemon = True
            self.ui_reader.start()

    def _read(self):
        for line in iter(self.p.stdout.readline, b""):
            with self.lock:
                self.chunks.append(line.decode("utf-8", "replace"))

    def _read_ui(self):
        while True:
            try:
                data = os.read(self.ui, 65536)
            except OSError:  # (the other side has gone)
                break
            if not data:
                break
            with self.lock:
                self.ui_chunks.append(data.decode("utf-8", "replace"))

    def send(self, text):
        """Type `text` at the process: at its second UI if it has one."""
        try:
            if self.ui is not None:
                os.write(self.ui, text.encode())
            else:
                self.p.stdin.write(text.encode())
                self.p.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    @property
    def output(self):
        with self.lock:
            return "".join(self.chunks) + "".join(self.ui_chunks)

    def wait_for(self, pattern, timeout, flags=0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            m = re.search(pattern, self.output, flags)
            if m:
                return m
            if self.p.poll() is not None:
                self.reader.join(1)
                return re.search(pattern, self.output, flags)
            time.sleep(0.02)
        return None

    def wait(self, timeout):
        try:
            rc = self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            return None
        self.reader.join(5)
        return rc

    def alive(self):
        return self.p.poll() is None

    def kill(self):
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGKILL)
            except OSError:
                pass
            self.p.wait()
        self.reader.join(5)


def run_gdb(args, info, config, target_args, tmp):
    """Run one gdb session for `info` and return (output, protocol,
    timed_out, gdb's exit status).  The harness reports to the protocol file
    (see realm_gdb_tests._out), which works the same whichever interpreter
    gdb's stdout belongs to."""
    cmd = gdb_base_cmd(args)
    if info.init:
        # Before anything else, as a user's ~/.gdbinit would be.
        init = os.path.join(tmp, "init.gdb")
        with open(init, "w") as f:
            f.write(info.init)
        cmd[1:1] = ["-ix", init]
    # First of all, as if by an extension loaded by ~/.gdbinit: a Python exit
    # handler registered before realm-gdb.py is loaded (the harness registers
    # one after), which must run as gdb exits.
    cmd[1:1] = ["-iex", exit_handler_cmd(config["protocol"], "before")] + [
        a for c in getattr(args, "extra_init", ()) for a in ("-iex", c)]
    mode = info.interaction
    if mode == "batch":
        cmd[1:1] = ["-batch"]
    elif mode == "mi":
        cmd[1:1] = ["--interpreter=mi2"]
    cmd += start_harness_cmds(info, config)
    if mode == "batch":
        cmd += ["-ex", "python rgt.run_all()"]
    ui = ui_tty = None
    if mode == "mi_new_ui":
        # A console gdb driven through a second UI that runs MI on a
        # terminal of its own, as some front ends do, with the console idle.
        ui, ui_tty = pty.openpty()
        tty.setraw(ui_tty)
        cmd += ["-ex", "new-ui mi2 " + os.ttyname(ui_tty)]
    cmd += target_args
    gdb_proc = Proc(cmd, clean_env(), tmp,
                    stdin=subprocess.DEVNULL if mode == "batch" else subprocess.PIPE,
                    ui=ui)
    protocol = Protocol(config["protocol"])
    timed_out = False
    if mode != "batch":
        timed_out = not drive(gdb_proc, protocol, args.timeout,
                              mi=mode in ("mi", "mi_new_ui"))
    # Once the test is over gdb should exit promptly (on EOF, in batch mode,
    # or because the test quit).  It may not after some failures (e.g. a
    # failed `run` leaves gdb 9 ignoring EOF), so don't wait forever.
    rc = gdb_proc.wait(args.timeout if mode == "batch" else 20)
    gdb_proc.kill()
    if ui is not None:
        os.close(ui_tty)
        gdb_proc.ui_reader.join(5)
        os.close(ui)
    out = gdb_proc.output
    if timed_out:
        out += "\nDRIVER: a step took longer than {}s\n".format(args.timeout)
    elif rc is None:
        out += "\nDRIVER: gdb did not exit after the test\n"
        timed_out = True
    return out, protocol.text(), timed_out, rc


class Protocol(object):
    """The harness's report file."""

    def __init__(self, path):
        self.path = path

    def text(self):
        try:
            with open(self.path) as f:
                return f.read()
        except OSError:
            return ""

    def wait_for(self, pattern, timeout, gdb_proc):
        deadline = time.time() + timeout
        while time.time() < deadline:
            m = re.search(pattern, self.text(), re.M)
            if m or not gdb_proc.alive():
                return m or re.search(pattern, self.text(), re.M)
            time.sleep(0.02)
        return None


def drive(gdb_proc, protocol, timeout, mi=False):
    """Feed an interactive (or MI) session one step at a time (see
    _Runner.step in realm_gdb_tests.py), typing any command a step asks for.
    Returns False if a step timed out."""
    send = gdb_proc.send

    step = ('-interpreter-exec console "python rgt.step()"\n' if mi
            else "python rgt.step()\n")
    n = 0
    ok = send(step)
    while ok:
        n += 1
        m = protocol.wait_for(r"^RGT: (STEP-END|DONE) {}(?: TYPE (.*))?$".format(n),
                              timeout, gdb_proc)
        if m is None:
            if gdb_proc.alive():
                gdb_proc.kill()
                return False
            break  # gdb exited, e.g. after a typed `quit`
        if m.group(1) == "DONE":
            break
        cmd = m.group(2)
        if cmd is not None and mi and cmd.startswith("-"):
            # Tag the MI command, and record its result for the test to check
            # (Context.mi_result).
            token = 1000 + n
            ok = send("{}{}\n".format(token, cmd))
            r = gdb_proc.wait_for(r"^{}\^(\w+)(.*)$".format(token), timeout, flags=re.M)
            with open(protocol.path, "a") as f:
                f.write("RGT-MI-RESULT {}\n".format(
                    "{} {}".format(r.group(1), r.group(2)) if r else "none"))
        elif cmd is not None:
            ok = send(cmd + "\n")
        ok = ok and send(step)
    # EOF, which gdb treats as `quit`.
    try:
        gdb_proc.p.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    return True


def ptrace_blocked(out):
    return ("ptrace: Operation not permitted" in out
            or "Could not attach to process" in out)


def exit_status_problem(rc):
    """What is wrong with a process's exit status (from subprocess), if
    anything."""
    if rc is None or rc == 0:
        return None
    if rc < 0:
        try:
            return "killed by {}".format(signal.Signals(-rc).name)
        except ValueError:
            return "killed by signal {}".format(-rc)
    return "exit status {}".format(rc)


def check_gdb_output(name, out, protocol, rc=None):
    """Return (status, detail) from the harness's report, gdb's output as a
    whole, and gdb's exit status: gdb must exit cleanly, whatever the test
    concluded (crashing as it exits may leave no message to find)."""
    fails = re.findall(r"^RGT: FAIL (.*)$", protocol, re.M)
    if fails:
        return "FAIL", fails[0]
    for bad in BAD_OUTPUT:
        if bad in out:
            return "FAIL", "gdb output contains {!r}".format(bad)
    problem = exit_status_problem(rc)
    if problem:
        return "FAIL", "gdb failed: " + problem
    m = re.search(r"^RGT: SKIP (.*)$", protocol, re.M)
    if m:
        return "SKIP", m.group(1)
    if not re.search(r"^RGT: PASS {}$".format(re.escape(name)), protocol, re.M):
        return "FAIL", "test did not finish"
    # Other extensions' Python exit handlers, registered before or after
    # realm-gdb.py was loaded, must have run, once each.
    for when in ("before", "after"):
        n = len(re.findall(r"^RGT-EXIT-HANDLER {}$".format(when), protocol, re.M))
        if n != 1:
            return "FAIL", ("a Python exit handler registered {} realm-gdb.py was "
                            "loaded ran {} times as gdb exited".format(when, n))
    return "PASS", ""


def make_gcore(args, core):
    """Write a core file of the program stopped in realm_gdb_test_stop.
    Returns (path, status, detail): path is None on failure."""
    tmp = os.path.dirname(core)
    cmd = [args.gdb, "-nx", "-nh", "-q", "-batch",
           "-iex", "set auto-load python-scripts off",
           "-ex", "break realm_gdb_test_stop",
           # (`run ARGS` replaces the arguments, so give them again.)
           "-ex", "run {} < /dev/null".format(" ".join(program_args() + ["-mode", "stop"])),
           "-ex", "gcore " + core,
           "-ex", "kill",
           "--args", args.program] + program_args() + ["-mode", "stop"]
    p = Proc(cmd, clean_env(), tmp)
    rc = p.wait(args.timeout)
    p.kill()
    if not os.path.exists(core) or rc != 0:
        if ptrace_blocked(p.output):
            return None, "SKIP", "ptrace is not permitted here"
        return None, "FAIL", "gcore failed ({}):\n{}".format(
            exit_status_problem(rc) or "no core file", p.output)
    return core, None, None


class CoreFiles(object):
    """Core files shared by every test of a run, made when first needed."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.made = {}
        self.dir = None

    def get(self, kind):
        with self.lock:
            given = self.args.core if kind == "core" else self.args.kcore
            if given:
                if not os.path.exists(given):
                    return None, "SKIP", "no core file {} (did making it fail or " \
                        "get skipped?)".format(given)
                return given, None, None
            if kind not in self.made:
                if self.dir is None:
                    self.dir = tempfile.mkdtemp(prefix="rgt-cores-")
                if kind == "core":
                    self.made[kind] = make_gcore(
                        self.args, os.path.join(self.dir, "realm_gdb_test.core"))
                else:
                    self.made[kind] = make_kernel_core(self.args, self.dir)
            return self.made[kind]

    def cleanup(self):
        if self.dir is not None and not self.args.keep:
            shutil.rmtree(self.dir, ignore_errors=True)


def kernel_core_pattern():
    try:
        with open("/proc/sys/kernel/core_pattern") as f:
            return f.read().strip()
    except OSError:
        return None


def make_kernel_core(args, tmp):
    pattern = kernel_core_pattern()
    if pattern is None or pattern.startswith("|") or "/" in pattern:
        return None, "SKIP", "kernel core_pattern {!r} doesn't write core files to " \
            "the working directory".format(pattern)

    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    if hard != resource.RLIM_INFINITY and hard < 512 << 20:
        return None, "SKIP", "the hard core file size limit ({} bytes) is too " \
            "small".format(hard)

    def unlimited_core():
        resource.setrlimit(resource.RLIMIT_CORE, (hard, hard))

    cwd = os.path.join(tmp, "kcore")
    os.mkdir(cwd)
    p = Proc([args.program] + program_args() + ["-mode", "crash"], clean_env(), cwd,
             preexec_fn=unlimited_core)
    p.wait(args.timeout)
    p.kill()
    cores = [f for f in os.listdir(cwd) if f.startswith("core")]
    if not cores:
        return None, "SKIP", "the kernel didn't write a core file " \
            "(core_pattern {!r}, hard RLIMIT_CORE?)".format(pattern)
    return os.path.join(cwd, cores[0]), None, None


def no_core():
    # (A test may leave the program to crash after gdb has let go of it.)
    resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))


def start_program(args, mode, tmp, extra_env=None):
    env = clean_env()
    if extra_env:
        env.update(extra_env)
    return Proc([args.program] + program_args() + ["-mode", mode, "-timeout",
                                                 str(args.timeout)], env, tmp,
                preexec_fn=no_core)


def run_test(args, info, cores):
    tmp = tempfile.mkdtemp(prefix="rgt-")
    config = {"script": args.script, "embedded": not args.source_script,
              "program": args.program, "tmp": tmp,
              "program_args": program_args(info) + ["-mode", "stop"],
              "protocol": os.path.join(tmp, "protocol.txt")}
    program = None
    log = []
    try:
        s = info.scenario
        if s in ("live", "noprocess"):
            target = ["--args", args.program] + program_args(info) + ["-mode", "stop"]
        elif s in ("core", "kcore"):
            core, status, detail = cores.get(s)
            if core is None:
                return Result(info.name, status, detail)
            target = [args.program, core]
        elif s in ("attach", "freeze"):
            if s == "attach":
                program = start_program(args, "attach", tmp)
                ready = r"REALM_GDB_TEST: READY (\d+)"
            else:
                program = start_program(args, "freeze", tmp,
                                        {"REALM_FREEZE_ON_ERROR": "1"})
                ready = r"Process (\d+) on node .* is frozen"
            m = program.wait_for(ready, 60)
            if m is None:
                return Result(info.name, "FAIL", "program never got ready",
                              program.output)
            target = ["-p", m.group(1), args.program]
        else:
            return Result(info.name, "FAIL", "unknown scenario " + s)

        out, protocol, timed_out, rc = run_gdb(args, info, config, target, tmp)
        log.append(out)
        if ptrace_blocked(out) and "RGT: PASS" not in protocol:
            return Result(info.name, "SKIP", "ptrace is not permitted here", out)
        status, detail = check_gdb_output(info.name, out, protocol, rc)
        if timed_out and status != "FAIL":
            status, detail = "FAIL", "gdb hung (see the end of the log)"

        # What happened to the program?  (Its output may follow a gdb
        # prompt on the same line.)
        if info.ending == "exits" and status == "PASS":
            if "REALM_GDB_TEST: FAIL" in out:
                status, detail = "FAIL", "the program reported a failure: " + \
                    re.search(r"REALM_GDB_TEST: FAIL.*", out).group(0)
            elif "REALM_GDB_TEST: PASS" not in out:
                status, detail = "FAIL", "the program did not report PASS"
        elif info.ending == "release" and program is not None and status != "SKIP":
            rc = program.wait(60)
            log.append("--- program output ---\n" + program.output)
            if rc is None:
                if status == "PASS":
                    status, detail = "FAIL", "the program did not finish after gdb left"
            elif rc != 0 or "REALM_GDB_TEST: PASS" not in program.output:
                if status == "PASS":
                    how = ("killed by signal {}".format(-rc) if rc < 0
                           else "exit code {}".format(rc))
                    status, detail = "FAIL", "the program failed after gdb left ({})".format(how)
        elif info.ending == "alive" and program is not None:
            time.sleep(1)
            log.append("--- program output ---\n" + program.output)
            if not program.alive() and status == "PASS":
                status, detail = "FAIL", "the frozen program died after gdb left " \
                    "(exit status {})".format(program.p.returncode)
        log = re.sub(r"(\(gdb\) )+", "(gdb) ", "\n".join(log))
        return Result(info.name, status, detail, log)
    finally:
        if program is not None:
            program.kill()
        if args.keep:
            sys.stderr.write("kept {}\n".format(tmp))
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def environment_problem(args):
    """Return a reason to skip everything, or None."""
    if not args.gdb or not shutil.which(args.gdb):
        return "gdb not found"
    try:
        out = subprocess.run([args.gdb, "-nx", "-nh", "-batch", "-ex",
                              "python print('RGT-PYTHON-OK')"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=clean_env(), timeout=60).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return "cannot run gdb: {}".format(e)
    if "RGT-PYTHON-OK" not in out:
        return "gdb has no Python support"
    if os.uname().machine not in ("x86_64", "amd64", "aarch64", "arm64"):
        return "realm-gdb.py does not support " + os.uname().machine
    return None


def self_test(args):
    """Check that a test fails if gdb exits badly after the harness has
    reported PASS: run one test with gdb, then with gdb wrapped to exit with
    status 42 or die of SIGSEGV afterwards, and with a Python exit handler
    that ends gdb before the one registered first can run.  Returns the
    exit status."""
    gdb = shutil.which(args.gdb)
    tmp = tempfile.mkdtemp(prefix="rgt-selftest-")
    failed = 0
    skips_handler = 'python __import__("atexit").register(lambda: __import__("os")._exit(0))'
    try:
        for label, then, init, want in (
                ("plain", None, None, "PASS"),
                ("exit 42", "exit 42", None, "exit status 42"),
                ("SIGSEGV", "kill -SEGV $$", None, "killed by SIGSEGV"),
                ("skipping exit handlers", None, skips_handler,
                 "registered before realm-gdb.py was loaded ran 0 times")):
            run_args = argparse.Namespace(**vars(args))
            if then is not None:
                run_args.gdb = os.path.join(tmp, "gdb-" + label.replace(" ", "-"))
                with open(run_args.gdb, "w") as f:
                    f.write('#!/bin/sh\n"{}" "$@"\n{}\n'.format(gdb, then))
                os.chmod(run_args.gdb, 0o755)
            if init is not None:
                run_args.extra_init = [init]
            r = run_test(run_args, realm_gdb_tests.TESTS["live.loaded"], None)
            ok = r.status == want if then is None and init is None else (
                r.status == "FAIL" and want in r.detail)
            print("{:4}  self-test: gdb then {:22} -> {} {}".format(
                "PASS" if ok else "FAIL", label, r.status, r.detail))
            if not ok:
                failed += 1
                print(r.log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


class _Dap(object):
    """A DAP client talking to gdb (-i=dap)."""

    def __init__(self, cmd, timeout):
        self.timeout = timeout
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=clean_env())
        self.messages = []  # everything gdb has sent
        self.queue = []
        self.cond = threading.Condition()
        self.seq = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        f = self.p.stdout
        while True:
            line = f.readline()
            if not line:
                break
            if line.lower().startswith(b"content-length:"):
                n = int(line.split(b":")[1])
                while f.readline().strip():
                    pass
                msg = json.loads(f.read(n).decode("utf-8", "replace"))
                with self.cond:
                    self.messages.append(msg)
                    self.queue.append(msg)
                    self.cond.notify_all()
        with self.cond:
            self.queue.append(None)
            self.cond.notify_all()

    def send(self, command, **arguments):
        self.seq += 1
        body = json.dumps({"seq": self.seq, "type": "request", "command": command,
                           "arguments": arguments}).encode()
        self.p.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.p.stdin.flush()
        return self.seq

    def wait(self, pred, what):
        deadline = time.time() + self.timeout
        with self.cond:
            while True:
                for i, msg in enumerate(self.queue):
                    if msg is None:
                        raise RuntimeError("gdb exited waiting for " + what)
                    if pred(msg):
                        del self.queue[:i + 1]
                        return msg
                del self.queue[:]
                left = deadline - time.time()
                if left <= 0:
                    raise RuntimeError("timed out waiting for " + what)
                self.cond.wait(left)

    def request(self, command, **arguments):
        seq = self.send(command, **arguments)
        r = self.wait(lambda m: m.get("type") == "response" and m.get("request_seq") == seq,
                      command)
        if not r.get("success"):
            raise RuntimeError("{} {} failed: {}".format(command, arguments, r.get("message")))
        return r.get("body") or {}

    def repl(self, expression, frame=None):
        kw = {"expression": expression, "context": "repl"}
        if frame is not None:
            kw["frameId"] = frame
        return self.request("evaluate", **kw).get("result", "")

    def stack(self, thread):
        return self.request("stackTrace", threadId=thread, startFrame=0,
                            levels=60)["stackFrames"]

    def close(self):
        try:
            self.send("disconnect", terminateDebuggee=True)
            self.p.wait(10)
        except Exception:
            pass
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait()


def dap_test(args):
    """Drive gdb's DAP (gdb >= 14) as an IDE would, from its debug console:
    `realm-thread apply` shows a parked user thread's stack (from frame #0)
    and makes function calls on it, and `thread apply all bt` shows them
    all; afterwards the stopped thread's stack trace and variables are its
    own, as before, and the settings realm-gdb changes meanwhile are the
    user's again.  Returns the exit status."""
    probe = subprocess.run([args.gdb, "-nx", "-nh", "-batch", "-ex", "python import gdb.dap"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=clean_env(), timeout=60).stdout.decode("utf-8", "replace")
    if "Error" in probe or "Traceback" in probe:
        print("SKIP: this gdb has no DAP")
        return SKIP
    dap = _Dap(gdb_base_cmd(args) + ["-i=dap"], args.timeout)
    failures = []

    def check(ok, what):
        print("{:4}  dap: {}".format("PASS" if ok else "FAIL", what))
        if not ok:
            failures.append(what)

    def variables(frame):
        names = []
        for scope in dap.request("scopes", frameId=frame["id"])["scopes"]:
            if scope.get("presentationHint") == "registers":
                continue
            names += [v["name"] for v in dap.request(
                "variables", variablesReference=scope["variablesReference"])["variables"]]
        return names

    try:
        dap.request("initialize", adapterID="realm-gdb-tests", supportsInvalidatedEvent=True)
        dap.wait(lambda m: m.get("event") == "initialized", "the initialized event")
        dap.request("setFunctionBreakpoints", breakpoints=[{"name": "realm_gdb_test_stop"}])
        # (gdb 15 runs the program when it gets launch, later ones once
        # configuration is done, and answer accordingly.)
        dap.send("launch", program=args.program, args=program_args() + ["-mode", "stop"])
        dap.send("configurationDone")
        thread = dap.wait(lambda m: m.get("event") == "stopped", "a stop")["body"]["threadId"]
        own = dap.stack(thread)
        probe_task = [f for f in own if f["name"] == "probe_task"]
        own_vars = variables(probe_task[0]) if probe_task else []
        check("probe_marker" in own_vars, "probe_task's variables: {}".format(own_vars))
        unwind = dap.repl("show unwindonsignal")
        parked = re.findall(r"(0x[0-9a-f]+)\s+parked", dap.repl("realm-thread"))
        check(parked, "realm-thread lists parked user threads")
        chain = None
        out = ""
        for ptr in parked:
            out = dap.repl("realm-thread apply {} bt".format(ptr))
            if "rgt_chain_level2" in out:
                chain = ptr
                break
        check(chain is not None, "a chain task's stack:\n{}".format(out))
        check(re.match(r"#0\s+.*Realm::UserThread::user_switch", out),
              "its frame #0 is the parked thread's own")
        if chain is not None:
            check("= 5" in dap.repl("realm-thread apply {} print realm_gdb_test_add(2, 3)"
                                    .format(chain)), "a function call on a parked thread")
        check("Parked Realm user threads" in dap.repl("thread apply all bt"),
              "thread apply all bt")
        check(dap.repl("show unwindonsignal") == unwind, "unwindonsignal is the user's")
        after = dap.stack(thread)
        check([f["name"] for f in after] == [f["name"] for f in own],
              "the thread's own stack afterwards: {}".format([f["name"] for f in after]))
        probe_task = [f for f in after if f["name"] == "probe_task"]
        check(probe_task and variables(probe_task[0]) == own_vars,
              "probe_task's variables afterwards")
        check("= 9" in dap.repl("print realm_gdb_test_add(4, 5)"),
              "a function call afterwards")
    except Exception as e:
        failures.append("DAP session: {}".format(e))
        print("FAIL  dap: {}".format(e))
    finally:
        dap.close()
    output = "".join(m.get("body", {}).get("output", "") for m in dap.messages
                     if m.get("event") == "output")
    bad = [b for b in BAD_OUTPUT if b in output]
    check(not bad, "gdb's output has no {}".format(bad or "errors"))
    if failures:
        print(output)
    return 1 if failures else 0


def select_tests(patterns):
    names = sorted(realm_gdb_tests.TESTS)
    if not patterns:
        return names
    out = []
    for pat in patterns:
        matched = [n for n in names if fnmatch.fnmatchcase(n, pat)]
        if not matched:
            sys.exit("no test matches {!r}".format(pat))
        out += [n for n in matched if n not in out]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*", help="test names or glob patterns")
    ap.add_argument("--list", action="store_true", help="list tests and exit")
    ap.add_argument("--gdb", default="gdb")
    ap.add_argument("--program", help="path to realm_gdb_test")
    ap.add_argument("--script", default=os.path.join(HERE, "..", "..", "src", "realm",
                                                     "realm-gdb.py"),
                    help="realm-gdb.py, for tests that source it explicitly")
    ap.add_argument("--source-script", action="store_true",
                    help="source --script instead of using the copy embedded in "
                         "librealm (to test edits without rebuilding; this turns "
                         "off auto-loading Python scripts, so libstdc++'s "
                         "pretty-printers are left out too)")
    ap.add_argument("--timeout", type=int, default=180,
                    help="seconds allowed for each gdb session")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the gdb log of every test, not just failures")
    ap.add_argument("--keep", action="store_true", help="keep temporary directories")
    ap.add_argument("--core", help="core file for the core.* tests (see --make-core)")
    ap.add_argument("--make-core", metavar="FILE",
                    help="write the core file the core.* tests use, and exit")
    ap.add_argument("--kcore", help="kernel-written core file for the kcore.* tests "
                                    "(see --make-kcore)")
    ap.add_argument("--make-kcore", metavar="FILE",
                    help="have the kernel write the core file the kcore.* tests "
                         "use, and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="check that this driver fails a test whose gdb exits "
                         "badly, and exit")
    ap.add_argument("--dap-test", action="store_true",
                    help="drive gdb's DAP (gdb >= 14), showing parked user "
                         "threads from its console, and exit")
    args = ap.parse_args()

    if args.list:
        for name in sorted(realm_gdb_tests.TESTS):
            print(name)
        return 0
    if not args.program:
        ap.error("--program is required")
    args.program = os.path.abspath(args.program)
    args.script = os.path.abspath(args.script)
    if args.core:
        args.core = os.path.abspath(args.core)
    if args.kcore:
        args.kcore = os.path.abspath(args.kcore)

    problem = environment_problem(args)
    if problem:
        print("SKIP: " + problem)
        return SKIP

    if args.self_test:
        return self_test(args)
    if args.dap_test:
        return dap_test(args)

    if args.make_core or args.make_kcore:
        path = os.path.abspath(args.make_core or args.make_kcore)
        if os.path.exists(path):
            os.unlink(path)
        if args.make_core:
            core, status, detail = make_gcore(args, path)
        else:
            tmp = tempfile.mkdtemp(prefix="rgt-kcore-")
            try:
                core, status, detail = make_kernel_core(args, tmp)
                if core:
                    shutil.move(core, path)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        print("{}: {}".format(path, detail or "written"))
        return 0 if core else (SKIP if status == "SKIP" else 1)

    names = select_tests(args.tests)
    cores = CoreFiles(args)
    results = []
    lock = threading.Lock()
    queue = list(names)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                name = queue.pop(0)
            t0 = time.time()
            try:
                r = run_test(args, realm_gdb_tests.TESTS[name], cores)
            except Exception:
                import traceback
                r = Result(name, "FAIL", "driver exception: " + traceback.format_exc())
            with lock:
                results.append(r)
                print("{:4}  {:60} {:5.1f}s  {}".format(
                    r.status, name, time.time() - t0, r.detail.splitlines()[0]
                    if r.detail else ""))
                if r.log and (args.verbose or r.status == "FAIL"):
                    print("----- log of {} -----".format(name))
                    print(r.log.rstrip())
                    if r.status == "FAIL":
                        print("----- failure: {}".format(r.detail))
                    print("----- end of {} -----".format(name))
                sys.stdout.flush()

    threads = [threading.Thread(target=worker) for _ in range(max(1, args.jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cores.cleanup()

    counts = {s: sum(1 for r in results if r.status == s) for s in ("PASS", "FAIL", "SKIP")}
    print("{} passed, {} failed, {} skipped".format(counts["PASS"], counts["FAIL"],
                                                   counts["SKIP"]))
    if counts["FAIL"]:
        for r in sorted(results, key=lambda r: r.name):
            if r.status == "FAIL":
                print("  FAILED: {}  ({})".format(r.name, r.detail.splitlines()[0]
                                                  if r.detail else ""))
        return 1
    if len(results) != len(names):
        print("only {} of {} tests ran".format(len(results), len(names)))
        return 1
    if counts["SKIP"] == len(results):
        return SKIP
    return 0


if __name__ == "__main__":
    sys.exit(main())
