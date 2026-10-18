#!/bin/sh
# a64diag.sh PROGRAM: why does `continue` hang after a window call that
# fails with "Command aborted" (may-write-memory off) on gdb 15.1 aarch64?
P=$(readlink -f "$1")
A="-ll:cpu 1 -ll:util 1 -ll:csize 16 -ll:networks none -mode stop"
D=$(mktemp -d); cd "$D"
gdb --version | head -n 1
UT="python import re; ut = re.findall(r'(0x[0-9a-f]+)\s+parked', gdb.execute('realm-thread', to_string=True))[0]"
run_case() {
  name=$1; shift
  echo; echo "===================== $name"
  gdb -nx -nh -q -batch -iex "set auto-load safe-path /" -iex "set confirm off" \
      -ex "break realm_gdb_test_stop" -ex run "$@" -ex "info threads" \
      -ex "maint info breakpoints" -ex "delete" -ex "set debug infrun on" -ex continue \
      --args "$P" $A > $name.log 2>&1 &
  g=$!
  i=0
  while kill -0 $g 2>/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i+1)); done
  if kill -0 $g 2>/dev/null; then
    echo "##### HUNG after 60s"
    prog=$(pgrep -P $g | head -n 1)
    echo "--- program $prog threads:"
    ps -L -o pid,lwp,stat,wchan:32,comm -p "$prog"
    for t in /proc/$prog/task/*; do
      echo "$(basename $t): state $(cut -d' ' -f3 $t/stat) syscall $(cat $t/syscall 2>/dev/null | cut -d' ' -f1-3)"
    done
    echo "--- gdb's own threads:"
    gdb -nx -batch -p $g -ex "thread apply all bt 30" 2>&1 | grep -v '^\[New\|^Reading\|^warning\|^Download' | head -n 120
    kill -9 $g 2>/dev/null; [ -n "$prog" ] && kill -9 $prog 2>/dev/null
    sleep 1
  else
    wait $g; echo "##### gdb finished, rc $?"
  fi
  echo "--- log (without the banner), head:"
  grep -v '^  \|^=====\|^Realm\|^In a live\|^registers\|^putting\|^program\|^refused\|^breakpoints\|^were killed\|^corrupted\|^while the\|^Linux' $name.log | grep -v '^\[infrun\]\|^  \[infrun\]' | head -n 80
  echo "--- log, infrun tail:"
  grep 'infrun' $name.log | tail -n 60
}
run_case A_plain -iex "set auto-load python-scripts off" \
  -ex "with may-write-memory off -- print realm_gdb_test_add(1, 2)" -ex "bt 4"
run_case A2_plain_return -iex "set auto-load python-scripts off" \
  -ex "with may-write-memory off -- print realm_gdb_test_add(1, 2)" -ex "bt 4" -ex "return" -ex "bt 4"
run_case B_realm_one -ex "$UT" \
  -ex "python gdb.execute('realm-thread apply %s with may-write-memory off -- print realm_gdb_test_add(1, 2)' % ut)" -ex "bt 4"
run_case C_realm_all \
  -ex "realm-thread apply all with may-write-memory off -- print realm_gdb_test_add(1, 2)" -ex "bt 4"
run_case D_realm_ok -ex "$UT" \
  -ex "python gdb.execute('realm-thread apply %s print realm_gdb_test_add(1, 2)' % ut)" -ex "bt 4"
true
