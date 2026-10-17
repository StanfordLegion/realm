set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v gdb >/dev/null; then
  apt-get update -qq >/dev/null
  apt-get install -y -qq --no-install-recommends gcc libc6-dev gdb >/dev/null
fi
gcc -g -O0 -o /tmp/t /p/t.c
echo "PROBE $(gdb --version | head -n 1)"
gdb -nx -batch -x /p/probe.py /tmp/t 2>&1 | grep PROBE
