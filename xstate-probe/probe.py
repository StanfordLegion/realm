# Plain gdb, no realm-gdb: write a general register, an SSE register (every
# x86-64 has xmm0), and make a function call.
import gdb
gdb.execute("break main")
gdb.execute("run")
for c in ["set var $rax = 1",
          "set var $xmm0.v2_int64[0] = 0x1122334455667788",
          "print/x $xmm0.v2_int64[0]",
          "print add(2, 3)"]:
    try:
        out = gdb.execute(c, to_string=True).strip()
        print("PROBE ok     %-50s %s" % (c, out))
    except gdb.error as e:
        print("PROBE error  %-50s %s" % (c, e))
gdb.execute("kill")
