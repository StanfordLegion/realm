/* What the kernel's xstate regset expects, against what gdb <= 13 sends. */
#define _GNU_SOURCE
#include <cpuid.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

#define NT_X86_XSTATE 0x202
#define GDB13_SIZE 2696 /* X86_XSTATE_MAX_SIZE in gdb 8.2 to 13.2 */

static void setregset(pid_t pid, size_t len)
{
  static char buf[65536];
  struct iovec iov = {buf, len};
  long r = ptrace(PTRACE_GETREGSET, pid, (void *)NT_X86_XSTATE, &iov);
  if (r < 0) {
    printf("GETREGSET(%zu): %s\n", len, strerror(errno));
    return;
  }
  size_t got = iov.iov_len;
  r = ptrace(PTRACE_SETREGSET, pid, (void *)NT_X86_XSTATE, &iov);
  printf("get %zu bytes then set them back: %s\n", got, r < 0 ? strerror(errno) : "ok");
}

int main(void)
{
  unsigned a, b, c, d, lo, hi;
  __cpuid_count(7, 0, a, b, c, d);
  printf("cpuid 7: avx512f=%u amx-tile=%u amx-bf16=%u amx-int8=%u\n", (b >> 16) & 1,
         (d >> 24) & 1, (d >> 22) & 1, (d >> 25) & 1);
  __cpuid_count(0xd, 0, a, b, c, d);
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  printf("xcr0=%#llx  xsave size for xcr0 (cpuid 0xd ebx)=%u  max (ecx)=%u\n",
         ((unsigned long long)hi << 32) | lo, b, c);

  pid_t pid = fork();
  if (pid == 0) {
    ptrace(PTRACE_TRACEME, 0, 0, 0);
    raise(SIGSTOP);
    _exit(0);
  }
  int st;
  waitpid(pid, &st, 0);
  static char buf[65536];
  struct iovec iov = {buf, sizeof buf};
  if (ptrace(PTRACE_GETREGSET, pid, (void *)NT_X86_XSTATE, &iov) == 0)
    printf("kernel's xstate regset: %zu bytes\n", iov.iov_len);
  setregset(pid, GDB13_SIZE);
  setregset(pid, sizeof buf);
  kill(pid, SIGKILL);
  return 0;
}
