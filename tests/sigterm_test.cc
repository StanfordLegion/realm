#include "realm.h"
#include "realm/cmdline.h"
#include "realm/logging.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#include <fstream>
#include <string>

// Synchronization signal
static volatile sig_atomic_t sigusr1_received = 0;

void sigusr1_handler(int signum)
{
    sigusr1_received = 1;
}

using namespace Realm;

Logger log_app("sigterm_test");

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  LONG_RUNNING_TASK,
};

namespace TestConfig {
  std::string log_file = "sigterm_test.log";
  int test_duration_ms = 1000;
};

void long_running_task(const void *args, size_t arglen,
                      const void *userdata, size_t userlen, Processor p)
{
  log_app.info() << "Long running task started on processor " << p;

  // Log some messages to ensure there's buffered content
  for(int i = 0; i < 50; i++) {
    log_app.info() << "Task iteration " << i << " - generating log content for flush test";
    usleep(TestConfig::test_duration_ms * 1000 / 50); // Sleep for portion of test duration
  }

  log_app.info() << "Long running task completed normally";
}

void top_level_task(const void *args, size_t arglen,
                   const void *userdata, size_t userlen, Processor p)
{
  log_app.info() << "SIGTERM test starting";

  // Launch a long-running task that will be interrupted by SIGTERM
  Event task_event = p.spawn(LONG_RUNNING_TASK, 0, 0);

  log_app.info() << "Long running task launched, waiting for completion or signal";

  // Wait for the task (which should be interrupted by SIGTERM)
  task_event.wait();

  log_app.info() << "Top level task completed";
}

int test_sigterm_handling()
{
  printf("Starting SIGTERM handling test\n");

  pid_t child_pid = fork();

  if (child_pid == 0) {
    // Child process - run the Realm application

    // Redirect logs to a file for analysis
    freopen(TestConfig::log_file.c_str(), "w", stderr);

    Runtime rt;
    rt.init(0, nullptr);

    rt.register_task(TOP_LEVEL_TASK, top_level_task);
    rt.register_task(LONG_RUNNING_TASK, long_running_task);

    // Signal parent that we are ready for SIGTERM
    kill(getppid(), SIGUSR1);

    // Get a processor to run on
    Processor p = Processor::NO_PROC;
    std::set<Processor> all_procs;
    Machine::get_machine().get_all_processors(all_procs);
    for(std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++) {
      if(it->kind() == Processor::LOC_PROC) {
        p = *it;
        break;
      }
    }
    assert(p.exists());

    // Start the task and wait for shutdown
    Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
    rt.shutdown(e);
    rt.wait_for_shutdown();

    exit(0);

  } else if (child_pid > 0) {
    // Parent process - send SIGTERM to child after a short delay

    printf("Parent: Child process %d started\n", child_pid);

    // Wait for the child to signal it's ready
    struct sigaction sa;
    sa.sa_handler = sigusr1_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGUSR1, &sa, NULL);

    printf("Parent: Waiting for child to be ready...\n");
    while (!sigusr1_received) {
      sleep(1);
    }
    printf("Parent: Child is ready.\n");

    // Give the child time to start up and begin logging
    usleep(TestConfig::test_duration_ms * 1000 / 2);

    printf("Parent: Sending SIGTERM to child process %d\n", child_pid);

    // Send SIGTERM to the child process
    if (kill(child_pid, SIGTERM) != 0) {
      printf("Failed to send SIGTERM to child process\n");
      return 1;
    }

    // Wait for child to terminate
    int status;
    pid_t result = waitpid(child_pid, &status, 0);

    if (result == -1) {
      printf("Failed to wait for child process\n");
      return 1;
    }

    printf("Parent: Child process terminated with status %d\n", status);

    // Check if child was terminated by SIGTERM
    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM) {
      printf("Child process correctly terminated by SIGTERM\n");
    } else {
      printf("Child process did not terminate by SIGTERM as expected (status=%d)\n", status);
      return 1;
    }

    // Analyze log file to verify SIGTERM handler was called
    std::ifstream logfile(TestConfig::log_file);
    if (!logfile.is_open()) {
      printf("Failed to open log file for analysis\n");
      return 1;
    }

    std::string line;
    bool found_sigterm_message = false;
    bool found_flush_message = false;

    while (std::getline(logfile, line)) {
      if (line.find("Realm caught SIGTERM") != std::string::npos) {
        found_sigterm_message = true;
        printf("Found SIGTERM catch message in logs\n");
      }
      if (line.find("flushing logs") != std::string::npos ||
          line.find("logs flushed") != std::string::npos) {
        found_flush_message = true;
        printf("Found log flush message in logs\n");
      }
    }

    logfile.close();

    if (!found_sigterm_message) {
      printf("SIGTERM catch message not found in logs\n");
      return 1;
    }

    if (!found_flush_message) {
      printf("Log flush message not found in logs\n");
      return 1;
    }

    printf("SIGTERM handling test PASSED\n");

    // Clean up log file only on success
    remove(TestConfig::log_file.c_str());
    return 0;

  } else {
    // Fork failed
    printf("Failed to fork child process\n");
    return 1;
  }
}

int main(int argc, char **argv)
{
  // Simple command line parsing without Realm runtime
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-logfile") && i+1 < argc) {
      TestConfig::log_file = argv[++i];
      continue;
    }
    if(!strcmp(argv[i], "-duration") && i+1 < argc) {
      TestConfig::test_duration_ms = atoi(argv[++i]);
      continue;
    }
  }

  printf("SIGTERM test configuration:\n");
  printf("  log_file = %s\n", TestConfig::log_file.c_str());
  printf("  test_duration_ms = %d\n", TestConfig::test_duration_ms);

  int result = test_sigterm_handling();

  return result;
}