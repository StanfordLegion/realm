#include "realm.h"

enum {
  FID_INPUT = 100,
  FID_OUTPUT = 101,
};

struct StencilArgs {
  Realm::RegionInstance buffer = Realm::RegionInstance::NO_INST;
  Realm::IndexSpace<2> local_space = Realm::IndexSpace<2>();
  int64_t hx = -1, hy = -1;
};

struct IncrementArgs {
  Realm::RegionInstance buffer = Realm::RegionInstance::NO_INST;
  Realm::IndexSpace<2> local_space = Realm::IndexSpace<2>();
};

#ifdef REALM_USE_CUDA
void stencil_task_gpu(const void *_args, size_t arglen,
                  const void *userdata, size_t userlen, Realm::Processor p);
void increment_task_gpu(const void *_args, size_t arglen,
                  const void *userdata, size_t userlen, Realm::Processor p);
#endif
