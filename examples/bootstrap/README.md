# Realm Bootstrap Example

Shows how an application can configure Realm's bootstrap using a key-value interface.

The `sample_app.cc` main function represents an application like Dynamo or Flashcache that links against Realm. The `realm_bootstrap.h` and `realm_bootstrap.cc` files implement Realm's `bootstrap_handle_t` contract, providing the required interface.

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Running

```bash
mpirun -n 4 ./sample_app
```

## What It Does

The app implements a `bootstrap_handle_t` using environment variables as the communication mode. The `get()` function reads from `getenv()`, and `init()` writes with `setenv()` - both use the same mechanism.

The `init()` function sets example keys that Realm could read:
- `REALM_COMM_MODE=env` - communication mode (env, etcd, pmix)
- `REALM_BOOTSTRAP_TYPE=mpi` - bootstrap mechanism (mpi, p2p, etc)

Realm uses the same `bootstrap_handle_t` interface regardless of the underlying communication mode. The application chooses the mode (environment variables, etcd, pmix, etc.) and implements all functions consistently with that choice. Future versions could switch to etcd or other distributed stores by changing only the implementation, not the interface.

