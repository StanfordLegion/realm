cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(RealmExample_vtable)

if(NOT TARGET Realm::Realm)
  find_package(Realm REQUIRED)
endif()

find_package(MPI REQUIRED)

add_executable(vtable vtable.cc realm_bootstrap.cc)
target_link_libraries(vtable Realm::Realm MPI::MPI_CXX)

find_library(UCP_LIBRARY NAMES ucp PATHS ENV LD_LIBRARY_PATH)
find_library(UCC_LIBRARY NAMES ucc PATHS ENV LD_LIBRARY_PATH)
find_library(UCS_LIBRARY NAMES ucs PATHS ENV LD_LIBRARY_PATH)
find_library(UCT_LIBRARY NAMES uct PATHS ENV LD_LIBRARY_PATH)

if(UCP_LIBRARY)
  target_link_libraries(vtable ${UCP_LIBRARY})
endif()
if(UCC_LIBRARY)
  target_link_libraries(vtable ${UCC_LIBRARY})
endif()
if(UCS_LIBRARY)
  target_link_libraries(vtable ${UCS_LIBRARY})
endif()
if(UCT_LIBRARY)
  target_link_libraries(vtable ${UCT_LIBRARY})
endif()

target_include_directories(vtable PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

