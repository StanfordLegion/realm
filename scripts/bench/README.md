<!--
Copyright 2023 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->

Legion Quickstart Script
========================

Legion Quickstart provides scripts for building Legion from source
and running Legion programs with appropriate defaults for a number of supported
clusters (and auto-detected settings for local installs).

The scripts in this repository will detect if you are running on the login node
of a supported cluster, and automatically use the appropriate flags to build and
run Legion.

The scripts will automatically invoke the appropriate job scheduler commands, so
you don't need to create jobscripts yourself. Please run the commands directly
from the login node.

Even if your specific cluster is not covered, you may be able to adapt an
existing workflow; look for all the places where the `PLATFORM` variable is
checked and add a case for your cluster.

You can use the same scripts on your local machine, in which case the build/run
flags will be set according to the detected hardware resources.

Invoke any script with `-h` to see more available options.


Build Legion
============

```
git clone https://gitlab.com/StanfordLegion/legion.git <legion-dir>
cd <legion-dir>
USE_CUDA=1 NETWORK=gasnet1 CONDUIT=ibv <Other options> ./script/build.sh
```
Wait for the compilation to be done

Run Legate programs
===================

```
JOB_NAME=<name of the job> ./script/run.sh <num-nodes> ./build/bin/<legion-program> <legion_options> <program_options>
```