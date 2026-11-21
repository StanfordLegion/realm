<!--
Copyright 2026 Stanford University, NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0

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

# Formal Specifications for Realm Protocols

TLA+ models, model-checking scenarios, and design documents for Realm's
distributed protocols. One subdirectory per protocol family:

| Directory | Protocols |
|---|---|
| [`barrier/`](barrier/README.md) | Scalable barrier arrival (`BarrierArrive.tla`) and notification (`BarrierNotify.tla`) — see `barrier/README.md` for the documents, the scenario suite, and the mutation battery |

The working convention, earned the hard way in the barrier effort (details in
`barrier/README.md`): a protocol rule counts as verified only when a scenario
catches its removal, every property is validated in both directions, and the
battery re-runs whenever a rule changes.
