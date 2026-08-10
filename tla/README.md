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
