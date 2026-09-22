# TLA+ Model Design: Realm Deferred Instance Allocation

Authoritative C++ → TLA+ mapping for `LocalManagedMemory`'s deferred
allocation/deletion protocol. Phase 2 spec authors implement this document
verbatim. Where this document and the code disagree, the code wins and the
document must be fixed. Where the code and the talk transcript disagree, the
code wins (differences are flagged inline).

All citations are against the current tree:
- `cc`  = `src/realm/mem_impl.cc`
- `h`   = `src/realm/mem_impl.h`
- `inl` = `src/realm/mem_impl.inl`
- `ii`  = `src/realm/inst_impl.cc`

---

## 1. Scope & abstractions (v1)

Modeled: one `LocalManagedMemory` instance and the client/event environment
around its five allocator entry points.

**Atomicity.** Every protocol action corresponds to exactly one
`allocator_mutex`-holding region (`h:398`). All heap-state mutation in
`allocate_storage_deferrable` (cc:732), `attempt_deferrable_allocation`
(precondition: "must be called with allocator_mutex held", cc:747-748),
`release_storage_deferrable` (cc:838), `allocate_storage_immediate` (cc:1096),
`release_storage_immediate` (cc:1619), and `attempt_release_reordering`
(called only under the mutex) happens under one mutex acquisition, so each is
one atomic TLA+ action. Notifications (`notify_allocation` /
`notify_deallocation`) fire after the mutex is dropped; in the model they are
folded into the same action as event-state updates (see §5 for why this is
sound: the only externally visible effects are event triggers/poisons, and
interleaving between mutex-drop and notification only delays those triggers,
which the environment's nondeterministic trigger order already covers).

**Abstractions:**

| Aspect | v1 choice | Rationale / code excluded |
|---|---|---|
| Heap | `0..HEAP_SIZE-1`, HEAP_SIZE ≈ 4-8 units | sizes in bytes are irrelevant; fragmentation shape is what matters |
| Sizes | naturals `>= 1` | size-0 instances take the `SENTINEL` path (`inl:413-416`, `inl:183-198`) and never occupy heap; they cannot affect fragmentation or ordering decisions, only tag bookkeeping. Excluded in v1; revisit in v2 only if tag-lifetime bugs are in scope. |
| Alignment | none (alignment = 0 everywhere) | `calculate_offset` (`inl:154-165`) and the carve-a-front-block path (`inl:441-463`) never fire with alignment 0. v2 extension: alignment adds fragmentation modes and is cheap to add to the FirstFit operator. |
| Instances | fixed set `INSTANCES` ≈ 3-5, each with a model-assigned size | bounded state space; instance IDs never reused (ID-reuse interaction with #442 is v2) |
| Redistricting | **out of scope v1** | `reuse_storage_deferrable` (cc:926-1085), `reuse_storage_immediate` (cc:1326-1536), `split_range` (`inl:168-274`), `record_redistrict` (cc:1822-1837). v2. |
| External resources | out | `ext_resource` branches cc:721-729, cc:831-832, cc:1117-1134, cc:1605-1611 |
| Remote/network | out | creator-node forwarding `ii:996-1032`, active messages `h:594-627`. Single node: `assert(target == Network::my_node_id)` (cc:698-699) holds. |
| Duplicate releases | out | the "network delays → multiple deallocations of same instance" tolerance (comment cc:773-777) cannot arise single-node with one `deferred_destroy` slot per instance. v2 (see §9). |
| `Config::deferred_instance_allocation` | TRUE (default, cc:40) | the `FALSE` branch (cc:762) degenerates to a blocking allocator; not interesting |
| Poison | **intrinsic poison always-on; user poison toggled** | Intrinsic failure poison is core Realm behavior, not optional: `notify_allocation` poisons `eCreated(i)` on INSTANT_FAILURE/CANCELLED (`ii:1121-1122`) with no user poison involved, and under C2 that makes `preD[i]` fire poisoned, routing to `remove_pending_release` (cc:1754-1755). INSTANT_FAILURE is reachable in every config, and the DELAYEDDESTROY path (cc:845-849 → cc:1146-1147) queues a release for an instance that may INSTANT_FAIL at create-trigger — with poison omitted that queued destroy could never fire, producing **false deadlocks** on the full-cleanup client. The `USER_POISON` toggle gates only client-poisoned ballistic events (§5). |
| Profiling / `need_alloc_result` | out | affects only which messages are sent, not heap state |
| Dealloc-completion feedback | **out (v2)** | `InstanceStatus`/`InstanceTimeline` profiling responses on destruction (`ii:1248-1262`) let real clients derive user-event triggers from *destruction completion*, creating cycle shapes v1 cannot express. v1 clients observe only `eCreated` results. |
| `deferred_dealloc_notify` | **modeled, simplified** (see §3) | it is protocol-relevant state (drives when `notify_deallocation` fires; recent fix for slot-recycle double-tracking, `h:427-436`); we track it as one bit + check a notify-exactly-once property |

---

## 2. The deterministic allocator as TLA+ operators

An allocator state is a partial function

```
Alloc == [tag ∈ SUBSET INSTANCES -> [first: 0..HEAP_SIZE, size: 1..HEAP_SIZE]]
```

represented in TLA+ as a function with domain `DOMAIN a` = allocated tags.
Free space is **derived**: the complement of the union of allocated intervals
within `[0, HEAP_SIZE)`.

**Faithfulness argument.** `BasicRangeAllocator` keeps two doubly linked
lists: all ranges in address order, and free ranges. The free list is a
sublist of the address-ordered range list: on `deallocate`, the insertion
neighbors `pf_idx`/`nf_idx` are found by walking the address-ordered `prev` /
`next` chains (`inl:528-537`) and the freed block is linked between them
(`inl:544-551`), and merges (cases 2-4, `inl:552-605`) splice in place. On
`allocate`, leftover blocks replace the consumed block in position
(`inl:473-495`). `add_range` establishes the base case (`inl:100-128`).
Therefore the free list is always in ascending address order, and the
first-fit walk from `ranges[SENTINEL].next_free` (`inl:424-435`,
`can_allocate` `inl:386-402`) selects the **lowest-address free gap that
fits**. Adjacent-free merging (`inl:539-605`) is implicit in the derived-gap
representation: two adjacent free intervals and one merged interval have the
same derived gap set. Hence the representation below is exactly faithful and
deterministic.

Operators (pure, no TLA+ variables):

```
Gaps(a)              == maximal intervals of [0,HEAP_SIZE) not covered by ranges of a
                        (compute: sort allocated intervals by first; walk)
FirstFitOff(a, sz)   == the smallest g.first among gaps g with g.size >= sz
CanAlloc(a, sz)      == ∃ gap g : g.size >= sz
DoAlloc(a, tag, sz)  == a ++ (tag :> [first |-> FirstFitOff(a,sz), size |-> sz])
                        (precondition CanAlloc; caller branches on it)
HasTag(a, tag)       == tag ∈ DOMAIN a
DoFree(a, tag)       == [restrict a to DOMAIN a \ {tag}]
```

`missing_ok` mapping: C++ `deallocate(tag, missing_ok)` asserts the tag is
present when `missing_ok = false` (`inl:612-614`). In the model:
`missing_ok = true` call sites use `IF HasTag(a,tag) THEN DoFree(a,tag) ELSE a`;
`missing_ok = false` call sites emit the invariant-checked form — the enclosing
action asserts `HasTag` via an invariant (§6, INV_TriggeredDeallocPresent /
INV_InOrderReleasePresent) and then frees. Do **not** guard the action on
`HasTag`: a missing tag must produce an invariant violation (that is the bug
detector), not a disabled action.

`can_allocate` (`inl:377-406`) reads state without mutating: maps to
`CanAlloc`. Note `allocate`/`can_allocate` agree on the chosen gap
(same walk), so `CanAlloc(a,sz) => DoAlloc` well-defined is automatic.

---

## 3. Protocol state variables

```
VARIABLES
  cur,   \* current_allocator  (h:411) — ground truth of completed heap ops
  fut,   \* future_allocator   (h:411) — heap after all pending ops
  rel,   \* release_allocator  (h:411) — completed allocs + ready releases
  pendingAllocs,    \* Seq of [inst, size, lastSeq]        (h:413-419, 444)
  pendingReleases,  \* Seq of [inst, isReady, seq, defNote] (h:420-443, 445)
  seqCtr,           \* cur_release_seqid (h:412), incremented pre-push (++x)
  instState,        \* [INSTANCES -> status] — see below
  instOffset,       \* [INSTANCES -> 0..HEAP_SIZE ∪ {OFF_NONE, OFF_FAILED}]
  ... client/event vars (§5)
```

**`instState` values** (derived from the `INSTOFFSET_*` sentinels
`inst_impl.h:167-172` plus create/destroy bookkeeping):

| Model status | C++ correspondence |
|---|---|
| `UNREQUESTED` | instance not yet created by client |
| `CREATE_PENDING` | `INSTOFFSET_DELAYEDALLOC` — create requested, precondition untriggered (cc:714) |
| `CREATE_PENDING_DESTROY` | `INSTOFFSET_DELAYEDDESTROY` — destroy requested while create still pending (cc:847) |
| `ALLOC_DEFERRED` | entry in `pending_allocs`; `ALLOC_DEFERRED` returned (cc:788, 802) |
| `ALLOCATED` | offset valid; `eCreated` triggered (`ii:1201-1202`) |
| `FAILED` | `INSTOFFSET_FAILED`; `eCreated` poisoned (`ii:1121-1122`) |
| `DESTROY_QUEUED` | destroy requested (deferred or bad-path-ready), entry in `pending_releases` or awaiting trigger |
| `DESTROYED` | storage freed from `cur` and `notify_deallocation` fired |

(The Phase 2 author may split `DESTROY_QUEUED` into
untriggered/ready sub-states or derive them from `pendingReleases`; derived is
preferred — keep `instState` minimal and compute the rest.)

**`ALLOC_DEFERRED` note:** while deferred, the instance also appears in
`pendingAllocs`; on eventual success `notify_allocation(ALLOC_EVENTUAL_SUCCESS)`
fires `eCreated` (`ii:1201-1202`); on `ALLOC_EVENTUAL_FAILURE` (poison replay
only) it is poisoned (`ii:1121-1122`).

**Validity convention.** `fut` is meaningful only when `pending_allocs` is
nonempty — comment `h:399-405`; it is (re)built from scratch at cc:768-779
whenever the first deferred alloc is admitted, and the code even notes
"future_allocator becomes invalid" on the failure path (cc:791). `rel` is
meaningful only when `pending_allocs` is nonempty (established cc:787,
maintained thereafter). **Model decision: keep `fut` and `rel` as ordinary
variables holding whatever the code would hold, including stale garbage.** Do
not reset them to a sentinel when they become "invalid": staleness bugs (a
path reading `fut`/`rel` when the code's implicit validity convention says it
shouldn't) are precisely the class of bug we want TLC to find. All invariants
that mention `fut`/`rel` must therefore be guarded by `pendingAllocs /= <<>>`.

**`deferred_dealloc_notify` / notify-once.** Kept as the `defNote` bit on
each `pendingReleases` entry (`h:436`), plus a per-instance ghost counter
`notifyCount` to state PROP_NotifyOnce (§6). The bit is set on the three
bad-path sites (cc:886, cc:1469 [v2], cc:1739) and drained at cc:1244-1245,
cc:1295-1296, cc:1375-1376 [v2], cc:1644-1645. v1 models the non-redistrict
sites only.

**seqid semantics.** `cur_release_seqid` starts 0 (cc:675) and every push
uses `++cur_release_seqid` (pre-increment: cc:859, 885, 895, 1147, 1158).
A `PendingAlloc` records `last_release_seqid := cur_release_seqid` **at
admission time** (cc:784, 801) — i.e. the seqid of the newest release pushed
so far, whether or not that release is still in the list. Model identically:
`seqCtr' = seqCtr + 1` with the new entry carrying `seqCtr'`; allocs carry
`seqCtr` unchanged.

---

## 4. Actions

One action per mutex-holding path. Pseudocode below is normative; C++ lines
cited per step. `Head`/`Tail`/`Append` are TLA+ sequence ops; "erase set S
from seq" keeps relative order.

### 4.1 `ADA(i, sz)` — attempt_deferrable_allocation (cc:749-807)

Helper (not a standalone action; inlined into 4.2/4.4). Returns one of
`INSTANT_SUCCESS | DEFERRED | INSTANT_FAILURE` plus state updates.

```
IF pendingAllocs = <<>> THEN                                   \* cc:754
  IF CanAlloc(cur, sz) THEN
    cur' := DoAlloc(cur, i, sz); result := INSTANT_SUCCESS     \* cc:755-757
  ELSE IF pendingReleases = <<>> THEN
    result := INSTANT_FAILURE                                  \* cc:762-765
  ELSE
    \* rebuild future from scratch                             \* cc:768-779
    INVARIANT CHECK: ∀ r ∈ pendingReleases: ¬r.isReady         \* cc:772 assert
    f := cur; FOR r ∈ pendingReleases (in order):
                f := FreeMissingOk(f, r.inst)                  \* cc:778, missing_ok=TRUE
    IF CanAlloc(f, sz) THEN                                    \* cc:781
      fut' := DoAlloc(f, i, sz)
      pendingAllocs' := << [inst|->i, size|->sz, lastSeq|->seqCtr] >>  \* cc:783-784
      rel' := cur                                              \* cc:787
      result := DEFERRED                                       \* cc:788
    ELSE
      fut' := f (stale — code computed it and left it)         \* cc:790-791
      result := INSTANT_FAILURE
ELSE                                                           \* cc:795-806
  IF CanAlloc(fut, sz) THEN                                    \* cc:798
    fut' := DoAlloc(fut, i, sz)
    pendingAllocs' := Append(pendingAllocs,
                             [inst|->i, size|->sz, lastSeq|->seqCtr])  \* cc:800-801
    result := DEFERRED                                         \* cc:802
  ELSE result := INSTANT_FAILURE                               \* cc:804
```

Transcript flag: the talk describes mode-2 needing a "future fix-up" after an
instant success; the code instead **rebuilds `fut` lazily from scratch**
(cc:768) every time the first deferred alloc is admitted, so no fix-up path
exists. Model the code.

### 4.2 `RequestCreate(i)` — allocate_storage_deferrable (cc:693-745)

Client action; enabled when `instState[i] = UNREQUESTED` and client contract
allows (§5). The create carries precondition `preC[i]`.

- Precondition already triggered+poisoned → `ALLOC_CANCELLED`,
  `instState[i]' = FAILED`, poison `eCreated(i)` (cc:703-708; `ii:1121-1122`).
- Precondition untriggered → `instState[i]' = CREATE_PENDING`
  (`INSTOFFSET_DELAYEDALLOC`, cc:714), waiter registered (cc:715), return
  `DEFERRED` (cc:716). No heap state touched.
- Precondition triggered clean → run `ADA(i, sz_i)` (cc:734-736); apply
  result: `INSTANT_SUCCESS` → `ALLOCATED` + fire `eCreated(i)`;
  `INSTANT_FAILURE` → `FAILED` + poison `eCreated(i)`; `DEFERRED` →
  `ALLOC_DEFERRED` (eCreated not fired yet).

### 4.3 `TriggerCreate(i)` — DeferredCreate::event_triggered → allocate_storage_immediate (ii:49-56; cc:1087-1176)

Environment action; enabled when `instState[i] ∈ {CREATE_PENDING,
CREATE_PENDING_DESTROY}` and `preC[i]` has fired (clean or poisoned).

```
ddExists := (instState[i] = CREATE_PENDING_DESTROY)            \* cc:1106-1107
IF preC[i] poisoned THEN
  result := CANCELLED; instState[i]' = FAILED                  \* cc:1113-1115
  poison eCreated(i)                                           \* ii:1038,1121-1122
ELSE
  result := ADA(i, sz_i)                                       \* cc:1136-1138
IF ddExists THEN                                               \* cc:1145-1154
  seqCtr' := seqCtr + 1
  newRel := [inst|->i, isReady|->FALSE, seq|->seqCtr', defNote|->FALSE]
  pendingReleases' := Append(pendingReleases, newRel)          \* cc:1146-1147  (unconditional!)
  IF result ∈ {INSTANT_SUCCESS, DEFERRED} ∧ pendingAllocs' /= <<>> THEN
    fut' := FreeMissingOk(fut', i)                             \* cc:1150-1153
```

Critical details, all deliberate model targets:
1. The destroy's `PendingRelease` gets its seqid **at create-trigger time**,
   not at destroy-request time (cc:1147) — the total-order insertion point.
2. It is pushed **even when the allocation failed or was cancelled**
   ("success or fail … we have to add it to our list so that we can find it
   later", cc:1142-1143) — but applied to `fut` only on success (cc:1150).
   Seeded bug #2 territory: a later in-order drain frees it from `cur` with
   `missing_ok=false` (cc:1641 → cc:1847 → `inl:614`).
3. Note the ordering: `ADA` pushes the deferred alloc *before* the destroy's
   seqid is assigned, so `lastSeq(alloc for i) < seq(release of i)` — an
   instance never depends on its own release.
4. The pushed release's precondition is the destroy precondition already
   registered via `deferred_destroy.defer` (from 4.4's DELAYEDALLOC path);
   `TriggerDestroy(i)` fires later against this entry.

### 4.4 `RequestDestroy(i)` — release_storage_deferrable (cc:810-924)

Client action; carries precondition `preD[i]`; contract per §5.

```
IF preD[i] triggered ∧ poisoned THEN no-op                     \* cc:818-825 (silent cancel)
ELSE IF instState[i] = CREATE_PENDING THEN
  IF triggered THEN structuralAssertFailed' := TRUE            \* cc:846 assert(!triggered) — ghost flag, NOT an enabledness guard;
                                                               \* C2-on makes this unreachable, C2-off configs hunt it
  instState[i]' := CREATE_PENDING_DESTROY                      \* cc:845-849 (code proceeds as if untriggered)
  (waiter registered on preD[i], cc:920)
ELSE IF pendingAllocs = <<>> THEN                              \* cc:851-860
  IF preD[i] triggered THEN
    IF instState[i] /= FAILED THEN cur' := DoFree(cur, i)      \* cc:854-855, missing_ok=FALSE → missingFree ghost site
    notify_deallocation(i)                                     \* cc:915-917
  ELSE
    seqCtr' := seqCtr+1
    pendingReleases' := Append(.., [i, FALSE, seqCtr', FALSE]) \* cc:858-859
    \* fut NOT touched: invalid while pendingAllocs empty; lazy rebuild covers it
ELSE                                                           \* cc:861-897
  IF preD[i] triggered THEN
    IF instState[i] = FAILED THEN skip                         \* cc:866-868
    ELSE
      rel' := DoFree(rel, i);  fut' := DoFree(fut, i)          \* cc:871-872, missing_ok=FALSE
      INVARIANT: HasTag(rel,i) ∧ HasTag(fut,i)                 \* (INV_TriggeredDeallocPresent)
      IF ARR() THEN (notifications inside)                     \* cc:875-876  (§4.6)
      ELSE
        seqCtr' := seqCtr+1
        pendingReleases' := Append(.., [i, TRUE, seqCtr', TRUE])  \* cc:884-887 ready, defNote
        \* notify_deallocation deferred until entry drained    \* cc:886-887, 916
  ELSE
    IF instState[i] /= FAILED THEN fut' := FreeMissingOk(fut,i) \* cc:892-893
    seqCtr' := seqCtr+1
    pendingReleases' := Append(.., [i, FALSE, seqCtr', FALSE])  \* cc:894-895
```

Untriggered paths register the `TriggerDestroy(i)` waiter (cc:920).

### 4.5 `TriggerDestroy(i)` — DeferredDestroy::event_triggered → release_storage_immediate (ii:81-99; cc:1600-1794)

Environment action; enabled when a destroy for `i` was deferred and `preD[i]`
has fired.

**Poisoned:** `RemovePendingRelease(i)` (§4.7) (cc:1754-1755), no
`notify_deallocation` (cc:1789).

**Clean (cc:1628-1753):** let `pr := pendingReleases`. If `pr = <<>>` then
`structuralAssertFailed' := TRUE` (cc:1630 `assert(!pending_releases.empty())`
— ghost flag, not a guard; the action still fires where the code would abort).

*Oldest path* — `Head(pr).inst = i` (cc:1634):

```
IF pendingAllocs /= <<>> THEN rel' := DoFree(rel, i)           \* cc:1635-1637 (missing_ok=FALSE)
k := 1
REPEAT                                                          \* cc:1640-1700 do-while
  e := pr[k]
  cur' := DoFree(cur', e.inst)                                  \* cc:1641 (missing_ok=FALSE → INV)
  IF e.inst /= i ∧ e.defNote THEN queue notify(e.inst)          \* cc:1644-1645
  \* unblock scan                                               \* cc:1649-1695
  WHILE pendingAllocs' /= <<>> :
    a := Head(pendingAllocs')
    IF k+1 <= Len(pr) ∧ a.lastSeq >= pr[k+1].seq THEN break     \* cc:1654-1657
    DEBUG-INV: a.lastSeq >= e.seq                               \* cc:1662
    INVARIANT: CanAlloc(cur', a.size)                           \* cc:1670 assert(ok)
    off := FirstFitOff(cur', a.size); cur' := DoAlloc(cur', a.inst, a.size)
    INVARIANT (FutureOffsetConsistency):                        \* cc:1674-1691
      IF HasTag(fut, a.inst) THEN fut[a.inst].first = off ∧ fut[a.inst].size = a.size
      ELSE LET m == FIRST entry of the FULL not-yet-erased pendingReleases
                    (searched from index 1, cc:1679-1683) with m.inst = a.inst
           IN  m exists (else structuralAssertFailed, cc:1682 off-end assert)
               ∧ ¬m.isReady                                     \* cc:1687
      \* Under v1's one-release-entry-per-instance uniqueness the FIRST-match
      \* form equals "the entry is ¬ready", but the exact from-begin() FIRST-
      \* match semantics must be kept — it diverges once v2 duplicates exist.
    fire eCreated(a.inst) EVENTUAL_SUCCESS; instState' ALLOCATED
    pendingAllocs' := Tail(pendingAllocs')                      \* cc:1693-1697
  k := k+1
UNTIL k > Len(pr) ∨ ¬pr[k].isReady                              \* cc:1700
pendingReleases' := SubSeq(pr, k, Len(pr))                      \* cc:1702
IF someAllocsSucceeded ∧ pendingAllocs' /= <<>> THEN            \* cc:1704-1717
  rel' := cur'
  FOR r ∈ pendingReleases' with r.isReady: rel' := DoFree(rel', r.inst)  \* cc:1713-1714, missing_ok=FALSE
notify_deallocation(i) at end                                   \* cc:1789-1790
```

Confirmed-correct staleness to model as-is (adversarial review): when the
drain frees entries but **no** allocs succeed, the cc:1704-1717 rebuild is
skipped. This is sound in the code because the drained entries were applied
to both `rel` (cc:1636 / earlier ready marks) and `cur` (cc:1641), so
`rel` stays consistent without a rebuild; the rebuild exists only because
successful allocs are applied to `cur` but not `rel`. Model exactly this
conditional — do not "fix" it.

*Non-oldest path* (cc:1718-1742): find the FIRST entry `e` with
`e.inst = i`, searching forward; if none exists,
`structuralAssertFailed' := TRUE` (cc:1720-1723 off-end assert; ghost flag,
action still proceeds as the code would into UB). Unique in v1 single-node.
`e.isReady' := TRUE` (cc:1724). Then:
- `pendingAllocs = <<>>` → `cur' := DoFree(cur, i)` (cc:1728, missing_ok=FALSE),
  erase `e` (cc:1730), `notify_deallocation(i)` (cc:1789-1790).
- else → `rel' := DoFree(rel, i)` (cc:1738, missing_ok=FALSE),
  `e.defNote' := TRUE`; notify deferred (cc:1739-1740, 1789).

*Tail step, both clean paths:* if `pendingAllocs' /= <<>>` then run `ARR()`
(cc:1751-1753); its notifications (incl. possibly `i`'s own deferred one via
defNote, cc:1744-1750) fire in-action.

### 4.6 `ARR()` — attempt_release_reordering (cc:1207-1324)

Helper, called only with `pendingAllocs /= <<>>` and the mutex held. Returns
TRUE/FALSE; on TRUE state is rewritten, on FALSE **no state change** (unwind,
cc:1320).

```
front := Head(pendingAllocs)
IF ¬CanAlloc(rel, front.size) THEN return FALSE                \* cc:1211-1215
test := rel
n := largest prefix length s.t. allocs 1..n fit greedily:      \* cc:1220-1231
  FOR j = 1.. : IF CanAlloc(test, a_j.size) THEN test := DoAlloc(test, a_j.inst, a_j.size) ELSE break
ASSERT n >= 1                                                  \* cc:1233
IF n = Len(pendingAllocs) THEN                                  \* full success cc:1236-1252
  cur' := test                                                  \* cc:1239 swap
  pendingAllocs' := <<>>                                        \* cc:1240
  pendingReleases' := erase all entries with isReady            \* cc:1241-1250
  (queue notify for erased entries with defNote)                \* cc:1244-1245
  fire eCreated EVENTUAL_SUCCESS for allocs 1..n
  return TRUE
  \* note: fut left stale; rel left stale (both invalid now: pendingAllocs empty)
ELSE                                                            \* partial cc:1253-1322
  tf := test                                                    \* cc:1255
  it3 walks pendingReleases from the front
  FOR j = n+1 .. Len(pendingAllocs):                            \* cc:1258-1277
    advance it3 over entries with seq <= a_j.lastSeq:
      IF ¬entry.isReady THEN tf := FreeMissingOk(tf, entry.inst)  \* cc:1260-1266
    IF CanAlloc(tf, a_j.size) THEN tf := DoAlloc(tf, a_j.inst, a_j.size)  \* cc:1270-1272
    ELSE return FALSE (no state change)                         \* cc:1274-1276, 1311-1321
  \* all future allocs replayed OK                              \* cc:1280
  FOR remaining entries after it3: IF ¬isReady THEN tf := FreeMissingOk(tf, entry.inst)  \* cc:1284-1289
  pendingReleases' := erase all isReady entries                 \* cc:1292-1301 (+defNote notifies)
  pendingAllocs' := SubSeq(pendingAllocs, n+1, ..)              \* cc:1304
  cur' := test; fut' := tf                                      \* cc:1306-1307
  rel' := cur'                                                  \* cc:1309
  fire eCreated EVENTUAL_SUCCESS for allocs 1..n
  return TRUE
```

Note the asymmetry that is a prime model target: ready releases already
applied to `rel` are **not** re-applied to `tf` (they're inside `test`), while
non-ready ones are replayed positionally by seqid. The replay applies
non-ready releases *only up to each alloc's lastSeq* before that alloc, and
the rest after all allocs — faithful to cc:1258-1289.

### 4.7 `RemovePendingRelease(i)` — remove_pending_release (cc:1538-1597)

Called on poisoned destroy trigger (cc:1754-1755). **This action is
always-on v1 core**, not toggled: intrinsic failure poison (INSTANT_FAILURE/
CANCELLED poisoning `eCreated(i)`, `ii:1121-1122`) propagates through C2 into
`preD[i]` in every config, so poisoned destroy triggers are reachable without
any user poison (e.g. the DELAYEDDESTROY entry queued at cc:1146-1147 for an
instance that INSTANT_FAILs at create-trigger). The `USER_POISON` toggle
(§5) gates only client-poisoned ballistic events, which widen the reachable
shapes here (needed for BUG-4-escalated / BUG-6(b)).

```
IF pendingAllocs = <<>> THEN
  erase first entry with inst = i                               \* cc:1547-1553
  (if none exists: structuralAssertFailed' := TRUE — cc:1550 off-end assert)
ELSE
  fut' := cur; rel' := cur                                      \* cc:1556-1557
  found := FALSE; it2 walks pendingAllocs
  FOR each entry e ∈ pendingReleases (in order):                \* cc:1562-1595
    IF e.inst = i ∧ ¬found THEN found := TRUE; erase e          \* cc:1567-1570
    ELSE fut' := FreeMissingOk(fut', e.inst)                    \* cc:1573
    FOR allocs a with a.lastSeq <= e.seq (advance it2):         \* cc:1579
      IF CanAlloc(fut', a.size) THEN fut' := DoAlloc(fut', a.inst, a.size)  \* cc:1581
      ELSE INVARIANT: found                                     \* cc:1587 assert
           fail a: poison eCreated(a.inst) EVENTUAL_FAILURE     \* cc:1591; ii:1121-1122
           erase a from pendingAllocs                           \* cc:1592
```

Subtle (verify in model review): allocs are replayed onto `fut'` only as the
walk passes entries with `e.seq >= a.lastSeq`; entries erased still
contribute their saved `seqid` (cc:1564). **There is no trailing replay
after the loop**: a pending alloc whose `lastSeq` exceeds every walked seqid
(possible after ARR-partial erased the ready release whose seqid the alloc
recorded) is never re-allocated into the rebuilt `fut'` at all — later
admissions test against a `fut` missing that alloc and may claim overlapping
future space. Registered as **BUG-5** (§8); the model keeps exact code
behavior so TLC adjudicates. Also note `rel'` is set to `cur` and *not*
subsequently given the ready releases — ready entries can exist here
(bad-path entries with `isReady`): each is replayed onto `fut` via cc:1573
like any other entry but **`rel` is left without it**, violating the
`rel = current + ready releases` definition (h:399-405). Registered as
**BUG-4** (§8, escalated) — keep exact code behavior.

---

## 5. Client / environment model

**Events.** Each instance `i` has:
- `eCreated(i)`: output event; fired clean on INSTANT/EVENTUAL success
  (`ii:1201-1202`), poisoned on failure/cancel (`ii:1121-1122`).
- `preC[i]`: create precondition — either `NOW` (already triggered) or an
  abstract dependency term.
- `preD[i]`: destroy precondition — same.

A dependency term is `[deps: SUBSET INSTANCES, ballistic: BOOLEAN]`.
**Firing rule (exact):** the term fires only when **all** its deps have
*resolved* — each `eCreated(j)`, `j ∈ deps`, has fired either clean or
poisoned — **and**, if `ballistic = TRUE`, its user event has fired. It
fires **poisoned iff at least one dep resolved poisoned** (or its ballistic
event was user-poisoned, `USER_POISON` configs only); otherwise clean.
There is no early-fire on first poison: the merger waits for all inputs to
resolve before firing. Ballistic user events always eventually fire (weak
fairness on their trigger action). Trigger order among enabled events is
nondeterministic.

**Intrinsic vs user poison.** Intrinsic poison is core v1 in every config:
INSTANT_FAILURE / CANCELLED poisons `eCreated(i)` (`ii:1121-1122`) with no
client involvement, and under C2 that poisons `preD[i]`, routing
`TriggerDestroy(i)` to `RemovePendingRelease` (cc:1754-1755). The
`USER_POISON` toggle adds only one capability: the client may poison a
ballistic user event instead of triggering it cleanly.

Environment actions `TriggerCreate(i)` / `TriggerDestroy(i)` (§4.3/§4.5) are
enabled once the respective term has fired. Trigger delivery order is
nondeterministic (models event-waiter callback scheduling).

**Contract constraints** (named, individually toggleable; violation of a
constraint = illegal client, excluded from legal-client runs):

- **C1 — topological sort / no back edges** (user's stated assumption): the
  dependency set of `preC[i]` / `preD[i]` may contain only instances whose
  *create was requested strictly earlier in the client's request order*, and
  ballistic user events are always eventually triggered by the environment
  regardless of Realm results. This encodes "no untriggered user event whose
  trigger depends on a deferred allocation result" (transcript ~34:45-35:53).
- **C2 — destroy-after-create**: `i ∈ preD[i].deps` always. Consequences:
  `preD[i]` cannot fire clean before `eCreated(i)` fires clean, and a
  failed/cancelled create poisons the destroy (which the code then silently
  cancels, cc:818-825, or removes, cc:1754-1755). Source: transcript
  ~20:51-21:58 ("you can't destroy the instance until … it was actually
  created", "I have the check in Legion to make sure"); the code asserts
  the consequence at cc:846 (`assert(!triggered)` for a destroy of a
  `DELAYEDALLOC` instance).
- **C3 — destroy request ordering**: the transcript mentions Realm "doesn't
  want the control plane to destroy an instance until it's at least attempted
  the deferrable allocation" (~20:28-20:51) — but the code fully handles
  destroy-before-create-trigger via `INSTOFFSET_DELAYEDDESTROY`
  (cc:845-849, cc:1106-1110). **Model decision: allow destroy request any
  time after the create request.** That path is implemented and is one of the
  most delicate (seqid assigned at create-trigger time), so it must be
  reachable.

**Crucially legal and required expressible (seeded bug #1 shape):** the
client may give `preD[j]` a dependency on `eCreated(i)` where `i`'s create
was *requested earlier* but *triggers later* (or never resolves before `j`'s
destroy is requested). C1 permits it (request-order forward edge); nothing in
the code forbids it.

**Worked example the model must express (bug #1, 2 instances + 1 ballistic
event):** heap size H; sizes `sz(I0) = H`, `sz(I1) = H`.
1. `RequestCreate(I0, preC=NOW)` → INSTANT_SUCCESS (fills heap).
2. `RequestCreate(I1, preC=[deps={}, ballistic=TRUE])` → CREATE_PENDING
   (cc:712-717). `eCreated(I1)` handed out.
3. `RequestDestroy(I0, preD=[deps={I0, I1}])` → untriggered (needs
   `eCreated(I1)`); pushed as pending release, seq=1 (cc:858-859).
4. Ballistic event fires → `TriggerCreate(I1)` → `ADA`: `pendingAllocs`
   empty; `cur` full → no; `pendingReleases` nonempty → `fut = cur - I0`;
   I1 fits → **DEFERRED** with `lastSeq = 1` (cc:781-788): Realm plans I1
   into I0's space.
5. Deadlock: `TriggerDestroy(I0)` needs `preD` → needs `eCreated(I1)` →
   fires only on I1's EVENTUAL_SUCCESS → needs the release of I0.
   `LIVE_NoStuckAllocs` fails; in a terminating client TLC reports deadlock.

**Result feedback wiring:** DEFERRED = `eCreated(i)` not yet fired;
EVENTUAL_SUCCESS fires it clean (`ii:1201-1202` via cc:904, 1774, etc.);
INSTANT_FAILURE / EVENTUAL_FAILURE / CANCELLED poison it (`ii:1121-1122`).
The model treats `notify_allocation`'s event trigger as part of the acting
step (§1). `notify_deallocation` sets a ghost bit (for PROP_NotifyOnce);
it fires no event visible to the client in v1.

**Client termination:** each instance is created at most once and destroyed
at most once; when all requested work is drained, only environment stutter
remains. For deadlock-check configs the client must *request destroy for
every instance it creates* (a full-cleanup client), so quiescence =
everything freed.

---

## 6. Invariants and properties

Allocator-state soundness (representation makes overlap impossible by
construction **only if** DoAlloc is used correctly; state them anyway to
catch spec typos — they're nearly free):

- **INV_NoOverlap / INV_InBounds**: allocated intervals of `cur` are
  pairwise disjoint and within `[0, HEAP_SIZE)`. (Conservation of total size
  is implied by disjoint+derived gaps; no separate check needed.)
- **INV_CurrentMatchesGround**: `DOMAIN cur =`
  `{i : instState[i] ∈ {ALLOCATED, ALLOC-completed-awaiting-destroy}} ∪`
  `{i : ∃ e ∈ pendingReleases : e.inst = i ∧ (¬e.isReady ∨ e.defNote)}`
  — precise form: an instance's tag is in `cur` iff it completed a real
  allocation (offset valid) and no release of it has yet been *applied to
  `cur`*. Ready releases queued on the bad path (cc:884-887) and non-oldest
  ready marks (cc:1738-1740) have been applied to `rel`/`fut` but NOT `cur` —
  the defining subtlety. Phase 2: define via a ghost variable
  `curFreed ⊆ INSTANCES` set exactly where the code calls
  `cur.deallocate` — then `INV_CurrentMatchesGround` is
  `DOMAIN cur = allocatedEver \ curFreed`, and staleness is checked by
  construction.
- **INV_NoReadyWhenNoPendingAllocs** (strengthening of cc:772 `assert`):
  `pendingAllocs = <<>> => ∀ e ∈ pendingReleases : ¬e.isReady`.
  **Expected to FAIL — this is BUG-6 (§8).** Both adversarial reviewers
  constructed legal-client violations (one poison-free with 4 instances, one
  via the poison path with 3); the stranded-ready state then trips the real
  cc:772 assert on the next oversized alloc request that reaches the rebuild.
  Model both this strengthened state invariant AND the exact in-`ADA` check
  (a `structuralAssertFailed`-style flag at the rebuild site) so TLC shows
  the full request-to-abort trace, not just the stranded state.
- **INV_TriggeredDeallocPresent** (cc:871-872, missing_ok=FALSE):
  in `RequestDestroy` triggered/nonempty-pendingAllocs path with
  `instState[i] /= FAILED`: `HasTag(rel,i) ∧ HasTag(fut,i)` at the point of
  the free. Encode as action-local check (invariant on the pre-state guarded
  by action enabledness) or as a ghost "assertFailed" flag set when a
  missing_ok=FALSE free finds no tag — **recommended: one global ghost flag
  `missingFree` set by any missing_ok=FALSE free on a missing tag; invariant
  `¬missingFree`.** This covers cc:855, cc:871-872, cc:1636/1641/1728/1738,
  cc:1713-1714 uniformly (each is the code's `assert(missing_ok)` at
  `inl:614` firing).
- **INV_StructuralAsserts**: `¬structuralAssertFailed`. The ghost flag is set
  by the structural asserts the code makes about its own bookkeeping:
  cc:846 (`assert(!triggered)` on a DELAYEDALLOC destroy), cc:1630
  (`assert(!pending_releases.empty())`), and the find-loop off-end asserts
  cc:1720-1723 and cc:1548-1551 (plus cc:1682 inside the DEBUG cross-check).
  These must be ghost flags, NOT enabledness guards: guarding would silently
  *disable* exactly the transitions that crash the real code, hiding bugs in
  C2-off (BUG-2 hunt) configs. Each action still performs the step the code
  would take as it proceeds into UB/abort; once the flag is set the remainder
  of the trace is not meaningful — TLC halts and reports at the violating
  state anyway.
- **INV_NoOrphanTags**: every tag in `DOMAIN cur` either belongs to a live
  instance (allocated, `notify_deallocation` not yet fired) or has an entry
  in `pendingReleases`. Backstop: `Quiescent => DOMAIN cur = {}` (full-
  cleanup client). This is the detector for the escalated BUG-4 mechanism
  (§8): a tag stranded in `cur` after its release entry was erased and its
  dealloc notify fired is invisible to the `curFreed` ghost form of
  INV_CurrentMatchesGround, because `deallocate` genuinely is never called.
- **INV_InOrderUnblockSucceeds** (cc:1670 `assert(ok)`): ghost flag
  `unblockFailed` set in §4.5's oldest path if `¬CanAlloc(cur', a.size)`;
  invariant `¬unblockFailed`. (Same pattern for the cc:1401 twin in v2.)
- **INV_FutureOffsetConsistency** (cc:1674-1691 DEBUG check): at the same
  point: `HasTag(fut, a.inst) => (fut[a.inst].first = off ∧
  fut[a.inst].size = a.size)`, and `¬HasTag(fut, a.inst) => ∃` a later
  `pendingReleases` entry for `a.inst` with `¬isReady`. Ghost-flag encoding.
- **INV_PoisonReplayOnlyFailsAfterPoint** (cc:1587 `assert(found)`):
  in §4.7's replay, an alloc failure before the poisoned entry was found is a
  violation. Ghost-flag. Checked in every config (intrinsic poison makes
  `RemovePendingRelease` reachable everywhere); the deep shapes need
  USER_POISON (Poison4/Big).
- **SAFETY_PromisesKept** (unconditional, all configs): an instance that
  ever entered `ALLOC_DEFERRED` reaches `FAILED` **only via
  `RemovePendingRelease`** (the EVENTUAL_FAILURE at cc:1591, `ii:1121-1122`)
  — goal 1, transcript ~6:28-6:54: "we never want to say that we can
  allocate something that can't actually work". Since intrinsic poison is
  core, EVENTUAL_FAILURE is reachable in every config; the property is that
  no *other* path ever fails a promised allocation. Ghost variables
  `wasDeferred` + a `failedVia` tag recorded at the failing action.
- **PROP_NotifyOnce**: `notifyCount[i] <= 1` always, and (liveness, below)
  `= 1` eventually for every destroyed instance. Safety half is an invariant.
- **LIVE_NoStuckAllocs**: `∀ i : (instState[i] = ALLOC_DEFERRED) ~>
  (instState[i] ∈ {ALLOCATED, FAILED})` under weak fairness on all
  environment trigger actions and ballistic events. **Expected to FAIL —
  bug #1.** Mark the config with the expected-fail annotation like the
  barrier campaign's `LiveG2NoFlush`.
- **LIVE_AllDrains / deadlock detection**: with a terminating full-cleanup
  client, run TLC **with** deadlock checking (do not pass `-deadlock`).
  Caveat: a client that drains *successfully* also reaches a no-successor
  state, which TLC would misreport as deadlock. Fix: define
  `Quiescent == all requests issued ∧ pendingAllocs = <<>> ∧
  pendingReleases = <<>> ∧ ∀ i : instState[i] ∈ {DESTROYED, FAILED-with-
  destroy-resolved}` and add an explicit self-loop disjunct
  `Done == Quiescent ∧ UNCHANGED vars` to `Next`. TLC deadlock reports then
  fire **exactly** on stuck non-quiescent states — the cheap primary
  detector, catching BUG-1 without temporal checking. Backstop invariant:
  `Quiescent => DOMAIN cur = {}` (also the BUG-4 detector, see
  INV_NoOrphanTags). Temporal `LIVE_*` configs run with `-deadlock`
  (deadlock checking OFF — the `Done` stutter-loop plus fairness handles
  termination there).

---

## 7. Module & config plan

```
tla/allocation/
  DESIGN.md              (this file)
  DeferredAlloc.tla      protocol: allocator operators (§2), state (§3), actions (§4)
  MCDeferredAlloc.tla    client/env (§5), constants, ghost vars, invariants (§6)
  Smoke.cfg  Safety.cfg  Liveness.cfg  EventLoop.cfg  Poison4.cfg  Big.cfg
  run.sh                 patterned on tla/barrier/run.sh (JAVA=/opt/homebrew/opt/openjdk/bin/java,
                         JAR=../barrier/tools/tla2tools.jar)
  sapling_tlc.sbatch     for runs projected > 1h
```

All configs: intrinsic poison on (core); `USER_POISON` off unless noted.
"dlk" = TLC deadlock check ON (with the `Done` self-loop, §6); temporal
configs pass `-deadlock` (check off).

| Config | Constants | Checks | Expectation | Where/Est. |
|---|---|---|---|---|
| Smoke | H=3, 2 insts (sz 2,2), free deps within C1/C2 | all INV_*, dlk | INV pass; deadlock traces possible (BUG-1 shape reachable even here) | local, seconds-minutes, <10^6 states |
| Safety | H=4, 4 insts (sz 2,1,1,2 — the BUG-6(a) shape; makes ARR-partial and the cc:772 rebuild reachable: needs ≥2 pending allocs + ≥2 pending releases with one ready) | all INV_*, SAFETY_*, PROP_NotifyOnce(safety), dlk | **FAIL expected: INV_NoReadyWhenNoPendingAllocs (BUG-6(a))**; other INVs hunt (BUG-3) | local, minutes-hours, ~10^7-10^9; sapling fallback |
| EventLoop | the §5 worked-example client, hardcoded shape (H=3, 2 insts sz 3,3, 1 ballistic) | dlk | **FAIL (BUG-1)** | local, seconds |
| Liveness | H=4, 3 insts, WF on env, `-deadlock` | LIVE_NoStuckAllocs | **FAIL (BUG-1)**; re-run with a client constraint excluding the BUG-1 shape → expect pass | local, tens of minutes |
| Poison4 | H=4, 4 insts (Safety sizes) + USER_POISON on | + INV_PoisonReplayOnlyFailsAfterPoint, INV_NoOrphanTags, dlk | hunt: BUG-4-escalated (needs 4 insts + user poison), BUG-5, BUG-6(b) | **sapling** likely; try constrained-client local first |
| Big | H=5-6, 4-5 insts, mixed sizes, USER_POISON on | Safety set + Poison4 set | open hunt | **sapling**, >1h → sbatch |

State-space control: sizes fixed per instance (not chosen nondeterministically)
in v1 configs; dependency sets chosen nondeterministically at request time
within C1/C2 (this is where the client-behavior branching lives). Add a
`StateConstraint` on `seqCtr` (≤ ~2×INSTANCES) as a backstop; it should be
naturally bounded since each instance releases at most once in v1.

**Symmetry: none in v1.** Instances are not symmetric: they carry distinct
sizes, and even equal-sized instances are distinguished by request order,
seqids, and first-fit offsets. TLC symmetry sets over `INSTANCES` would be
unsound (order-sensitive state) — skip.

---

## 8. Seeded bug hypotheses

- **BUG-1 — deferred-create ordered at trigger time (event-loop deadlock).**
  A pending alloc's `last_release_seqid` and its future-allocator basis are
  fixed when the *precondition triggers* (cc:781-788 via cc:1136, admission
  at cc:783-784/800-801), not when the create was *requested* — although
  `eCreated(i)` is handed out at request time. Releases requested after the
  create but before its trigger are in `pendingReleases` and get consumed by
  the plan (cc:768-779), yet their preconditions may legally depend on
  `eCreated(i)` (C1 forward edge). Cycle: alloc waits on release, release's
  precondition waits on alloc's completion. The transcript speaker suspects
  exactly this (~24:47-25:35, "doesn't necessarily get put in the right spot
  in the overall ordering… can cause an event loop"). Detector: EventLoop
  config deadlock check; LIVE_NoStuckAllocs. Fix direction (for the report,
  not to implement): snapshot `cur_release_seqid` at request time in
  `DeferredCreate` and restrict the future-rebuild replay (cc:769) and the
  admission seqid (cc:784) to releases with `seqid <=` snapshot; or refuse to
  defer against newer releases and fail instantly.
- **BUG-2 — failed alloc + deferred destroy → missing-tag free.**
  cc:1144-1154 pushes the destroy's `PendingRelease` even when `result` is
  INSTANT_FAILURE/CANCELLED; the tag was never allocated. The in-order drain
  frees with missing_ok=FALSE (cc:1641 → cc:1839-1848 → `inl:609-614`
  `assert(missing_ok)`), as does the non-oldest path (cc:1728/1738).
  Contract C2 should force `preD` poisoned whenever the alloc failed
  (poisoned `eCreated` propagates), routing to `remove_pending_release`
  instead — the model must confirm C2 closes *every* such path (incl. the
  CANCELLED create with clean-then-poisoned orderings). Detector:
  INV_TriggeredDeallocPresent / the `missingFree` ghost under the Poison
  config and under a C2-off config (to document what happens to
  contract-violating clients).
- **BUG-3 — replay soundness after partial reordering.** The `assert(ok)`
  at cc:1670 and the determinism cross-check cc:1674-1691 rest on "current
  replays the future's history in the same order"; after a partial
  `attempt_release_reordering` (cc:1253-1310) history has been rewritten
  (`cur`/`fut`/`rel` swapped, ready entries erased, allocs erased) and the
  surviving `lastSeq` values refer to erased entries' seqids. Whether every
  interleaving preserves the assert is exactly what TLC will decide.
  Detector: INV_InOrderUnblockSucceeds + INV_FutureOffsetConsistency under
  Safety/Big configs.
- **BUG-4 (flagged in §4.7, ESCALATED to candidate SAFETY bug) — `rel`
  rebuilt without ready releases in `remove_pending_release`.** cc:1556-1557
  rebuilds `rel := cur` but the replay at cc:1573 applies surviving entries
  (ready ones included) only to `fut`; `rel` never gets the ready frees,
  violating the `rel = current + ready releases` definition (h:399-405).
  Escalated mechanism (reviewer B): a surviving READY entry's tag is still
  in `cur` — and now also still in `rel`. A later triggered destroy runs
  `ARR()` with `test := rel` *still containing that tag*; on the
  full-success path `cur' := test` (cc:1239) and the ready entry is erased
  with its `defNote` notify fired (cc:1241-1250). The tag was never
  deallocated from the allocator state that became `cur`: **permanent range
  leak with `notify_deallocation` already fired while the tag is live** —
  the #442 slot-recycle double-tracking class. The `curFreed` ghost form of
  INV_CurrentMatchesGround does NOT catch this (deallocate genuinely never
  called); detector is **INV_NoOrphanTags** + the `Quiescent => DOMAIN cur
  = {}` backstop. Reaching it needs ≥4 instances + user poison → Poison4
  config. Secondary (conservative) consequence: missed unblocks via a `rel`
  missing ready frees — liveness only.
- **BUG-5 (reviewer A) — poison rebuild drops trailing pending allocs from
  `fut`.** The `remove_pending_release` rebuild loop (cc:1562-1595) replays
  a pending alloc onto the rebuilt `fut` only when the walk passes an entry
  with `seq >= lastSeq` (cc:1579); there is **no trailing alloc replay after
  the loop**. Reachable shape: ARR-partial (cc:1253-1310) erases the ready
  release whose seqid a surviving alloc recorded as `lastSeq`; a strictly
  older release is then poisoned → the rebuild walk's seqids all fall below
  that alloc's `lastSeq` → the alloc is silently omitted from `fut'`. Later
  admissions (cc:798) test against a `fut` missing a promised allocation and
  may claim overlapping future space; the overlap materializes when both
  unblock into `cur`. Candidate safety bug. Detectors:
  INV_InOrderUnblockSucceeds / INV_FutureOffsetConsistency /
  INV_NoOverlap-on-`cur` under Poison4.
- **BUG-6 (both reviewers) — cc:772 `assert(!it->is_ready)` reachable on
  legal input; INV_NoReadyWhenNoPendingAllocs is violable.**
  Variant (a), poison-free, 4 instances, H=4: `I1`(sz2)@0, `I2`(sz1)@2,
  `I3`(sz1)@3 fill the heap; create `I4`(sz2) → DEFERRED with
  `lastSeq = seq(R1)` where `R1` = pending destroy(`I1`); destroy(`I2`)
  untriggered → `R2`; destroy(`I3`) triggered → cc:871 frees `rel`/`fut`,
  ARR front-gate fails → `R3` pushed READY (cc:884-887); TriggerDestroy(`I1`):
  oldest drain frees `I1`, unblock scan admits `I4`
  (`I4.lastSeq < R2.seq`), `pendingAllocs` EMPTIES; the do-while stops at
  `R2` (non-ready), leaving `[R2 ¬ready, R3 ready]` with no pending allocs.
  The next oversized alloc request reaches the cc:768 rebuild and trips
  cc:772 (abort in a DEBUG build; in release the ready release is replayed
  with missing_ok — benign-looking, but `rel := cur` at cc:787 then omits
  `R3`'s readiness → BUG-4-shape conservatism). Variant (b), poison path,
  3 instances: `[R1 ¬ready, R2 ready(defNote)]` + pending alloc `A`;
  poison `R1` → `remove_pending_release` erases `R1`, replay fails `A`
  (cc:1587-1592) → `pendingAllocs` empties with `R2` ready stranded.
  Detectors: INV_NoReadyWhenNoPendingAllocs (expected FAIL, §6) + the
  in-`ADA` rebuild-site flag for the full request-to-abort trace. TLC
  adjudicates downstream severity.

---

## 9. v2 roadmap

1. **Redistricting**: model `split_range` (`inl:168-274`) — carve N new tags
   out of one old range **in place** at ascending offsets, partial success
   returns count `i` of tags placed and *deallocates the old tag in every
   exit path* (`inl:189, 207, 254, 264`); zero-sized handling; then
   `reuse_storage_deferrable` (cc:926-1085) and `reuse_storage_immediate`
   (cc:1326-1536) as new actions, `PendingRelease.redistrict_*` payloads
   (h:422-424), the `deferred_redistrict` handoff (cc:1078-1080; ii:84-98).
   The `offsets` out-param feeds `notify_allocation` per new instance
   (cc:1527-1533).
2. **Alignment**: add `align` to instance parameters, extend `FirstFitOff`
   with `calculate_offset` semantics (`inl:154-165`) incl. the freed-padding
   behavior in allocate (`inl:441-463`) — pure operator change.
3. **Duplicate releases** (multi-node artifact, comment cc:773-777):
   introduce an action that enqueues a second release for one instance and
   check the missing_ok=FALSE drain paths (cc:1641) against it.
4. **notify-once / instance-slot reuse**: full `deferred_dealloc_notify`
   semantics vs. `new_instance` slot recycling (#442, h:427-436) — needs a
   model of the instance-slot free list.
5. **Size-0 instances** via the SENTINEL path if tag-lifetime bugs become
   interesting.
5b. **Dealloc-completion feedback to the client**: model the
   destruction-side profiling responses (`InstanceStatus` /
   `InstanceTimeline`, `ii:1248-1262`) as a client-visible "destroyed(i)"
   event that user-event triggers may depend on — adds cycle shapes through
   destruction completion that v1 cannot express (§1 exclusion).
6. **`SizedRangeAllocator`** (`inl:645-1169`): different fit policy
   (size-binned, not address-ordered first-fit) — swap the FirstFit operator
   and re-run; determinism assumptions of the protocol must hold for ANY
   deterministic allocator, so this is a cheap second data point.
