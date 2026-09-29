# Deeper Verification Options — Deferred Allocation Model

Written 2026-08-30, when the C++ port of the fix bundle (FIX_CAP + FIX_SWEEP
+ FIX_RPR) began on the evidence below. This records what we would do if we
ever wanted more detailed verification than the campaign has already
delivered — cheap wins first, structural levers second, then the model
extensions that carry real verification debt, and the C++-side options that
complement TLC entirely.

## 1. Where verification stands (the baseline)

Local, **fully exhaustive** (state space completely drained, all green):

| Config | Gen / distinct | Notes |
|---|---|---|
| SafetyMiniFixed | 64.7M / 23.5M | full battery incl. `INV_NoDupAlloc`, depth 19 |
| SafetyMiniSweepOnly | 53.3M / 20.6M | sweep-alone attribution |
| Composite4Fixed | 245k / 120k | the BUG-6→BUG-4 leak composite, closed |
| Inversion | 478 / 255 | **deadlock checking ON** — monotone-cap + trailing replay |
| GCRipple / EventLoopFixed / SmokeFixed | small | scripted intent invariants |
| LivenessFixed | 13.3k / 6.3k | temporal `LIVE_NoStuckAllocs` holds |

Toggles-off regressions reproduce the four known counterexamples at exact
trace depths (EventLoop deadlock@7, SafetyMini violation@9, etc.).

Sapling, **bounded but very large** (all violation-free at death):

- **Job 77858** (SafetyFixed4, bundle, `-gzip`): 23.9B generated / **7.52B
  distinct**, depth 11 COMPLETE at 3.58B, ~3.9B states into depth 12; died
  at 25.5h on the fingerprint-set merge ("No space left", 261G free at
  start — the node carried 435G of dead-job leftovers).
- **Poison4Fixed** (77843): 1.90B distinct clean through depth 10 — the
  poison paths, where FIX_SWEEP's third site and FIX_RPR live.
- **Hunts** (77854/77855): SafetyHunt 1.67B, PoisonHunt 1.42B distinct,
  clean through depth 10-11 — still **no witness** for BUG-3
  (mem_impl.cc:1668-1691 replay soundness), BUG-4-standalone, or unfixed
  BUG-5 in the current-code model.

The pre-registered gate (SafetyFixed4 clean through depth 13, `Progress(14)`)
was **not met**: completing depth 13 plausibly needs 15-25B distinct states
and 600-800G+ of node-local disk — beyond single-node sapling hardware.
Accepted rationale for proceeding anyway: 7.52B clean distinct states is
~2x the total coverage at which the only scale-level defect ever observed
(the round-1 TrailingRPR wiring artifact, a spec bug — corrected and
re-validated) appeared (depth 12, 4.0B distinct), on top of the local full
exhaustions of every behavior class the model expresses.

## 2. Cheap wins if we resume scale runs (in effort order)

1. **Sweep node /tmp first.** 77858 ran with only 261G of a 733G disk
   (435G of leftovers from scancel'ed jobs whose cleanup traps never ran).
   A swept node is ~2.7x the capacity for zero engineering.
2. **Keep the fingerprint set in RAM**: submit with `--mem=256G` (if nodes
   allow) and pass TLC `-fpmem 0.6` (fraction of heap for the fpset).
   Arithmetic: 7.5-15B fingerprints × 8B = 60-120G, which fits a ~200G heap
   at fpmem 0.6-0.7. This eliminates the on-disk fp files AND the merge
   transient that actually killed 77858, leaving the whole disk budget to
   the gzipped queue (measured ~34-44 B/state → a swept 733G node holds
   roughly 16-20B queued states).
3. **Wider nodes**: raise `--cpus-per-task` past 40 if sapling nodes have
   more cores — TLC worker scaling is roughly linear until memory
   bandwidth saturates (77858 sustained 17-21M states/min at 40 workers).

With all three: depth-12 completion is plausible in ~40-70h; depth 13
likely remains out of reach on a single node.

## 3. Structural options (bigger levers)

- **Distributed TLC** (TLCServer + TLCWorker + distributed fpset servers
  across several nodes): removes the single-node disk/memory ceiling
  entirely. Real setup cost, fragile interactions with `-gzip` and
  checkpointing, and cluster etiquette concerns — worth it only if a
  specific claim (e.g., "depth 13 complete") becomes load-bearing.
- **Targeted adversarial configs instead of blanket depth**: scripted
  clients aimed at specific interleavings (the pattern that confirmed the
  BUG-6→BUG-4 composite locally in 5 seconds via `Composite4.cfg` after
  blanket Safety needed sapling). Note the claim changes from "all
  behaviors ≤ depth D" to "all behaviors of shape S" — pre-register the
  shape in EXPECTED.md as we did for Composite4.
- **Symmetry reduction**: the no-symmetry decision (DESIGN.md §7) holds for
  mixed-size configs, but configs with EQUAL instance sizes admit
  permutation symmetry (TLC `SYMMETRY` over a model-value instance set),
  typically 10-100x state-space reduction for equal-size hunts. Caveat:
  TLC's symmetry is unsound for liveness checking — safety configs only.
- **Simulation mode** (`-simulate` with `-depth N`): probabilistic
  deep-trace sampling far past any BFS frontier. No completeness claim,
  but the best bug-hunting per node-hour for "is there anything lurking at
  depth 20+" questions, and it barely touches disk.

## 4. Model-extension verification debt (v2 roadmap)

Each of these is UNVERIFIED territory today; extending the model is the
prerequisite for trusting the corresponding code path or fix arm:

- **Redistricting** (split_range: mem_impl.inl:168-274;
  reuse_storage_deferrable/immediate: cc:926-1085, cc:1326-1536).
  **Required before trusting the sweep fix's redistrict arm in
  production** — bugs/BUG-6.md fix A is redistrict-aware by design, but v1
  models plain frees only. Needs: redistrict actions + PendingRelease
  redistrict fields + a split_range operator in DeferredAlloc.tla;
  redistrict variants of SafetyMini/Composite4; child-offset-consistency
  invariants and the INV_NoOrphanTags extension to child tags.
- **The BUG-1 union rule** ("cap ∪ clean-triggered releases",
  bugs/BUG-1.md): only needed if the GC-ripple pattern under memory
  pressure produces unacceptable spurious instant-failures. It is
  currently UNVERIFIED — extending the funding gate in `ADAResCap` and
  re-running the FULL validation matrix is mandatory before any C++ use.
- **Alignment** (calculate_offset, mem_impl.inl:154-165): richer
  fragmentation; extends the §2 allocator operators and the first-fit
  equivalence argument.
- **Duplicate releases via network delays** (tolerated by cc:773-778):
  relax v1's one-release-per-instance uniqueness; the exact first-match
  form of INV_FutureOffsetConsistency was written to survive this. Note:
  the sweep fix's strict void free would debug-assert on the second entry
  of a duplicate ready pair — identical strictness to the pre-existing
  in-order drain (so not a regression), but the duplicates model must
  account for it when this item is taken up.
- **Multi-node create ordering** (the remote-create snapshot window): a
  creation issued on a non-owner node publishes `e_created` at the creator
  before the owner takes the `release_seqid_cap` snapshot on
  MemStorageAllocRequest receipt, so a release requested inside that
  window can slip under the cap and readmit the BUG-1 funding cycle
  across nodes. Fix directions to evaluate in a multi-node model: take the
  snapshot at the creator and carry it in the active message, or adjust
  the cap on the owner at AM receipt (e.g. exclude releases whose request
  provably raced the create's AM).
- **Dealloc-completion feedback shapes** (clients deriving triggers from
  destruction profiling responses — excluded in DESIGN.md §1).
- **Multi-memory / remote request paths** (MemStorageAlloc/Release
  messages, remote notify forwarding).

## 5. C++-side verification (complementary to TLC)

- **Randomized stress harness**: drive the public API with
  create/destroy/user-event-trigger sequences shaped like the model's
  client contract (topologically sorted, destroy-after-create, ballistic
  triggers) at sizes chosen to force deferral, reordering, and poison.
  `tests/random_config_test.cc` is an existing repo pattern to follow.
- **Debug-build soak**: the fix bundle restores mem_impl.cc:772's
  `assert(!it->is_ready)` as a true invariant — long debug-build runs with
  the cc:772-family asserts active are now meaningful regression signal
  rather than known-false alarms.
- **Runtime shadow-checker**: maintain the model's key ghosts as
  DEBUG_REALM counters inside LocalManagedMemory — a tag-vs-live-instance
  audit (INV_NoOrphanTags) at pending-queue-empty transitions and a
  notify-once counter per instance — turning the two #442-class detectors
  into cheap in-situ checks.

## 6. The standing regression oracle

The local matrix is the durable payoff: `./run.sh` (Smoke → EventLoop →
SafetyMini → Composite4 → Liveness → LivenessNoCross, plus the Fixed
family) runs in seconds-to-minutes and must be green before and after ANY
change to mem_impl.cc allocator logic — with the toggles mirroring whether
the change is pre- or post-fix-bundle semantics. The sapling configs only
re-enter the picture when the MODEL itself changes (v2 extensions above);
routine code work never needs them.
