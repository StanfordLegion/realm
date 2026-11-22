// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RealmDialect.h"
#include "RealmPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

namespace mlir::realm {
  namespace {
    llvm::DenseMap<Value, size_t> rankSpawnEvents(ArrayRef<SpawnOp> tasks)
    {
      // SSA spawn-to-spawn edges form a DAG. Postorder gives each spawn a
      // rank above its prerequisites, including across blocks and regions.
      // Block arguments and other event producers are opaque roots at rank 0.
      llvm::DenseMap<Value, size_t> ranks;
      SmallVector<std::pair<SpawnOp, bool>> pending;
      for(SpawnOp task : tasks) {
        pending.emplace_back(task, false);
        while(!pending.empty()) {
          auto [current, expanded] = pending.pop_back_val();
          if(ranks.contains(current.getDone()))
            continue;
          if(expanded) {
            ranks.try_emplace(current.getDone(), ranks.size() + 1);
            continue;
          }
          pending.emplace_back(current, true);
          for(Value input : current.getDependencies()) {
            if(auto predecessor = input.getDefiningOp<SpawnOp>())
              pending.emplace_back(predecessor, false);
          }
        }
      }
      return ranks;
    }

    SmallVector<Value> reduceDependencies(OperandRange dependencies,
                                          const llvm::DenseMap<Value, size_t> &ranks)
    {
      llvm::SmallDenseSet<Value> candidates(dependencies.begin(), dependencies.end());
      size_t minimumRank = std::numeric_limits<size_t>::max();
      for(Value event : candidates)
        minimumRank = std::min(minimumRank, ranks.lookup(event));

      SmallVector<Value> pending;
      llvm::SmallDenseSet<Value> visited;
      llvm::SmallDenseSet<Value> redundant;
      auto enqueueInputs = [&](Value event) {
        if(ranks.lookup(event) <= minimumRank)
          return;
        for(Value input : event.getDefiningOp<SpawnOp>().getDependencies()) {
          if(candidates.contains(input))
            redundant.insert(input);
          if(ranks.lookup(input) > minimumRank && visited.insert(input).second)
            pending.push_back(input);
        }
      };
      for(Value event : candidates)
        enqueueInputs(event);

      // Traverse once from the prerequisites of the direct inputs. No input
      // can reach itself, and the highest-ranked input cannot be redundant.
      while(redundant.size() + 1 < candidates.size() && !pending.empty())
        enqueueInputs(pending.pop_back_val());

      SmallVector<Value> required;
      for(Value event : dependencies) {
        if(candidates.erase(event) && !redundant.contains(event))
          required.push_back(event);
      }
      return required;
    }

    bool isCleanupJoin(JoinSettledOp join, CreateInstanceOp allocation, CopyOp second)
    {
      return join && allocation &&
             llvm::is_contained(join.getEvents(), allocation.getReady()) &&
             llvm::is_contained(join.getEvents(), second.getDone());
    }

    void eliminateStagingCopy(CopyOp second)
    {
      if(second.getDependencies().size() != 1)
        return;
      auto first = second.getDependencies().front().getDefiningOp<CopyOp>();
      if(!first || first->getNextNode() != second.getOperation() ||
         first.getTarget() != second.getSource() ||
         first.getSource() == second.getTarget() ||
         first.getTarget() == first.getSource() ||
         first.getTarget() == second.getTarget() ||
         first.getSpace() != second.getSpace() ||
         first.getBytesAttr() != second.getBytesAttr() ||
         first.getTargetFieldAttr() != second.getSourceFieldAttr() ||
         !first->getDiscardableAttrs().empty() || !second->getDiscardableAttrs().empty())
        return;
      auto allocation = first.getTarget().getDefiningOp<CreateInstanceOp>();
      for(Operation *user : first.getTarget().getUsers()) {
        if(user == first.getOperation() || user == second.getOperation())
          continue;
        if(!allocation)
          return;
        if(isa<InstanceExistsOp>(user))
          continue;
        auto destroy = dyn_cast<DestroyInstanceOp>(user);
        if(!destroy || destroy.getDependencies().size() != 1 ||
           !isCleanupJoin(
               destroy.getDependencies().front().getDefiningOp<JoinSettledOp>(),
               allocation, second))
          return;
      }

      SmallVector<JoinSettledOp> cleanupJoins;
      for(Operation *user : first.getDone().getUsers()) {
        if(user == second.getOperation())
          continue;
        auto join = dyn_cast<JoinSettledOp>(user);
        if(!isCleanupJoin(join, allocation, second))
          return;
        cleanupJoins.push_back(join);
      }

      second.getSourceMutable().set(first.getSource());
      second.setSourceFieldAttr(first.getSourceFieldAttr());
      second.getDependenciesMutable().assign(first.getDependencies());
      for(JoinSettledOp join : cleanupJoins) {
        SmallVector<Value> retained;
        llvm::copy_if(join.getEvents(), std::back_inserter(retained),
                      [&](Value event) { return event != first.getDone(); });
        join.getEventsMutable().assign(retained);
      }
      first.erase();
    }

    struct SimplifyRealmDataflow
      : PassWrapper<SimplifyRealmDataflow, OperationPass<ModuleOp>> {
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyRealmDataflow)

      StringRef getArgument() const override { return "simplify-realm-dataflow"; }
      StringRef getDescription() const override
      {
        return "Simplify event and copy edges under explicit successful dataflow "
               "contracts";
      }

      void runOnOperation() override
      {
        getOperation().walk([](func::FuncOp function) {
          // These rewrites preserve successful dataflow. The attribute opts out
          // of preserving the exact timing of poison propagation and profiling.
          if(!function->hasAttrOfType<UnitAttr>("realm.assume_success"))
            return;

          SmallVector<SpawnOp> tasks;
          function.walk([&](SpawnOp op) { tasks.push_back(op); });
          auto ranks = rankSpawnEvents(tasks);
          // Analyze the original edges so a shortcut remains available to
          // later queries even when it will itself be removed.
          SmallVector<std::pair<SpawnOp, SmallVector<Value>>> changes;
          for(SpawnOp task : tasks) {
            auto dependencies = task.getDependencies();
            if(dependencies.size() < 2)
              continue;
            auto retained = reduceDependencies(dependencies, ranks);
            if(retained.size() != dependencies.size())
              changes.emplace_back(task, std::move(retained));
          }
          for(auto &[task, retained] : changes)
            task.getDependenciesMutable().assign(retained);
          // The caller guarantees a supported direct copy, private staging,
          // disjoint instances, a ready destination, and stable source data.
          if(function->hasAttrOfType<UnitAttr>("realm.private_staging")) {
            SmallVector<CopyOp> copies;
            function.walk([&](CopyOp op) { copies.push_back(op); });
            for(CopyOp copy : copies)
              eliminateStagingCopy(copy);
          }
        });
      }
    };
  } // namespace

  std::unique_ptr<Pass> createSimplifyRealmDataflowPass()
  {
    return std::make_unique<SimplifyRealmDataflow>();
  }

  void registerRealmPasses()
  {
    registerPass(createLowerRealmToEmitCPass);
    registerPass(createLowerRealmToLLVMPass);
    registerPass(createLowerAsyncToRealmPass);
    registerPass(createLowerGPUToRealmPass);
    registerPass(createSimplifyRealmDataflowPass);
    PassPipelineRegistration<>(
        "convert-realm-machine-to-emitc", "Alias for convert-realm-to-emitc",
        [](OpPassManager &manager) { manager.addPass(createLowerRealmToEmitCPass()); });
  }
} // namespace mlir::realm
