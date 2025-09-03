#ifndef MEMBERSHIP_H
#define MEMBERSHIP_H

#include <stdint.h>
#include <stddef.h>
#include <functional>
#include <memory>
#include "realm/realm_c.h"

struct NodeInfo {
  int32_t node_id;
  int32_t seed_id;
  bool announce_mm;
};

class IMembership {
public:
  enum class ChangeKind
  {
    PreJoin,
    PostJoin,
    PreLeave,
    PostLeave
  };

  using ChangeCallback =
      std::function<void(const NodeInfo &, bool joined /*true: join, false: leave*/)>;
  using FilterCallback = std::function<bool(const NodeInfo &)>;

  struct Hooks {
    ChangeCallback pre_join;
    ChangeCallback post_join;
    ChangeCallback pre_leave;
    ChangeCallback post_leave;
    FilterCallback filter;
  };

  virtual ~IMembership() = default;
  virtual bool join(const NodeInfo &self) = 0;
  virtual bool leave(const NodeInfo &self) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
};

std::unique_ptr<IMembership> create_membership(const std::string &backend,
                                               const IMembership::Hooks &hooks);

#endif /* MEMBERSHIP_H */
