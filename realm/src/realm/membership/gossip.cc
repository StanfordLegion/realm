#include "realm/membership/heartbeat.h"


namespace Realm {

std::unique_ptr<GossipBackend> make_default_gossip_backend(GossipMonitor &owner)
{
 return std::unique_ptr<GossipBackend>(new HeartbeatBackend(owner));
}


} // namespace Realm

