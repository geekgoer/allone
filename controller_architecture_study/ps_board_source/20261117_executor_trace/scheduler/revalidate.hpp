#pragma once
#include "scheduler.hpp"
namespace ps_scheduler {
// One proposed future delta, one latest snapshot, one full evaluation. No search.
// Returns {accepted,status,decision?}; supplied costs remain empirical envelopes.
Json revalidate_candidate(const Json& old_snapshot,const Json& old_decision,
                          const Json& latest_snapshot,double reserve_ms);
}
