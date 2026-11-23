"""Replay a frozen PS/DPU trace with whole-job or two-segment EDF."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path


@dataclass(frozen=True)
class Job:
    name: str
    ready: int
    service: int
    deadline: int
    model: int


def replay(jobs, split_model=None, gap_us=0, prefix_fraction=.5):
    arrivals = sorted(jobs, key=lambda j: (j.ready, j.name))
    waiting, start, done = [], {}, {}
    now = maximum_block = index = switches = 0
    busy = 0
    previous_boundary = None
    while len(done) < len(jobs):
        if not waiting:
            now = max(now, arrivals[index].ready)
        while index < len(arrivals) and arrivals[index].ready <= now:
            job = arrivals[index]
            pieces = (job.service,)
            if job.model == split_model and job.service >= 2:
                first = max(1, min(job.service - 1, int(job.service * prefix_fraction)))
                # Copy/dispatch/requantization is pessimistically serialized
                # ahead of the next dispatch, even when another job is ready.
                pieces = (first + gap_us, job.service - first)
            heapq.heappush(waiting, (job.deadline, job.name, job, pieces))
            index += 1
        deadline, name, job, pieces = heapq.heappop(waiting)
        if previous_boundary is not None and previous_boundary != name:
            switches += 1
        previous_boundary = None
        start.setdefault(name, now)
        end = now + pieces[0]
        if index < len(arrivals) and arrivals[index].ready < end:
            maximum_block = max(maximum_block, end - arrivals[index].ready)
        now = end
        busy += pieces[0]
        if len(pieces) == 1:
            done[name] = now
        else:
            heapq.heappush(waiting, (deadline, name, job, pieces[1:]))
            previous_boundary = name
    misses = {j.name for j in jobs if done[j.name] > j.deadline}
    waits = sorted(start[j.name] - j.ready for j in jobs)
    return dict(misses=misses, wait_p99_us=waits[int(.99 * (len(waits)-1))],
                wait_max_us=waits[-1], max_arrival_block_us=maximum_block,
                busy_us=busy, boundary_switch_opportunities=switches)


def experiment(trial, gap_grid, prefix_fraction):
    trace = trial / "requests.jsonl"
    identity = json.loads((trial / "execution_identity.json").read_text(encoding="utf-8"))
    reserve = round(identity["live_config"]["command_reserve_ms"] * 1_000)
    jobs, statuses, missed_outside, late_stage = [], Counter(), Counter(), Counter()
    digest = hashlib.sha256()
    with trace.open("rb") as stream:
        for line in stream:
            digest.update(line)
            raw = json.loads(line)
            status = raw["status"]
            statuses[status] += 1
            if not raw.get("inference_dispatched"):
                missed_outside[status] += 1
                continue
            ready = raw["dpu_enqueue_us"]
            service = raw["dpu_local_return_us"] - raw["dpu_dequeue_us"]
            deadline = raw["actuation_deadline_us"] - reserve
            jobs.append(Job(f"{raw['vehicle']}:{raw['cycle']}", ready, service,
                            deadline, raw["model_id"]))
            if status == "dropped_before_command":
                if ready >= deadline:
                    late_stage["ready_after_reserved_deadline"] += 1
                elif ready + service > deadline:
                    late_stage["insufficient_even_on_idle_DPU"] += 1
                else:
                    late_stage["fits_in_isolation"] += 1
    baseline = replay(jobs)
    comparisons = []
    for gap in gap_grid:
        alternative = replay(jobs, 11, gap, prefix_fraction)
        rescued = baseline["misses"] - alternative["misses"]
        harmed = alternative["misses"] - baseline["misses"]
        comparisons.append(dict(gap_us=gap, recovered=len(rescued), new_misses=len(harmed),
                                **metrics(alternative)))
    return dict(trace_sha256=digest.hexdigest(), actual_statuses=statuses,
                not_replayed=missed_outside, late_stage=late_stage,
                replayed=len(jobs), replayed_model_counts=Counter(j.model for j in jobs),
                fixed_command_reserve_us=reserve,
                split_prefix_fraction=prefix_fraction,
                independent_infeasible=sum(j.ready+j.service > j.deadline for j in jobs),
                whole_edf=metrics(baseline), split_id11_edf=comparisons)


def metrics(result):
    return {**result, "misses": len(result["misses"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-fraction", type=float, default=.5)
    parser.add_argument("--gaps-us", nargs="+", type=int, default=(0, 50, 100, 250, 500, 1000))
    args = parser.parse_args()
    if not 0 < args.prefix_fraction < 1 or min(args.gaps_us) < 0:
        parser.error("prefix fraction must be in (0, 1) and gap costs must be nonnegative")
    # One blocked urgent job, one donor. At larger overhead the donor misses.
    demo = [Job("donor", 0, 8, 11, 11), Job("urgent", 3, 2, 7, 1)]
    assert replay(demo)["misses"] == {"urgent"}
    assert not replay(demo, 11, 0)["misses"]
    assert replay(demo, 11, 2)["misses"] == {"urgent", "donor"}
    report = experiment(args.trial, args.gaps_us, args.prefix_fraction)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
