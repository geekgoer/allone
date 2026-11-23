"""Audit selective DNN splitting at historical single-DPU dispatch instants."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path


STAGES = ("planned_release_us", "actual_publish_us", "sample_start_us", "state_ready_us",
          "analysis_ready_us", "feature_ready_us")
WINDOW_US = 5_000
SPLIT_MODEL = 11


@dataclass(frozen=True)
class Job:
    name: str
    model: int
    release_model: int
    observed: tuple
    ready: int | None
    due: int | None
    slot_due: int
    dequeue: int | None
    finished: int
    service: int | None
    run_wall: int | None
    admitted: bool
    status: str
    rejection: str | None


def quantile(samples, p):
    values = sorted(samples)
    return values[round(p * (len(values) - 1))]


def read_trial(trial):
    identity = json.loads((trial / "execution_identity.json").read_text(encoding="utf-8"))
    reserve = round(identity["live_config"]["command_reserve_ms"] * 1000)
    nominal = {row["numeric_id"]: round(row["service_ms"] * 1000) for row in identity["initial_modes"]}
    jobs, reserve_updates, service_updates = [], [], []
    last_release = {}
    period_us = round(identity["live_config"]["period_ms"] * 1000)
    sha = hashlib.sha256()
    with (trial / "requests.jsonl").open("rb") as stream:
        for line in stream:
            sha.update(line)
            row = json.loads(line)
            release = row["planned_release_us"]
            previous = last_release.get(row["vehicle"])
            if previous is not None and release - previous != period_us:
                raise ValueError("This forecast audit requires fixed per-vehicle release phases")
            last_release[row["vehicle"]] = release
            queued = row.get("calendar_queued") or {}
            admitted = queued.get("phase") == "queued" and not queued.get("cancelled", False)
            actual_model = row["model_id"] if row.get("inference_dispatched") else queued.get("model_id", row["model_id"])
            service = row["dpu_local_return_us"] - row["dpu_dequeue_us"] if row.get("inference_dispatched") else None
            run_wall = row["dpu_result_readable_us"] - row["dpu_submit_us"] if service else None
            due = row.get("actuation_deadline_us")
            if row.get("paired_sample_start_us") is not None:
                reserve_updates.append((row["paired_sample_start_us"], row["command_return_us"] - row["control_converted_us"]))
            recorded = row["calendar_finished"]
            if "board_done_ms" in recorded:
                start, done = (round(recorded[k] * 1000) for k in ("board_start_ms", "board_done_ms"))
                service_updates.append((done, actual_model, done - start))
            jobs.append(Job(f"{row['vehicle']}:{row['cycle']}", actual_model,
                            row["calendar_release"]["model_id"], tuple(row.get(k) for k in STAGES),
                            row.get("dpu_enqueue_us"), None if due is None else due - reserve,
                            row["period_deadline_us"] - reserve, row.get("dpu_dequeue_us"),
                            row["slot_return_us"], service, run_wall, admitted, row["status"],
                            (row.get("calendar_dispatch") or {}).get("dispatch_rejection")))
    candidates = identity["live_config"].get("candidate_ids", [identity["live_config"]["model_id"]])
    final = json.loads((trial / "calendar_final.json").read_text(encoding="utf-8"))
    reserve_history = [(0, reserve)]
    for stamp, value in sorted(reserve_updates):
        if value > reserve_history[-1][1]:
            reserve_history.append((stamp, value))
    expected = round(final["config"]["command_reserve_ms"] * 1000)
    if reserve_history[-1][1] != expected:
        raise ValueError(f"Incomplete reserve reconstruction in {trial}")
    modes = nominal.copy()
    floor_history = [(0, min(modes[i] for i in candidates))]
    for stamp, model, service in sorted(service_updates):
        modes[model] = max(modes[model], service)
        floor = min(modes[i] for i in candidates)
        if floor > floor_history[-1][1]:
            floor_history.append((stamp, floor))
    expected_modes = {int(i): round(mode["service_ms"] * 1000) for i, mode in final["modes"].items()}
    if modes != expected_modes:
        raise ValueError(f"Incomplete model cost reconstruction in {trial}")
    return jobs, nominal, candidates, sha.hexdigest(), reserve_history, floor_history, modes


def learn(jobs, cutoff):
    deltas = defaultdict(list)
    service = defaultdict(list)
    for job in jobs:
        if job.ready is not None and job.ready < cutoff:
            for stage, stamp in enumerate(job.observed):
                if stamp is not None and stamp <= job.ready:
                    deltas[stage].append(job.ready - stamp)
        if job.service and job.dequeue + job.service < cutoff:
            service[job.model].append(job.service)
    return ({stage: sorted(values) for stage, values in deltas.items()},
            {model: quantile(values, .99) for model, values in service.items() if len(values) >= 30})


def estimate(job, now, samples, q):
    stage = max((i for i, stamp in enumerate(job.observed) if stamp is not None and stamp <= now), default=0)
    stamp = job.observed[stage]
    values = samples.get(stage, [])
    # The input is still not ready. Condition the estimate on survival past now.
    lo = bisect_right(values, now - stamp)
    if len(values) - lo < 10:
        return None
    return stage, stamp + values[lo + round(q * (len(values) - lo - 1))]


def pair(now, ready, donor_cost, urgent_cost, donor_due, urgent_due):
    prefix = round(donor_cost * .428) + 31
    suffix = donor_cost - round(donor_cost * .428)
    whole_urgent = max(now + donor_cost, ready) + urgent_cost
    defer_urgent = max(now, ready) + urgent_cost
    split_urgent = max(now + prefix, ready) + urgent_cost
    return {"whole": (now + donor_cost, whole_urgent),
            "defer": (defer_urgent + donor_cost, defer_urgent),
            "split": (split_urgent + suffix, split_urgent)}, (donor_due, urgent_due)


def fits(finishes, dues):
    return all(finish <= due for finish, due in zip(finishes, dues))


def audit_rejection_floor(jobs, reserve_history, floor_history):
    reserve_time, reserve_value = zip(*reserve_history)
    cost_time, cost_value = zip(*floor_history)
    result = Counter()
    for job in jobs:
        if job.rejection != "no_model_fits_remaining_delivery_budget" or job.dequeue is None:
            continue
        at_rejection = job.dequeue
        reserve = reserve_value[bisect_right(reserve_time, at_rejection) - 1]
        service = cost_value[bisect_right(cost_time, at_rejection) - 1]
        actuation = job.due + reserve_value[0]
        budget = actuation - at_rejection - reserve
        result["logged"] += 1
        if service > budget:
            result["reproduced_with_observed_max_costs"] += 1
            if cost_value[0] <= budget:
                result["would_pass_using_initial_cost_and_same_reserve"] += 1
            else:
                result["initial_cost_also_too_large"] += 1
        else:
            result["not_reproduced_at_logged_dequeue"] += 1
    return result


def check_tail(now, jobs, cost, reserve_increase=0):
    """EDF over the other obligations in the local window after the forced prefix."""
    arrivals = iter(sorted(jobs, key=lambda row: row.ready))
    queue = []
    following = next(arrivals, None)
    while queue or following:
        if not queue:
            now = max(now, following.ready)
        while following and following.ready <= now:
            heapq.heappush(queue, (following.due, following.name, following))
            following = next(arrivals, None)
        due, _, job = heapq.heappop(queue)
        now += cost(job)
        if now > due - reserve_increase:
            return False
    return True


def audit(jobs, cutoff, estimates, nominal, candidates, reserve_history):
    samples, empirical = estimates
    predicted_cost = nominal | empirical
    fastest_profile = min(nominal[model] for model in candidates)

    def cost(job):
        return job.service or predicted_cost[job.model]

    reserve_stamps, reserves = zip(*reserve_history)

    def extra_at(stamp):
        return reserves[bisect_right(reserve_stamps, stamp) - 1] - reserves[0]

    admitted = sorted((job for job in jobs if job.admitted and job.ready is not None and job.due is not None),
                      key=lambda job: job.ready)
    ready_times = [job.ready for job in admitted]
    calendar = sorted(jobs, key=lambda job: job.observed[0])
    release_times = [job.observed[0] for job in calendar]
    max_span = max(job.finished - job.observed[0] for job in calendar)
    counts, forecast_counts = Counter(), Counter()
    cases, pair_misses, prediction_error = [], [], defaultdict(list)
    rejected = Counter()
    for job in admitted:
        if job.ready < cutoff or job.status != "dropped_inference":
            continue
        rejected["admitted_then_rejected"] += 1
        rejected[f"logged/{job.rejection or 'absolute_deadline_or_cancelled'}"] += 1
        if job.ready + fastest_profile > job.due - extra_at(job.ready):
            rejected["no_model_fits_at_ready"] += 1
        elif job.dequeue is not None and job.dequeue + fastest_profile > job.due - extra_at(job.dequeue):
            rejected["fit_at_ready_only"] += 1
        else:
            rejected["fits_at_rejection"] += 1
    donors = [job for job in jobs if job.service and job.model == SPLIT_MODEL and job.dequeue >= cutoff]
    for donor in sorted(donors, key=lambda job: job.dequeue):
        now = donor.dequeue
        reserve_increase = extra_at(now)
        counts["donor_starts"] += 1
        local = admitted[bisect_left(ready_times, now-max_span):bisect_right(ready_times, now+WINDOW_US)]
        local = [j for j in local if j != donor and (j.dequeue is None or j.dequeue >= now)
                 and j.finished >= now]
        whole_end = now + cost(donor)
        whole_ok = whole_end <= donor.due - reserve_increase and check_tail(whole_end, local, cost, reserve_increase)
        positives = set()
        for urgent in local:
            if not now < urgent.ready < now + cost(donor):
                continue
            counts["within_whole_block"] += 1
            actions, dues = pair(now, urgent.ready, cost(donor), cost(urgent),
                                 donor.due - reserve_increase, urgent.due - reserve_increase)
            record = {"donor": donor.name, "urgent": urgent.name, "start_us": now,
                      "urgent_ready_us": urgent.ready, "urgent_status": urgent.status,
                      "donor_service_us": cost(donor), "donor_dnndk_run_wall_us": donor.run_wall,
                      "due_us": dues, "finish_us": actions}
            defer_fits = fits(actions["defer"], dues)
            split_fits = fits(actions["split"], dues)
            if not fits(actions["whole"], dues):
                counts["whole_pair_miss"] += 1
                pair_misses.append(record)
                if defer_fits:
                    classification = "pair_defer_fits"
                elif split_fits:
                    classification = "pair_split_only"
                elif urgent.ready + fastest_profile > dues[1]:
                    classification = "urgent_too_late_even_fastest_profile"
                elif urgent.ready + cost(urgent) > dues[1]:
                    classification = "urgent_too_late_at_frozen_model"
                elif actions["split"][1] > urgent.due:
                    classification = "prefix_still_blocks_urgent"
                else:
                    classification = "donor_misses_after_resume"
                record["classification"] = classification
                counts[classification] += 1
            if whole_ok:
                continue
            counts["whole_local_miss_candidates"] += 1
            others = [job for job in local if job != urgent]
            if defer_fits and check_tail(max(actions["defer"]), others, cost, reserve_increase):
                counts["local_defer_fits"] += 1
                continue
            if not split_fits or not check_tail(max(actions["split"]), others, cost, reserve_increase):
                continue
            counts["local_split_only"] += 1
            positives.add(urgent.name)
            cases.append(record)

        active = calendar[bisect_left(release_times, now-max_span):bisect_right(release_times, now+WINDOW_US)]
        for job in active:
            if job == donor or job.finished <= now or (job.ready is not None and job.ready <= now):
                continue
            if job.observed[1] is None or job.observed[1] > now:
                forecast_counts["calendar_only"] += 1
            else:
                forecast_counts["visible_preparing"] += 1
            for q in (.5, .95):
                forecast = estimate(job, now, samples, q)
                key = f"q{round(q * 100)}"
                if forecast is None:
                    forecast_counts[f"{key}_insufficient_samples"] += 1
                    continue
                stage, ready = forecast
                if job.ready is not None:
                    prediction_error[(key, STAGES[stage])].append(ready - job.ready)
                donor_cost = predicted_cost[donor.model]
                if ready >= now + donor_cost:
                    continue
                forecast_counts[f"{key}_predicted_within_block"] += 1
                if job.admitted and job.ready is not None and job.ready < now + cost(donor):
                    forecast_counts[f"{key}_observed_within_block"] += 1
                due = job.due if stage == len(STAGES) - 1 and job.due is not None else job.slot_due
                actions, dues = pair(now, ready, donor_cost, predicted_cost[job.release_model],
                                     donor.due - reserve_increase, due - reserve_increase)
                if fits(actions["whole"], dues) or fits(actions["defer"], dues) or not fits(actions["split"], dues):
                    continue
                forecast_counts[f"{key}_suggested_split"] += 1
                if job.name in positives:
                    forecast_counts[f"{key}_matched_local_split_only"] += 1
    return {"statuses": Counter(job.status for job in jobs), "counts": counts, "forecasts": forecast_counts,
            "admitted_rejections": rejected, "fastest_nominal_cost_us": fastest_profile,
            "prediction_error_us": {f"{key}/{stage}": {"n": len(values),
                                    "absolute_median": quantile([abs(v) for v in values], .5),
                                    "absolute_p95": quantile([abs(v) for v in values], .95),
                                    "later_than_forecast_fraction": sum(v < 0 for v in values) / len(values)}
                                    for (key, stage), values in sorted(prediction_error.items())},
            "examples": cases[:30], "whole_pair_misses": pair_misses[:100]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = iter(args.trial)
    first = next(trials)
    jobs, nominal, candidates, sha, reserve, floor, modes = read_trial(first)
    cutoff = min(job.observed[0] for job in jobs if job.observed[0] is not None) + 8_000_000
    estimates = learn(jobs, cutoff)
    report = {"window_us": WINDOW_US, "training_trial": str(first), "training_end_us": cutoff,
              "learned_cost_us_p99": estimates[1], "trials": []}
    for trial in [first, *trials]:
        test_start = cutoff if trial == first else 0
        if trial != first:
            jobs, nominal, candidates, sha, reserve, floor, modes = read_trial(trial)
        fixed = audit(jobs, test_start, estimates, nominal, candidates, reserve[:1])
        dynamic = audit(jobs, test_start, estimates, nominal, candidates, reserve)
        report["trials"].append({"trial": str(trial), "trace_sha256": sha, "reserve_history_us": reserve,
                                 "fastest_model_history_us": floor, "final_candidate_costs_us": {i: modes[i] for i in candidates},
                                 "logged_model_rejections_full_trial": audit_rejection_floor(jobs, reserve, floor),
                                 "fixed_reserve": fixed, "reconstructed_reserve": dynamic})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
