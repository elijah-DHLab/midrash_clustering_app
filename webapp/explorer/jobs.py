"""Runs a clustering in the background, so the page can show where it is.

A request that computes for two minutes and says nothing is indistinguishable from
a hung one. The work is therefore done in a thread that records which stage it is
in, and the page polls for that; the estimate below turns the stage plus the size
of the selection into a horizon, which is what makes a progress bar worth drawing.

The estimates are calibrated on this machine — 9,300 chunks at k=8 took 6.5s end to
end, and they scale with chunks x k. They are used only for the bar; the stages
themselves are reported as they actually happen.
"""
import threading
import time
import uuid

# job id -> {stage, stage_index, stages, started, chunks, k, estimate, done, error, token}
JOBS = {}
_LOCK = threading.Lock()

# Seconds per chunk per unit of k, measured stage by stage.
STAGE_COST = {
    "טעינת הצ'אנקים":        0.00004,
    "קלאסטרינג":             0.00045,
    "שיוך לקלאסטרים":        0.00018,
    "צירופי מילים אופייניים": 0.00012,
    "נקודות שינוי":          0.00004,
    "ציור":                  0.00030,
}
STAGES = list(STAGE_COST)


def estimate(chunks, k, with_cpd=True):
    """Seconds the whole run should take, and the cumulative fraction of each stage."""
    costs = [STAGE_COST[s] * chunks * max(k, 1) / 8 for s in STAGES]
    if not with_cpd:
        costs[STAGES.index("change points")] = 0.0
    total = sum(costs) + 0.5
    cum, acc = [], 0.0
    for c in costs:
        acc += c
        cum.append(acc / total if total else 1.0)
    return total, cum


def create(chunks, k, with_cpd):
    job_id = uuid.uuid4().hex[:12]
    total, cum = estimate(chunks, k, with_cpd)
    with _LOCK:
        JOBS[job_id] = {
            "stage": STAGES[0], "stage_index": 0, "stages": STAGES,
            "started": time.time(), "chunks": chunks, "k": k,
            "estimate": total, "cumulative": cum, "done": False, "error": None,
            "token": None, "summary": None,
        }
    return job_id


def set_stage(job_id, stage):
    with _LOCK:
        job = JOBS.get(job_id)
        if job:
            job["stage"] = stage
            job["stage_index"] = STAGES.index(stage) if stage in STAGES else job["stage_index"]


def finish(job_id, token, summary):
    with _LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(done=True, token=token, summary=summary,
                       seconds=round(time.time() - job["started"], 1))


def fail(job_id, message):
    with _LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(done=True, error=message,
                       seconds=round(time.time() - job["started"], 1))


def status(job_id):
    """What the page needs to draw the bar: stage, elapsed, and a horizon."""
    with _LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        elapsed = time.time() - job["started"]
        floor = job["cumulative"][job["stage_index"] - 1] if job["stage_index"] else 0.0
        ceiling = job["cumulative"][job["stage_index"]]
        # Inside a stage, move from its floor toward its ceiling on the clock, but never
        # past it: the stage boundaries are the facts, the rest is interpolation.
        within = min(max((elapsed / job["estimate"]) if job["estimate"] else 1.0, floor), ceiling)
        return {
            "done": job["done"], "error": job["error"], "token": job["token"],
            "stage": job["stage"], "stage_index": job["stage_index"],
            "stages": job["stages"], "elapsed": round(elapsed, 1),
            "estimate": round(job["estimate"], 1),
            # Past the estimate the honest answer is that we no longer know, so the
            # page is told to say so rather than to keep counting down to zero.
            "remaining": max(0, round(job["estimate"] - elapsed)),
            "overrun": elapsed > job["estimate"],
            "fraction": round(1.0 if job["done"] else within, 3),
            "chunks": job["chunks"],
        }
