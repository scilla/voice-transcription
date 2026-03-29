from __future__ import annotations

import argparse
import time
from typing import Iterable

from cill.app import hydrate_state, process_job, storage


def iter_pending_states(job_id: str | None = None) -> Iterable[dict]:
    if job_id:
        state = storage.load_state(job_id)
        if state:
            yield hydrate_state(state)
        return

    states = storage.list_states()
    states.sort(key=lambda item: (item.get("created_at") or "", item.get("updated_at") or ""))
    for state in states:
        hydrated = hydrate_state(state)
        if hydrated["status"] != "queued" and state.get("status") != "queued":
            continue
        yield hydrated


def process_pending_jobs(job_id: str | None = None) -> int:
    processed = 0
    for state in iter_pending_states(job_id=job_id):
        processed += 1
        print(f"Processing queued job {state['job_id']} for {state['source_url']}")
        result = process_job(state)
        print(f"Finished {state['job_id']} with status={result['status']}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process queued cill.app transcript jobs from the configured storage backend.",
    )
    parser.add_argument("--job-id", help="Process a single queued job by ID.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep polling for queued jobs instead of exiting after one pass.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=10.0,
        help="Polling interval when --loop is enabled.",
    )
    args = parser.parse_args()

    if args.loop:
        while True:
            processed = process_pending_jobs(job_id=args.job_id)
            if processed == 0:
                print("No queued jobs found.")
            time.sleep(max(args.interval_seconds, 1.0))
    else:
        processed = process_pending_jobs(job_id=args.job_id)
        if processed == 0:
            print("No queued jobs found.")


if __name__ == "__main__":
    main()
