#!/usr/bin/env python3

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


START_RE = re.compile(
    r"session (?P<session_id>\S+) start:.*?"
    r"sample_index=(?P<sample_index>\d+)\s+"
    r"session_index=(?P<session_index>\d+)\s+"
    r"global_steps=(?P<global_steps>\S+)"
)
SESSION_ID_RE = re.compile(r"session-sample-\d+-rollout-\d+-[0-9a-fA-F]+")
MARKER = "[ROLLOUT_SAMPLE]"


@dataclass
class SessionStart:
    session_id: str
    sample_index: int
    session_index: int
    global_steps: str
    log_path: str
    start_line: int


def parse_logs(paths: list[str]):
    starts: dict[str, SessionStart] = {}
    completed: dict[str, list[dict]] = {}
    last_activity: dict[str, tuple[str, int, str]] = {}
    malformed_records: list[tuple[str, int, str]] = []

    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                text = line.rstrip("\r\n")

                start_match = START_RE.search(text)
                if start_match:
                    session_id = start_match.group("session_id")
                    starts[session_id] = SessionStart(
                        session_id=session_id,
                        sample_index=int(start_match.group("sample_index")),
                        session_index=int(start_match.group("session_index")),
                        global_steps=start_match.group("global_steps"),
                        log_path=path,
                        start_line=line_number,
                    )

                for session_id in SESSION_ID_RE.findall(text):
                    last_activity[session_id] = (
                        path,
                        line_number,
                        text[-500:],
                    )

                if MARKER not in text:
                    continue
                payload = text.split(MARKER, 1)[1].lstrip()
                try:
                    record, _ = json.JSONDecoder().raw_decode(payload)
                except json.JSONDecodeError as exc:
                    malformed_records.append((path, line_number, str(exc)))
                    continue
                session_id = str(record.get("session_id", ""))
                if session_id:
                    completed.setdefault(session_id, []).append(record)

    return starts, completed, last_activity, malformed_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find rollout sessions that started but emitted no ROLLOUT_SAMPLE"
    )
    parser.add_argument("--log", nargs="+", required=True)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--output", default="incomplete_rollouts.csv")
    args = parser.parse_args()

    if args.rollout_n <= 0:
        parser.error("--rollout-n must be greater than zero")

    starts, completed, last_activity, malformed = parse_logs(args.log)
    if not starts:
        raise ValueError(
            "No session start records were found. The log must include framework INFO lines "
            "containing 'session ... start: ... sample_index=...'."
        )

    group_instance_ids: dict[tuple[str, int], set[str]] = {}
    for session_id, records in completed.items():
        start = starts.get(session_id)
        if start is None:
            continue
        instance_ids = {
            str(record.get("instance_id", ""))
            for record in records
            if str(record.get("instance_id", ""))
        }
        group_instance_ids.setdefault(
            (start.global_steps, start.sample_index), set()
        ).update(instance_ids)

    rows = []
    for session_id, start in starts.items():
        if session_id in completed:
            continue
        group_key = (start.global_steps, start.sample_index)
        instance_ids = sorted(group_instance_ids.get(group_key, set()))
        activity = last_activity.get(session_id)
        rows.append(
            {
                "instance_id": instance_ids[0] if len(instance_ids) == 1 else "",
                "instance_id_status": (
                    "mapped"
                    if len(instance_ids) == 1
                    else "unknown"
                    if not instance_ids
                    else "ambiguous"
                ),
                "global_steps": start.global_steps,
                "sample_index": start.sample_index,
                "session_index": start.session_index,
                "session_id": session_id,
                "start_log": start.log_path,
                "start_line": start.start_line,
                "last_log": activity[0] if activity else start.log_path,
                "last_line": activity[1] if activity else start.start_line,
                "last_activity": activity[2] if activity else "session start",
            }
        )

    rows.sort(
        key=lambda row: (
            str(row["global_steps"]),
            int(row["sample_index"]),
            int(row["session_index"]),
        )
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_id",
        "instance_id_status",
        "global_steps",
        "sample_index",
        "session_index",
        "session_id",
        "start_log",
        "start_line",
        "last_log",
        "last_line",
        "last_activity",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    started_groups: dict[tuple[str, int], set[int]] = {}
    completed_groups: dict[tuple[str, int], set[int]] = {}
    for session_id, start in starts.items():
        key = (start.global_steps, start.sample_index)
        started_groups.setdefault(key, set()).add(start.session_index)
        if session_id in completed:
            completed_groups.setdefault(key, set()).add(start.session_index)

    incomplete_groups = 0
    expected_indices = set(range(args.rollout_n))
    for key in started_groups:
        if completed_groups.get(key, set()) != expected_indices:
            incomplete_groups += 1

    print(f"started_sessions={len(starts)}")
    print(f"completed_sessions={len(set(starts) & set(completed))}")
    print(f"incomplete_sessions={len(rows)}")
    print(f"incomplete_sample_groups={incomplete_groups}")
    print(f"malformed_rollout_records={len(malformed)}")
    print(f"output={args.output}")
    for row in rows:
        print(
            f"instance_id={row['instance_id'] or 'UNKNOWN'} "
            f"global_steps={row['global_steps']} "
            f"sample_index={row['sample_index']} "
            f"session_index={row['session_index']} "
            f"session_id={row['session_id']} "
            f"last={row['last_log']}:{row['last_line']}"
        )


if __name__ == "__main__":
    main()
