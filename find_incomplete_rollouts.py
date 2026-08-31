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

    session_rows = []
    for session_id, start in starts.items():
        if session_id in completed:
            continue
        group_key = (start.global_steps, start.sample_index)
        instance_ids = sorted(group_instance_ids.get(group_key, set()))
        activity = last_activity.get(session_id)
        session_rows.append(
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

    session_rows.sort(
        key=lambda row: (
            str(row["global_steps"]),
            int(row["sample_index"]),
            int(row["session_index"]),
        )
    )

    session_fieldnames = [
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
    started_groups: dict[tuple[str, int], set[int]] = {}
    completed_groups: dict[tuple[str, int], set[int]] = {}
    group_session_ids: dict[tuple[str, int], list[str]] = {}
    for session_id, start in starts.items():
        key = (start.global_steps, start.sample_index)
        started_groups.setdefault(key, set()).add(start.session_index)
        group_session_ids.setdefault(key, []).append(session_id)
        if session_id in completed:
            completed_groups.setdefault(key, set()).add(start.session_index)

    expected_indices = set(range(args.rollout_n))
    group_rows = []
    for key, started_indices in started_groups.items():
        completed_indices = completed_groups.get(key, set())
        incomplete_session_ids = [
            session_id
            for session_id in group_session_ids[key]
            if session_id not in completed
        ]
        if completed_indices == expected_indices and not incomplete_session_ids:
            continue
        instance_ids = sorted(group_instance_ids.get(key, set()))
        group_rows.append(
            {
                "instance_id": instance_ids[0] if len(instance_ids) == 1 else "",
                "instance_id_status": (
                    "mapped"
                    if len(instance_ids) == 1
                    else "unknown"
                    if not instance_ids
                    else "ambiguous"
                ),
                "global_steps": key[0],
                "sample_index": key[1],
                "started_session_indices": ",".join(map(str, sorted(started_indices))),
                "completed_session_indices": ",".join(map(str, sorted(completed_indices))),
                "missing_session_indices": ",".join(
                    map(str, sorted(expected_indices - completed_indices))
                ),
                "started_session_count": len(group_session_ids[key]),
                "completed_session_count": sum(
                    session_id in completed for session_id in group_session_ids[key]
                ),
                "incomplete_session_ids": ",".join(incomplete_session_ids),
            }
        )

    group_rows.sort(
        key=lambda row: (str(row["global_steps"]), int(row["sample_index"]))
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_fieldnames = [
        "instance_id",
        "instance_id_status",
        "global_steps",
        "sample_index",
        "started_session_indices",
        "completed_session_indices",
        "missing_session_indices",
        "started_session_count",
        "completed_session_count",
        "incomplete_session_ids",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=group_fieldnames)
        writer.writeheader()
        writer.writerows(group_rows)

    session_output = output_path.with_name(
        f"{output_path.stem}_sessions{output_path.suffix or '.csv'}"
    )
    with open(session_output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=session_fieldnames)
        writer.writeheader()
        writer.writerows(session_rows)

    print(f"started_sessions={len(starts)}")
    print(f"completed_sessions={len(set(starts) & set(completed))}")
    print(f"incomplete_sessions={len(session_rows)}")
    print(f"incomplete_instances={len(group_rows)}")
    print(f"malformed_rollout_records={len(malformed)}")
    print(f"output={args.output}")
    print(f"session_output={session_output}")
    for row in group_rows:
        print(
            f"instance_id={row['instance_id'] or 'UNKNOWN'} "
            f"global_steps={row['global_steps']} "
            f"sample_index={row['sample_index']} "
            f"completed={row['completed_session_indices'] or '-'} "
            f"missing={row['missing_session_indices'] or '-'}"
        )


if __name__ == "__main__":
    main()
