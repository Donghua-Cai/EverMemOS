"""
Preprocess random LongMemEval variants into Locomo-style JSON.

Input (per file): one LongMemEval-like sample dict with message-level `time`.
Output (single file): Locomo-style list[dict], preserving message order.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


RAW_TIME_FMT = "%Y/%m/%d (%a) %H:%M"


def parse_raw_time(time_str: str) -> datetime:
    """Parse random variant time string, e.g. '2023/05/20 (Sat) 02:21'."""
    return datetime.strptime(time_str.strip(), RAW_TIME_FMT)


def to_locomo_session_time(dt: datetime) -> str:
    """Convert datetime to LoCoMo session timestamp format."""
    hour = str(int(dt.strftime("%I")))
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p").lower()
    day = str(dt.day)
    month = dt.strftime("%B")
    year = str(dt.year)
    return f"{hour}:{minute} {ampm} on {day} {month}, {year}"


def to_iso_seconds(dt: datetime) -> str:
    """Convert datetime to ISO format with seconds."""
    return dt.isoformat(timespec="seconds")


def build_evidence(sample: dict[str, Any]) -> list[str]:
    """Build evidence list in D{session}:{msg} format."""
    evidence_session_idx: list[int] = []
    for idx, session_id in enumerate(sample["haystack_session_ids"]):
        if session_id in sample["answer_session_ids"]:
            evidence_session_idx.append(idx)

    evidence: list[str] = []
    for idx, session in enumerate(sample["haystack_sessions"]):
        if idx in evidence_session_idx:
            for i, _ in enumerate(session):
                evidence.append(f"D{idx}:{i}")
    return evidence


def convert_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one LongMemEval-like sample to Locomo style."""
    question_id = sample["question_id"]

    item: dict[str, Any] = {"qa": [], "conversation": {}}
    item["qa"].append(
        {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "evidence": build_evidence(sample),
            "category": sample["question_type"],
        }
    )

    conv = item["conversation"]
    conv["speaker_a"] = f"user_{question_id}"
    conv["speaker_b"] = f"assistant_{question_id}"

    for idx, session in enumerate(sample["haystack_sessions"]):
        # Keep session-level timestamp for compatibility/readability.
        # Prefer haystack_dates[idx], fallback to first message time.
        session_time_raw = None
        if idx < len(sample.get("haystack_dates", [])):
            session_time_raw = sample["haystack_dates"][idx]
        elif session and session[0].get("time"):
            session_time_raw = session[0]["time"]

        if session_time_raw:
            session_dt = parse_raw_time(session_time_raw)
            conv[f"session_{idx}_date_time"] = to_locomo_session_time(session_dt)
        else:
            conv[f"session_{idx}_date_time"] = "Unknown"

        session_out: list[dict[str, Any]] = []
        for msg_idx, msg in enumerate(session):
            msg_time_raw = msg.get("time")
            if not msg_time_raw:
                raise ValueError(
                    f"Missing message time: question_id={question_id}, session={idx}, msg={msg_idx}"
                )
            msg_dt = parse_raw_time(msg_time_raw)
            session_out.append(
                {
                    "speaker": f"{msg['role']}_{question_id}",
                    "text": msg["content"],
                    "dia_id": f"D{idx}:{msg_idx}",
                    # Critical: preserve each message's own timestamp.
                    "time": to_iso_seconds(msg_dt),
                }
            )
        conv[f"session_{idx}"] = session_out

    return item


def convert_dir(input_dir: Path, output_file: Path) -> tuple[int, int]:
    """Convert all sample JSON files in a directory to one Locomo-style file."""
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    records: list[dict[str, Any]] = []
    skipped = 0
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Skip already-converted aggregate files (list at top level).
        if isinstance(data, list):
            skipped += 1
            continue
        if not isinstance(data, dict) or "haystack_sessions" not in data:
            skipped += 1
            continue

        records.append(convert_sample(data))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return len(records), skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess random LongMemEval variants to Locomo style."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory containing per-sample JSON files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output Locomo-style JSON file path.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert both random_merge and random_bucket_K5 using default paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_data_root = Path(__file__).resolve().parents[1] / "data"

    tasks: list[tuple[Path, Path]] = []
    if args.all:
        tasks.extend(
            [
                (
                    eval_data_root / "random_merge",
                    eval_data_root / "random_merge" / "random_merge_locomo_style.json",
                ),
                (
                    eval_data_root / "random_bucket_K5",
                    eval_data_root
                    / "random_bucket_K5"
                    / "random_bucket_K5_locomo_style.json",
                ),
            ]
        )
    else:
        if not args.input_dir or not args.output_file:
            raise ValueError("Use --all, or provide both --input-dir and --output-file.")
        tasks.append((args.input_dir, args.output_file))

    for input_dir, output_file in tasks:
        count, skipped = convert_dir(input_dir, output_file)
        print(f"✅ Converted {count} samples -> {output_file}")
        if skipped:
            print(f"   ℹ️ Skipped {skipped} non-sample JSON file(s)")


if __name__ == "__main__":
    main()
