"""将三层 topic 名称回写到 extraction_results/new_data/*.json。"""

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_DATA = PROJECT_ROOT / "extraction_results" / "new_data"
TOPIC_ROOT = PROJECT_ROOT / "topic_results"

LAYER_CONFIG = [
    ("outer", "outer_layer"),
    ("middle", "middle_layer"),
    ("inner", "inner_layer"),
]


def read_name_map(topic_root: Path) -> dict[tuple[str, int], str]:
    name_map = {}
    path = topic_root / "topic_names_suggested.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            layer = row["layer"].strip()
            topic_id = int(row["topic"])
            name_map[(layer, topic_id)] = row["suggested_name"].strip()
    return name_map


def read_signal_topic_map(topic_root: Path) -> dict[tuple[str, str], int]:
    signal_topic = {}
    for short_layer, _ in LAYER_CONFIG:
        path = topic_root / short_layer / f"{short_layer}_layer_assignments.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                signal = row["Doc"].strip()
                topic_id = int(row["Topic"])
                signal_topic[(short_layer, signal)] = topic_id
    return signal_topic


def backfill_topics(new_data: Path = NEW_DATA, topic_root: Path = TOPIC_ROOT) -> int:
    name_map = read_name_map(topic_root)
    signal_topic = read_signal_topic_map(topic_root)

    updated = 0
    for fp in sorted(new_data.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))

        for short_layer, layer_key in LAYER_CONFIG:
            layer_data = data.get(layer_key, {})
            signals = layer_data.get("signals", [])
            topic_names = []
            for signal in signals:
                clean_signal = signal.strip()
                topic_id = signal_topic.get((short_layer, clean_signal))
                topic_name = name_map.get((short_layer, topic_id), "") if topic_id is not None else ""
                topic_names.append(topic_name)
            layer_data["signal_topics"] = topic_names

        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1

    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将三层 topic 名称回写到 JSON")
    parser.add_argument("--new-data", type=Path, default=NEW_DATA, help="待回写 JSON 目录")
    parser.add_argument("--topic-root", type=Path, default=TOPIC_ROOT, help="topic 结果根目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = backfill_topics(args.new_data, args.topic_root)
    print(f"Done. Updated {count} files.")
