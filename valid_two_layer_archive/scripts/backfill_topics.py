"""将 topic_names_suggested.csv 中的簇名称回标到 new_data/*.json"""
import json
import csv
from pathlib import Path

NEW_DATA = Path("./new_data")

# 1. 读取命名表: (layer, topic_id) → name
name_map = {}  # key: ("outer"|"inner", topic_id_int) → name_str
with open("topic_names_suggested.csv", "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        layer = row["layer"].strip()
        tid = int(row["topic"])
        name_map[(layer, tid)] = row["suggested_name"].strip()

# 2. 读取 assignments: signal_text → topic_id (两层分别)
signal_topic = {}  # key: signal_text → (layer, topic_id)
for layer_key, csv_path in [("outer", "outer/outer_layer_assignments.csv"),
                             ("inner", "inner/inner_layer_assignments.csv")]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            doc = row["Doc"].strip()
            tid = int(row["Topic"])
            signal_topic[(layer_key, doc)] = tid

# 3. 回标到每个 JSON
updated = 0
for fp in sorted(NEW_DATA.glob("*.json")):
    data = json.loads(fp.read_text(encoding="utf-8"))

    for layer_key in ["outer_layer", "inner_layer"]:
        layer_data = data.get(layer_key, {})
        signals = layer_data.get("signals", [])
        if not signals:
            continue
        short_key = "outer" if layer_key == "outer_layer" else "inner"
        topic_names = []
        for s in signals:
            s_clean = s.strip()
            tid = signal_topic.get((short_key, s_clean))
            tname = name_map.get((short_key, tid), "") if tid is not None else ""
            topic_names.append(tname)
        layer_data["signal_topics"] = topic_names

    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    updated += 1

print(f"Done. Updated {updated} files.")
