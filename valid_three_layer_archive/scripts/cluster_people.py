"""
基于三层 topic 能力标签对人物聚类。

默认读取 extraction_results/new_data/*.json 中的:
  - outer_layer.signal_topics
  - middle_layer.signal_topics
  - inner_layer.signal_topics
输出到 person_cluster_results/person_clusters。
"""

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_DATA = PROJECT_ROOT / "extraction_results" / "new_data"
OUT_DIR = PROJECT_ROOT / "person_cluster_results" / "person_clusters"
LAYERS = ["outer_layer", "middle_layer", "inner_layer"]


def load_person_profiles(new_data: Path = NEW_DATA) -> tuple[list[tuple[str, dict[str, int]]], list[str]]:
    all_topic_names = set()
    person_profiles = []

    for fp in sorted(new_data.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        profile = Counter()
        for layer_key in LAYERS:
            topics = data.get(layer_key, {}).get("signal_topics", [])
            for topic in topics:
                if topic:
                    profile[topic] += 1
        if profile:
            person_profiles.append((fp.stem, dict(profile)))
            all_topic_names.update(profile.keys())

    return person_profiles, sorted(all_topic_names)


def build_matrix(person_profiles: list[tuple[str, dict[str, int]]], topic_list: list[str]):
    import numpy as np

    matrix = np.zeros((len(person_profiles), len(topic_list)))
    names = []
    for i, (name, profile) in enumerate(person_profiles):
        names.append(name)
        for j, topic in enumerate(topic_list):
            matrix[i, j] = profile.get(topic, 0)
    return names, matrix


def run(new_data: Path = NEW_DATA, out_dir: Path = OUT_DIR, best_k: int = 5) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    person_profiles, topic_list = load_person_profiles(new_data)
    print(f"{len(person_profiles)} 人, {len(topic_list)} 个能力标签")
    if len(person_profiles) < 3:
        raise RuntimeError("可聚类样本少于 3 个，请先完成更多三层提取和 topic 回标。")
    if len(topic_list) < 2:
        raise RuntimeError("能力标签少于 2 个，请先检查 signal_topics 回标结果。")

    names, matrix = build_matrix(person_profiles, topic_list)
    x_raw = StandardScaler().fit_transform(matrix)

    print("\n--- UMAP 降维到 5D ---")
    umap_5d = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    x_5d = umap_5d.fit_transform(x_raw)

    print("\n--- 轮廓系数选 K ---")
    max_k = min(12, len(person_profiles) - 1)
    k_range = range(2, max_k + 1)
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(x_5d)
        sil = silhouette_score(x_5d, labels)
        silhouettes.append(sil)
        print(f"  K={k:>2}: silhouette={sil:.4f}")

    best_k = min(best_k, max_k)
    print(f"\n使用 K = {best_k}")

    plt.figure(figsize=(12, 7))
    plt.plot(list(k_range), silhouettes, "bo-", markersize=8, linewidth=2)
    plt.axvline(best_k, color="red", linestyle="--", label=f"Chosen K={best_k}")
    plt.title("Silhouette Score vs K", fontsize=16)
    plt.xlabel("K", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
    plt.xticks(list(k_range))
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(str(out_dir / "silhouette.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n--- 最终聚类 (K={best_k}) ---")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(x_5d)

    umap_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
    x_2d = umap_2d.fit_transform(x_raw)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    for c in range(best_k):
        mask = labels == c
        plt.scatter(
            x_2d[mask, 0],
            x_2d[mask, 1],
            c=[colors[c]],
            label=f"簇{c} ({mask.sum()}人)",
            s=50,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )
    plt.title(f"Person Clusters (K={best_k}) - UMAP 2D", fontsize=16)
    plt.legend(fontsize=11, loc="lower left", ncol=2)
    plt.tight_layout()
    plt.savefig(str(out_dir / "person_umap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    cluster_profiles = []
    for c in range(best_k):
        mask = labels == c
        cluster_profiles.append(matrix[mask].mean(axis=0))

    df_profile = pd.DataFrame(cluster_profiles, columns=topic_list)
    df_profile.index = [f"簇{c}" for c in range(best_k)]

    print("\n--- 各簇能力画像 ---")
    for c in range(best_k):
        n = int((labels == c).sum())
        print(f"\n簇{c} ({n}人):")
        top = df_profile.iloc[c].sort_values(ascending=False).head(5)
        for topic, val in top.items():
            bar = "#" * int(val * 5)
            print(f"  {topic:<12} {val:>5.2f}  {bar}")

    df_profile.to_csv(str(out_dir / "cluster_profiles.csv"), encoding="utf-8-sig")
    pd.DataFrame({"name": names, "cluster": labels}).to_csv(
        str(out_dir / "person_assignments.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    for c in range(best_k):
        members = [names[i] for i in range(len(names)) if labels[i] == c]
        suffix = "..." if len(members) > 15 else ""
        print(f"\n簇{c} 成员 ({len(members)}人): {', '.join(members[:15])}{suffix}")

    print(f"\n结果保存至 {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于三层 topic 标签进行人才聚类")
    parser.add_argument("--new-data", type=Path, default=NEW_DATA, help="已回标 JSON 目录")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="聚类结果输出目录")
    parser.add_argument("--best-k", type=int, default=5, help="最终聚类簇数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.new_data, args.out_dir, args.best_k)
