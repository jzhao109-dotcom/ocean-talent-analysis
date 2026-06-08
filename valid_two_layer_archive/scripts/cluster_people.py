"""
基于 topic 能力标签对人物聚类 —— UMAP 降维 + KMeans + 轮廓系数选K
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from umap import UMAP
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NEW_DATA = Path("./new_data")
OUT_DIR = Path("./person_clusters")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
all_topic_names = set()
person_profiles = []

for fp in sorted(NEW_DATA.glob("*.json")):
    data = json.loads(fp.read_text(encoding="utf-8"))
    name = fp.stem
    profile = Counter()
    for layer_key in ["outer_layer", "inner_layer"]:
        topics = data.get(layer_key, {}).get("signal_topics", [])
        for t in topics:
            if t:
                profile[t] += 1
    if profile:
        person_profiles.append((name, dict(profile)))
        all_topic_names.update(profile.keys())

topic_list = sorted(all_topic_names)
print(f"{len(person_profiles)} 人, {len(topic_list)} 个能力标签")

# 构建矩阵
matrix = np.zeros((len(person_profiles), len(topic_list)))
names = []
for i, (name, profile) in enumerate(person_profiles):
    names.append(name)
    for j, topic in enumerate(topic_list):
        matrix[i, j] = profile.get(topic, 0)

# 标准化
X_raw = StandardScaler().fit_transform(matrix)

# ============================================================
# 2. UMAP 降维到 5 维 → 再做聚类
# ============================================================
print("\n--- UMAP 降维 (19D → 5D) ---")
umap_5d = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
X_5d = umap_5d.fit_transform(X_raw)

# ============================================================
# 3. 轮廓系数选 K
# ============================================================
print("\n--- 轮廓系数选 K ---")
K_range = range(2, 13)
silhouettes = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_5d)
    sil = silhouette_score(X_5d, labels)
    silhouettes.append(sil)
    print(f"  K={k:>2}: silhouette={sil:.4f}")

best_k = 5  # 手动指定，避免簇数过多等同标签数
print(f"\n使用 K = {best_k}")

plt.figure(figsize=(12, 7))
plt.plot(K_range, silhouettes, "bo-", markersize=8, linewidth=2)
plt.axvline(best_k, color="red", linestyle="--", label=f"Chosen K={best_k}")
plt.title("Silhouette Score vs K", fontsize=16)
plt.xlabel("K", fontsize=14)
plt.ylabel("Silhouette Score", fontsize=14)
plt.xticks(K_range)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "silhouette.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 4. 用最优 K 做最终聚类
# ============================================================
print(f"\n--- 最终聚类 (K={best_k}) ---")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_5d)

# UMAP → 2D 可视化
umap_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
X_2d = umap_2d.fit_transform(X_raw)

plt.figure(figsize=(12, 10))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for c in range(best_k):
    mask = labels == c
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=[colors[c]], label=f"簇{c} ({mask.sum()}人)",
                s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
plt.title(f"Person Clusters (K={best_k}) — UMAP 2D", fontsize=16)
plt.legend(fontsize=11, loc="lower left", ncol=2)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "person_umap.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 5. 每簇能力画像
# ============================================================
print(f"\n--- 各簇能力画像 ---")
cluster_profiles = []
for c in range(best_k):
    mask = labels == c
    avg = matrix[mask].mean(axis=0)
    cluster_profiles.append(avg)

df_profile = pd.DataFrame(cluster_profiles, columns=topic_list)
df_profile.index = [f"簇{c}" for c in range(best_k)]

for c in range(best_k):
    n = (labels == c).sum()
    print(f"\n簇{c} ({n}人):")
    top = df_profile.iloc[c].sort_values(ascending=False).head(5)
    for topic, val in top.items():
        bar = "█" * int(val * 5)
        print(f"  {topic:<12} {val:>5.2f}  {bar}")

# ============================================================
# 6. 输出
# ============================================================
df_profile.to_csv(str(OUT_DIR / "cluster_profiles.csv"), encoding="utf-8-sig")
pd.DataFrame({"name": names, "cluster": labels}).to_csv(
    str(OUT_DIR / "person_assignments.csv"), index=False, encoding="utf-8-sig"
)

for c in range(best_k):
    members = [names[i] for i in range(len(names)) if labels[i] == c]
    print(f"\n簇{c} 成员 ({len(members)}人): {', '.join(members[:15])}{'...' if len(members) > 15 else ''}")

print(f"\n结果保存至 {OUT_DIR.resolve()}")
