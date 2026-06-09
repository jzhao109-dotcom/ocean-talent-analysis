# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Analysis of 209 Chinese marine science & technology leading talents' public interviews/biographies. Uses DeepSeek API to extract multi-layer competency signals, then BERTopic + UMAP + HDBSCAN for topic clustering and K-Means for person profiling. Two parallel analysis pipelines: two-layer (outer/inner) and three-layer (outer/middle/inner), each in their own archive.

## Repository structure

```
valid_two_layer_archive/          # Two-layer analysis (12 outer + 7 inner → 5 profiles)
  scripts/                        # extract (test.py), backfill, cluster
  raw_data/data/*.txt             # 209 interview texts
  extraction_results/new_data/    # Extracted JSON with signal_topics
  topic_results/                  # BERTopic outputs + visualizations
  person_cluster_results/         # K-Means cluster assignments + summary

valid_three_layer_archive/        # Three-layer analysis (14 + 6 + 7 → 8 profiles)
  scripts/                        # extract_three_layers.py, backfill_topics.py, cluster_people.py
  test.ipynb                      # BERTopic clustering notebook (elbow + BisectingKMeans)
  tests/                          # pytest tests for scripts and notebook
  raw_data/, extraction_results/, topic_results/, person_cluster_results/

literature_hr_theory/             # HR theory backing (competency modeling, AMO, human capital resources)
  pdfs/, metadata/sources.csv, metadata/references.bib
```

## Two-layer vs three-layer

- **Two-layer**: outer (observable marine practice) + inner (values/mission/drivers). Simpler, 5 person profiles.
- **Three-layer**: 蓝色胜任表征维度 (observable domain-specific professional capabilities) + 协同创新支撑维度 (KSAO methods/resources — "how things get done") + 深蓝精神内核维度 (values/drivers). 8 person profiles with finer granularity.
- **Three-layer is the primary/active pipeline.** The 协同创新支撑维度 is the key differentiator, bridging surface competencies and deep drivers.

## Pipeline workflow

1. **Extract signals** — `extract_three_layers.py` calls DeepSeek API to extract outer/middle/inner signals from `.txt` → `.json`
2. **Topic modeling** — `test.ipynb` runs BERTopic (BisectingKMeans) + elbow method on each layer, producing topic assignments and `topic_names_suggested.csv`
3. **Backfill** — `backfill_topics.py` writes topic names into each JSON's `signal_topics` field
4. **Person clustering** — `cluster_people.py` aggregates topic profiles per person, runs K-Means with silhouette score, outputs cluster assignments + UMAP visualizations + profile CSVs

## Commands

```bash
# Set API key before extraction
export DEEPSEEK_API_KEY="your-key"

# Three-layer extraction (from repo root)
python valid_three_layer_archive/scripts/extract_three_layers.py
python valid_three_layer_archive/scripts/extract_three_layers.py --limit 3   # test run

# Topic backfill (requires topic_results/ populated by the notebook)
python valid_three_layer_archive/scripts/backfill_topics.py

# Person clustering (--best-k from silhouette analysis)
python valid_three_layer_archive/scripts/cluster_people.py --best-k 8

# Two-layer extraction (must run from valid_two_layer_archive/ directory)
cd valid_two_layer_archive && python scripts/test.py

# Run tests
python -m pytest valid_three_layer_archive/tests/ -v
```

## Key technical details

- **DeepSeek API** is called via the OpenAI Python client (`base_url=https://api.deepseek.com`, model `deepseek-v4-pro`). Max 12 concurrent threads, 3 retries with exponential backoff.
- **Text encoding**: source `.txt` files may be GBK/GB2312/UTF-8 — scripts try multiple encodings in order.
- **Three-layer scripts use location-independent paths** via `PROJECT_ROOT = Path(__file__).resolve().parents[1]` — they can run from any directory. Two-layer scripts use `Path("./data")` and must run from within `valid_two_layer_archive/`.
- **BERTopic** in the notebook uses `BisectingKMeans` (not HDBSCAN) for controlled cluster counts, with `paraphrase-multilingual-MiniLM-L12-v2` embeddings and cosine-distance UMAP.
- **Person clustering** uses UMAP → 5D (cosine, `min_dist=0.0`), then K-Means with silhouette score for K selection, plus UMAP → 2D for visualization.
- **Matplotlib** is configured for Chinese font rendering (`SimHei`, `Microsoft YaHei` fallbacks).
