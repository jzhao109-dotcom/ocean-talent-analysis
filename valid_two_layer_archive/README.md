# 有效两层提取归档

本文件夹只保留当前可复现的“两层提取”有效链路材料，采用复制方式整理，原目录文件未移动。

## 内容

- `scripts/`: 两层流程脚本
  - `test.py`: 从 `data/*.txt` 调用 DeepSeek API 提取 `outer_layer` 和 `inner_layer`
  - `backfill_topics.py`: 将两层 topic 名称回写到 `new_data/*.json`
  - `cluster_people.py`: 基于 topic 标签进行人才聚类
- `raw_data/data/`: 209 个有效样本对应的原始 txt
- `extraction_results/new_data/`: 209 个有效样本对应的两层 JSON
- `topic_results/`: 两层 topic 聚类结果、主题命名表和可视化/汇总文件
- `person_cluster_results/`: 人才聚类结果和总结文档

## 有效样本口径

有效样本定义为同时具备以下字段的 `new_data/*.json`:

- `outer_layer.signal_topics`
- `inner_layer.signal_topics`
- `person_cluster`

原始 `data/` 与 `new_data/` 各有 219 个文件；其中 209 个已完整进入 topic 回标和人才聚类。

未纳入本归档有效样本的 10 个为：

唐波、曾芸先、朱作言、林枫、桂建芳、熊盛青、王海斌、胡章立、袁东亮（文件名为 `袁东亮 .json`，人名后有空格）、陈宜瑜

## 注意

三层旧流程脚本和结果未放入本文件夹，避免与当前两层分析口径混淆。
