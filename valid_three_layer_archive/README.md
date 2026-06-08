# 有效三层提取归档

本文件夹用于重新跑“三层提取”链路，和 `valid_two_layer_archive` 并列保存，避免两层结果与三层结果混用。

## 路径方法

三层脚本不再依赖当前命令行所在目录，而是通过脚本位置推导归档根目录：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

默认路径如下：

- 输入文本：`raw_data/data/*.txt`
- 三层提取 JSON：`extraction_results/new_data/*.json`
- topic 聚类/命名结果：`topic_results/`
- 人才聚类结果：`person_cluster_results/person_clusters/`

因此可以在项目根目录运行，也可以直接运行 `scripts/` 下的脚本，路径都会指向本归档内部。

## 三层字段

- `outer_layer`: 外层 · 海洋专业实践层，对应“认识海洋”，提取可观察的海洋相关实践、专业活动与产出。
- `middle_layer`: 中层 · 海洋能力资源层，对应支撑海洋实践的 KSAO 与人力资本资源，提取可迁移的能力机制、方法经验、知识结构、现场经验与资源组织方式。中层应避免复述外层的项目、成果、装备、物种或具体业务名称。
- `inner_layer`: 内层 · 海洋深层素养层，对应“关心海洋”与“经略海洋”，提取海洋相关的认知方式、价值取向、情感联结、使命驱动、战略眼光和长期愿景。

## 脚本

- `scripts/extract_three_layers.py`: 从 `raw_data/data/*.txt` 调用 DeepSeek API 提取三层 JSON。
- `scripts/backfill_topics.py`: 将 `topic_results/topic_names_suggested.csv` 与三层 assignments 回写到 JSON 的 `signal_topics`。
- `scripts/cluster_people.py`: 基于三层 `signal_topics` 生成人才聚类结果。

## 运行

先设置 API Key：

```powershell
$env:DEEPSEEK_API_KEY = "你的 key"
```

提取三层：

```powershell
python valid_three_layer_archive\scripts\extract_three_layers.py
```

只试跑前 3 个待处理文件：

```powershell
python valid_three_layer_archive\scripts\extract_three_layers.py --limit 3
```

topic 回标要求 `topic_results/` 中存在以下文件：

```text
topic_results/topic_names_suggested.csv
topic_results/outer/outer_layer_assignments.csv
topic_results/middle/middle_layer_assignments.csv
topic_results/inner/inner_layer_assignments.csv
```

回标：

```powershell
python valid_three_layer_archive\scripts\backfill_topics.py
```

人才聚类：

```powershell
python valid_three_layer_archive\scripts\cluster_people.py --best-k 5
```

## 说明

原始 txt 已从两层归档的 `raw_data/data/` 复制到本目录。`extraction_results/new_data/` 当前留给三层提取结果写入。
