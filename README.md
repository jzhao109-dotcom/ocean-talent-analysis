# 海洋人才 · 双层能力聚类分析

基于 **209 位中国海洋科技领军人才** 的公开访谈与传记文本，使用 BERTopic + UMAP + HDBSCAN 进行双层主题聚类，再用 K-Means 形成五类人才画像。

## 目录结构

```
.
├── valid_two_layer_archive/        # 当前主分析口径（两层结构）
│   ├── scripts/                    # 提取/回写/聚类脚本
│   ├── raw_data/data/              # 209 份访谈原文
│   ├── extraction_results/         # 两层提取 JSON
│   ├── topic_results/              # 主题聚类结果 + 可视化
│   └── person_cluster_results/     # 人员聚类 + 总结文档
├── valid_three_layer_archive/      # 历史三层流程存档
└── literature_hr_theory/           # 相关人力资源理论文献
```

## 双层能力结构

- **外层（外显行动能力）— 12 个主题**：海洋生物资源转化、深海装备攻关、船舰远航实践、学科平台搭建、关键技术装备研发、重大项目领衔与成果转化、观海识变预报、风浪声场预警、近海治污修复、极地冰区工程、深水油气勘探、珊瑚保种修礁
- **内层（内隐驱动与认知）— 7 个主题**：向海报国担当、自主创新、海洋权益维护、谋海布局、绿色护海意识、深蓝探路、中国方案探索

## 五类人才画像

| 簇 | 类型 | 人数 | 占比 |
|---|---|---|---|
| 0 | 深海装备攻坚型 | 63 | 30.1% |
| 1 | 近海治理经略型 | 33 | 15.8% |
| 2 | 船舰远航观测型 | 55 | 26.3% |
| 3 | 生物资源专精型 | 46 | 22.0% |
| 4 | 综合均衡型 | 18 | 8.6% |

详细分析见 [`valid_two_layer_archive/person_cluster_results/海洋人才能力聚类分析结果总结.md`](valid_two_layer_archive/person_cluster_results/海洋人才能力聚类分析结果总结.md)。

## 使用脚本

提取脚本依赖 DeepSeek API，需先设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "你的密钥"
python valid_two_layer_archive/scripts/test.py
```
