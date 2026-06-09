import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_script_uses_archive_relative_paths_and_three_layers():
    module = load_module("extract_three_layers")

    assert module.PROJECT_ROOT == ROOT
    assert module.DATA_DIR == ROOT / "raw_data" / "data"
    assert module.OUT_DIR == ROOT / "extraction_results" / "new_data"
    assert module.LAYERS == ["outer_layer", "middle_layer", "inner_layer"]

    prompt = module.build_user_prompt("sample.txt", "访谈文本")
    assert '"outer_layer"' in prompt
    assert '"middle_layer"' in prompt
    assert '"inner_layer"' in prompt


def test_extract_prompt_links_hr_theory_to_marine_talent_layers():
    module = load_module("extract_three_layers")

    prompt = module.build_user_prompt("sample.txt", "访谈文本")

    for phrase in [
        "胜任力模型",
        "人力资本资源理论",
        "KSAO",
        "海洋人才",
        "关心海洋",
        "认识海洋",
        "经略海洋",
        "外层（蓝色胜任表征维度）= 与海洋直接相关的可观察实践",
        "中层（协同创新支撑维度）= 支撑这些实践的可迁移能力机制",
        "内层（深蓝精神内核维度）= 与海洋相关的内在认知、价值、使命与战略取向",
    ]:
        assert phrase in prompt


def test_extract_prompt_preserves_two_layer_marine_sensitivity():
    module = load_module("extract_three_layers")

    prompt = module.build_user_prompt("sample.txt", "访谈文本")

    for phrase in [
        "凡是文本中出现的任何与海洋相关的实践行为",
        "这个行为是否与海洋直接相关",
        "信号应直接从原文中涌现，而非套用预设分类",
        "不要将中层泛化为一般管理能力",
        "海洋可以以多种方式内化于一个人",
        "首创浪致混合理论",
    ]:
        assert phrase in prompt


def test_extract_script_defaults_to_deepseek_v4_pro():
    module = load_module("extract_three_layers")

    assert module.DEEPSEEK_MODEL == "deepseek-v4-pro"


def test_backfill_supports_three_layers(tmp_path):
    module = load_module("backfill_topics")

    topic_root = tmp_path / "topic_results"
    new_data = tmp_path / "new_data"
    new_data.mkdir()
    for layer in ["outer", "middle", "inner"]:
        (topic_root / layer).mkdir(parents=True)

    (topic_root / "topic_names_suggested.csv").write_text(
        "layer,topic,suggested_name\n"
        "outer,0,外层主题\n"
        "middle,0,中层主题\n"
        "inner,0,内层主题\n",
        encoding="utf-8-sig",
    )
    for layer in ["outer", "middle", "inner"]:
        (topic_root / layer / f"{layer}_layer_assignments.csv").write_text(
            "Doc,Topic\n信号,0\n",
            encoding="utf-8-sig",
        )

    sample = {
        "outer_layer": {"signals": ["信号"]},
        "middle_layer": {"signals": ["信号"]},
        "inner_layer": {"signals": ["信号"]},
    }
    fp = new_data / "sample.json"
    fp.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    assert module.backfill_topics(new_data, topic_root) == 1
    result = json.loads(fp.read_text(encoding="utf-8"))
    assert result["outer_layer"]["signal_topics"] == ["外层主题"]
    assert result["middle_layer"]["signal_topics"] == ["中层主题"]
    assert result["inner_layer"]["signal_topics"] == ["内层主题"]


def test_cluster_people_collects_topics_from_three_layers(tmp_path):
    module = load_module("cluster_people")

    data_dir = tmp_path / "new_data"
    data_dir.mkdir()
    (data_dir / "sample.json").write_text(
        json.dumps(
            {
                "outer_layer": {"signal_topics": ["外层主题"]},
                "middle_layer": {"signal_topics": ["中层主题"]},
                "inner_layer": {"signal_topics": ["内层主题"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    profiles, topics = module.load_person_profiles(data_dir)

    assert profiles == [("sample", {"外层主题": 1, "中层主题": 1, "内层主题": 1})]
    assert topics == ["中层主题", "内层主题", "外层主题"]
