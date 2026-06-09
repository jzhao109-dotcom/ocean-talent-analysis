import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "test.ipynb"


def notebook_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


def test_notebook_targets_three_layer_archive_paths():
    src = notebook_source()

    assert 'DATA_DIR = PROJECT_ROOT / "extraction_results" / "new_data_prompt_test_10"' in src
    assert 'OUTER_OUT = TOPIC_ROOT / "outer"' in src
    assert 'MIDDLE_OUT = TOPIC_ROOT / "middle"' in src
    assert 'INNER_OUT = TOPIC_ROOT / "inner"' in src
    assert "EXPECTED_JSON_COUNT = 209" in src


def test_notebook_defines_and_runs_three_layers():
    src = notebook_source()

    assert "OUTER_K" in src
    assert "MIDDLE_K" in src
    assert "INNER_K" in src
    assert 'analyze_layer("outer_layer", "外层 · 蓝色胜任表征维度", OUTER_OUT' in src
    assert 'analyze_layer("middle_layer", "中层 · 协同创新支撑维度", MIDDLE_OUT' in src
    assert 'analyze_layer("inner_layer", "内层 · 深蓝精神内核维度", INNER_OUT' in src
    assert "海洋人才三层能力聚类分析" in src
