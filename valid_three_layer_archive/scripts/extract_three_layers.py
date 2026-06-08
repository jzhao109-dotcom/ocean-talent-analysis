"""
海洋人才访谈文本分析 —— 三层能力信号提取

流程:
  1. 读取 raw_data/data/*.txt 原始访谈文本
  2. 调用 DeepSeek API，一次性提取外层/中层/内层三类能力信号
  3. 多线程并发处理
  4. 结果写入 extraction_results/new_data/*.json
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "raw_data" / "data"
OUT_DIR = PROJECT_ROOT / "extraction_results" / "new_data"
LAYERS = ["outer_layer", "middle_layer", "inner_layer"]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "12"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

_write_lock = Lock()


SYSTEM_PROMPT = (
    "你是一名资深的海洋领域人才测评专家，同时熟悉人力资源胜任力模型与人力资本资源理论。"
    "你必须严格基于用户提供的访谈文本进行深度剖析，"
    "以「关心海洋、认识海洋、经略海洋」为总体框架，重点关注与海洋相关的能力特征。"
    "不要编造、不要扩展、不要引入外部知识。输出必须是合法 JSON。"
)


def build_user_prompt(file_name: str, text: str) -> str:
    return f"""
文件名：{file_name}

请作为一名资深的海洋领域人才测评专家，对下方【访谈文本】进行深度剖析。
请从以下外、中、内三个层次提取该人物的海洋能力信号。

理论口径说明：本任务借鉴人力资源中的胜任力模型、人力资本资源理论与 KSAO 框架，但所有提取都必须回到海洋人才主题。
三层关系可以理解为：
- 外层 = 与海洋直接相关的可观察实践、项目、成果与活动（做了什么、在哪个海洋领域做）
- 中层 = 支撑这些实践的可迁移能力机制、方法经验、知识结构与资源组织方式（凭什么能做、怎样做成）
- 内层 = 与海洋相关的内在认知、价值、使命与战略取向（为什么长期投入、相信什么、要推动什么）

━━━━━━━━━━━━━━━━━━━━━━━━
外层 · 海洋专业实践层
━━━━━━━━━━━━━━━━━━━━━━━━
关注该人物"在海洋领域做了什么"——所有可被外部观察到的海洋相关实践行为、专业活动、项目、产品、成果与产出。这一层对应"认识海洋"的维度：通过科学探索、技术研发、工程实践等方式去认知海洋、触及海洋。

提取时请保持以下敏感性：
- 凡是文本中出现的任何与海洋相关的实践行为——无论是科研、工程、装备、航行、探测、资源开发、环境保护、海上作业、水产养殖、海洋法务、海洋教育、海洋文化等等——都应纳入外层视野
- 外层 signal 可以保留具体海洋对象、专业领域、项目/工程/成果名称，如深海潜器、海洋遥感、珊瑚修复、海上风电、船舶总体设计、远洋科考等
- 关键不是"属于哪个子类"，而是"这个行为是否与海洋直接相关、是否能说明此人的外显实践轨迹"
- 信号应直接从原文中涌现，而非套用预设分类

━━━━━━━━━━━━━━━━━━━━━━━━
中层 · 海洋能力资源层
━━━━━━━━━━━━━━━━━━━━━━━━
关注该人物"凭什么能够在海洋领域持续做成事情"——支撑其海洋实践的可迁移能力机制、方法经验、知识结构、工程/科研范式与资源组织方式。这一层对应人力资源胜任力模型中的 KSAO 与人力资本资源：它不是已经做出的外显行为，也不是内在价值信念，而是连接行动与深层驱动的能力资源。

中层必须刻意与外层拉开距离：
- 外层回答"做了什么/做成了什么"，中层回答"靠什么方法、能力、经验、组织资源把事情做成"
- 外层可以像专业领域目录；中层应像能力机制目录
- 中层不要复述外层的项目、成果、产品、海域、装备、物种或具体业务名称
- 如果原文只说"主持某项目/研发某装备/完成某成果"，外层可以提取；中层只有在文本显示其方法、经验、知识、组织方式时才提取

中层 signal 的优先类型：
- 方法能力：数值建模、系统集成、实验设计、数据同化、遥感反演、风险评估、标准制定、方案论证、现场诊断
- 经验能力：海试组织、航次统筹、极端环境作业、长期观测、工程调试、产业转化、技术推广
- 知识结构：跨学科整合、海气耦合认知、地质解释训练、生物组学分析、法规条约解读
- 资源组织：团队协同、产学研联动、国际合作、平台建设、项目管理、标准/规范推动

提取时请保持以下敏感性：
- 凡是文本中显示其海洋相关知识、技术方法、研究范式、工程能力、组织能力、跨学科能力、数据能力、现场经验、平台资源的内容，都可纳入中层
- 关键不是"这个人性格如何"或"管理能力如何"，而是"这些能力资源是否支撑了其海洋实践"
- 不要将中层泛化为一般管理能力、一般学习能力或泛泛个人素质；必须能够回扣到海洋领域的实践场景
- 中层信号可以从外层实践中推断其能力基础，但必须有文本依据，不能凭空补写；推断时要抽象到"能力机制"，不要照搬外层名词
- 中层 signal 尽量避免使用"海洋/深海/远洋/极地/船舶/水产/油气/珊瑚"等领域对象作为开头，除非不写会导致含义失真；优先写"系统集成能力""海试组织经验""多源数据融合""跨学科建模""标准制定能力"这类机制型标签
- 禁止把中层写成外层的同义改写：例如外层已有"深海潜器研制"，中层不要写"深海装备能力"，应写"复杂系统集成"、"海试组织经验"或"极端环境验证"；外层已有"珊瑚修复工程"，中层不要写"珊瑚修复能力"，应写"生态工程设计"、"长期监测评估"或"社会协同组织"
- 信号应直接从原文中涌现，而非套用预设分类


━━━━━━━━━━━━━━━━━━━━━━━━
内层 · 海洋深层素养层
━━━━━━━━━━━━━━━━━━━━━━━━
关注该人物"与海洋相关的内在世界"——认知方式、价值取向、信念体系、情感联结等一切无法直接观察但驱动其海洋实践的内在因素。这一层对应"关心海洋"与"经略海洋"两个维度：对海洋发自内心的关切与热爱，以及从战略高度经略海洋的格局与使命感。

提取时请保持以下敏感性：
- 凡是文本中透露出海洋相关认知、情感、信念、价值判断的内容，都属于内层
- "关心海洋"可以体现为对海洋生态的忧虑、对海洋事业的热忱、对海洋未知的好奇、对海洋权益的关切等
- "经略海洋"可以体现为海洋战略眼光、资源布局思维、学科/产业规划意识、海洋强国信念等
- 不要将内层局限于固定的维度——文本中实际呈现什么就提取什么
- 海洋可以以多种方式内化于一个人：它可以是研究对象、事业舞台、精神家园、报国路径、探索前沿、童年记忆……关注文本中真实浮现的那种联结方式
- 信号应直接从原文中涌现，而非套用预设分类

━━━━━━━━━━━━━━━━━━━━━━━━

请输出标准 JSON 格式，字段定义如下（严格遵守，不要增减字段）：

{{
  "file": "{file_name}",
  "outer_layer": {{
    "summary": "1-2句话概括该人物在海洋专业实践方面的特征",
    "signals": [
      "≤10字关键词，如：深海采矿技术、带队远洋科考、船舶总体设计",
      "（建议5-10条，每条≤10字）"
    ]
  }},
  "middle_layer": {{
    "summary": "1-2句话概括该人物支撑海洋实践的能力机制、方法经验与资源组织特征",
    "signals": [
      "≤10字关键词，如：系统集成、跨学科建模、海试组织经验、数据同化、标准制定",
      "（建议5-10条，每条≤10字）"
    ]
  }},
  "inner_layer": {{
    "summary": "1-2句话概括该人物在海洋深层素养方面的特征",
    "signals": [
      "≤10字关键词，如：海洋强国战略视野、深海探索使命感、向海图强信念",
      "（建议5-10条，每条≤10字）"
    ]
  }}
}}

注意：
1. 若某一层未提取到明确信号，signals 返回空数组 []，summary 写明“未提取到足够信息”。
2. 每条 signal 必须是 ≤10字的简短关键词或标签，从原文提炼核心词，禁止写成完整句子。
3. 严格区分三层：外层 = 与海洋直接相关的可观察实践、项目、成果与活动（做了什么）；中层 = 支撑海洋实践的能力机制、方法经验、知识结构与资源组织方式（凭什么能做、怎样做成）；内层 = 与海洋相关的内在认知、价值、使命与战略取向（驱动什么、相信什么、要推动什么）。
4. 不要让预设的分类框架限制你的提取——文本中出现什么就提取什么，框架只是方向指引，不是边界。
5. 三层都必须紧扣海洋人才主题。禁止提取与海洋无关的一般管理能力、泛泛人格评价或普通履历标签。
6. signal 的价值在于区分度。请反问自己：这条 signal 能把这个人和同领域其他人区分开吗？避免生成"海洋科研""海洋报国""海洋强国""能力突出"等放之四海而皆准的通用标签。好的 signal 是"首创浪致混合理论"，而不是"海洋理论研究"。
7. 尤其注意外层与中层不要重复：若两个 signal 只是在"实践名称"后加"能力/经验"，请重写中层，抽象为方法、经验、知识结构或组织机制。

【访谈文本】
{text}
""".strip()


def get_client():
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("请先安装 openai 包：pip install openai") from exc
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def read_text_file(fp: Path) -> str:
    raw_bytes = fp.read_bytes()
    for enc in ["gbk", "gb2312", "utf-8", "gb18030"]:
        try:
            return raw_bytes.decode(enc).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode("utf-8", errors="ignore").strip()


def call_llm(text: str, file_name: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(file_name, text)},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=8000,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw, flags=re.S)
                if match:
                    return json.loads(match.group(0))
                return {"_parse_error": True, "raw": raw}
        except Exception as exc:
            retryable = exc.__class__.__name__ in {
                "APIError",
                "APITimeoutError",
                "APIConnectionError",
            }
            if not retryable:
                return {"_error": "Unexpected exception", "details": str(exc)}
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            return {"_error": "API error after retries", "details": str(exc)}

    return {"_error": "Max retries exceeded"}


def normalize_result(result: dict, file_name: str) -> dict:
    result.setdefault("file", file_name)
    for layer in LAYERS:
        layer_data = result.setdefault(layer, {})
        layer_data.setdefault("summary", "未提取到足够信息")
        layer_data.setdefault("signals", [])
    return result


def process_one_file(fp: Path, out_dir: Path = OUT_DIR) -> None:
    out_path = out_dir / f"{fp.stem}.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    try:
        text = read_text_file(fp)
    except Exception as exc:
        tqdm.write(f"[Read Error] {fp.name}: {exc}")
        return

    if not text:
        tqdm.write(f"[skip empty] {fp.name}")
        return

    result = normalize_result(call_llm(text, fp.name), fp.name)
    with _write_lock:
        try:
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            tqdm.write(f"[Write Error] {fp.name}: {exc}")


def run(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR, limit: int | None = None) -> None:
    if not data_dir.exists():
        raise RuntimeError(f"数据目录不存在：{data_dir.resolve()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_dir.glob("*.txt"))
    if not all_files:
        raise RuntimeError(f"{data_dir.resolve()} 下未找到 .txt 文件")

    pending = []
    skipped = 0
    for fp in all_files:
        out_path = out_dir / f"{fp.stem}.json"
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
        else:
            pending.append(fp)

    if limit is not None:
        pending = pending[:limit]

    print(f"扫描到 {len(all_files)} 个文件，跳过 {skipped} 个已完成")
    if not pending:
        print("所有文件已处理完毕。")
        return

    print(f"开始处理 {len(pending)} 个文件")
    print(f"模型: {DEEPSEEK_MODEL} | 并发线程: {MAX_WORKERS} | 模式: 三层整文档提取")
    print("-" * 50)

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_one_file, fp, out_dir): fp.name for fp in pending}
        pbar = tqdm(total=len(pending), desc="Processing", unit="file", ncols=100)
        for future in as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                future.result()
                success += 1
            except Exception as exc:
                failed += 1
                tqdm.write(f"[Error] {fname}: {exc}")
            pbar.update(1)
        pbar.close()

    print("-" * 50)
    print(f"完成: {success} 成功, {failed} 失败, {skipped} 跳过")
    print(f"结果输出目录: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取海洋人才访谈文本的三层能力信号")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个待处理文件")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="输入 txt 目录")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="输出 JSON 目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args.data_dir, args.out_dir, args.limit)
    except KeyboardInterrupt:
        print("\n用户中断。")
