"""
海洋人才访谈文本分析 —— 内外两层能力提取

流程概览:
  1. 读取 data/*.txt 原始访谈文本（整文档，不切块）
  2. 直接调用 DeepSeek API，一次性提取内外两层能力信号
  3. 多线程并发处理
  4. 结果写入 new_data/*.json
"""

import os
import re
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError
from tqdm import tqdm


# ============================================================
# 0. 全局配置
# ============================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"

MAX_WORKERS = 12
MAX_RETRIES = 3
TEST_LIMIT = None  # None 表示不限，测试时设为数字

DATA_DIR = Path("./data")
OUT_DIR = Path("./new_data")

# ============================================================
# 1. 初始化
# ============================================================

if not DEEPSEEK_API_KEY:
    raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")

OUT_DIR.mkdir(parents=True, exist_ok=True)

_write_lock = Lock()


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


# ============================================================
# 2. Prompt 模板 —— 内外两层能力提取
# ============================================================

SYSTEM_PROMPT = (
    "你是一名资深的海洋领域人才测评专家。你必须严格基于用户提供的访谈文本进行深度剖析，"
    "以「关心海洋、认识海洋、经略海洋」为总体框架，重点关注与海洋相关的能力特征。"
    "不要编造、不要扩展、不要引入外部知识。输出必须是合法 JSON。"
)


def build_user_prompt(file_name: str, text: str) -> str:
    return f"""
文件名：{file_name}

请作为一名资深的海洋领域人才测评专家，对下方【访谈文本】进行深度剖析。
请从以下内外两个层次提取该人物的能力信号：

━━━━━━━━━━━━━━━━━━━━━━━━
外层 · 海洋专业实践层
━━━━━━━━━━━━━━━━━━━━━━━━
关注该人物"在海洋领域做了什么"——所有可被外部观察到的海洋相关实践行为、专业活动与产出。这一层对应"认识海洋"的维度：通过科学探索、技术研发、工程实践等方式去认知海洋、触及海洋。

提取时请保持以下敏感性：
- 凡是文本中出现的任何与海洋相关的实践行为——无论是科研、工程、装备、航行、探测、资源开发、环境保护、海上作业、水产养殖、海洋法务、海洋教育、海洋文化等等——都应纳入外层视野
- 关键不是"属于哪个子类"，而是"这个行为是否与海洋直接相关"
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
  "inner_layer": {{
    "summary": "1-2句话概括该人物在海洋深层素养方面的特征",
    "signals": [
      "≤10字关键词，如：海洋强国战略视野、深海探索使命感、向海图强信念",
      "（建议5-10条，每条≤10字）"
    ]
  }}
}}

注意：
1. 若某一层未提取到明确信号，signals 返回空数组 []，summary 写明"未提取到足够信息"。
2. 每条 signal 必须是 ≤10字 的简短关键词或标签，从原文提炼核心词，禁止写成完整句子。
3. 严格区分两层：外层 = 与海洋直接相关的可观察实践（做了什么）；内层 = 与海洋相关的内在认知与价值（驱动什么、相信什么、感受到什么）。
4. 不要让预设的分类框架限制你的提取——文本中出现什么就提取什么，框架只是方向指引，不是边界。
5. signal 的价值在于区分度。请反问自己：这条 signal 能把这个人和同领域其他人区分开吗？避免生成"海洋科研""海洋报国""海洋强国"等放之四海而皆准的通用标签。好的 signal 是"首创浪致混合理论"，而不是"海洋理论研究"。

【访谈文本】
{text}
""".strip()


# ============================================================
# 3. LLM 调用 —— 含自动重试 & JSON 容错解析
# ============================================================

def call_llm(text: str, file_name: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(file_name, text)},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            client = _get_client()
            resp = client.chat.completions.create(
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
                pass

            match = re.search(r"\{.*\}", raw, flags=re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass

            return {"_parse_error": True, "raw": raw}

        except (APIError, APITimeoutError, APIConnectionError) as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return {"_error": "API error after retries", "details": str(exc)}

        except Exception as exc:
            return {"_error": "Unexpected exception", "details": str(exc)}

    return {"_error": "Max retries exceeded"}


# ============================================================
# 4. 单文件处理（线程内同步执行）
# ============================================================

def process_one_file(fp: Path) -> None:
    file_name = fp.name
    out_path = OUT_DIR / f"{fp.stem}.json"

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    try:
        raw_bytes = fp.read_bytes()
        text = None
        for enc in ["gbk", "gb2312", "utf-8", "gb18030"]:
            try:
                text = raw_bytes.decode(enc).strip()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if text is None:
            text = raw_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        tqdm.write(f"[Read Error] {file_name}: {exc}")
        return

    if not text:
        tqdm.write(f"[skip empty] {file_name}")
        return

    result = call_llm(text, file_name)
    result.setdefault("file", file_name)

    with _write_lock:
        try:
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            tqdm.write(f"[Write Error] {file_name}: {exc}")


# ============================================================
# 5. 主入口
# ============================================================

def main() -> None:
    if not DATA_DIR.exists():
        raise RuntimeError(
            f"数据目录不存在：{DATA_DIR.resolve()}（请确认 data/ 在当前工作目录下）"
        )

    all_files = sorted(DATA_DIR.glob("*.txt"))
    if not all_files:
        raise RuntimeError(f"{DATA_DIR.resolve()} 下未找到 .txt 文件")

    pending = []
    skipped = 0
    for fp in all_files:
        out_path = OUT_DIR / f"{fp.stem}.json"
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
        else:
            pending.append(fp)

    if TEST_LIMIT is not None:
        pending = pending[:TEST_LIMIT]

    print(f"扫描到 {len(all_files)} 个文件，跳过 {skipped} 个已完成")
    if not pending:
        print("所有文件已处理完毕。")
        return

    print(f"开始处理 {len(pending)} 个文件")
    print(f"模型: {DEEPSEEK_MODEL} | 并发线程: {MAX_WORKERS} | 模式: 整文档（不切块）")
    print("-" * 50)

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_one_file, fp): fp.name
            for fp in pending
        }

        pbar = tqdm(
            total=len(pending),
            desc="Processing",
            unit="file",
            ncols=100,
        )
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
    print(f"结果输出目录: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。")
