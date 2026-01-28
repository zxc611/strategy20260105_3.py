import argparse
import hashlib
import json
import os
import sys
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TARGET_FILE = os.path.join(ROOT, "Strategy20260105_3.py")
BASELINE_FILE = os.path.join(ROOT, ".core_lock_hash.json")
MARKER_PAIRS: List[Tuple[str, str]] = [
    ("CORE_LOCK_START_WIDTH_FUTURES", "CORE_LOCK_END_WIDTH_FUTURES"),
    ("CORE_LOCK_START_WIDTH_GROUP", "CORE_LOCK_END_WIDTH_GROUP"),
    ("CORE_LOCK_START_MONTH_CHECK_SPECIFIED", "CORE_LOCK_END_MONTH_CHECK_SPECIFIED"),
    ("CORE_LOCK_START_MONTH_CHECK_SPECIFIED_NEXT", "CORE_LOCK_END_MONTH_CHECK_SPECIFIED_NEXT"),
    ("CORE_LOCK_START_KLINE_TRIGGER", "CORE_LOCK_END_KLINE_TRIGGER"),
    ("CORE_LOCK_START_BATCH_FILTER", "CORE_LOCK_END_BATCH_FILTER"),
    ("CORE_LOCK_START_BATCH_GROUPS", "CORE_LOCK_END_BATCH_GROUPS"),
]


def _extract_sections(text: str) -> List[str]:
    lines = text.splitlines()
    sections: List[str] = []
    for start, end in MARKER_PAIRS:
        collecting = False
        buf: List[str] = []
        for line in lines:
            if start in line:
                collecting = True
                buf.append(line)
                continue
            if collecting:
                buf.append(line)
                if end in line:
                    sections.append("\n".join(buf))
                    collecting = False
                    buf = []
        if collecting:
            raise RuntimeError(f"未找到结束标记: {end}")
    return sections


def compute_hash(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    sections = _extract_sections(content)
    if not sections:
        raise RuntimeError("未找到任何标记区块")
    joined = "\n".join(sections)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_baseline() -> dict:
    if not os.path.exists(BASELINE_FILE):
        return {}
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(data: dict) -> None:
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check core lock hash")
    parser.add_argument("--update", action="store_true", help="更新基线哈希")
    args = parser.parse_args()

    h = compute_hash(TARGET_FILE)
    baseline = load_baseline()

    if args.update:
        baseline[os.path.basename(TARGET_FILE)] = h
        save_baseline(baseline)
        print(f"[core-lock] 基线已更新: {h}")
        return 0

    stored = baseline.get(os.path.basename(TARGET_FILE))
    if not stored:
        print("[core-lock] 未找到基线，请先执行 --update", file=sys.stderr)
        return 1
    if stored != h:
        print("[core-lock] 核心代码已变更，禁止提交。如确认调整，请先运行 --update 重新固化基线。", file=sys.stderr)
        return 1
    print("[core-lock] 校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
