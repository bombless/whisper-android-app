#!/usr/bin/env python3
"""
Scan Whisper vocab.json, find tokens containing Traditional Chinese characters,
and output a Kotlin source file with token IDs to mask in logits.

Usage:
    python find_traditional_tokens.py
    python find_traditional_tokens.py > ../app/src/main/java/com/example/whisperapp/asr/TraditionalChineseBlocklist.kt
"""

import json
import sys
from pathlib import Path

# ── GPT-2 byte-level encoder (matches WhisperTokenizer.buildByteEncoder) ──

def build_byte_decoder():
    bytes_list = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    encoder = {}
    for b in bytes_list:
        encoder[b] = chr(b)
    extra = 0
    for b in range(256):
        if b not in encoder:
            encoder[b] = chr(256 + extra)
            extra += 1
    return {v: k for k, v in encoder.items()}

BYTE_DECODER = build_byte_decoder()

def decode_token(raw: str) -> str:
    """Decode a GPT-2 byte-level BPE token string back to UTF-8 text."""
    byte_vals = []
    for ch in raw:
        b = BYTE_DECODER.get(ch)
        if b is None:
            return ""
        byte_vals.append(b)
    try:
        return bytes(byte_vals).decode("utf-8", errors="replace")
    except Exception:
        return ""

# ── Traditional → Simplified Chinese mapping ──
# Characters where Traditional form differs from Simplified form.

TRAD_TO_SIMP = {
    # Single characters
    "這": "这", "對": "对", "個": "个", "來": "来", "時": "时",
    "國": "国", "華": "华", "說": "说", "話": "话", "見": "见",
    "問": "问", "們": "们", "經": "经", "會": "会", "將": "将",
    "與": "与", "從": "从", "進": "进", "過": "过", "還": "还",
    "開": "开", "關": "关", "給": "给", "實": "实", "處": "处",
    "機": "机", "車": "车", "門": "门", "電": "电", "東": "东",
    "頭": "头", "學": "学", "書": "书", "點": "点", "認": "认",
    "場": "场", "長": "长", "無": "无", "後": "后", "號": "号",
    "單": "单", "頁": "页", "業": "业", "種": "种", "萬": "万",
    "藝": "艺", "選": "选", "邊": "边", "錯": "错", "條": "条",
    "報": "报", "計": "计", "記": "记", "設": "设", "請": "请",
    "論": "论", "證": "证", "詞": "词", "課": "课", "調": "调",
    "負": "负", "費": "费", "資": "资", "買": "买", "賣": "卖",
    "較": "较", "轉": "转", "運": "运", "達": "达", "響": "响",
    "雜": "杂", "難": "难", "準": "准", "滿": "满", "漢": "汉",
    "決": "决", "測": "测", "濟": "济", "燈": "灯", "營": "营",
    "獎": "奖", "獨": "独", "異": "异", "畫": "画", "當": "当",
    "監": "监", "離": "离", "積": "积", "穩": "稳", "競": "竞",
    "筆": "笔", "節": "节", "簡": "简", "據": "据", "舉": "举",
    "舊": "旧", "補": "补", "裝": "装", "豐": "丰", "許": "许",
    "輕": "轻", "辦": "办", "隊": "队", "隨": "随", "順": "顺",
    "預": "预", "飛": "飞", "驗": "验", "體": "体", "風": "风",
    "馬": "马", "魚": "鱼", "鳥": "鸟", "龍": "龙", "軟": "软",
    "環": "环", "產": "产", "畢": "毕", "盡": "尽", "氣": "气",
    "溫": "温", "減": "减", "煙": "烟", "牆": "墙", "狀": "状",
    "發": "发", "幣": "币", "壓": "压", "廠": "厂", "廣": "广",
    "慶": "庆", "戰": "战", "戲": "戏", "歡": "欢", "殘": "残",
    "歸": "归", "殺": "杀", "氫": "氢", "災": "灾", "寶": "宝",
    "導": "导", "師": "师", "歷": "历", "獅": "狮", "嗎": "吗",
    "為": "为", "麼": "么", "裡": "里", "鬆": "松", "乾": "干",
    "麵": "面", "薑": "姜", "闆": "板", "砲": "炮", "徵": "征",
    "鍾": "钟", "歲": "岁", "幾": "几", "歡": "欢", "氣": "气",
    "萬": "万", "葉": "叶", "鍾": "钟", "裡": "里", "髮": "发",
    "衝": "冲", "剋": "克", "佈": "布", "麼": "么",
    # Common multi-character words
    "什麼": "什么", "怎麼": "怎么", "為什麼": "为什么", "哪裡": "哪里",
    "這麼": "这么", "那麼": "那么", "哪些": "哪些", "這些": "这些",
    "那些": "那些", "每個": "每个", "各種": "各种", "整個": "整个",
    "問題": "问题", "應該": "应该", "開始": "开始", "時間": "时间",
    "知道": "知道", "現在": "现在", "非常": "非常", "因為": "因为",
    "所以": "所以", "但是": "但是", "可以": "可以", "已經": "已经",
    "他們": "他们", "我們": "我们", "你們": "你们", "自己": "自己",
    "這裏": "这里", "那裏": "那里", "這兒": "这儿", "那兒": "那儿",
    "哪兒": "哪儿", "後來": "后来", "然後": "然后", "最後": "最后",
    "還是": "还是", "只是": "只是", "就是": "就是", "不是": "不是",
    "這樣": "这样", "那樣": "那样", "怎樣": "怎样", "樣子": "样子",
    "時候": "时候", "東西": "东西", "關係": "关系", "圖片": "图片",
    "電話": "电话", "電腦": "电脑", "電影": "电影", "電視": "电视",
    "醫院": "医院", "學校": "学校", "馬路": "马路", "飛機": "飞机",
    "鐵路": "铁路", "點兒": "点儿", "玩兒": "玩儿", "其實": "其实",
    "這樣子": "这样子", "那樣子": "那样子", "過來": "过来", "過去": "过去",
    "起來": "起来", "下來": "下来", "上去": "上去", "回來": "回来",
    "出去": "出去", "進來": "进来", "買賣": "买卖", "計劃": "计划",
    "認識": "认识", "感覺": "感觉", "記憶": "记忆", "經驗": "经验",
    "繼續": "继续", "觀察": "观察", "討論": "讨论", "報告": "报告",
    "設計": "设计", "選擇": "选择", "準備": "准备", "運動": "运动",
    "練習": "练习", "考試": "考试", "畢業": "毕业", "圖書館": "图书馆",
    "辦公室": "办公室", "實驗室": "实验室", "會議": "会议", "節目": "节目",
    "風景": "风景", "歷史": "历史", "經濟": "经济", "環境": "环境",
    "藝術": "艺术", "音樂": "音乐", "體育": "体育", "衛生": "卫生",
    "系統": "系统", "軟件": "软件", "網络": "网络", "數據": "数据",
    "資料": "资料", "標準": "标准", "規則": "规则", "程序": "程序",
    "結構": "结构", "組織": "组织", "集團": "集团", "機構": "机构",
    "單位": "单位", "行業": "行业", "產品": "产品", "服務": "服务",
    "市場": "市场", "客戶": "客户", "品牌": "品牌", "質量": "质量",
    "價格": "价格", "投資": "投资", "經營": "经营", "貿易": "贸易",
    "工業": "工业", "農業": "农业", "商業": "商业", "教育": "教育",
    "科學": "科学", "技術": "技术", "研究": "研究", "發展": "发展",
    "進步": "进步", "改變": "改变", "影響": "影响", "效率": "效率",
    "能力": "能力", "權力": "权力", "義務": "义务", "責任": "责任",
    "利益": "利益", "風險": "风险", "機會": "机会", "挑戰": "挑战",
    "成功": "成功", "失敗": "失败", "勝利": "胜利", "進展": "进展",
    "目標": "目标", "方向": "方向", "策略": "策略", "方案": "方案",
    "措施": "措施", "辦法": "办法", "條件": "条件", "情況": "情况",
    "問題": "问题", "答案": "答案", "解釋": "解释", "說明": "说明",
    "介紹": "介绍", "描述": "描述", "表達": "表达", "溝通": "沟通",
    "交流": "交流", "合作": "合作", "競爭": "竞争", "協調": "协调",
    "談判": "谈判", "協議": "协议", "合同": "合同", "條款": "条款",
    "規定": "规定", "制度": "制度", "政策": "政策", "法律": "法律",
    "法規": "法规", "條例": "条例", "規範": "规范", "標準": "标准",
}

# Build set of Traditional Chinese characters (single chars where trad != simp)
TRADITIONAL_CHARS = set()
for trad, simp in TRAD_TO_SIMP.items():
    if trad != simp and len(trad) == 1 and len(simp) == 1:
        TRADITIONAL_CHARS.add(trad)


def contains_traditional(text: str) -> bool:
    """Check if text contains any Traditional Chinese character."""
    return any(ch in TRADITIONAL_CHARS for ch in text)


def main():
    vocab_path = Path(__file__).parent.parent / "lib" / "assets" / "models" / "whisper" / "tokenizer" / "vocab.json"
    if not vocab_path.exists():
        print(f"ERROR: vocab.json not found at {vocab_path}", file=sys.stderr)
        sys.exit(1)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    trad_token_ids = []
    examples = []

    for raw_token, token_id in vocab.items():
        decoded = decode_token(raw_token)
        if not decoded or decoded.startswith("<|"):
            continue
        if contains_traditional(decoded):
            trad_token_ids.append(token_id)
            if len(examples) < 50:
                examples.append((token_id, decoded))

    trad_token_ids.sort()

    # Print summary to stderr
    print(f"Found {len(trad_token_ids)} tokens with Traditional Chinese characters", file=sys.stderr)
    print(f"\nSamples:", file=sys.stderr)
    for tid, text in examples:
        print(f"  id={tid:>5}  {text}", file=sys.stderr)

    # Output Kotlin source to stdout
    ids_str = ", ".join(str(x) for x in trad_token_ids)
    print(f"""package com.example.whisperapp.asr

/**
 * Auto-generated blocklist of token IDs containing Traditional Chinese characters.
 * Generated by tools/find_traditional_tokens.py from vocab.json.
 *
 * When filtering logits, mask these token IDs to force Simplified Chinese output.
 * Total: {len(trad_token_ids)} tokens
 */
object TraditionalChineseBlocklist {{
    /** Set of token IDs that decode to text containing Traditional Chinese characters. */
    val blockedTokenIds: Set<Int> = setOf({ids_str})

    /** Check if a token ID should be blocked. */
    fun isBlocked(tokenId: Int): Boolean = tokenId in blockedTokenIds
}}
""")


if __name__ == "__main__":
    main()
