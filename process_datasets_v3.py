"""
process_datasets.py — v3
=========================
从三个独立数据源（fi-deepl JSONL、de CSV、en TSV）各提取 17491 条，
按毒性:非毒性 = 36:64 的比例抽样，合并为最终三语数据集。

v3 更新说明（相对于 v2）：
  - clean_text 升级为 v3：新增翻译幻觉修复（连字符/普通词/全大写词重复）
  - 英语数据也应用 clean_text（之前不清洗英语，但英语中有 2% 的 URL、
    0.1% 的 @用户名、40% 的连续空格等噪声，不清洗会导致训练-验证不一致）
  - #标签处理：只对 #字母开头 的标签去掉 # 符号，保留 #数字（如 #2 是编号引用）
  - 新增移除极短文本（<5字符）和去重步骤
"""

import pandas as pd
import re
import json

# ============================================================
# clean_text v3
# ============================================================
def clean_text(text):
    """
    清洗毒性检测文本，移除对分类无关的噪声内容。
    设计原则：宁可少删，不可误伤有语义价值的内容。

    清洗内容：
      第一层 - 结构性噪声：URL、邮箱、HTML标签、Wiki标记
      第二层 - 社交媒体噪声：@用户名、#标签（仅去#保留文本）
      第三层 - 时间/日期噪声：UTC时间戳、ISO日期、三语日期格式
      第四层 - 翻译幻觉修复：连字符词重复、普通词重复、全大写词重复
      第五层 - 格式归一化：首尾引号、制表符/换行符、连续空白
    """
    if not isinstance(text, str):
        return ""

    # ---- 第一层：结构性噪声 ----
    # 1. URL（含 t.co 短链接）
    text = re.sub(r'https?://\S+', '', text)
    # 2. 邮箱（严格格式，避免误伤 @username）
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    # 3. HTML 标签（要求 < 后紧跟字母或 /，避免误伤 <<Oops>>）
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    # 4. Wiki 模板 {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # 5. Wiki 链接 [[...]]，保留显示文本
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    # 6. Wiki 签名 ~~~~
    text = re.sub(r'~{3,5}', '', text)
    # 7. "- Talk -" 签名
    text = re.sub(r'-\s*Talk\s*-', '', text, flags=re.IGNORECASE)
    # 8. User:xxx / Special:Contributions/xxx
    text = re.sub(r'(?:User|Special:Contributions)[:/]\S+', '', text, flags=re.IGNORECASE)

    # ---- 第二层：社交媒体噪声 ----
    # 9. 行首 @username（Twitter 回复前缀），可能连续多个
    text = re.sub(r'^(?:@\w+\s*)+', '', text)
    # 10. 文中的 @username（前面不能是字母数字，避免误伤邮箱）
    text = re.sub(r'(?<![A-Za-z0-9.])@\w+', '', text)
    # 11. #标签：仅对 #字母开头 去掉 # 保留文本；#数字（如 #2）是编号引用，保留原样
    text = re.sub(r'#([a-zA-Z]\w*)', r'\1', text)

    # ---- 第三层：时间/日期噪声 ----
    # 12. 带 UTC/GMT 的完整时间戳
    text = re.sub(
        r'\d{1,2}:\d{2}(?::\d{2})?\s*,?\s*\d{0,2}\s*\w*\s*\(?\s*(?:UTC|GMT)(?:[+-]\d{1,2})?\s*\)?',
        '', text, flags=re.IGNORECASE)
    # 13. 孤立的 UTC/GMT 标记
    text = re.sub(r'\(?\s*\b(?:UTC|GMT)(?:[+-]\d{1,2})?\b\s*\)?', '', text, flags=re.IGNORECASE)
    # 14. 时间戳 HH:MM:SS（三段格式，排除圣经引用 John 6:35）
    text = re.sub(r'(?<![A-Za-z])\b\d{1,2}:\d{2}(?::\d{2})\b', '', text)
    # 15. ISO 日期 YYYY-MM-DD
    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', text)
    # 16. 英文完整日期
    text = re.sub(
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{1,2},?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r',?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    # 17. 芬兰语完整日期
    text = re.sub(
        r'\b\d{1,2}\s+(?:tammikuu(?:ta)?|helmikuu(?:ta)?|maaliskuu(?:ta)?|huhtikuu(?:ta)?|'
        r'toukokuu(?:ta)?|kesäkuu(?:ta)?|heinäkuu(?:ta)?|elokuu(?:ta)?|syyskuu(?:ta)?|'
        r'lokakuu(?:ta)?|marraskuu(?:ta)?|joulukuu(?:ta)?)\b', '', text, flags=re.IGNORECASE)
    # 18. 德语完整日期
    text = re.sub(
        r'\b\d{1,2}\.?\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)'
        r'(?:\s+\d{4})?\b', '', text, flags=re.IGNORECASE)
    # 19. 孤立年份（整行只有年份 / 行首年份后跟空白）
    text = re.sub(r'^\d{4}\s*$', '', text)
    text = re.sub(r'^\d{4}\s+', '', text)
    # 20. Wiki 编辑标记 "Korjattu"（芬兰语"已修正"）
    text = re.sub(r'\bKorjattu\b', '', text, flags=re.IGNORECASE)

    # ---- 第四层：翻译幻觉修复 ----
    # 21. 连字符词异常重复：win-win-win-win... → win-win
    text = re.sub(r'\b((\w+)-\2(?:-\2){2,})\b',
                  lambda m: m.group(2) + '-' + m.group(2), text)
    # 22. 普通词异常重复（3次+）：保留口语感叹词，缩减其余
    def fix_word_repeat(match):
        word = match.group(1)
        if word.lower() in {'ha', 'he', 'ho', 'no', 'na', 'la', 'ya', 'go',
                             'bla', 'blah', 'etc', 'ding', 'boom', 'bang',
                             'haha', 'very', 'really', 'rah', 'nah', 'meh'}:
            return match.group(0)
        return word
    text = re.sub(r'\b(\w{3,})(?:\s+\1){2,}\b', fix_word_repeat, text, flags=re.IGNORECASE)
    # 23. 全大写词异常重复
    text = re.sub(r'\b([A-ZÄÖÜ]{3,})(?:\s+\1){2,}\b', r'\1', text)

    # ---- 第五层：格式归一化 ----
    # 24. 首尾成对引号
    text = text.strip()
    if len(text) >= 2 and \
       ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        text = text[1:-1].strip()
    # 25. 制表符/换行符→空格
    text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    # 26. 连续空白归一化
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    return text


# ============================================================
# 按 36:64（毒性:非毒性）比例抽样
# ============================================================
def sample_by_ratio(df, label_col, total_n, toxic_ratio=0.36, random_state=42):
    """从 df 中按 toxic_ratio : (1-toxic_ratio) 比例抽样 total_n 条。"""
    n_toxic = int(total_n * toxic_ratio)
    n_nontoxic = total_n - n_toxic

    toxic_df = df[df[label_col] == 1].copy()
    nontoxic_df = df[df[label_col] == 0].copy()

    print(f"  可用毒性样本: {len(toxic_df)}, 需要: {n_toxic}")
    print(f"  可用非毒性样本: {len(nontoxic_df)}, 需要: {n_nontoxic}")

    if len(toxic_df) < n_toxic:
        print(f"  [警告] 毒性样本不足，使用全部 {len(toxic_df)} 条")
        toxic_sampled = toxic_df
        n_nontoxic = min(total_n - len(toxic_df), len(nontoxic_df))
        nontoxic_sampled = nontoxic_df.sample(n=n_nontoxic, random_state=random_state)
    elif len(nontoxic_df) < n_nontoxic:
        print(f"  [警告] 非毒性样本不足，使用全部 {len(nontoxic_df)} 条")
        nontoxic_sampled = nontoxic_df
        n_toxic = min(total_n - len(nontoxic_df), len(toxic_df))
        toxic_sampled = toxic_df.sample(n=n_toxic, random_state=random_state)
    else:
        toxic_sampled = toxic_df.sample(n=n_toxic, random_state=random_state)
        nontoxic_sampled = nontoxic_df.sample(n=n_nontoxic, random_state=random_state)

    combined = pd.concat([toxic_sampled, nontoxic_sampled]) \
                 .sample(frac=1, random_state=random_state) \
                 .reset_index(drop=True)
    return combined


# ============================================================
# 通用后处理：移除空文本、极短文本、去重
# ============================================================
def post_process(df, lang_name):
    """清洗后的通用后处理步骤。"""
    before = len(df)

    # 移除空文本
    empty = df['text'].str.len() == 0
    if empty.sum() > 0:
        print(f"  移除空文本: {empty.sum()} 条")
        df = df[~empty].copy()

    # 移除极短文本（<5字符，如 "oh", "und"，对分类无信息量）
    short = df['text'].str.len() < 5
    if short.sum() > 0:
        print(f"  移除极短文本(<5字符): {short.sum()} 条")
        df = df[~short].copy()

    # 去除完全重复
    dup_before = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    n_dup = dup_before - len(df)
    if n_dup > 0:
        print(f"  去除重复文本: {n_dup} 条")

    print(f"  后处理: {before} → {len(df)} 条 (移除 {before - len(df)})")
    return df


# ============================================================
# 第一部分：处理芬兰语 fi-deepl 数据（JSONL）
# ============================================================
print("=" * 60)
print("第一部分：处理芬兰语数据 (train_fi_deepl.jsonl)")
print("=" * 60)

fi_records = []
with open('train_fi_deepl.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            fi_records.append(json.loads(line))

fi_df = pd.DataFrame(fi_records)

# 合并毒性标签：任何一个毒性标签为 1 即视为毒性文本
toxicity_cols = [
    'label_identity_attack', 'label_insult', 'label_obscene',
    'label_severe_toxicity', 'label_threat', 'label_toxicity'
]
fi_df['label'] = fi_df[toxicity_cols].max(axis=1).astype(int)

# 清洗文本
fi_df['text'] = fi_df['text'].apply(clean_text)

# 后处理
fi_df = post_process(fi_df, 'fi')

# 按 36:64 抽样 17491 条
fi_sampled = sample_by_ratio(fi_df, 'label', total_n=17491)
fi_sampled['lang'] = 'fi'
fi_sampled['id'] = [f"fi_train{i}" for i in range(len(fi_sampled))]
fi_final = fi_sampled[['id', 'text', 'label', 'lang']].copy()

print(f"芬兰语最终: {len(fi_final)} 条 "
      f"(毒性: {(fi_final['label']==1).sum()}, 非毒性: {(fi_final['label']==0).sum()})")
print()


# ============================================================
# 第二部分：处理德语 de 数据（CSV）
# ============================================================
print("=" * 60)
print("第二部分：处理德语数据 (de_hf_112024.csv)")
print("=" * 60)

de_df = pd.read_csv('de_hf_112024.csv', engine='python', on_bad_lines='skip')
de_df['labels'] = pd.to_numeric(de_df['labels'], errors='coerce')
de_df = de_df.dropna(subset=['labels'])
de_df['label'] = de_df['labels'].astype(int)

# 清洗文本
de_df['text'] = de_df['text'].astype(str).apply(clean_text)

# 后处理
de_df = post_process(de_df, 'de')

# 按 36:64 抽样 17491 条
de_sampled = sample_by_ratio(de_df, 'label', total_n=17491)
de_sampled['lang'] = 'de'
de_sampled['id'] = [f"de_train{i}" for i in range(len(de_sampled))]
de_final = de_sampled[['id', 'text', 'label', 'lang']].copy()

print(f"德语最终: {len(de_final)} 条 "
      f"(毒性: {(de_final['label']==1).sum()}, 非毒性: {(de_final['label']==0).sum()})")
print()


# ============================================================
# 第三部分：处理英语 en 数据（TSV）
# ============================================================
print("=" * 60)
print("第三部分：处理英语数据 (train.tsv)")
print("=" * 60)

en_df = pd.read_csv('train.tsv', sep='\t', header=0, quoting=3)
en_df['label'] = pd.to_numeric(en_df['label'], errors='coerce')
en_df = en_df.dropna(subset=['label'])
en_df['label'] = en_df['label'].astype(int)

# 英语数据也应用 clean_text（v3 变更：与其他语言保持一致）
en_df['text'] = en_df['text'].astype(str).apply(clean_text)

# 后处理
en_df = post_process(en_df, 'en')

# 按 36:64 抽样 17491 条
en_sampled = sample_by_ratio(en_df, 'label', total_n=17491)
en_sampled['lang'] = 'en'
en_sampled['id'] = [f"eng_train{i}" for i in range(len(en_sampled))]
en_final = en_sampled[['id', 'text', 'label', 'lang']].copy()

print(f"英语最终: {len(en_final)} 条 "
      f"(毒性: {(en_final['label']==1).sum()}, 非毒性: {(en_final['label']==0).sum()})")
print()


# ============================================================
# 第四部分：合并并导出
# ============================================================
print("=" * 60)
print("第四部分：合并并导出")
print("=" * 60)

final_df = pd.concat([fi_final, en_final, de_final], ignore_index=True)

output_file = 'final_trilingual_dataset.tsv'
final_df.to_csv(output_file, sep='\t', index=False, quoting=3, escapechar='\\')

print(f"最终合并数据集: {len(final_df)} 条")
print(f"  芬兰语 (fi): {(final_df['lang']=='fi').sum()}")
print(f"  英语 (en):   {(final_df['lang']=='en').sum()}")
print(f"  德语 (de):   {(final_df['lang']=='de').sum()}")
print(f"  毒性总计:    {(final_df['label']==1).sum()}")
print(f"  非毒性总计:  {(final_df['label']==0).sum()}")
print(f"\n已保存至: {output_file}")
