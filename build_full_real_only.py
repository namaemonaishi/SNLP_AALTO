"""
build_full_real_only.py — 全量真实数据构建（不用翻译数据补足比例）
================================================================
以德语（~50,000条）为基准，各语言下采样到约50,000条。
芬兰语 = Suomi24 原生全量 + deepl 补足到50,000条。
不强制毒性比例平衡，保留各数据源原始的毒性分布。

数据源：
  1. train.tsv — 英语，~99,000 条 → 下采样到 ~50,000
  2. de_hf_112024.csv — 德语，~50,000 条 → 全量
  3. train_fi_deepl.jsonl — 芬兰语翻译，~159,000 条 → 下采样
  4. all_annotations.tsv — 芬兰语原生，~2,200 条 → 全量保留

使用方式：
  将四个数据文件放在同一目录下，运行：
  python3 build_full_real_only.py
"""

import pandas as pd
import re
import json

# ============================================================
# clean_text v3
# ============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 第一层：结构性噪声
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'~{3,5}', '', text)
    text = re.sub(r'-\s*Talk\s*-', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:User|Special:Contributions)[:/]\S+', '', text, flags=re.IGNORECASE)
    # 第二层：社交媒体噪声
    text = re.sub(r'^(?:@\w+\s*)+', '', text)
    text = re.sub(r'(?<![A-Za-z0-9.])@\w+', '', text)
    text = re.sub(r'#([a-zA-Z]\w*)', r'\1', text)
    # 第三层：时间/日期噪声
    text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?\s*,?\s*\d{0,2}\s*\w*\s*\(?\s*(?:UTC|GMT)(?:[+-]\d{1,2})?\s*\)?',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(?\s*\b(?:UTC|GMT)(?:[+-]\d{1,2})?\b\s*\)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![A-Za-z])\b\d{1,2}:\d{2}(?::\d{2})\b', '', text)
    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', text)
    text = re.sub(
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{1,2},?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r',?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b\d{1,2}\s+(?:tammikuu(?:ta)?|helmikuu(?:ta)?|maaliskuu(?:ta)?|huhtikuu(?:ta)?|'
        r'toukokuu(?:ta)?|kesäkuu(?:ta)?|heinäkuu(?:ta)?|elokuu(?:ta)?|syyskuu(?:ta)?|'
        r'lokakuu(?:ta)?|marraskuu(?:ta)?|joulukuu(?:ta)?)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b\d{1,2}\.?\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)'
        r'(?:\s+\d{4})?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d{4}\s*$', '', text)
    text = re.sub(r'^\d{4}\s+', '', text)
    text = re.sub(r'\bKorjattu\b', '', text, flags=re.IGNORECASE)
    # 第四层：翻译幻觉修复
    text = re.sub(r'\b((\w+)-\2(?:-\2){2,})\b',
                  lambda m: m.group(2) + '-' + m.group(2), text)
    def fix_word_repeat(match):
        word = match.group(1)
        if word.lower() in {'ha', 'he', 'ho', 'no', 'na', 'la', 'ya', 'go',
                             'bla', 'blah', 'etc', 'ding', 'boom', 'bang',
                             'haha', 'very', 'really', 'rah', 'nah', 'meh'}:
            return match.group(0)
        return word
    text = re.sub(r'\b(\w{3,})(?:\s+\1){2,}\b', fix_word_repeat, text, flags=re.IGNORECASE)
    text = re.sub(r'\b([A-ZÄÖÜ]{3,})(?:\s+\1){2,}\b', r'\1', text)
    # 第五层：格式归一化
    text = text.strip()
    if len(text) >= 2 and \
       ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        text = text[1:-1].strip()
    text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    return text


def post_process(df):
    """移除空/极短/重复文本。"""
    before = len(df)
    df = df[df['text'].str.len() >= 5].copy()
    df = df.drop_duplicates(subset=['text'], keep='first')
    removed = before - len(df)
    if removed > 0:
        print(f"  后处理移除: {removed} 条 ({before} → {len(df)})")
    return df


RANDOM_STATE = 42

# ============================================================
# 1. 英语 (train.tsv)
# ============================================================
print("=" * 60)
print("1. 英语 (train.tsv)")
print("=" * 60)

en = pd.read_csv('train.tsv', sep='\t', header=0, quoting=3)
en['label'] = pd.to_numeric(en['label'], errors='coerce')
en = en.dropna(subset=['label'])
en['label'] = en['label'].astype(int)
en['text'] = en['text'].astype(str).apply(clean_text)
en['lang'] = 'en'
en = en[['id', 'text', 'label', 'lang']]
en = post_process(en)
print(f"  可用: {len(en)} 条")
print()

# ============================================================
# 2. 德语 (de_hf_112024.csv)
# ============================================================
print("=" * 60)
print("2. 德语 (de_hf_112024.csv)")
print("=" * 60)

de = pd.read_csv('de_hf_112024.csv', engine='python', on_bad_lines='skip')
de['labels'] = pd.to_numeric(de['labels'], errors='coerce')
de = de.dropna(subset=['labels'])
de['label'] = de['labels'].astype(int)
de['text'] = de['text'].astype(str).apply(clean_text)
de['lang'] = 'de'
de['id'] = [f"de_hf_{i}" for i in range(len(de))]
de = de[['id', 'text', 'label', 'lang']]
de = post_process(de)
print(f"  可用: {len(de)} 条")
print()

# ============================================================
# 3. 芬兰语 — 合并 deepl + Suomi24 原生
# ============================================================
print("=" * 60)
print("3. 芬兰语 (deepl + Suomi24 原生)")
print("=" * 60)

# --- deepl ---
fi_records = []
with open('train_fi_deepl.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            fi_records.append(json.loads(line))

fi_deepl = pd.DataFrame(fi_records)
toxicity_cols = [
    'label_identity_attack', 'label_insult', 'label_obscene',
    'label_severe_toxicity', 'label_threat', 'label_toxicity'
]
fi_deepl['label'] = fi_deepl[toxicity_cols].max(axis=1).astype(int)
fi_deepl['text'] = fi_deepl['text'].apply(clean_text)
fi_deepl['lang'] = 'fi'
fi_deepl['source'] = 'deepl'
fi_deepl = fi_deepl[['id', 'text', 'label', 'lang', 'source']]

# --- Suomi24 原生 ---
fi_raw = pd.read_csv('all_annotations.tsv', sep='\t')
fi_raw['is_toxic'] = fi_raw['label'].apply(
    lambda x: 0 if str(x).startswith('not-') else 1
).astype(int)
fi_native = fi_raw.groupby('text').agg({
    'ID': 'first', 'is_toxic': 'max'
}).reset_index()
fi_native = pd.DataFrame({
    'id': fi_native['ID'],
    'text': fi_native['text'].astype(str).apply(clean_text),
    'label': fi_native['is_toxic'],
    'lang': 'fi',
    'source': 'native',
})

# 合并（原生优先去重）
fi_all = pd.concat([fi_native, fi_deepl], ignore_index=True)
fi_all = post_process(fi_all)
n_native = (fi_all['source'] == 'native').sum()
n_deepl = (fi_all['source'] == 'deepl').sum()
print(f"  合并可用: {len(fi_all)} 条 (原生: {n_native}, deepl: {n_deepl})")
print()

# ============================================================
# 4. 下采样：以德语为基准
# ============================================================
print("=" * 60)
print("4. 下采样平衡")
print("=" * 60)

target_n = len(de)
print(f"基准: 德语 {target_n} 条")
print()

# 德语：全量
de_final = de.copy()

# 英语：随机下采样到 target_n
en_final = en.sample(n=target_n, random_state=RANDOM_STATE).reset_index(drop=True)

# 芬兰语：原生全量 + deepl 补足到 target_n
fi_native_part = fi_all[fi_all['source'] == 'native'].copy()
fi_deepl_part = fi_all[fi_all['source'] == 'deepl'].copy()
fi_deepl_needed = target_n - len(fi_native_part)
fi_deepl_sampled = fi_deepl_part.sample(n=fi_deepl_needed, random_state=RANDOM_STATE)
fi_final = pd.concat([fi_native_part, fi_deepl_sampled], ignore_index=True)
fi_final = fi_final.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 去掉 source 列
for df in [de_final, en_final, fi_final]:
    if 'source' in df.columns:
        df.drop(columns=['source'], inplace=True)

# 合并
final = pd.concat([en_final, de_final, fi_final], ignore_index=True)

# ============================================================
# 5. 输出统计
# ============================================================
print("=" * 60)
print("最终数据集")
print("=" * 60)

print(f"\n{'语言':<4} {'总数':>7} {'毒性':>7} {'非毒性':>7} {'毒性比例':>8} {'语言占比':>8}")
print("-" * 50)
for lang in sorted(final['lang'].unique()):
    sub = final[final['lang'] == lang]
    t = (sub['label'] == 1).sum()
    nt = (sub['label'] == 0).sum()
    print(f"{lang:<4} {len(sub):>7} {t:>7} {nt:>7} {t/len(sub)*100:>7.1f}% {len(sub)/len(final)*100:>7.1f}%")

tt = (final['label'] == 1).sum()
tnt = (final['label'] == 0).sum()
print("-" * 50)
print(f"{'总计':<4} {len(final):>7} {tt:>7} {tnt:>7} {tt/len(final)*100:>7.1f}%")

# 芬兰语来源细分
fi_sub = final[final['lang'] == 'fi']
n_native_final = fi_sub['id'].astype(str).str.startswith('s24:').sum()
n_deepl_final = len(fi_sub) - n_native_final
print(f"\n芬兰语来源细分:")
print(f"  原生(Suomi24): {n_native_final} 条（全量）")
print(f"  翻译(deepl):   {n_deepl_final} 条")

# 导出
output_file = 'full_train_real_only.tsv'
final.to_csv(output_file, sep='\t', index=False, quoting=3, escapechar='\\')
print(f"\n已保存至: {output_file}")
