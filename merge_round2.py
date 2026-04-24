"""
merge_round2.py — 第二轮微调数据集合并
==========================================
将两个新数据集（GermEval21 德语、Suomi24 芬兰语）与现有的
final_trilingual_dataset.tsv 合并，生成第二轮微调数据集。

新数据集处理：
  - GermEval21: Sub1_Toxic → label, 应用 clean_text v3
  - Suomi24 Finnish: 多标签去重聚合为二分类，应用 clean_text v3
  - 两个新数据集都是原生标注（非翻译），质量高于翻译数据
"""

import pandas as pd
import re

# ============================================================
# clean_text v3（与所有其他脚本完全一致）
# ============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'~{3,5}', '', text)
    text = re.sub(r'-\s*Talk\s*-', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:User|Special:Contributions)[:/]\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:@\w+\s*)+', '', text)
    text = re.sub(r'(?<![A-Za-z0-9.])@\w+', '', text)
    text = re.sub(r'#([a-zA-Z]\w*)', r'\1', text)
    text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?\s*,?\s*\d{0,2}\s*\w*\s*\(?\s*(?:UTC|GMT)(?:[+-]\d{1,2})?\s*\)?',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(?\s*\b(?:UTC|GMT)(?:[+-]\d{1,2})?\b\s*\)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![A-Za-z])\b\d{1,2}:\d{2}(?::\d{2})\b', '', text)
    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', text)
    text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}\s+(?:tammikuu(?:ta)?|helmikuu(?:ta)?|maaliskuu(?:ta)?|huhtikuu(?:ta)?|toukokuu(?:ta)?|kesäkuu(?:ta)?|heinäkuu(?:ta)?|elokuu(?:ta)?|syyskuu(?:ta)?|lokakuu(?:ta)?|marraskuu(?:ta)?|joulukuu(?:ta)?)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}\.?\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)(?:\s+\d{4})?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d{4}\s*$', '', text)
    text = re.sub(r'^\d{4}\s+', '', text)
    text = re.sub(r'\bKorjattu\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b((\w+)-\2(?:-\2){2,})\b', lambda m: m.group(2) + '-' + m.group(2), text)
    def fix_word_repeat(match):
        word = match.group(1)
        if word.lower() in {'ha', 'he', 'ho', 'no', 'na', 'la', 'ya', 'go',
                             'bla', 'blah', 'etc', 'ding', 'boom', 'bang',
                             'haha', 'very', 'really', 'rah', 'nah', 'meh'}:
            return match.group(0)
        return word
    text = re.sub(r'\b(\w{3,})(?:\s+\1){2,}\b', fix_word_repeat, text, flags=re.IGNORECASE)
    text = re.sub(r'\b([A-ZÄÖÜ]{3,})(?:\s+\1){2,}\b', r'\1', text)
    text = text.strip()
    if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        text = text[1:-1].strip()
    text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    return text


# ============================================================
# 读取现有数据集
# ============================================================
print("=" * 60)
print("读取现有数据集")
print("=" * 60)

existing = pd.read_csv('final_trilingual_dataset.tsv', sep='\t', header=0, quoting=3)
existing['text'] = existing['text'].astype(str)
print(f"现有数据: {len(existing)} 条")
for lang in sorted(existing['lang'].unique()):
    sub = existing[existing['lang'] == lang]
    print(f"  [{lang}] {len(sub)} 条 (毒性: {(sub['label']==1).sum()})")
print()


# ============================================================
# 处理 GermEval21 德语数据
# ============================================================
print("=" * 60)
print("处理 GermEval21 德语数据")
print("=" * 60)

ge = pd.read_csv('GermEval21_TrainData.csv')
print(f"原始: {len(ge)} 条")

# 映射字段
ge_df = pd.DataFrame({
    'text': ge['comment_text'].astype(str).apply(clean_text),
    'label': ge['Sub1_Toxic'].astype(int),
    'lang': 'de',
})

# 后处理
before = len(ge_df)
ge_df = ge_df[ge_df['text'].str.len() >= 5].copy()
ge_df = ge_df.drop_duplicates(subset=['text'], keep='first')
print(f"清洗后: {len(ge_df)} 条 (移除 {before - len(ge_df)})")
print(f"  毒性: {(ge_df['label']==1).sum()} ({(ge_df['label']==1).sum()/len(ge_df)*100:.1f}%)")
print(f"  非毒性: {(ge_df['label']==0).sum()}")

# 检查与现有德语数据是否有文本重叠
existing_de_texts = set(existing[existing['lang'] == 'de']['text'].tolist())
overlap = ge_df['text'].isin(existing_de_texts).sum()
print(f"  与现有德语数据重叠: {overlap} 条")
if overlap > 0:
    ge_df = ge_df[~ge_df['text'].isin(existing_de_texts)].copy()
    print(f"  去重叠后: {len(ge_df)} 条")

# 生成ID
ge_df['id'] = [f"de_germeval_{i}" for i in range(len(ge_df))]
ge_df = ge_df[['id', 'text', 'label', 'lang']]
print()


# ============================================================
# 处理芬兰语 Suomi24 数据
# ============================================================
print("=" * 60)
print("处理芬兰语 Suomi24 数据")
print("=" * 60)

fi = pd.read_csv('all_annotations__1_.tsv', sep='\t')
print(f"原始: {len(fi)} 条 ({fi['label'].nunique()} 个标签类别)")

# 转换为二分类：not- 开头 = 0，其余 = 1
fi['is_toxic'] = fi['label'].apply(lambda x: 0 if x.startswith('not-') else 1).astype(int)

# 按文本聚合去重：同一文本有多个标签时，任一毒性标签即为毒性
fi_dedup = fi.groupby('text').agg({
    'ID': 'first',
    'is_toxic': 'max'
}).reset_index()

print(f"去重聚合后: {len(fi_dedup)} 条")

# 清洗
fi_df = pd.DataFrame({
    'text': fi_dedup['text'].astype(str).apply(clean_text),
    'label': fi_dedup['is_toxic'],
    'lang': 'fi',
    'id': fi_dedup['ID'],
})

# 后处理
before = len(fi_df)
fi_df = fi_df[fi_df['text'].str.len() >= 5].copy()
fi_df = fi_df.drop_duplicates(subset=['text'], keep='first')
print(f"清洗后: {len(fi_df)} 条 (移除 {before - len(fi_df)})")
print(f"  毒性: {(fi_df['label']==1).sum()} ({(fi_df['label']==1).sum()/len(fi_df)*100:.1f}%)")
print(f"  非毒性: {(fi_df['label']==0).sum()}")

# 检查与现有芬兰语数据是否有文本重叠
existing_fi_texts = set(existing[existing['lang'] == 'fi']['text'].tolist())
overlap = fi_df['text'].isin(existing_fi_texts).sum()
print(f"  与现有芬兰语数据重叠: {overlap} 条")
if overlap > 0:
    fi_df = fi_df[~fi_df['text'].isin(existing_fi_texts)].copy()
    print(f"  去重叠后: {len(fi_df)} 条")

# 重新生成ID
fi_df['id'] = [f"fi_suomi24_{i}" for i in range(len(fi_df))]
fi_df = fi_df[['id', 'text', 'label', 'lang']]
print()


# ============================================================
# 合并
# ============================================================
print("=" * 60)
print("合并数据集")
print("=" * 60)

final = pd.concat([existing, ge_df, fi_df], ignore_index=True)

# 最终统计
print(f"最终数据集: {len(final)} 条\n")
print(f"{'语言':<6} {'总数':>6} {'毒性':>6} {'非毒性':>6} {'毒性%':>7} {'占总比%':>8}")
print("-" * 45)
for lang in sorted(final['lang'].unique()):
    sub = final[final['lang'] == lang]
    toxic = (sub['label'] == 1).sum()
    nontoxic = (sub['label'] == 0).sum()
    print(f"{lang:<6} {len(sub):>6} {toxic:>6} {nontoxic:>6} {toxic/len(sub)*100:>6.1f}% {len(sub)/len(final)*100:>7.1f}%")

total_toxic = (final['label'] == 1).sum()
total_nontoxic = (final['label'] == 0).sum()
print("-" * 45)
print(f"{'总计':<6} {len(final):>6} {total_toxic:>6} {total_nontoxic:>6} {total_toxic/len(final)*100:>6.1f}%")

# 导出
output_file = 'round2_trilingual_dataset.tsv'
final.to_csv(output_file, sep='\t', index=False, quoting=3, escapechar='\\')
print(f"\n已保存至: {output_file}")
