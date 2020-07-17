import pandas as pd
import argparse
import unicodedata
import jieba
import numpy as np
import os
import re
import string
en_chars = set(string.ascii_lowercase)
en_chars_num = set(string.ascii_lowercase + string.digits)
square_character = [92432, 78855, 159655]
root = os.path.dirname(os.path.abspath(__file__))
file_en = os.path.join(root, 'dev_en.csv')
file_zh = os.path.join(root, 'dev_tcn.csv')
file_en_text_dev = os.path.join(root, 'shopdev2020-ref.en')
file_zh_text_dev = os.path.join(root, 'shopdev2020-ref.zh')
file_en_text_test = os.path.join(root, 'shoptest2020-ref.en')
file_zh_text_test = os.path.join(root, 'shoptest2020-ref.zh')


file_en_cleaned = os.path.join(root, 'dev_en_cleaned.csv')
file_zh_cleaned = os.path.join(root, 'dev_tcn_cleaned.csv')


def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)


def clean(x, lg):
    x = x.lower().strip()
    x = unicodedata.normalize('NFKC', x)
    x = remove_emoji(x).replace('\\n', ' ')
    if lg == 'zh':
        x = ''.join(x for x in jieba.cut(x, cut_all=False))
    x = ''.join([t if (t.isalpha() or t.isdigit() or t == ' ') else ' ' for t in x])
    x = ''.join([t if (ord(t) not in square_character) else ' ' for t in x])
    l = x.split()
    x = [t for t in l if (len(set(t) - en_chars_num) > 0 or len(t) > 1)] #remove alon english character
    x = ' '.join(x)
    x = x.strip()
    return x

#
# csv2text(file_en, file_en_text_dev, file_en_cleaned, 'translation_output', 'en')
# csv2text(file_zh, file_zh_text_dev, file_zh_cleaned, 'text', 'zh')
#

df1 = pd.read_csv(file_en)
df2 = pd.read_csv(file_zh)
assert len(df1) == len(df2)
df = pd.concat([df2['text'], df1['translation_output']], axis=1)
print('len df original', len(df))
df.drop_duplicates(inplace=True)
df = df.replace('', np.nan)
df.dropna(inplace=True)
df['translation_output'] = df['translation_output'].apply(lambda x: clean(x, 'en'))
df['text'] = df['text'].apply(lambda x: clean(x, 'zh'))
print('len df first time', len(df))
print(df[df.duplicated()])
df.drop_duplicates(inplace=True)
df = df.replace('', np.nan)
df.dropna(inplace=True)
print('len df second time', len(df))

for col, file_text_dev, file_text_test, file_csv_cleaned in [('translation_output', file_en_text_dev, file_en_text_test, file_en_cleaned),
                                         ('text', file_zh_text_dev, file_zh_text_test, file_zh_cleaned)]:
    l = df[col].values.tolist()
    with open(file_text_dev, 'w') as f:
        for line in l[:800]:
            f.write(line + '\n')
    with open(file_text_test, 'w') as f:
        for line in l[800:]:
            f.write(line + '\n')

    df[[col]].to_csv(file_csv_cleaned, index=False)