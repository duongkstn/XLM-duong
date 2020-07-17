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
title = 'product_title'
root = os.path.dirname(os.path.abspath(__file__))
file_en = os.path.join(root, 'en', 'train_en.csv')
file_zh = os.path.join(root, 'zh', 'train_tcn.csv')
file_en_text = os.path.join(root, 'en', 'all.en')
file_zh_text = os.path.join(root, 'zh', 'all.zh')
file_en_cleaned = os.path.join(root, 'en', 'train_en_cleaned.csv')
file_zh_cleaned = os.path.join(root, 'zh', 'train_tcn_cleaned.csv')


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
    x = [t for t in l if (len(set(t) - en_chars_num) > 0 or len(t) > 1) ] #remove alon english character
    x = ' '.join(x)
    x = x.strip()
    return x


def csv2text(file_csv, file_text, file_csv_cleaned, lg):
    df = pd.read_csv(file_csv)
    df.drop_duplicates(inplace=True)
    df = df.replace('', np.nan)
    df.dropna(inplace=True)
    df[title] = df[title].apply(lambda x: clean(x, lg))
    df.drop_duplicates(inplace=True)
    df = df.replace('', np.nan)
    df.dropna(inplace=True)
    l = df[title].values.tolist()
    with open(file_text, 'w') as f:
        for line in l:
            f.write(line + '\n')
    df.to_csv(file_csv_cleaned, index=False)

csv2text(file_en, file_en_text, file_en_cleaned, 'en')
csv2text(file_zh, file_zh_text, file_zh_cleaned, 'zh')
