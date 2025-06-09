"""
MeCab を使って分かち書きした日本語の文を保存する
"""

import csv
import MeCab
from pathlib import Path

input_file = Path("news-commentary-corpu.tsv")
output_file = Path("data/en-ja.txt")

tagger = MeCab.Tagger("-Owakati")

writer = csv.writer(output_file.open("w", encoding="utf-8"), delimiter="\t")
reader = csv.reader(input_file.open("r", encoding="utf-8"), delimiter="\t")

for row in reader:
    if len(row) != 2:
        continue  # Skip rows that do not have exactly two columns
    en_sentence, ja_sentence = row
    ja_tokenized = tagger.parse(ja_sentence).strip()
    writer.writerow([en_sentence, ja_tokenized])


