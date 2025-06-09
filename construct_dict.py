"""
コーパスの語彙を作成するスクリプト。

```py
uv run construct_dict.py news-commentary-v18.en-ja.tsv news-commentary-corpus
```

のように実行すると、`news-commentary-corpus.en.json` と `news-commentary-corpus.ja.json` が生成される。

日本語は、MeCab を、英語は NLTK を使用して分かち書きする。
"""

import csv
from wordembedding import (
    Dictionary,
    tokenize_sentence_japanese,
    tokenize_sentence_english,
)
from pathlib import Path
import argparse


def construct_japanese_dict(sentences: list[str], output_file: Path):
    """
    日本語の分かち書きされた文のリストを受け取り、単語とインデックスの辞書を作成し、
    指定されたファイルに保存。
    最大で [word_embedding_dim] 個の単語をサポート。
    """

    # 分かち書きする
    split_sentences: list[list[str]] = []
    for sentence in sentences:
        tokenized_sentence = tokenize_sentence_japanese(sentence)
        split_sentences.append(tokenized_sentence)

    print(f"Constructing dictionary for {len(split_sentences)} sentences.")

    # 辞書を作成
    Dictionary.construct_dict_file(split_sentences, output_file)


def construct_english_dict(sentences: list[str], output_file: Path):
    """
    英語の分かち書きされた文のリストを受け取り、単語とインデックスの辞書を作成し、
    指定されたファイルに保存。
    最大で [word_embedding_dim] 個の単語をサポート。
    """
    # 空白で分かち書きする。ピリオドやカンマは、一つの単語として扱う。
    tokenized_sentences: list[str] = []
    for sentence in sentences:
        tokenized_sentence = tokenize_sentence_english(sentence)
        # lower にしておく
        tokenized_sentences.append(tokenized_sentence)

    print(f"Constructing dictionary for {len(tokenized_sentences)} sentences.")

    # 辞書を作成
    Dictionary.construct_dict_file(tokenized_sentences, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct a word dictionary from sentences."
    )
    parser.add_argument(
        "input_file", type=Path, help="Input file containing sentences."
    )
    parser.add_argument(
        "output_file_base", type=Path, help="Output file to save the word dictionary."
    )

    args = parser.parse_args()
    input_file: Path = args.input_file
    if not isinstance(input_file, Path):
        raise ValueError("input_file must be a Path object")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    output_file_base: Path = args.output_file_base
    if not isinstance(output_file_base, Path):
        raise ValueError("output_file_base must be a Path object")

    # load tsv such as:
    # America’s Misguided Immigration Debate	アメリカの見当違いな移民討論

    sentences_en = []
    sentences_ja = []
    with input_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                raise ValueError(
                    "Input file must contain two columns: English and Japanese sentences."
                )
            sentences_en.append(row[0])
            sentences_ja.append(row[1])

    assert len(sentences_en) == len(sentences_ja), (
        "The number of English and Japanese sentences must match."
    )

    print(f"Loaded {len(sentences_en)} sentences.")

    print("Constructing English Dictionary...")
    construct_english_dict(sentences_en, output_file_base.with_suffix(".en.json"))
    print("Constructing Japanese Dictionary...")
    construct_japanese_dict(sentences_ja, output_file_base.with_suffix(".ja.json"))
