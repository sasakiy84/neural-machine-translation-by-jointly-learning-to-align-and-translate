from pathlib import Path
from typing import Callable
import torch
from torch import Tensor, nn
import json
from collections import Counter

import nltk
import MeCab
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
tagger = MeCab.Tagger("-Owakati")

VOCABULARY_SIZE = 3000
BEGIN_OF_SENTENCE_SYMBOL = "<BOS>"
END_OF_SENTENCE_SYMBOL = "<EOS>"


def tokenize_sentence_japanese(sentence: str) -> list[str]:
    """
    日本語の文を分かち書きする関数。
    MeCab を使用して、文を単語のリストに変換する。
    """
    tokenized_sentence: str = tagger.parse(sentence).strip()
    return tokenized_sentence.split(" ")


def tokenize_sentence_english(sentence: str) -> list[str]:
    """
    英語の文を分かち書きする関数。
    NLTK を使用して、文を単語のリストに変換する。
    """
    tokenized_sentence = word_tokenize(sentence, language="english")
    return [word.lower() for word in tokenized_sentence]


class Dictionary:
    def __init__(
        self,
        word_to_index: dict[str, int],
        tokenizer_function=Callable[[str], list[str]],
    ):
        """
        dict[str, int] の形式で保存されている json ファイルを読み込み、
        辞書を保持しておく
        辞書には、必ず <BOS>, <EOS>, <UNK> のキーが含まれていることを前提とする
        """
        self.word_embedding_dim = VOCABULARY_SIZE
        self.word_to_index: dict[str, int] = word_to_index
        self.index_to_word: dict[int, str] = {}
        self.tokenizer_function = tokenizer_function
        # init pytorch embeddings with word_to_index
        self.pytorch_embeddings: nn.Embedding = nn.Embedding.from_pretrained(
            torch.eye(self.word_embedding_dim), freeze=True
        )

        for key in [BEGIN_OF_SENTENCE_SYMBOL, END_OF_SENTENCE_SYMBOL, "<UNK>"]:
            if key not in word_to_index:
                raise ValueError(f"word_embedding_file must contain the key '{key}'")

        self.word_to_index = word_to_index
        self.index_to_word = {v: k for k, v in word_to_index.items()}

    def get_bos_vector(self) -> Tensor:
        """
        Returns the one-hot vector for the <BOS> token.
        """
        bos_index = self.word_to_index[BEGIN_OF_SENTENCE_SYMBOL]
        return self.pytorch_embeddings(torch.tensor(bos_index, dtype=torch.long))

    def sentence_to_indices(self, sentence: str) -> Tensor:
        words = self.tokenizer_function(sentence)
        return torch.tensor(
            [
                self.word_to_index.get(word, self.word_to_index["<UNK>"])
                for word in words
            ],
            dtype=torch.long,
        )

    def get_word_from_vector(self, vector: Tensor) -> str:
        """
        Returns the word corresponding to the one-hot vector.
        If the vector does not correspond to any word, returns <UNK>.
        """
        if not isinstance(vector, Tensor) or vector.shape[0] != self.word_embedding_dim:
            raise ValueError("vector must be a one-hot vector of the correct dimension")

        index = torch.argmax(vector).item()
        print(f"Index from vector: {index}")  # Debugging line
        # print(f"Index to word mapping: {self.index_to_word}")  # Debugging line
        return self.index_to_word.get(index, "<UNK>")

    def sentence_to_one_hot_vectors(self, sentence: str) -> Tensor:
        """
        Returns a list of one-hot vectors for the words in the sentence.
        If a word is not in the vocabulary, it uses the <UNK> symbol.
        """
        indices = self.sentence_to_indices(sentence)
        return self.pytorch_embeddings.forward(indices)

    def vectors_to_sentence(self, vectors: list[Tensor]) -> list[str]:
        """
        Returns a list of words corresponding to the one-hot vectors.
        If a vector does not correspond to any word, it uses the <UNK> symbol.
        """
        return [self.get_word_from_vector(vector) for vector in vectors]

    @classmethod
    def construct_dict_file(cls, sentences: list[list[str]], output_file: Path):
        """
        分かち書きされた文のリストを受け取り、単語とインデックスの辞書を作成し、
        指定されたファイルに保存。
        最大で [word_embedding_dim] 個の単語をサポート。
        ただし、<BOS>, <EOS>, <UNK> の3つの特別なトークンは必ず含まれる
        """
        word_to_index = {
            BEGIN_OF_SENTENCE_SYMBOL: 0,
            END_OF_SENTENCE_SYMBOL: 1,
            "<UNK>": 2,
        }
        index = 3

        word_counter = Counter()
        for sentence in sentences:
            for word in sentence:
                word_counter[word] += 1
        for word, _count in word_counter.most_common():
            if index >= VOCABULARY_SIZE:
                break
            if word not in word_to_index:
                word_to_index[word] = index
                index += 1

        assert len(word_to_index) <= VOCABULARY_SIZE, (
            f"The number of unique words exceeds the maximum limit of {VOCABULARY_SIZE}."
        )
        output_file.write_text(json.dumps(word_to_index, ensure_ascii=False, indent=2))
