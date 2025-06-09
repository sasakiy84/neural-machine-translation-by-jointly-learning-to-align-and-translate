import json
from pathlib import Path
from utils import unimplemented
import torch
from torch import nn, Tensor
from wordembedding import (
    END_OF_SENTENCE_SYMBOL,
    VOCABULARY_SIZE,
    Dictionary,
    tokenize_sentence_english,
    tokenize_sentence_japanese,
)


MAX_SENTENCE_LENGTH = 50

class Encoder(nn.Module):
    """
    Encoder RNN の実装。
    入力は、単語の one-hot ベクトルで、出力は、隠れ状態のベクトル。
    """

    def __init__(self, input_size: int = VOCABULARY_SIZE, hidden_size: int = 1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, input_vector: Tensor, previous_hidden_state: Tensor | None
    ) -> Tensor:
        """
        `h_t = f(x_t, h_(t-1))`
        """

        if previous_hidden_state is None:
            previous_hidden_state = torch.zeros(1, self.hidden_size)

        current_state = torch.tanh(
            self.W_xh(input_vector) + self.W_hh(previous_hidden_state)
        )
        return current_state


class Decoder(nn.Module):
    def __init__(
        self,
        input_size=VOCABULARY_SIZE,
        hidden_size=1000,
        context_size=1000,
        output_size=VOCABULARY_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.context_size = context_size
        self.output_size = output_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_ch = nn.Linear(context_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        input_vector: Tensor,
        previous_hidden_state: Tensor | None,
        context_vector: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if previous_hidden_state is None:
            previous_hidden_state = torch.zeros(1, self.hidden_size)
        hidden_state = torch.tanh(
            self.W_xh(input_vector)
            + self.W_hh(previous_hidden_state)
            + self.W_ch(context_vector)
        )
        output = self.softmax(self.W_ho(hidden_state))

        assert output.shape[1] == self.output_size, (
            f"Expected output size {self.output_size}, got {output.shape[1]}"
        )
        return output, hidden_state


class PytorchEncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)


class EncoderDecoderWithAttention(nn.Module):
    def __init__(
        self,
        encoder_word_embedding: Dictionary,
        decoder_word_embedding: Dictionary,
        hidden_layer_dim=1000,
        word_embedding_dim=VOCABULARY_SIZE,
        _maxout_hidden_layer_dim=500,
    ):
        super().__init__()
        self.forward_encoder = Encoder(
            input_size=word_embedding_dim, hidden_size=hidden_layer_dim // 2
        )
        self.backward_encoder = Encoder(
            input_size=word_embedding_dim, hidden_size=hidden_layer_dim // 2
        )
        self.encoder_word_embedding = encoder_word_embedding
        self.decoder_word_embedding = decoder_word_embedding
        self.decoder = Decoder()
        # `n`
        if hidden_layer_dim % 2 != 0:
            raise ValueError(f"hidden_layer_dim must be even, got {hidden_layer_dim}")
        self.hidden_layer_dim = hidden_layer_dim
        # `m`
        self.word_embedding_dim = word_embedding_dim
        # `l`
        # self.maxout_hidden_layer_dim = maxout_hidden_layer_dim

    def vectors_to_sentence_in_target_language(
        self, vectors: list[Tensor]
    ) -> list[str]:
        return self.encoder_word_embedding.vectors_to_sentence(vectors)

    def vector_to_word(self, vector: Tensor) -> str:
        return self.encoder_word_embedding.get_word_from_vector(vector)

    def calc_attention() -> Tensor:
        unimplemented()

    def get_word_vecotr_from_prob_distribution():
        unimplemented()

    def translate(self, source_sentence: str) -> str:
        """
        sentence を受け取って翻訳する
        sentence を one-hot ベクトルとしてエンコードし、
        encoder に渡し、両方向の隠れ状態を取得する。
        その後、decoder に渡して、while 文で翻訳された文を生成する。
        """
        word_vector_sequence = self.encoder_word_embedding.sentence_to_one_hot_vectors(source_sentence)

        # encoder
        current_forward_hidden_state: Tensor | None = None
        for word_vector in word_vector_sequence:
            print(
                f"processing token '{self.vector_to_word(word_vector)}' in forward encoder"
            )
            # self.encoder_word_embedding.index_to_word[torch.argmax(word_vector).item()]
            current_forward_hidden_state: Tensor = self.forward_encoder.forward(
                word_vector, current_forward_hidden_state
            )

        current_backward_hidden_state: Tensor | None = None
        for word_vector in reversed(word_vector_sequence):
            print(
                f"processing token '{self.vector_to_word(word_vector)}' in backward encoder"
            )
            current_backward_hidden_state: Tensor = self.backward_encoder.forward(
                word_vector, current_backward_hidden_state
            )

        # forward と backward の隠れ状態を結合する
        sentence_vector: Tensor = torch.cat(
            (
                current_forward_hidden_state.squeeze(0),
                current_backward_hidden_state.squeeze(0),
            ),
            dim=0,
        )
        assert sentence_vector.shape[0] == self.hidden_layer_dim, (
            f"Expected hidden layer dimension {self.hidden_layer_dim}, got {sentence_vector.shape[0]}"
        )
        print(f"sentence_vector: {sentence_vector.shape}")


        # decoder
        # 初期状態は BOS トークンのベクトル
        current_decoder_hidden_state: Tensor | None = None
        generated_words = []
        bos_vector = self.encoder_word_embedding.get_bos_vector().unsqueeze(0)
        for _ in range(MAX_SENTENCE_LENGTH):
            # attention_i = self.calc_attention(sentence_vector, current_decoder_hidden_state)
            # context_i = torch.matmul(sentence_vector, attention_i)

            print(bos_vector.shape)
            next_word_prob_distribution, current_decoder_hidden_state = (
                self.decoder.forward(
                    bos_vector,
                    current_decoder_hidden_state,
                    sentence_vector,
                )
            )
            print(f"next_word_prob_distribution: {next_word_prob_distribution.shape}")
            word = self.vectors_to_sentence_in_target_language(
                [next_word_prob_distribution.squeeze()]
            )[0]
            generated_words.append(word)
            print(f"Generated word: {word}")
            print(f"{' '.join(generated_words)}")

            if word == END_OF_SENTENCE_SYMBOL:
                break

        return " ".join(
            self.vectors_to_sentence_in_target_language(word_vector_sequence)
        )


def setup_word_embeddings() -> tuple[Dictionary, Dictionary]:
    """
    日本語、英語の単語埋め込みを返す
    事前に用意していた JSON ファイルを読み込む
    JSON ファイルは、`construct_dict.py` で生成する
    """
    english_wordembedding_file = Path("news-commentary-corpus.en.json")
    japanese_wordembedding_file = Path("news-commentary-corpus.ja.json")

    word_to_index_en: dict[str, int] = json.loads(
        english_wordembedding_file.read_text()
    )
    if not isinstance(word_to_index_en, dict):
        raise ValueError("word_embedding_file must contain a dictionary of str to int")
    english_word_embedding = Dictionary(
        word_to_index=word_to_index_en, tokenizer_function=tokenize_sentence_english
    )

    word_to_index_ja: dict[str, int] = json.loads(
        japanese_wordembedding_file.read_text()
    )
    if not isinstance(word_to_index_ja, dict):
        raise ValueError("word_embedding_file must contain a dictionary of str to int")
    japanese_word_embedding = Dictionary(
        word_to_index=word_to_index_ja, tokenizer_function=tokenize_sentence_japanese
    )

    return english_word_embedding, japanese_word_embedding


def main():
    print("loading word embeddings...")
    english_word_embedding, japanese_word_embedding = setup_word_embeddings()

    encoder_decoder_model = EncoderDecoderWithAttention(
        encoder_word_embedding=english_word_embedding,
        decoder_word_embedding=japanese_word_embedding,
        hidden_layer_dim=1000,
        word_embedding_dim=VOCABULARY_SIZE,
        _maxout_hidden_layer_dim=500,
    )

    en_sentence = "This is a test sentence."

    print(f"Translating from English to Japanese: {en_sentence}")
    translated_sentence = encoder_decoder_model.translate(en_sentence)
    print(f"Translated sentence: {translated_sentence}")


if __name__ == "__main__":
    main()
