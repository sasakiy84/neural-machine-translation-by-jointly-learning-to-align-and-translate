from utils import unimplemented
import torch
from torch import nn, Tensor

BEGIN_OF_SENTENCE_SYMBOL = "<BOS>"
END_OF_SENTENCE_SYMBOL = "<EOS>"

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        unimplemented()

    def forward(input: Tensor, previous_hidden_state: Tensor | None) -> Tensor:
        """
        `h_t = f(x_t, h_(t-1))`
        """

        current_hidden_state: Tensor = Tensor() # TODO
        if previous_hidden_state is not None:
            assert previous_hidden_state.shape == current_hidden_state.shape

        unimplemented()

        return current_hidden_state
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        unimplemented()

    def forward(sentence: list) -> tuple[str, Tensor]:
        unimplemented()

class EncoderDecoderWithAttention(nn.Module):
    def __init__(self, hidden_layer_dim = 1000, word_embedding_dim = 620, maxout_hidden_layer_dim = 500):
        self.forward_encoder = Encoder()
        self.backward_encoder = Encoder()
        self.decoder = Decoder()
        # `n`
        if hidden_layer_dim // 2 != 0:
            raise ValueError("hidden_layer_dim must be even")
        self.hidden_layer_dim = hidden_layer_dim
        # `m`
        self.word_embedding_dim = word_embedding_dim
        # `l`
        self.maxout_hidden_layer_dim = maxout_hidden_layer_dim

        unimplemented()

    def words_to_vector(self, words: list[str]) -> list[Tensor]:
        unimplemented()
    
    def vector_to_word(self, vector: Tensor) -> str:
        unimplemented()

    def calc_attention() -> Tensor:
        unimplemented()

    def get_word_vecotr_from_prob_distribution():
        unimplemented()

    def translate(self, source_sentence: list[str]) -> list[str]:
        word_vector_sequence = self.words_to_vector(source_sentence)

        # -------------------------------------
        # Bi-Directional RNN encoder
        # -------------------------------------
        each_hidden_layer_dim = self.hidden_layer_dim / 2
        # forward encode
        forward_hidden_states: list[Tensor] = []
        for word_vector in word_vector_sequence:
            assert word_vector.shape[0] == self.word_embedding_dim

            previous_state = forward_hidden_states[-1] if len(forward_hidden_states) > 0 else None
            current_hidden_state: Tensor = self.forward_encoder.forward(word_vector, previous_state)
            forward_hidden_states.append(current_hidden_state)

            assert current_hidden_state.shape[0] == each_hidden_layer_dim
        
        # backfoard encode
        backward_hidden_states: list[Tensor] = []
        for word_vector in Tensor.flip(word_vector_sequence):
            previous_state = backward_hidden_states[-1] if len(backward_hidden_states) > 0 else None
            current_hidden_state: Tensor = self.backward_encoder.forward(word_vector, previous_state)
            backward_hidden_states.append(current_hidden_state)

            assert current_hidden_state.shape[0] == each_hidden_layer_dim

        assert len(backward_hidden_states) == len(forward_hidden_states)
        
        hidden_states: list[Tensor] = []
        for forward_hidden_state, backward_hidden_state in zip(forward_hidden_states, reversed(backward_hidden_states)):
            hidden_states.append(
                torch.cat((forward_hidden_state, backward_hidden_state), dim=0)
            )

        # h_i は x_i に対応している
        assert len(hidden_states) == len(word_vector_sequence)

        # -------------------------------------
        # RNN Decoder with attention
        # -------------------------------------

        previous_word_vector = self.words_to_vector(BEGIN_OF_SENTENCE_SYMBOL)
        previous_state: Tensor | None = None
        generated_seq: list[str] = []
        while generated_seq[-1] != END_OF_SENTENCE_SYMBOL:
            attention_i = self.calc_attention(hidden_states, previous_state)
            # TODO: matrix として計算したい。各関数の返り値をあわせる　x....
            context_i = torch.matmul(hidden_states, attention_i)
            # TODO: ここの probability まわりの流れを確認する
            # p. 3 の g, f の関数が何をやっているかわからない
            next_word_prob_distribution, current_state = self.decoder.forward(previous_word_vector, previous_state, context_i)
            current_word_vector = self.get_word_vecotr_from_prob_distribution(next_word_prob_distribution)
            generated_seq.append(
                self.vector_to_word(current_word_vector)
            )

            previous_state = current_state
            previous_word_vector = current_word_vector
            
        
        # remove END_OF_SENTENCE_SYMBOL
        generated_seq.pop()

        return generated_seq


def main():
    unimplemented()

main()