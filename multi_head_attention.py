import math
import unittest
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from utils import construct_future_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * self.qkv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.qkv_dim, hidden_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        """ Weight initialization taken from the UvA DL1 PyTorch Transformer tutorial. """
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Perform multi-head attention using one projection matrix. Self attention is performed when encoder_hidden_states
        is None, in which case input x represents encoder token embeddings. Otherwise, cross-attention is performed.
        In that case, input x represents the decoder hidden states.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality

        :param x: Either encoder or decoder hidden states. Shape: (N, S or T, E)
        :param encoder_hidden_states: Encoder hidden states to perform cross-attention with. Shape: (N, S, E)
        :param src_padding_mask: Used for encoder self-attention and cross-attention to handle pad tokens.
        Masks all incoming "connections" or "logits" from any token position to any pad token in a sequence.
        Shape: (N, S)
        :param future_mask: Used for decoder self-attention to avoid any token i attending to a token >i, i.e. "peaking"
        Shape: (T, T).
        :return: Contextualized token embeddings. Shape depends on attention type. (N, S, E) for encoder self-attention
        and decoder cross-attention. (N, T, E) for decoder self-attention.
        """
        if encoder_hidden_states is None:
            query, key, value = self._self_attention_projection(x)
        else:
            query, key, value = self._cross_attention_projection(encoder_hidden_states, x)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        embeddings, attention_score = self.scaled_dot_product(query, key, value, src_padding_mask, future_mask)
        N, H, S, E = embeddings.shape
        embeddings = embeddings.transpose(1, 2).reshape(N, S, H * E)
        return self.o_proj(embeddings)


    def _self_attention_projection(self, x: torch.Tensor):
        """
        Project x and interpret the result as chunks that represent q, k and v vectors for every head.
        Input x can be encoder or decoder hidden states, depending on which one calls this MHA module.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        :param x: Encoder or decoder hidden states. (N, S or T, E)
        :return: query, key and value vectors. (N, S or T, H, E/H)
        """

        qkv_states = self.qkv_proj(x)
        N, S, E = x.shape 

        # N S 3* E
        query, key, value = torch.chunk(qkv_states, 3, dim=-1)
        query = query.reshape(N, S, self.num_heads, -1)
        key = key.reshape(N, S, self.num_heads, -1)
        value = value.reshape(N, S, self.num_heads, -1)
        return query, key, value

    def _cross_attention_projection(
        self, encoder_hidden_states: torch.Tensor, decoder_hidden_states: torch.Tensor,
    ):
        """
        Projects decoder hidden states into query vectors and encoder hidden states into key and value vectors.
        The columns of W_proj determine how much independent linear combinations of the input we obtain - which we
        then interpret as heads and qkv vectors. Thus we can simply split the weight matrix and project the decoder
        hidden states x into q separately from projecting the encoder_hidden_states into k and v.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        :param encoder_hidden_states: Shape: (N, S, E)
        :param decoder_hidden_states: Shape: (N, T, E)
        :return: query vector: Shape: (N, T, H, E/H) and key and value vectors both (N, S, H, E/H)
        """
        qkv_states = self.qkv_proj(decoder_hidden_states) 
        # N S 3* E
        query, _, _ = torch.chunk(qkv_states, 3, dim=-1)
        N, S, E = decoder_hidden_states.shape  
        query = query.reshape(N, S, self.num_heads, -1)

        qkv_states = self.qkv_proj(encoder_hidden_states) 
        # N S 3* E
        N, T, E = encoder_hidden_states.shape 
        _, key, value = torch.chunk(qkv_states, 3, dim=-1)
        key = key.reshape(N, T, self.num_heads, -1)
        value = value.reshape(N, T, self.num_heads, -1)
       
        return query, key, value



       

    def scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        For cross-attention, the sequence length of q and (k,v) may differ as q is projected from decoder hidden states
        and kv from encoder hidden states.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        :param q: Tensor stacking query vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param k: Tensor stacking key vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param v: Tensor stacking value vectors for all tokens and all heads. Shape: (N, H, S or T, E/H)
        :param src_padding_mask: Used for encoder self-attention and cross-attention to handle pad tokens.
        Masks all incoming "connections" or "logits" from any token position to any pad token in a sequence.
        Shape: (N, S)
        :param future_mask: Used for decoder self-attention to avoid any token i attending to a token >i, i.e. "peaking"
        Shape: (T, T).
        :return: values (N, H, S or T, E/H), attention scores (N, H, S or T, S or T)
        """
        if src_padding_mask is not None:
            N, H, S, E = q.shape
            q.masked_fill_(~src_padding_mask.reshape(N, 1, S, 1), 0.0)
        attention_matrix = torch.matmul(q, k.transpose(-1, -2))
        if future_mask is not None:
            attention_matrix = self.mask_logits(attention_matrix, future_mask=future_mask)
        if src_padding_mask is not None:
            attention_matrix = self.mask_logits(attention_matrix, src_padding_mask=src_padding_mask)
        E = q.size(-1)
        attention_scores = torch.softmax(attention_matrix / math.sqrt(E), dim=-1)
        values = torch.matmul(attention_scores, v)
        return values, attention_scores



    @staticmethod
    def mask_logits(
        logits: torch.Tensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Reshape masks to fit the shape of the logits and set all indices with "False" to -inf

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        H = number of heads

        :param logits: Tensor containing attention logits. Shape: (N, H, S or T, S or T)
        :param src_padding_mask: Used for encoder self-attention and cross-attention to handle pad tokens.
        Masks all incoming "connections" or "logits" from any token position to any pad token in a sequence.
        Shape: (N, S)
        :param future_mask: Used for decoder self-attention to avoid any token i attending to a token >i, i.e. "peaking"
        Shape: (T, T).
        :return: masked_logits (N, H, S or T, S or T)
        """
        if future_mask is not None:
            logits = logits.masked_fill(~future_mask, float("-inf"))
        if src_padding_mask is not None:
            N, S = src_padding_mask.shape
            logits = logits.masked_fill(~src_padding_mask.reshape(N, 1, 1, S), float("-inf"))
        return logits 
            
        


class TestMultiHeadAttention(unittest.TestCase):
    def test_scaled_dot_product(self):
        mha = MultiHeadAttention(512, 8)
        q = torch.randn(4, 8, 10, 512)
        k = torch.randn(4, 8, 10, 512)
        v = torch.randn(4, 8, 10, 512)

        values, attention_scores = mha.scaled_dot_product(q, k, v)

        self.assertEqual(values.shape, (4, 8, 10, 512))
        self.assertEqual(attention_scores.shape, (4, 8, 10, 10))

        # Each attention distribution should sum up to one
        expected = torch.Tensor([1.0]).repeat((4, 8, 10))
        torch.testing.assert_close(torch.sum(attention_scores, dim=-1), expected)

        self.assertEqual(torch.any(torch.isnan(values)), False)
        self.assertEqual(True in torch.isnan(attention_scores), False)

    def test_scaled_dot_product_encoder_self_attention_mask(self):
        mha = MultiHeadAttention(hidden_dim=512, num_heads=8)
        q = torch.randn(2, 8, 10, 512, dtype=torch.float)
        k = torch.randn(2, 8, 10, 512, dtype=torch.float)
        v = torch.randn(2, 8, 10, 512, dtype=torch.float)
        mask = torch.BoolTensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )

        _, attention_scores = mha.scaled_dot_product(q, k, v, src_padding_mask=mask)

        self.assertEqual(torch.any(torch.isnan(attention_scores)), False)

        # For the first sequence we expect the last two (8-10) attention scores for every attention distribution
        # for every head to be exactly zero due to the mask we defined above. The rest should be strictly non-zero.
        self.assertEqual(torch.all(attention_scores[0, :, :, 8:] == 0), True)
        self.assertEqual(torch.any(attention_scores[0, :, :, :8] == 0), False)

        # Each attention distribution should sum up to one (all values after summing should be 1)
        expected = torch.Tensor([1.0]).repeat((2, 8, 10))
        torch.testing.assert_close(torch.sum(attention_scores, dim=-1), expected)

        # For the second sequence in the batch all attention scores should be nonzero because the mask is all ones
        self.assertEqual(torch.any(attention_scores[1] == 0), False)
    
    def test_mha_self_attention_forward(self):
        mha = MultiHeadAttention(512, 8)
        x = torch.randn(4, 10, 512, dtype=torch.float)
        output = mha.forward(x)
        self.assertEqual(output.shape, (4, 10, 512))
        self.assertEqual(torch.any(torch.isnan(output)), False)

    def test_mha_cross_attention_forward(self):
        mha = MultiHeadAttention(512, 8)
        decoder_hidden_states = torch.randn(4, 2, 512, dtype=torch.float)
        encoder_hidden_states = torch.randn(4, 10, 512, dtype=torch.float)
        output = mha.forward(
            x=decoder_hidden_states, encoder_hidden_states=encoder_hidden_states
        )
        self.assertEqual(output.shape, (4, 2, 512))
        self.assertEqual(torch.any(torch.isnan(output)), False)


    def test_future_masking(self):
        batch_size, n_heads, seq_len = 2, 2, 3  # TODO add 2 heads and batch_size=3
        logits = torch.randn(batch_size, n_heads, seq_len, seq_len, dtype=torch.float)
        future_mask = construct_future_mask(seq_len)
        self.assertEqual(future_mask.shape, (3, 3))

        masked_logits = MultiHeadAttention(512, num_heads=n_heads).mask_logits(
            logits, future_mask=future_mask
        )
        torch.testing.assert_close(
            torch.isinf(masked_logits) == 0,
            torch.BoolTensor(
                [
                    [
                        [
                            [True, False, False],
                            [True, True, False],
                            [True, True, True],
                        ],
                        [
                            [True, False, False],
                            [True, True, False],
                            [True, True, True],
                        ],
                    ],
                    [
                        [
                            [True, False, False],
                            [True, True, False],
                            [True, True, True],
                        ],
                        [
                            [True, False, False],
                            [True, True, False],
                            [True, True, True],
                        ],
                    ],
                ]
            ),
        )

    def test_src_padding_masking(self):
        batch_size, n_heads, seq_len = 2, 2, 3
        logits = torch.randn(batch_size, n_heads, seq_len, seq_len, dtype=torch.float)
        src_padding_mask = torch.BoolTensor([[True, True, True], [True, False, False]])
        self.assertEqual(src_padding_mask.shape, (2, 3))
        masked_logits = MultiHeadAttention(512, num_heads=n_heads).mask_logits(
            logits, src_padding_mask=src_padding_mask
        )
        torch.testing.assert_close(
            torch.isinf(masked_logits) == 0,
            torch.BoolTensor(
                [
                    [
                        [[True, True, True], [True, True, True], [True, True, True],],
                        [[True, True, True], [True, True, True], [True, True, True],],
                    ],
                    [
                        [
                            [True, False, False],
                            [True, False, False],
                            [True, False, False],
                        ],
                        [
                            [True, False, False],
                            [True, False, False],
                            [True, False, False],
                        ],
                    ],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
