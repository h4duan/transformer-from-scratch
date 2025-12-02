import math
import unittest

import torch


class SinusoidEncoding(torch.nn.Module):
    """
    Mostly copied from
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, hidden_dim, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs

        exponents = torch.arange(start=0, end=hidden_dim, step=2) / hidden_dim
        step = 1.0 / torch.pow(10000, exponent=exponents)
        pos = torch.arange(max_len)
        inner = torch.outer(pos, step)
        self.pos_embed = torch.empty((max_len, hidden_dim))
        self.pos_embed[:, ::2] = torch.sin(inner)
        self.pos_embed[:, 1::2] = torch.cos(inner) 

        
    def forward(self, x):
        """
        Adds positional embeddings to token embeddings.
        N = batch size
        L = sequence length
        E = embedding dim

        :param x: token embeddings. Shape: (N, L, E)
        :return: token_embeddings + positional embeddings. Shape: (N, L, E)
        """
        x = x + self.pos_embed[: x.size(1), :]
        return x


class TestSinusoidEncoding(unittest.TestCase):
    def test_create_embedding(self):
        batch = 1
        dim = 8
        len = 3
        x = torch.zeros(batch, len, dim)
        encoding = SinusoidEncoding(dim).forward(x)
        expected = torch.Tensor(
            [
                [
                    [
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                    ],
                    [
                        8.4147e-01,
                        5.4030e-01,
                        9.9833e-02,
                        9.9500e-01,
                        9.9998e-03,
                        9.9995e-01,
                        1.0000e-03,
                        1.0000e00,
                    ],
                    [
                        9.0930e-01,
                        -4.1615e-01,
                        1.9867e-01,
                        9.8007e-01,
                        1.9999e-02,
                        9.9980e-01,
                        2.0000e-03,
                        1.0000e00,
                    ],
                ]
            ]
        )
        torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)

    def test_create_embedding_multi_batch(self):
        batch = 2
        dim = 8
        len = 3
        x = torch.zeros(batch, len, dim)
        encoding = SinusoidEncoding(dim).forward(x)
        expected = torch.Tensor(
            [
                [
                    [
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                    ],
                    [
                        8.4147e-01,
                        5.4030e-01,
                        9.9833e-02,
                        9.9500e-01,
                        9.9998e-03,
                        9.9995e-01,
                        1.0000e-03,
                        1.0000e00,
                    ],
                    [
                        9.0930e-01,
                        -4.1615e-01,
                        1.9867e-01,
                        9.8007e-01,
                        1.9999e-02,
                        9.9980e-01,
                        2.0000e-03,
                        1.0000e00,
                    ],
                ],
                [
                    [
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                    ],
                    [
                        8.4147e-01,
                        5.4030e-01,
                        9.9833e-02,
                        9.9500e-01,
                        9.9998e-03,
                        9.9995e-01,
                        1.0000e-03,
                        1.0000e00,
                    ],
                    [
                        9.0930e-01,
                        -4.1615e-01,
                        1.9867e-01,
                        9.8007e-01,
                        1.9999e-02,
                        9.9980e-01,
                        2.0000e-03,
                        1.0000e00,
                    ],
                ],
            ]
        )
        torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)


if __name__ == "__main__":
    unittest.main()
