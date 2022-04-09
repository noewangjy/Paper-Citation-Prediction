import torch
import torch.nn as nn


class EmbeddingClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,

                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

