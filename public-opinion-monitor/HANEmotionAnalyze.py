import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, num_input, num_hidden):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(num_input, num_hidden)
        self.U = nn.Linear(num_hidden, 1)

    def forward(self, x):
        u = F.tanh(self.W(x))
        a = F.softmax(self.U(u), dim=1)
        return torch.mul(a, x).sum(dim=1)


class HAN(nn.Module):
    def __init__(self, num_embeddings = 5845,
                 num_classes = 10,
                 num_words = 60,
                 embedding_dim = 200,
                 hidden_size_gru = 50,
                 hidden_size_att = 100,
                 ):
        super(HAN, self).__init__()

        self.num_words = num_words
        self.embed = nn.Embedding(num_embeddings, embedding_dim, 0)

        self.GRU1 = nn.GRU(embedding_dim,
                           hidden_size_gru,
                           bidirectional=True, # 双向  Default: ``False``
                           batch_first=True, # : If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
                           )
        self.self_attention1 = SelfAttention(hidden_size_gru * 2, hidden_size_att)


        self.GRU2 = nn.GRU(hidden_size_gru,
                           hidden_size_gru,
                           bidirectional=True,  # 双向  Default: ``False``
                           batch_first=True,    # : If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
                           )
        self.self_attention2 = SelfAttention(hidden_size_gru * 2, hidden_size_att)

        self.fc = nn.Linear(hidden_size_att, num_classes)

    def forward(self, x:torch.Tensor):
        # view() 说明：Returns a new tensor with the same data as the self tensor but of a different shape.
        # size() 说明：Returns the size of the self tensor. The returned value is a subclass of tuple.
        # contiguous() 说明： Returns a contiguous tensor containing the same data as self tensor. If self tensor is contiguous, this function returns the self tensor.
        x = x.view(x.size(0) * self.num_words, -1).contiguous()
        x = self.embed(x)
        x, _ = self.GRU1(x)
        x = self.self_attention1(x)
        x = x.view(x.size(0) // self.num_words, self.num_words, -1)
        x, _ = self.GRU2(x)
        x = self.self_attention2(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)

