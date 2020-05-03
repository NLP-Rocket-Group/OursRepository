import torch
import torch.nn as nn
import torch.nn.functional as F

isPrint = False

class SelfAttention(nn.Module):
    def __init__(self, num_input, num_hidden):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(num_input, num_hidden)
        self.U = nn.Linear(num_hidden, 1)

    def forward(self, x):
        u = F.tanh(self.W(x))
        a = F.softmax(self.U(u), dim=1)
        return torch.mul(a, x).sum(dim=1)

class SelfAttention2(nn.Module):
    def __init__(self, num_input,):
        super(SelfAttention2, self).__init__()
        self.W = nn.Linear(num_input, num_input)
        self.U = nn.Linear(num_input, 1)

    def forward(self, x):
        u = F.tanh(self.W(x))
        a = F.softmax(self.U(u), dim=1)
        return torch.mul(a, x).sum(dim=1)

class HAN(nn.Module):
    def __init__(self, num_embeddings = 5845,
                 num_classes = 10,
                 num_words = 100,
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


        self.GRU2 = nn.GRU(hidden_size_gru * 2,
                           hidden_size_gru,
                           bidirectional=True,  # 双向  Default: ``False``
                           batch_first=True,    # : If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
                           )
        self.self_attention2 = SelfAttention(hidden_size_gru * 2, hidden_size_att)

        # self.fc = nn.Linear(hidden_size_att, num_classes)
        self.fc = nn.Linear(hidden_size_gru * 2, num_classes)

    def forward(self, x:torch.Tensor):
        x = x.view(x.size(0) * self.num_words, -1).contiguous()
        x = self.embed(x)
        x, _ = self.GRU1(x)
        x = x.sum(1)
        x = x.view(x.size(0) // self.num_words, self.num_words, -1)
        x, _ = self.GRU2(x)
        x = self.self_attention2(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class SimpleHAN(HAN):
    def __init__(self, *args, **kwargs):
        super(SimpleHAN, self).__init__(*args, **kwargs)

    def forward(self, x:torch.Tensor):
        x = x.view(x.size(0) * self.num_words, -1).contiguous()
        x = self.embed(x)
        x, _ = self.GRU1(x)
        x = self.self_attention1(x)
        x = x.view(x.size(0) // self.num_words, self.num_words, -1)
        x, _ = self.GRU2(x)
        x = self.self_attention2(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class ClassicalHAN(nn.Module):
    def __init__(self, num_embeddings=5845,
                 num_classes=10,
                 num_words=20,  # 每句话最多多少个词
                 num_sentence=10,  # 一篇文章多少个句子
                 embedding_dim=200,
                 hidden_size_gru=50,
                 hidden_size_att=100,
                 ):
        super(ClassicalHAN, self).__init__()

        self.num_words = num_words
        self.num_sentence = num_sentence
        self.embed = nn.Embedding(num_embeddings, embedding_dim, 0)

        self.GRU1 = nn.GRU(embedding_dim,
                           hidden_size_gru,
                           bidirectional=True,  # 双向  Default: ``False``
                           batch_first=True,
                           # : If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
                           )
        self.self_attention1 = SelfAttention2(hidden_size_gru * 2)

        self.GRU2 = nn.GRU(hidden_size_gru * 2,
                           hidden_size_gru * 2,
                           bidirectional=True,  # 双向  Default: ``False``
                           batch_first=True,
                           # : If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
                           )
        self.self_attention2 = SelfAttention2(hidden_size_gru * 4)

        self.fc = nn.Linear(hidden_size_gru * 4, num_classes)

    def forward(self, x: torch.Tensor, isSentenceSplit: bool = True):
        if isPrint: print()
        if isPrint: print()
        if isPrint: print("x:", x.shape)
        if isSentenceSplit:
            sentences = []

            for i in range(self.num_sentence):
                sentence = x[:, i * self.num_words: (i + 1) * self.num_words]
                if isPrint: print("-> sentence:", sentence.shape)
                sentence = self.embed(sentence)
                if isPrint: print("-> embed 后 sentence:", sentence.shape)
                sentence, _ = self.GRU1(sentence)
                if isPrint: print("-> GRU1 后 sentence:", sentence.shape)
                sentence = self.self_attention1(sentence)
                if isPrint: print("-> self_attention1 后 sentence:", sentence.shape)
                sentences.append(sentence)
            sentences = torch.cat(sentences, dim=1)
            if isPrint: print()
            if isPrint: print("-> torch.cat(sentences, dim=1) 后 sentences:", sentences.shape)
            x = sentences.view(sentences.size(0), self.num_sentence, -1)
            if isPrint: print("-> sentences.view 后 x:", x.shape)
        else:
            sentences = self.embed(x)
            if isPrint: print("-> embed 后 sentences:", sentences.shape)
            sentences, _ = self.GRU1(sentences)
            if isPrint: print("-> GRU1 后 sentences:", sentences.shape)
            sentences = self.self_attention1(sentences)
            if isPrint: print("-> self_attention1 后 sentences:", sentences.shape)
            x = sentences
        if isPrint: print()
        if isPrint: print("view2 后 x:", x.shape)
        x, _ = self.GRU2(x)
        if isPrint: print("GRU2 后 x:", x.shape)
        x = self.self_attention2(x)
        if isPrint: print("self_attention2 后 x:", x.shape)
        x = self.fc(x)
        if isPrint: print("fc 后 x:", x.shape)
        return F.softmax(x, dim=1)