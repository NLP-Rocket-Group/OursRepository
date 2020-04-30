import torch
import torch.nn as nn
import torch.nn.functional as F

#参考 https://github.com/EdGENetworks/attention-networks-for-classification/blob/master/attention_model_validation_experiments.ipynb  华盛顿大学教授的源代码

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
                 num_words = 20,        # 每句话最多多少个词
                 num_sentence = 10,     # 一篇文章多少个句子
                 embedding_dim = 200,
                 hidden_size_gru = 50,
                 hidden_size_att = 100,
                 ):
        super(HAN, self).__init__()

        self.num_words = num_words
        self.num_sentence = num_sentence
        self.embed = nn.Embedding(num_embeddings, embedding_dim, 0)

        self.GRU1 = nn.GRU(embedding_dim,
                           hidden_size_gru,
                           bidirectional=True,  # 双向  Default: ``False``
                           batch_first=True,    # : If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
                           )
        self.self_attention1 = SelfAttention(hidden_size_gru * 2, hidden_size_att)


        self.GRU2 = nn.GRU(hidden_size_gru * num_sentence,
                           hidden_size_gru,
                           bidirectional=True,  # 双向  Default: ``False``
                           batch_first=True,    # : If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
                           )
        self.self_attention2 = SelfAttention(hidden_size_gru * num_sentence, hidden_size_att)

        # self.fc = nn.Linear(hidden_size_att, num_classes)
        self.fc = nn.Linear(hidden_size_gru * 2, num_classes)

    def forward(self, x:torch.Tensor):
        # print("x:", x.shape, self.num_words)
        sentences = []

        for i in range(self.num_sentence):
            sentence = x[i * self.num_words, (i + 1) * self.num_words]
            sentence = sentence.view(sentence.size(0) * self.num_words, -1).contiguous()
            # print("view 后 x:", x.shape)
            sentence = self.embed(sentence)
            # print("embed 后 x:", x.shape)
            sentence, _ = self.GRU1(sentence)
            # print("GRU1 后 x:", x.shape)
            sentence = self.self_attention1(sentence)
            # print("self_attention1 后 x:", x.shape)
            sentences.append(sentence)
        sentences = torch.cat(sentences)

        x = sentences.view(sentences.size(0) // self.num_words // self.num_sentence, self.num_words * self.num_sentence, -1)
        # print("view2 后 x:", x.shape)
        x, _ = self.GRU2(x)
        # print("GRU2 后 x:", x.shape)
        x = self.self_attention2(x)
        # print("self_attention2 后 x:", x.shape)
        x = self.fc(x)
        # print("fc 后 x:", x.shape)
        return F.softmax(x, dim=1)

