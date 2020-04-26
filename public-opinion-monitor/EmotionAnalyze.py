from HANModel import *
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import numpy
import time, math
import torch.utils.data as data
import os
import pandas as pd
import json
import thulac
thulac = thulac.thulac()
import jieba

from gensim.models import KeyedVectors, Word2Vec

def init():
    '''
    初始化
    :return:
    '''
    # 加载词向量
    file = '../../DataSets/Word2Vect/xiejunjie_300_jieba/wiki_han_word2vec_300维度.model'
    word2vec = Word2Vec.load(file)
    word2vec.init_sims(replace=True)

    wordEmbedding = [word2vec.wv[word]  for word in word2vec.wv.index2word]
    word2index = { word:i for i, word in enumerate(word2vec.wv.index2word)}

    # 创建模型
    wordEmbedding = torch.FloatTensor(wordEmbedding)
    num_embeddings = len(word2vec.wv.index2word)
    model = HAN(num_embeddings,
                num_classes=3,
                embedding_dim=word2vec.wv.vector_size,
                num_words=100,
                hidden_size_gru=300,
                hidden_size_att=600,
                )
    print(model)

    # 凯明初始化
    modelParams = model.parameters()
    for param in modelParams:
        if len(param.data.shape) > 1:
            torch.nn.init.kaiming_normal(param.data)

    model.embed.from_pretrained(wordEmbedding)

    # 载入模型数据
    model.load_state_dict(torch.load('EmotionAnalyzeModelData_300_600.model', map_location=torch.device('cpu')))
    model.eval()
    return model, word2index,

model, word2index = init()
print('舆情分析系统初始化完毕！调用 emotion_analyze(article) 方法来获得评价')

def emotion_analyze(article:str):
    '''
    情感分析
    :param article: 文章或句子
    :return: -1， 0 ，1    （分别代表负面评价、中性评价及正面评价）
    '''
    words = list(jieba.cut(article))
    token = [word2index[words[i]] if i < len(words) and words[i] in word2index else 0
             for i in range(100)]
    token = torch.from_numpy(numpy.array([token])).long()
    print('在词向量中的词汇：', [word for word in words if word in word2index], " 评分： ", model(token).data.numpy())
    res = model(token).data.max(1)[1].numpy()

    if res == 2: res = -1
    return res

if __name__ == "__main__":
    # test：
    article = "一点都不好玩！"
    print(article," ----评价---- ",emotion_analyze(article))
    article = "自从来了这家餐厅吃过饭，我就开始拉肚子。"
    print(article," ----评价---- ",emotion_analyze(article))
    article = "和女朋友一起来，很开心"
    print(article," ----评价---- ",emotion_analyze(article))
    article = "绝对好评！"
    print(article, " ----评价---- ", emotion_analyze(article))
    article = "卖家很热情，赞一个"
    print(article, " ----评价---- ", emotion_analyze(article))
    article = "卖家很热情，超赞！"
    print(article, " ----评价---- ", emotion_analyze(article))
    article = "武汉在院患者清零：来之不易更需珍惜"
    print(article," ----评价---- ",emotion_analyze(article))
    article = "相较于非重点地区，武汉现阶段的常态化防控复杂性还体现在，一方面，由于前期医疗资源承压吃紧，复工复产进度相对滞后，医疗秩序如何平稳过渡到正常状态面临考验；另一方面，复工复产加速也要做到与新的疫情防控形势“兼容”，这需要做好配套的调试和安排，尽快帮助城市运转秩序切换到常态化防控轨道上来。"
    print(article," ----评价---- ",emotion_analyze(article))



