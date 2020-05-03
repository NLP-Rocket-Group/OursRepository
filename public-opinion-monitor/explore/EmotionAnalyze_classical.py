from explore.HANModel import *
import torch
import numpy
import thulac
thulac = thulac.thulac()
import jieba
from gensim.models import KeyedVectors

isPrint = False

class SelfAttention(nn.Module):
    def __init__(self, num_input):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(num_input, num_input)
        self.U = nn.Linear(num_input, 1)

    def forward(self, x):
        if isPrint : print("SelfAttention 前 x.shape：", x.shape)
        u = F.tanh(self.W(x))
        if isPrint : print("--> F.tanh(self.W(x)) 后 u.shape：", u.shape, "self.W：", self.W)
        a = F.softmax(self.U(u), dim=1)
        if isPrint : print("--> F.softmax(self.U(u), dim=1) 后 a.shape：", a.shape, "self.U：", self.U)
        res = torch.mul(a, x).sum(dim=1)
        if isPrint : print("--> torch.mul(a, x).sum(dim=1) 后 res.shape：", res.shape)
        return res


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
        self.self_attention1 = SelfAttention(hidden_size_gru * 2)


        self.GRU2 = nn.GRU(hidden_size_gru * 2,
                           hidden_size_gru * 2,
                           bidirectional=True,  # 双向  Default: ``False``
                           batch_first=True,    # : If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
                           )
        self.self_attention2 = SelfAttention(hidden_size_gru * 4)

        self.fc = nn.Linear(hidden_size_att * 2, num_classes)

    def forward(self, x:torch.Tensor, isSentenceSplit:bool=True):
        if isPrint : print()
        if isPrint : print()
        if isPrint : print("x:", x.shape)
        if isSentenceSplit:
            sentences = []

            for i in range(self.num_sentence):
                sentence = x[:, i * self.num_words: (i + 1) * self.num_words]
                if isPrint : print("-> sentence:", sentence.shape)
                sentence = self.embed(sentence)
                if isPrint : print("-> embed 后 sentence:", sentence.shape)
                sentence, _ = self.GRU1(sentence)
                if isPrint : print("-> GRU1 后 sentence:", sentence.shape)
                sentence = self.self_attention1(sentence)
                if isPrint : print("-> self_attention1 后 sentence:", sentence.shape)
                sentences.append(sentence)
            sentences = torch.cat(sentences, dim=1)
            if isPrint : print()
            if isPrint : print("-> torch.cat(sentences, dim=1) 后 sentences:", sentences.shape)
            x = sentences.view(sentences.size(0), self.num_sentence, -1)
            if isPrint : print("-> sentences.view 后 x:", x.shape)
        else:
            sentences = self.embed(x)
            if isPrint : print("-> embed 后 sentences:", sentences.shape)
            sentences, _ = self.GRU1(sentences)
            if isPrint : print("-> GRU1 后 sentences:", sentences.shape)
            sentences = self.self_attention1(sentences)
            if isPrint : print("-> self_attention1 后 sentences:", sentences.shape)
            x = sentences
        if isPrint : print()
        if isPrint : print("view2 后 x:", x.shape)
        x, _ = self.GRU2(x)
        if isPrint : print("GRU2 后 x:", x.shape)
        x = self.self_attention2(x)
        if isPrint : print("self_attention2 后 x:", x.shape)
        x = self.fc(x)
        if isPrint : print("fc 后 x:", x.shape)
        return F.softmax(x, dim=1)


# def init(wordVecFilePath = '../../DataSets/Word2Vect/xiejunjie_300_jieba/wiki_han_word2vec_300维度.model'):
def init(wordVecFilePath = '../../DataSets/Word2Vect/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding_Min.txt'):
    '''
    初始化
    '''
    # 加载词向量
    # word2vec = Word2Vec.load(wordVecFilePath).wv
    word2vec = KeyedVectors.load_word2vec_format(wordVecFilePath, binary=False, limit=100000)
    word2vec.init_sims(replace=True)

    wordEmbedding = [word2vec[word] for word in word2vec.index2word]
    word2index = {word:i for i, word in enumerate(word2vec.index2word)}

    # 创建模型
    wordEmbedding = torch.FloatTensor(wordEmbedding)
    num_embeddings = len(word2vec.index2word)
    model = HAN(num_embeddings,
                num_classes=3,
                embedding_dim=word2vec.wv.vector_size,
                num_words=10,
                num_sentence=10,
                hidden_size_gru=200,
                hidden_size_att=400,
                )
    print(model)

    # 凯明初始化
    modelParams = model.parameters()
    for param in modelParams:
        if len(param.data.shape) > 1:
            torch.nn.init.kaiming_normal(param.data)

    model.embed.from_pretrained(wordEmbedding)

    # 载入模型数据
    # model.load_state_dict(torch.load('EmotionAnalyzeModelData_300_600.model', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('data/EmotionAnalyzeModelData_ClassicalHAN_Test.model', map_location=torch.device('cpu')))
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
    print('在词向量中的词汇：', [word for word in words if word in word2index], " 评分： ", (model(token).data.numpy()*100).astype('int') )
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
    article = "卖家很热情！"
    print(article, " ----评价---- ", emotion_analyze(article))
    article = "很热情！"
    print(article, " ----评价---- ", emotion_analyze(article))
    article = "赞一个"
    print(article, " ----评价---- ", emotion_analyze(article))
    article = "超赞！"
    print(article, " ----评价---- ", emotion_analyze(article))
    article = "武汉在院患者清零：来之不易更需珍惜"
    print(article," ----评价---- ",emotion_analyze(article))
    article = "相较于非重点地区，武汉现阶段的常态化防控复杂性还体现在，一方面，由于前期医疗资源承压吃紧，复工复产进度相对滞后，医疗秩序如何平稳过渡到正常状态面临考验；另一方面，复工复产加速也要做到与新的疫情防控形势“兼容”，这需要做好配套的调试和安排，尽快帮助城市运转秩序切换到常态化防控轨道上来。"
    print(article[0:20]," ----评价---- ",emotion_analyze(article))



