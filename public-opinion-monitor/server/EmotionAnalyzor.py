from server.Model import *
from server.DictionaryEmotionAnalyze import *
import torch
import numpy
import thulac
thulac = thulac.thulac()
import jieba

from gensim.models import Word2Vec, KeyedVectors

class EmotionAnalyzor:
    def __init__(self,
                modelSelect='HAN', # 可取值为 'HAN','SimpleHAN','ClassicalHAN', "ClassicalHAN600000",
                wordVecFilePath = '../../DataSets/Word2Vect/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding_Min.txt',
                modelData = 'data/EmotionAnalyzeModelData_300_600.model',
             ):

        # 加载词向量
        # word2vec = Word2Vec.load(wordVecFilePath).wv
        self.word2vec = KeyedVectors.load_word2vec_format(wordVecFilePath, binary=False, limit=100000)
        self.word2vec.init_sims(replace=True)

        wordEmbedding = [self.word2vec[word] for word in self.word2vec.index2word]
        self.word2index = {word:i for i, word in enumerate(self.word2vec.index2word)}

        # 创建模型
        wordEmbedding = torch.FloatTensor(wordEmbedding)
        num_embeddings = len(self.word2vec.index2word)
        if modelSelect == 'HAN':
            self.model = HAN(num_embeddings,
                        num_classes=3,
                        embedding_dim=self.word2vec.wv.vector_size,
                        num_words=100,
                        hidden_size_gru=300,
                        hidden_size_att=500,
                        )
        elif modelSelect == 'SimpleHAN':
            self.model = SimpleHAN(num_embeddings,
                        num_classes=3,
                        embedding_dim=self.word2vec.wv.vector_size,
                        num_words=100,
                        hidden_size_gru=300,
                        hidden_size_att=500,
                        )
        elif modelSelect == 'ClassicalHAN':
            self.model = ClassicalHAN(num_embeddings,
                        num_classes = 3,
                        embedding_dim = self.word2vec.wv.vector_size,
                        num_words = 10,
                        num_sentence = 10,
                        hidden_size_gru = 512,
                        hidden_size_att = 768,
                        )
        elif modelSelect == 'ClassicalHAN600000':
            self.model = ClassicalHAN(num_embeddings,
                                      num_classes=3,
                                      embedding_dim=self.word2vec.wv.vector_size,
                                      num_words=10,
                                      num_sentence=10,
                                      hidden_size_gru=200,
                                      hidden_size_att=400,
                        )
        print(self.model)

        # 凯明初始化
        modelParams = self.model.parameters()
        for param in modelParams:
            if len(param.data.shape) > 1:
                torch.nn.init.kaiming_normal(param.data)

        self.model.embed.from_pretrained(wordEmbedding)

        # 载入模型数据
        self.model.load_state_dict(torch.load(modelData, map_location=torch.device('cpu')))
        self.model.eval()
        print('舆情分析系统初始化完毕！调用 emotion_analyze(article) 方法来获得评价')

    def analyze(self, article:str):
        '''
        情感分析
        :param article: 文章或句子
        :return: -1， 0 ，1    （分别代表负面评价、中性评价及正面评价）
        '''
        words = list(jieba.cut(article))
        token = [self.word2index[words[i]] if i < len(words) and words[i] in self.word2index else 0
                 for i in range(100)]
        token = torch.from_numpy(numpy.array([token])).long()
        # print('在词向量中的词汇：', [word for word in words if word in self.word2index], " 评分： ", self.model(token).data.numpy())
        res = self.model(token).data.max(1)[1].numpy()

        if res == 2: res = -1
        return res

if __name__ == "__main__":
    # emotionAnalyzor = EmotionAnalyzor(
    #     modelSelect='SimpleHAN',
    #     wordVecFilePath = '../../../DataSets/Word2Vect/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding_Min.txt',
    #     modelData = '../data/EmotionAnalyzeModel_balance_TWV.model'
    #                                 )

    # emotionAnalyzor = EmotionAnalyzor(
    #     modelSelect='ClassicalHAN',
    #     wordVecFilePath = '../../../DataSets/Word2Vect/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding_Min.txt',
    #     modelData = '../data/EmotionAnalyzeModelData_ClassicalHAN_OB_plus.model'
    #                                 )

    emotionAnalyzor = EmotionAnalyzor(
        modelSelect='ClassicalHAN600000',
        wordVecFilePath = '../../../DataSets/Word2Vect/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding_Min.txt',
        modelData = '../data/EmotionAnalyzeModelData_ClassicalHAN_Test.model'
                                    )
    # test：
    testArticles = [
        "虽然购物的时候发生了一些不愉快，但货物质量挺好，最后还是给五星好评，下次再来",
        "一点都不好玩！",
        "自从来了这家餐厅吃过饭，我就开始拉肚子。",
        "和女朋友一起来，很开心",
        "绝对好评！",
        "卖家很热情，赞一个",
        "卖家很热情！",
        "很热情！",
        "赞一个",
        "超赞！",
        "赞个屁！",
        "赞你妹！",
        "菜都臭了，怎么吃？！",
        "根本没法下咽！",
        "好香！",
        "美味佳肴！",
        "武汉在院患者清零：来之不易更需珍惜",
        "相较于非重点地区，武汉现阶段的常态化防控复杂性还体现在，一方面，由于前期医疗资源承压吃紧，复工复产进度相对滞后，医疗秩序如何平稳过渡到正常状态面临考验；另一方面，复工复产加速也要做到与新的疫情防控形势“兼容”，这需要做好配套的调试和安排，尽快帮助城市运转秩序切换到常态化防控轨道上来。",
    ]
    for article in testArticles:
        print(article[: min(40, len(article))], " ----评价---- ", emotionAnalyzor.analyze(article))
