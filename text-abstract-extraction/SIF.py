# 参考：
# 1. https://www.jianshu.com/p/323a615c1599

# SIF 加权模型
# 文献 [1] 提出了一个简单但有效的加权词袋模型 SIF (Smooth Inverse Frequency)，其性能超过了简单的 RNN/CNN 模型
#
# SIF 的计算分为两步：
# 1） 对句子中的每个词向量，乘以一个权重 a/(a+p_w)，其中 a 是一个常数（原文取 0.0001），p_w 为该词的词频；对于出现频率越高的词，其权重越小；
# 2） 计算句向量矩阵的第一个主成分 u，让每个句向量减去它在 u 上的投影（类似 PCA）；

import jieba
import thulac
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import TruncatedSVD


##################################################################################################
def getWordWeight(word2weight, a=1e-3):
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0
    # try:
    for key, value in word2weight.items():
        if isinstance(value, float):
            continue
        else:
            word2weight[key] = a / (a + value.count / value.sample_int)
    # except AttributeError as error:
    #     print('##############',error, '##############', key, "##############")
    return word2weight


def getWeight(word_index_map, word2weight):
    weight4ind = {}
    for word, ind in word_index_map.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


def sentences2idx(sentences, words, splitWordFunc = jieba.cut):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq1 = []
    for sentence in sentences:
        seq1.append(getSeq(sentence, words, splitWordFunc))
    x1, m1 = prepare_data(seq1)
    return x1, m1


def getSeq(sentence, words, splitWordFunc = jieba.cut):
    cutedWords = splitWordFunc(sentence)
    indexes = []
    try:
        for word in cutedWords:
            indexes.append(getIndex(words, word))
    except AttributeError as err:
        print(sentence, err)

    return indexes


def getIndex(words, w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#", "")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1


def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    # x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    # weight = np.asarray(weight, dtype='float32')
    return weight


##################################################################################################
# get SIF embedding

# 移除纠正项
# 可以看到这里的移除项是通过SVD奇异值分解训练出来的，类似于PCA主成分分析，可用于降维。
# svd.components_是一个矩阵，每一行为主题在每个单词上的分布。我们可以通过这个矩阵得到哪些词对主题贡献最大。
# 接着，在remove_pc函数中将svd.components_这一项进行了移除，原文说的是：移出（减去）所有句子向量组成的矩阵的第一个主成分(principal component / singular vector)上的投影
# 论文实验表明该方法具有不错的竞争力，在大部分数据集上都比平均词向量或者使用TFIDF加权平均的效果好，在使用PSL作为词向量时甚至能达到最优结果。
# 根据论文中的实验结果来看，在句子相似度任务上超过平均水平，甚至超过部分复杂的模型。在句子分类上效果也很明显，甚至是最好成绩。

def SIF_embedding(vectors, x, w, isRemovePC):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param vectors: vectors[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(vectors, x, w)
    if isRemovePC > 0:
        emb = remove_pc(emb, isRemovePC)
    return emb


def get_weighted_average(vectors, x, w):
    """
    Compute the weighted average vectors
    :param vectors: vectors[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, vectors.shape[1]))
    for i in range(n_samples):
        emb[i, :] = w[i, :].dot(vectors[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


##################################################################################################

class SIF:
    def __init__(self, word2VecModelFilePath='Data/wiki_han_word2vec_300维度.model', weightpara=1e-3, isRemovePc=1, isUseThulac=True):
        self.weightpara = weightpara
        self.isRemovePc = isRemovePc
        self.model = Word2Vec.load(word2VecModelFilePath)
        self.word_index_map = {}
        for index, word in enumerate(self.model.wv.index2entity):
            self.word_index_map[word] = index
        self.vectors = self.model.wv.vectors
        self.isUseThulac = isUseThulac
        if isUseThulac == True:
            self._thulac = thulac.thulac(seg_only=True)

    def getSentencesEmbedding(self, sentences=['这是一个测试句子', '这是另一个测试句子']) -> list:
        # load word weights
        word2weight = getWordWeight(self.model.wv.vocab,
                                    self.weightpara)  # word2weight['str'] is the weight for the word 'str'
        weight4index = getWeight(self.word_index_map, word2weight)  # weight4ind[i] is the weight for the i-th word
        # load sentences
        # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        x, m = sentences2idx(sentences,
                             self.word_index_map,
                             self.thulacCutWord if self.isUseThulac == True else jieba.cut)
        w = seq2weight(x, m, weight4index)  # get word weights

        return SIF_embedding(self.vectors, x, w, self.isRemovePc)

    def thulacCutWord(self, sentence):
        return np.array(self._thulac.cut(sentence))[:, 0]

if __name__ == "__main__":
    sif = SIF()
    print(sif.getSentencesEmbedding())
