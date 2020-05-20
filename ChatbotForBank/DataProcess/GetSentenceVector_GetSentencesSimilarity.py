from collections import defaultdict
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
from scipy.optimize import linprog
import pandas as pd
import time
import pickle
from tqdm import tqdm

"""这里使用了electra_tiny预训练模型，为什么不适用bert？是因为bert的推理速度很慢，如果是不是NLG等任务，能不用bert就不用bert。
    electra模型是目前性能最优秀的预训练模型之一，在特征抽取和分类任务上，tiny模型可以达到bert-base水平，但推理速度是bert的10
    倍以上
"""

config_path = '/Users/junjiexie/Downloads/electra_tiny/bert_config_tiny.json'
checkpoint_path = '/Users/junjiexie/Downloads/electra_tiny/model.ckpt-1000000'
dict_path = '/Users/junjiexie/Downloads/electra_tiny/vocab.txt'


tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(model="electra", config_path=config_path, checkpoint_path=checkpoint_path)  # 建立模型，加载权重

def clock(func):
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        print('函数 {} 运行时间:{:.2f}s'.format(name,elapsed))
        return result
    return clocked

"""专用于比较bert抽取向量的句子相似度模块"""

class SentenceSimilarity:
    # 参考https://spaces.ac.cn/archives/7388
    def wasserstein_distance(self, p, q, D):
        """通过线性规划求Wasserstein距离
        p.shape=[m], q.shape=[n], D.shape=[m, n]
        p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
        """
        A_eq = []
        for i in range(len(p)):
            A = np.zeros_like(D)
            A[i, :] = 1
            A_eq.append(A.reshape(-1))
        for i in range(len(q)):
            A = np.zeros_like(D)
            A[:, i] = 1
            A_eq.append(A.reshape(-1))
        A_eq = np.array(A_eq)
        b_eq = np.concatenate([p, q])
        D = D.reshape(-1)
        result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
        return result.fun

    def word_rotator_distance(self, x, y):

        """WRD（Word Rotator's Distance）的参考实现
        x.shape=[m,d], y.shape=[n,d]
        """
        x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
        y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
        p = x_norm[:, 0] / x_norm.sum()
        q = y_norm[:, 0] / y_norm.sum()
        D = 1 - np.dot(x / x_norm, (y / y_norm).T)
        return self.wasserstein_distance(p, q, D)

    """
    Word Rotator's Distance（WRD），取值范围为[1,-1],该函数用于比较句子向量相似度
    """

    @clock
    def word_rotator_similarity(self, x, y):
        """1 - WRD
        x.shape=[m,d], y.shape=[n,d]
        """
        return 1 - self.word_rotator_distance(x, y)


"""获取句子向量"""

def get_sentence_vector(sentence):
    sentence = str(sentence)
    token_ids, segment_ids = tokenizer.encode(sentence)
    return model.predict([np.array([token_ids]), np.array([segment_ids])])[0]

"""创建问句查向量、向量查问句字典"""

def create_vector_dictionary():
    vector_to_sentence = defaultdict(str)
    sentence_to_vector = defaultdict(list)
    data = pd.read_csv("/Users/junjiexie/Downloads/项目4数据集.csv")

    for i in tqdm(range(len(data))):

        question = data.loc[i, "question"]
        question_vector = get_sentence_vector(question)
        question_vector_str = str(question_vector)

        vector_to_sentence[question_vector_str] = question
        sentence_to_vector[question] = question_vector

    with open('/Users/junjiexie/Downloads/data/vector_to_sentence', 'wb') as f:
        pickle.dump(vector_to_sentence, f)
    with open('/Users/junjiexie/Downloads/data/sentence_to_vector', 'wb') as f:
        pickle.dump(sentence_to_vector, f)
    print("字典已经创建完成")


if __name__ == '__main__':
    # a = get_sentence_vector("我真的好喜欢这个游戏啊")
    # b = get_sentence_vector("你玩不玩即时战略游戏呢")
    print(get_sentence_vector("我不是很喜欢这个游戏的").shape)
    #
    # print(word_rotator_similarity(a, b))
    # print(word_rotator_similarity(a, c))
    # a = create_vector_dictionary()