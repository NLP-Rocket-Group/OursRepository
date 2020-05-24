from sentence_transformers import SentenceTransformer, LoggingHandler, models
import numpy as np
from scipy import spatial
import time
import pickle
from tqdm import tqdm

def clock(func):
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        print('函数 {} 运行时间:{:.2f}s'.format(name,elapsed))
        return result
    return clocked

# Use BERT for mapping tokens to embeddings
word_embedding_model = models.Transformer('/Users/junjiexie/Downloads/chinese_simbert_L-12_H-768_A-12')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Corpus with example sentences
corpus = ['理财产品取号。',
          '查询名下所有账户',
          '现金取款。',
          '个人账户管理。',
          '其他信用卡业务。']

corpus = corpus

queries = ['我想知道卡里还有多少钱', '挂失操作']

corpus_embeddings = model.encode(corpus)

query_embeddings = model.encode(queries)

# 测试Bert编码一千个句子耗时1min，因此需要提前用bert编码语料库中的问句，然后用pickle存下来，直接拿来匹配的话，系统的响应就会很快
# 语义相似度匹配例子，https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search.py
# 聚类相似度匹配例子，https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering.py
# bert权重使用追一科技的simBERT，专用于相似句子检索和生成任务
# cos计算基本不耗时间
@clock
def run():


    closest_n = 1
    for query, query_embedding in zip(queries, query_embeddings):
        distances = spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx, distance in results[0:closest_n]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))

if __name__ == "__main__":
    run()