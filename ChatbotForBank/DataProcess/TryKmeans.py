import sklearn
import pickle
import time
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans

def clock(func):
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        print('函数 {} 运行时间:{:.2f}s'.format(name,elapsed))
        return result
    return clocked

"""获得文档中所有问句的向量，有些可能是空值，需要填上零矩阵"""
@clock
def get_all_vector():
    r = []
    with open('/Users/junjiexie/Downloads/data/sentence_to_vector', 'rb') as f:
        sentence_to_vector = pickle.load(f)

    for i in sentence_to_vector.values():
        if str(i.shape) != "(13, 312)":
            i = np.zeros((13, 312))
        r.append(i)
    return np.array(r).reshape(-1,1)


"""寻找合适的聚类簇，这里运行速度太慢了，可能得数天。不建议用k-means做。可以考虑使用LDA主题模型对文档进行分类完成意图识别"""
@clock
def run_Kmeans(data):
    score_list = []  # 储存系数的列表
    score_init = -1  # 初始轮廓系数
    good_k = 0
    for n_k in tqdm(range(2, 11)):
        model_kmeans = MiniBatchKMeans(n_clusters=n_k, random_state=0)  # 建立模型
        cluster_tmp = model_kmeans.fit_predict(data)  # 训练模型
        score_tmp = metrics.silhouette_score(data, cluster_tmp)  # 得到K值的轮廓系数
        if score_tmp > score_init:  # 如果这个系数更高
            good_k = n_k  # 储存K值
            score_init = score_tmp  # 储存轮廓系数，做下次比较
            good_model = model_kmeans  # 储存模型
            good_cluster = cluster_tmp  # 储存聚类标签
        score_list.append([n_k, score_tmp])

    print(score_list)
    print('Best K is:{0} with average silhouette of {1}'.
          format(good_k, score_init))



if __name__ == '__main__':
    data = get_all_vector()
    # print(data[0:-1].shape)
    # print(data.shape)
    run_Kmeans(data)
