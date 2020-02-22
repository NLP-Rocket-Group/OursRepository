import pandas as pd
from time import time
from opencc import OpenCC
import re
import jieba
import os
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec







# ----以下为汉语中文语料处理----

# 装饰器，用来统计函数执行时间
def time_func(f):
    def wrapper(*args):
        start = time()
        print("Start processing......")
        result = f(*args)
        end = time()
        print("End processing......")
        duration = end - start
        print("---Processed time in %ss---" % round(duration,2))

        return result
    return wrapper


#句子切分
def cut_sentence(sentence):
    sentence = re.sub('([。！？\?])([^”’])', r"\1\n\2", sentence)  # 单字符断句符
    sentence = re.sub('(\.{6})([^”’])', r"\1\n\2", sentence)  # 英文省略号
    sentence = re.sub('(\…{2})([^”’])', r"\1\n\2", sentence)  # 中文省略号
    sentence = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', sentence)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    sentence = sentence.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return sentence.split("\n")


#文本处理总函数，切分句子、去标点符号、分词
def string_process(string):
    # we will learn the regular expression next course.
    stay = []
    new_string = cut_sentence(string) #这个分句设计可能还是有问题的
    for sentence in new_string:
        # 匹配<...>标签
        label_pattern = '<.+>'
        # 匹配各类中英文标点
        punc_pattern = '[“”，。./（）@\(\)·《》：:\-\"「」‘’？?!！,、；]'
        clean_sentence = re.sub(label_pattern,"",sentence)
        clean_sentence = re.sub(punc_pattern,"",clean_sentence)
        # clean_sentence = DataPreProcess.sentence_decode(clean_sentence)
        sentence_words = [",".join(jieba.cut(clean_sentence))]
        stay.append(sentence_words)


    return stay

#处理汉语中文语料总函数,汉语中文语料较小，不需要批处理
@time_func
def han_path_laod(input_csv_path, output_txt_path):
    stay = []
    data = pd.read_csv(input_csv_path,encoding="GB18030")
    for count,article in enumerate(data.content):
        process_article = string_process(str(article))
        stay = stay + process_article

        if count % 500 == 0:
            print("已处理{}".format(count))

    #写入文件
    fileObject = open(output_txt_path,"w")
    for sentence in stay:
        fileObject.write(",".join(sentence))
        fileObject.write("\n")
    fileObject.close()

# ----以下为维基中文语料处理----

def read_all_wiki_path():
    stay_all_path = []
    file_path = "/Users/junjiexie/Documents/NLP学习/nlp文本摘要项目/wikiextractor/text/"
    first_names = os.listdir(file_path)
    for i in first_names[1:]:
        first_path = file_path + i
        second_names = os.listdir(first_path)
        for j in second_names:
            stay_all_path.append(first_path + "/" + j)

    return stay_all_path


def strip_wiki_source(wiki_source):
    # 简繁体转换器
    convertor = OpenCC('t2s')

    # 匹配<...>标签
    label_pattern = '<.+>'
    # 匹配各类中英文标点
    punc_pattern = '[“”，。（）\(\)·《》：:\-\"「」‘’？?!！,、；]'

    for count,path in enumerate(wiki_source):

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n': continue
                # 正则替换
                line = re.sub(label_pattern, '', line)
                line = re.sub(punc_pattern, '', line)
                # 由繁体转为简体
                simplified_line = convertor.convert(line)

                #追加模式，因此保证是个空文件
                output_file = open('wiki_stripped.txt', 'a', encoding='utf-8')
                output_file.write(simplified_line)
                output_file.close()

        print("完成{}个文件".format(count))


@time_func
def get_cut_lines(wiki_stripped):

    with open(wiki_stripped, 'r', encoding='utf-8') as f:

        new_line = ""

        for count,line in enumerate(f):

            # 空行把之前的一段切词
            if line == '\n':
                cut_line = ",".join(jieba.cut(new_line))
                new_line = ""
                output_file = open('cut_lines.txt', 'a')
                output_file.write(cut_line)
                output_file.write("\n")

                output_file.close()
            else:
                new_line = new_line + line

            if count % 5000 == 0 : print("已经切分{}行".format(count))


def train_word2vec(path):

    input_sentences = []
    # 训练一个批次的指针
    time = 0
    # 每次训练的训练量
    batch_size = 200000


    sentences = word2vec.LineSentence(path)
    for count, sentence in enumerate(sentences):
        input_sentences.append(sentence[0].split(","))
        # 每20万行训练一次，之后将input_sentences清空
        if count % batch_size == 0 and count > 0:
            print("载入数据")
            model = Word2Vec(input_sentences,size=300,window=5)
            input_sentences = []
            model.save('wiki_han_word2vec.model')
            time = time + 1
            print("成功训练{}次模型".format(time))
        elif count % batch_size == 0 and time > 0:
            #增量训练
            print("载入数据")
            model = Word2Vec.load('wiki_han_word2vec.model')
            model.build_vocab(input_sentences, update=True) #更新词汇表
            # epoch=iter语料库的迭代次数；（默认为5）  total_examples:句子数。
            model.train(input_sentences, total_example=model.corpus_count, epochs=model.iter)
            input_sentences = []
            model.save('wiki_han_word2vec.model')
            time = time + 1
            print("成功训练{}次模型".format(time))

    print('----Model Saved Successfully...')
    print("成功训练{}次模型".format(time))


def update_word2vec(data_path,model_path):


    input_sentences = []
    # 训练一个批次的指针
    time = 0
    # 每次训练的训练量
    batch_size = 200000

    sentences = word2vec.LineSentence(data_path)
    for count, sentence in enumerate(sentences):
        input_sentences.append(sentence[0].split(","))
        # 每20万行训练一次，之后将input_sentences清空
        if count % batch_size == 0 and count > 0:
            #增量训练
            print("载入数据")
            model = Word2Vec.load(model_path)
            model.build_vocab(input_sentences, update=True) #更新词汇表
            # epoch=iter语料库的迭代次数；（默认为5）  total_examples:句子数。
            model.train(input_sentences, total_examples=model.corpus_count, epochs=model.iter)
            input_sentences = []
            model.save('wiki_han_word2vec.model')
            time = time + 1
            print("成功训练{}次模型".format(time))


def use_word2vec_model():
    model = Word2Vec.load('wiki_han_word2vec.model')
    # 和某个词最接近的5个词
    word_list = ['武则天', '腾讯', '阿里', '马云', ]
    print('5 most similar words with ' + word_list[0] + ' are:')
    for item in model.wv.most_similar(word_list[0], topn=10):
        print(item[0], item[1])

    # 判断两个词之间的相似度
    sim_w1 = word_list[1]
    sim_w2 = word_list[2]
    sim_w3 = word_list[3]
    sim1 = model.wv.similarity(sim_w1, sim_w2)
    sim2 = model.wv.similarity(sim_w1, sim_w3)
    print('\nThe similarity between {w1} and {w2} is {sim}'.format(w1=sim_w1, w2=sim_w2, sim=str(sim1)))
    print('The similarity between {w1} and {w2} is {sim}'.format(w1=sim_w1, w2=sim_w3, sim=str(sim2)))


    

    


if __name__ == '__main__':

    #处理 汉语中文语料，40m大小，1百万项左右

    han_path = "/Users/junjiexie/Documents/NLP学习/nlp文本摘要项目/sqlResult_1558435.csv"
    txt_path = "han_data_process.txt"
    model_path = 'wiki_han_word2vec.model'

    # han_path_laod(han_path,txt_path)

    #处理 wiki中文语料，1.7G大小，1327个文件

    all_wiki_path = read_all_wiki_path()
    strip_wiki_source(all_wiki_path)

    #切分七千万条数据左右

    # get_cut_lines("wiki_stripped.txt")

    #wiki语料训练
    # train_word2vec("cut_lines.txt")

    #中文新闻预料训练
    # update_word2vec(data_path=txt_path,model_path=model_path)

    #展示相邻词
    # use_word2vec_model()

    model = Word2Vec.load('Data/wiki_han_word2vec_300维度.model')

    def analogy(x1,x2,y1):
        result = model.wv.most_similar(positive = [y1,x2],negative = [x1])
        return result[0][0]

    print(analogy("中国","汉语","美国"))
    print(analogy("美国", "奥巴马", "日本"))

    print(model.wv.most_similar(positive = '勇敢'))
    print(model.wv.most_similar(positive = '美女'))
    print(model.wv.most_similar('勇敢'))
    print(model.wv.most_similar('美女'))

