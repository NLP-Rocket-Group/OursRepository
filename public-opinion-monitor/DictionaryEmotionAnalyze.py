import pandas as pd
from collections import defaultdict
import re
import thulac

#获得情感极性和极性强度，构建词典

def get_emotion_words():
    table = pd.read_excel("data/情感词汇本体.xlsx")
    emotion_dict = defaultdict(int)
    vocabulary_emotion = []
    for count, i in enumerate(table["极性"]):
        if i == 1:
            vocabulary_emotion.append(1 * table["强度"][count])
        elif i == 2:
            vocabulary_emotion.append(-1 * table["强度"][count])
        else:
            vocabulary_emotion.append(0)

    for count, i in enumerate(table["词语"]):
        emotion_dict[i] = vocabulary_emotion[count]

    return emotion_dict

#停用词

def get_stop_words():
    stopwords = []
    with open("data/停用词表.txt", encoding="utf-8") as f:
        line_str = f.readline()
        while line_str != "":
            line_str = line_str.strip()
            stopwords.append(line_str)
            line_str = f.readline()
    return set(stopwords)

#原理是通过统计输入句子的情感词汇值，汇总计算，最终判断整个句子的情感值
#正确率在65%-85%区间
class DictionaryEmotionAnalyze:

    def __init__(self):
        self.stop_words = get_stop_words()
        self.emotion_words = get_emotion_words()
        self.thu1 = thulac.thulac(seg_only=True, filt=True)

    def token(self, string):
        return re.findall('\w+', string)

    def sentences_emotion(self, sentences):

        # input = "".join(self.token(sentences))
        # cut_list = ",".join(jieba.cut(input)).split(",")

        thu1 = self.thu1
        cut_list = thu1.cut(sentences, text=True).split(" ")

        stopwords = self.stop_words
        emotion_dict = self.emotion_words
        emotion = 0

        for element in cut_list:
            if element not in stopwords:
                emotion = emotion + emotion_dict[element]

        #目标句子情感值>3则判断为正面

        if emotion > 3:
            return 1
        #负面

        elif emotion < -3:
            return -1
        #中性

        else:
            return 0


if __name__ == '__main__':
    model = DictionaryEmotionAnalyze()
    output = model.sentences_emotion("虽然购物的时候发生了一些不愉快，但货物质量挺好，最后还是给五星好评，下次再来")
    print(output)




