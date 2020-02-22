import SIF
import similarity
import re

class Summarizor:
    def __init__(self, word2VecModelFilePath = 'Data/wiki_han_word2vec_300维度.model'):
        self.sif = SIF.SIF(word2VecModelFilePath)


    def summarize(self, content:str, title:str = None, splitChar = '(。|！|\!|\.|？|\?|\n|\t)', proportion = 0.3):
        if type(content) == type("str"):
            contents = re.split(splitChar, content)
            print('句子切分：')
            contents = ["".join(i) for i in zip(contents[0::2], contents[1::2])]
            for sen in contents:
                print(sen)
        else:
            contents = content

        # 获取标题向量
        if title != None :
            contents.insert(0, title)
        # 获取文章向量
        contents.append(content)

        sentenceVec = self.sif.getSentencesEmbedding(contents)

        # sentence4Index = {sentence : index for index, sentence in enumerate(contents)}
        # print("sentence4Index", sentence4Index)

        sentenceVec = list(sentenceVec)
        print(len(sentenceVec))
        contentVec = sentenceVec.pop()
        print(len(sentenceVec))

        similarities = [(similarity.cosine_similarity(sentenceVec, contentVec), index) for index, sentenceVec in enumerate(sentenceVec)]
        # 相似度平滑 KNN
        # ？？？

        # 排序
        similarities.sort(reverse=True)
        print("similarities:")
        for sim in similarities:
            print(sim)

        summarySentenceIndexes = similarities[0: int(len(similarities) * proportion)]
        print("summarySentenceIndexes:")
        for sim in summarySentenceIndexes:
            print(sim, contents[sim[1]])
        summarySentences = [ (index, contents[index]) for (cos, index) in summarySentenceIndexes ]

        summarySentences.sort()

        return [sentence for (index, sentence) in summarySentences]


if __name__ == "__main__":
    with open('Data/testArticle2.txt', 'r', encoding='utf-8') as testFile:
        test_content = testFile.read()
    # print(test_content)
    summarizor = Summarizor()
    summarySentences = summarizor.summarize(test_content, splitChar='(。|！|\!|\.|？|\?|\n|\t)')
    print("摘要：", "".join(summarySentences), "\n---------------------------------------------------")
