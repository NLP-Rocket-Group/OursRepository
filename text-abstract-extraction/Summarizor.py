import math
import SIF
import similarity
import re


class Summarizor:
    def __init__(self, word2VecModelFilePath = 'Data/wiki_han_word2vec_300维度.model', isUseThulac=False):
        self.sif = SIF.SIF(word2VecModelFilePath, isUseThulac=isUseThulac)
		
    def _splitText(self, text:str, splitChar = '(。|！|\!|？|\?|\n|\t)'):
        contents = re.split(splitChar, text)
        # print('句子切分分隔符：', splitChar)
        contents = ["".join([a, b]) if b != '\n' and b != '\t' else a + "。"
                    for a, b in zip(contents[0::2], contents[1::2])]
        contents = [content for content in contents if content.strip() != '' and content[0] != '。']
        for i, sen in enumerate(contents):
            print(i, sen)
        return contents

    def _knnSmooth(self, similarities, neighborCount = 1, neighborWeight = 0.3):
        """
        neighborCount: 考虑左边和右边分别 neighborCount 的邻居的值
        neighborWeight: 最相邻的邻居的权重
        """
        # print("平滑前：")
        # for i, item in enumerate(similarities):
        #     print(item[0])
        targetSimilarities = similarities.copy()
        for i, item in enumerate(similarities):
            sumValue = 0
            for j in range(i - neighborCount, i + neighborCount):
                if j == i:
                    sumValue += item[0] * (1 - neighborWeight)
                elif j >=0 and j < len(similarities):
                    sumValue = (neighborWeight ** abs(j - i)) * similarities[j][0]
                else:
                    continue
            targetSimilarities[i] = (sumValue, item[1])
        # print("平滑后：")
        # for i, item in enumerate(t        argetSimilarities):
        #     print(item[0])
        return targetSimilarities

    def _knnSmooth2(self, similarities):

        '''
        加入KNN平滑，控制为前后一句话，共计3
        '''
        similaritiesKnn = []
        for count, turple in enumerate(similarities):
            if count == 0:
                sentenceVec = (similarities[count][0] + similarities[count + 1][0] + similarities[count + 2][0]) / 3
                contentVec = similarities[count][1]
                similaritiesKnn.append((sentenceVec, contentVec))
            elif count == len(similarities) - 1:
                sentenceVec = (similarities[count][0] + similarities[count - 1][0] + similarities[count - 2][0]) / 3
                contentVec = similarities[count][1]
                similaritiesKnn.append((sentenceVec, contentVec))
            else:
                sentenceVec = (similarities[count][0] + similarities[count - 1][0] + similarities[count + 1][0]) / 3
                contentVec = similarities[count][1]
                similaritiesKnn.append((sentenceVec, contentVec))
        return similaritiesKnn

    def summarize(self, content: str, title: str = None, splitChar='(。|！|\!|？|\?|\n|\t)', proportion=0.3):
        contents = self._splitText(content, splitChar=splitChar)

        # 获取标题向量
        if title != None:
            title = title.strip()
            if splitChar.find(title[len(title) - 1]) == -1:
                title += '。'
            contents.insert(0, title)
        # 获取文章向量
        contents.append(content)

        # print(' len(contents)',  len(contents))
        if len(contents) <= 4:
            return contents

        sentencesVec = self.sif.getSentencesEmbedding(contents)

        sentencesVec = list(sentencesVec)
        contentVec = sentencesVec.pop()

        similarities = [(similarity.cosine_similarity(senVec, contentVec), index) for index, senVec in
                        enumerate(sentencesVec)]
        similarities2 = [(similarity.cosine_similarity(senVec, sentencesVec[0]), index) for index, senVec in
                         enumerate(sentencesVec)]
        similarities = [((sim1[0] * 0.382 + sim2[0] * 0.618), sim1[1]) for sim1, sim2 in
                        zip(similarities, similarities2)]
        # 相似度平滑 KNN
        similarities = self._knnSmooth(similarities)

        # 排序
        similarities.sort(reverse=True)

        summarySentenceIndexes = similarities[0: int(len(similarities) * proportion)]
        # print("summarySentenceIndexes:")
        # for i, sim in enumerate(summarySentenceIndexes):
        #     print(i, "index:", sim[1], sim, contents[sim[1]])
        summarySentences = [(index, contents[index]) for (cos, index) in summarySentenceIndexes]

        summarySentences.sort()

        return [sentence for (index, sentence) in summarySentences]


if __name__ == "__main__":
    with open('Data/testArticle.txt', 'r', encoding='utf-8') as testFile:
        test_content = testFile.read()
    summarizor = Summarizor(word2VecModelFilePath='Data/word2vect_50_w5.model', isUseThulac=True)
    summarySentences = summarizor.summarize(test_content, proportion = 0.3)

    # print("摘要：", "".join(summarySentences), "\n---------------------------------------------------")
    for i, sentence in enumerate(summarySentences):
        print(i, sentence)


# 0 神舟电脑谈起诉京东：未经许可将神舟产品降价销售后要返利
#
# 1 随后京东未经神舟许可，强行将神舟产品降价销售，并声称京东双11有百亿补贴，无需神舟承担任何费用。
# 2 神舟电脑还表示，为了逼迫神舟同意支付此部分降价损失，京东对神舟采用了五项措施：产品搜索降权、不让参加任何活动、缺货产品不予订货、全线产品下架、不予结算货款。
# 3 神舟电脑官网显示，深圳市神舟电脑股份有限公司成立于2001年，是一家以IT、IA为主业，以电脑技术开发为核心，集研发、生产、销售为一体的国家级高科技企业。


# 0 武汉中心医院急诊科主任艾芬辟谣：没感染新冠肺炎，仍在一线
#
# 1 针对湖北武汉市中心医院急诊科主任艾芬感染新冠肺炎遭遇不幸一事，艾芬2月20日中午向澎湃新闻辟谣称，她身体很好，也未感染新冠肺炎，目前仍在抗疫一线工作。
# 2 艾芬2月20日中午告诉澎湃新闻，自己好着呢，没有感染新冠肺炎，现在仍在抗疫一线工作。
# 3 武汉市中心医院官网显示，艾芬擅长各种急危重症患者，心跳呼吸骤停、中毒、休克、创伤、呼吸衰竭、严重感染及多脏器功能障碍的救治，是急诊科主任（副主任医师），也是一名教授、硕士研究生导师。
# 4 据《中国新闻周刊》此前报道，2019年12月18日，艾芬接触首例肺部感染表现为“双肺多发散在斑片状模糊影”的华南海鲜市场送货员，12月27日，她接诊第二例此类病人，但第二人无华南海鲜市场接触史。
# 5 这份检测报告，于12月30日下午被该院眼科医生李文亮发在同学微信群里，并被大量转发。