from textrank4zh import TextRank4Sentence


class Summarizor:
    def summarize(self, content:str, title:str = None, proportion = 0.3):
        tr4s = TextRank4Sentence()
        if title != None:
            text = "。".join([title, content])
        else:
            text = content
        tr4s.analyze(text=text, lower=True, source='all_filters')
        summarySentences = tr4s.get_key_sentences(num = len(tr4s.sentences) * proportion)
        summarySentences.sort(key=lambda item: item.index)
        return [sen.sentence + "。" for sen in summarySentences]


if __name__ == "__main__":
    with open('Data/testArticle3.txt', 'r', encoding='utf-8') as testFile:
        test_content = testFile.read()
    summarizor = Summarizor()
    summarySentences = summarizor.summarize(test_content, proportion = 0.1)

    print("摘要：", "".join(summarySentences), "\n---------------------------------------------------")
    for i, sentence in enumerate(summarySentences):
        print(i, sentence)

# 0 0.08609203119723863 神舟电脑谈起诉京东：未经许可将神舟产品降价销售后要返利
# 1 0.08136550317624627 神舟电脑披露了起诉京东背后双方的纠葛
# 3 0.08021868707679176 当天，京东方面做出回应称，因神舟违反双方签署的产品购销协议条款，导致其未结算货款被暂缓支付
# 4 0.0879412238264965 2月21日，神舟电脑发布微博列出了与京东的部分往来信函，披露了与京东之间的纠纷细节，争议焦点围绕在2019年的双11活动上
# 8 0.09931952312178195 神舟电脑还表示，为了逼迫神舟同意支付此部分降价损失，京东对神舟采用了五项措施：产品搜索降权、不让参加任何活动、缺货产品不予订货、全线产品下架、不予结算货款

# 摘要：
# 0 0.023901846926284467 武汉中心医院急诊科主任艾芬辟谣：没感染新冠肺炎，仍在一线
# 1 0.028862389656307885 针对湖北武汉市中心医院急诊科主任艾芬感染新冠肺炎遭遇不幸一事，艾芬2月20日中午向澎湃新闻辟谣称，她身体很好，也未感染新冠肺炎，目前仍在抗疫一线工作
# 4 0.027072332385385745 此前网络上有传言称，武汉中心医院急诊科主任艾芬，向医院提起新冠病毒可“人传人”的事实，但没有获得任何回应，她只得要求科室的医护人员先戴起N95口罩，还曾被训诫
# 6 0.024801756781123528 艾芬2月20日中午告诉澎湃新闻，自己好着呢，没有感染新冠肺炎，现在仍在抗疫一线工作
# 8 0.027661286059820735 武汉市中心医院官网显示，艾芬擅长各种急危重症患者，心跳呼吸骤停、中毒、休克、创伤、呼吸衰竭、严重感染及多脏器功能障碍的救治，是急诊科主任（副主任医师），也是一名教授、硕士研究生导师
# 20 0.029590855477737325 自疫情发生以来，武汉市中心医院急诊科含42名党员先锋在内的200人急救团队抗击疫情近40多天，日夜坚守发热门诊、留观病房、抢救室，把守诊断、救治新冠肺炎的第一道关口
# 36 0.032633649251833693 武汉市中心医院后湖院区急诊科留观病房专门收治新冠肺炎疑似患者，病痛、恐惧令留观病房内气氛压抑，但急诊科党支部书记、科主任艾芬说：“即使不能直接救人，至少我们能去安慰和关心病患