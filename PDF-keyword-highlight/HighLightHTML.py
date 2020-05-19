# coding:utf-8

def highLightTag(keyWords:[], htmlAddr, outputAddr, encoding='utf-8'):
    style = "<style> mark {background-color:#00ff90; font-weight:bold;}</style>"
    f2 = open(outputAddr, 'w', encoding=encoding)
    for pdf_html_line in open(htmlAddr, 'r', encoding=encoding):
        for keyWord in keyWords:
            if pdf_html_line.find("<meta") != -1:
                f2.write(style)
            if pdf_html_line.find(keyWord) != -1:
                temp = "<mark>" + keyWord + "</mark>"

                pdf_html_line = pdf_html_line.replace(keyWord, temp)
        f2.write(pdf_html_line)
    f2.close()


if __name__ == '__main__':
    htmlAddr = r'.\Datas\如何才能摆脱贫穷？穷人和富人有什么差别？.txt'
    outputAddr = r'.\Datas\如何才能摆脱贫穷？穷人和富人有什么差别？HighLight.html'

    highLightTag(["富人", "穷人"], htmlAddr, outputAddr, encoding='gbk')



