# coding:utf-8

def highLightTag(keyWordAddr, htmlAddr, outputAddr):
    list = []
    style = "<style> mark {background-color:#00ff90; font-weight:bold;}</style>"
    for lines in open(keyWordAddr, 'r', encoding='UTF-8'):
        list.extend(lines.split())

    f2 = open(outputAddr, 'w', encoding='utf-8')
    for pdf_html_line in open(htmlAddr, 'r', encoding='UTF-8'):
        for keyWord in list:
            if pdf_html_line.find("<meta") != -1:
                f2.write(style)
            if pdf_html_line.find(keyWord) != -1:
                temp = "<mark>" + keyWord + "</mark>"

                pdf_html_line = pdf_html_line.replace(keyWord, temp)
        f2.write(pdf_html_line)
    f2.close()


if __name__ == '__main__':
    keyWordAddr = r'.\Datas\keyWord.txt'
    htmlAddr = r'.\Datas\test.html'
    outputAddr = r'.\Datas\test2.html'

    highLightTag(keyWordAddr, htmlAddr, outputAddr)



