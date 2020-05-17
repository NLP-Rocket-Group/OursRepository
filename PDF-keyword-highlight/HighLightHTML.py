# coding:utf-8


if __name__ == '__main__':
    list = []
    style= "<style> mark {background-color:#00ff90; font-weight:bold;}</style>"
    for lines in open(r'.\Datas\keyWord.txt', 'r', encoding='UTF-8'):
            list.extend(lines.split())

    f2 = open(r'.\Datas\test2.html', 'w', encoding='utf-8')
    for pdf_html_line in open(r'.\Datas\test.html', 'r', encoding='UTF-8'):
        for keyWord in list:
            if pdf_html_line.find("<meta") != -1:
                f2.write(style)
            if pdf_html_line.find(keyWord) != -1:
                temp = "<mark>"+keyWord+"</mark>"

                pdf_html_line = pdf_html_line.replace(keyWord, temp)
        # print(pdf_html_line)
        f2.write(pdf_html_line)
    f2.close()


