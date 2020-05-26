import os

def pdf_to_html(filename:str, pdfname:str):
    """
    filename是路径，到当前目录就可以，
    pdfname是pdf的名称，需要带后缀pdf，
    该方法需要本地安装docker desktop才可使用
    目前设置pdf和html都在同一个路径
    """
    str = "docker run -i --rm -v "
    second_str = ":/pdf bwits/pdf2htmlex pdf2htmlEX --zoom 1.3 "
    all_str = str + filename + second_str + pdfname
    os.system(all_str)




if __name__ == '__main__':
    basepath = os.path.dirname(__file__)
    pdf_to_html(filename = basepath + "/Datas/", pdfname="如何才能摆脱贫穷？穷人和富人有什么差别？.pdf")

