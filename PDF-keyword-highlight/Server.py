import os
from flask import Flask, request, render_template
import json

from KeyWordGetter import *
from HighLightHTML import *
from PDFtoTxt import *

app = Flask(__name__)
print("初始化。。。")
keywordGetter = KeywordGetter()
print("初始化完成！")


def highlight_pdf2html(pdfPath, outputHtmlPath, encoding='gbk'):
    tempPath = "Datas/Temp.html"
    with open(pdfPath, 'rb') as pdf_html:
        parse(pdf_html, tempPath)

    with open(tempPath, 'r') as pdf_html:
        keywords = keywordGetter.get(pdf_html.read())

    highLightTag(keywords, tempPath, outputHtmlPath, encoding=encoding)

    os.remove(tempPath)

@app.route('/pdf2html', methods=['GET', 'POST'])
def pdf2html():
    title = request.args.get('tt')
    content = request.args.get('ctt')
    proportion = request.args.get('ppt')
    print("收到消息：", 'tt', title, 'ppt', proportion, 'ctt', content)

if __name__ == "__main__":
    app.run(port=9999)

    highlight_pdf2html("Datas/如何才能摆脱贫穷？穷人和富人有什么差别？.pdf",
                       "Datas/如何才能摆脱贫穷？穷人和富人有什么差别？.html")