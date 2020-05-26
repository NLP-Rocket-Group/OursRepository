import math
import os
import random

from flask import Flask, request, render_template
import json
from werkzeug.utils import secure_filename

from KeyWordGetter import *
from HighLightHTML import *
from PDFtoTxt import *

app = Flask(__name__)


print("初始化。。。")
keywordGetter = KeywordGetter()
print("初始化完成！")


def highlight_pdf2html(pdfPath, outputHtmlPath, encoding='utf-8'):
    tempPath = "Datas/Temp.html"
    with open(pdfPath, 'rb') as pdf_html:
        parse(pdf_html, tempPath)

    with open(tempPath, 'r', encoding='utf-8') as pdf_html:
        keywords = keywordGetter.get(pdf_html.read())

    highLightTag(keywords, tempPath, outputHtmlPath, encoding=encoding)

    os.remove(tempPath)


@app.route('/')
def index():
    with open('Client.html', 'rb') as htmlFile:
        return htmlFile.read()

@app.route('/pdf2html', methods=['GET', 'POST'])
def pdf2html():
    title = request.args.get('tt')
    content = request.args.get('ctt')
    proportion = request.args.get('ppt')
    print("收到消息：", 'tt', title, 'ppt', proportion, 'ctt', content)


# 设置允许的文件格式
ALLOWED_EXTENSIONS = {'pdf','PDF', 'Pdf'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 添加路由
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        # 通过file标签获取文件
        f = request.files['file']
        print(f.filename)
        if not (f and allowed_file(f.filename)):
            return json.dumps({"error": 1001, "msg": "类型：pdf"})
        # 当前文件所在路径
        basepath = os.path.dirname(__file__)
        print("basepath", basepath)
        # print("secure_filename(f.filename)", secure_filename(f.filename))
        # 一定要先创建该文件夹，不然会提示没有该路径
        newFileName = str(random.randrange(0, 9))
        path = ''.join([basepath, '/UploadFiles/'])
        if not os.path.isdir(path):
            os.mkdir(path)
        upload_path = ''.join([path, newFileName, '.pdf'])
        html_path = ''.join([path, newFileName, '.html'])
        print("upload_path", upload_path)
        # 保存文件
        f.save(upload_path)

        highlight_pdf2html(upload_path, html_path)

        with open(html_path, 'rb') as htmlFile:
            return htmlFile.read()
        # 返回上传成功界面
        # return '上传成功界面'
    # 重新返回上传界面
    return render_template('Client.html')

if __name__ == "__main__":
    app.run(port=9999)

    # highlight_pdf2html("Datas/如何才能摆脱贫穷？穷人和富人有什么差别？.pdf",
    #                    "Datas/如何才能摆脱贫穷？穷人和富人有什么差别？.html")