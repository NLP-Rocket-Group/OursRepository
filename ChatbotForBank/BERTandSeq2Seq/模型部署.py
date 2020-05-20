from __future__ import print_function
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import AutoRegressiveDecoder
import time

"""
参考
https://github.com/bojone/bert4keras/blob/master/examples/task_seq2seq_autotitle.py
利用seq2seq的思想完成简化版的银行客服，并添加了梯度扰动项，和一些随机替换，缓解seq2seq的不稳定性。
Unlim模型将bert和seq2seq统一起来，在NLG和NLU上有相当不俗的表现
"""

# 基本参数
maxlen = 256


# bert配置
config_path = '/Users/junjiexie/Downloads/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/junjiexie/Downloads/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/junjiexie/Downloads/chinese_L-12_H-768_A-12/vocab.txt'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

model.load_weights("/Users/junjiexie/Downloads/best_model_answer.weights")

def clock(func):
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        print('函数 {} 运行时间:{:.2f}s'.format(name,elapsed))
        return result
    return clocked


class AutoAnswer(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    @clock
    def generate(self, text, topk=3):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, max_length=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)




if __name__ == "__main__":
    autoanswer = AutoAnswer(start_id=None, end_id=tokenizer._token_end_id, maxlen=96)
    s = "你好，我想问问银行卡的余额还剩多少"
    print(u'生成回答:', autoanswer.generate(s))

    """
    函数 generate 运行时间:5.59s
    生成回答: 您好，目前我行暂时还不支持余额查询。
    在没有显卡支持下，推理速度较慢。硬件条件不足，最好使用抽取式对话模型
    """