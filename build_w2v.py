import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import time



# 输入文件路径和列名，将各列合并，输出一个迭代器，每次返回合并列数据一行的分词list
class MyCorpus:
    def __init__(self, train_path, train_cols, test_path, stop_list):
        self.train_cols = train_cols
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.stop_list = stop_list
        for key in self.train_cols:
            if key not in self.train_df.columns:
                self.train_cols.remove(key)
        if not len(self.train_cols) > 0:
            raise ValueError('invalid column')

    def __iter__(self):
        all_df = pd.concat([self.train_df, self.test_df], axis=0)
        text_sr = all_df[self.train_cols[0]].str.cat(all_df[self.train_cols[1:]], na_rep=' ', sep=' ')
        for item in text_sr[:100]:
            tokens = jieba.lcut(item.strip())
            new_tokens = tokens.copy()
            for token in tokens:
                if token in self.stop_list:
                    new_tokens.remove(token)
            yield new_tokens


class MyCorpus1:
    def __iter__(self):
        train_df = pd.read_csv('input/AutoMaster_TrainSet.csv')
        test_df = pd.read_csv('input/AutoMaster_TestSet.csv')
        all_df = pd.concat([train_df, test_df], axis=0)
        text_sr = all_df['Question'].str.cat(all_df[['Dialogue', 'Report']], na_rep=' ', sep=' ')
        stop_list = ['|', '[', ']', '语音', '图片', ' ']
        for item in text_sr[:100]:
            tokens = jieba.lcut(item.strip())
            for token in tokens:
                if token in stop_list:
                    tokens.remove(token)
            yield tokens


def build(sentences=None, size=100, skip_gram=1, hs=1, negative_num=-1, save_path='./'):
    # 训练词向量，并保存到硬盘
    model = Word2Vec(sentences=sentences, size=size, sg=skip_gram, hs=hs, negative=negative_num)
    # 问题2：
    # model.wv.save_word2vec_format(save_path, binary=True)
    model.wv.save(save_path)
    print("save w2v model ok.")
    print('车主和技师的词向量相似度为：{}'.format(model.wv.similarity('技师', '车主')))


if __name__ == '__main__':
    t1 = time.process_time()
    # 问题1：
    sentences = MyCorpus(train_path='input/AutoMaster_TrainSet.csv', train_cols=['Question', 'Dialogue', 'Report'],
                         test_path='input/AutoMaster_TestSet.csv', stop_list=['|', '[', ']', '语音', '图片', ' '])
    # sentences = MyCorpus1()

    model_path = 'input/model.wv'
    # build(sentences=sentences, size=256, skip_gram=1, hs=1, save_path='input/w2v_keyvec.bin')
    build(sentences=sentences, size=256, skip_gram=1, hs=1, save_path=model_path)

    # 问题2：
    # model1 = KeyedVectors.load_word2vec_format(model_path, binary=True)
    model1 = KeyedVectors.load(model_path)
    print('车主和技师的词向量相似度为：{}'.format(model1.similarity('技师', '车主')))
    t2 = time.process_time()
    print('程序运行{}秒'.format(t2-t1))
    print(model1.wv.get_vector('语音'))
    # 问题
    # 1 为什么加载语料时，使用MyCorpus1比使用MyCorpus慢一倍多一点
    # 2 使用save_word2vec_format和load_word2vec_format 会报错，但不是每次都报
