import sys
import os
import pandas as pd
import jieba
from gensim.models import Word2Vec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from tools.utils import load_model, save_model




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


def save_embedding_vocab(model, embed_path, vocab_path):
    """
    保存词向量矩阵和词典
    Args:
        model: gensim模型，Word2Vec实例或KeyedVectors实例
        embed_path: 词向量矩阵保存路径
        vocab_path: 词典保存路径
    """
    if isinstance(model, Word2Vec):
        model = model.wv
    # 保存词向量矩阵：2d-array
    with open(embed_path, 'wb') as f:
        save_model(model.vectors, f)

    # 保存字典：token to id
    vocab = {}
    with open(vocab_path, 'wb') as f:
        for i in range(len(model.index2word)):
            vocab[model.index2word[i]] = i
        save_model(vocab, f)


def build_model(sentences=None, size=100, skip_gram=1, hs=1, negative_num=-1, save_path='./undefined_model'):
    """
    训练词向量，并保存到文件
    Args:
        sentences: corpus
        size: Dimensionality of the word vectors.
        skip_gram: 1 for skip-gram; otherwise CBOW.
        hs:  If 1, hierarchical softmax will be used，  If 0, negative sampling
        negative_num: negative sampling numbers (usually between 5-20).
        save_path: model save path

    Returns:
        None
    """
    model = Word2Vec(sentences=sentences, size=size, sg=skip_gram, hs=hs, negative=negative_num, min_count=10)
    model.wv.save_word2vec_format(save_path, binary=True)  # 以KeyedVectors实例形式保存词向量
    # model.wv.save(save_path)  # 以Word2Vec形式保存模型，可继续训练
    print("save w2v model ok.")
    return model


if __name__ == '__main__':
    # 问题1：
    sentences = MyCorpus(train_path='input/AutoMaster_TrainSet.csv', train_cols=['Question', 'Dialogue', 'Report'],
                         test_path='input/AutoMaster_TestSet.csv', stop_list=['|', '[', ']', '语音', '图片', ' '])
    # # sentences = MyCorpus1()

    model_path = 'output/w2v.wv'
    embed_path = 'output/embedding.mat'
    vocab_path = 'output/vocab.dict'
    model = build_model(sentences=sentences, size=256, skip_gram=1, hs=1, save_path=model_path)
    print(model['说'])
    save_embedding_vocab(model, embed_path, vocab_path)
    with open(vocab_path, 'rb') as f:
        vocab = load_model(f)
    print(vocab['说'])
    with open(embed_path, 'rb') as f:
        matrix = load_model(f)
    print(matrix[1])

    # wv = KeyedVectors.load_word2vec_format(model_path, binary=True)  # 以KeyedVectors实例形式加载词向量
    # print(wv['车主'])

    # model1 = KeyedVectors.load(model_path)  # 以Word2Vec形式加载模型
    # print('车主和技师的词向量相似度为：{}'.format(model1.similarity('技师', '车主')))
    # print(model1.wv.get_vector('语音'))
    # 问题
    # 1 为什么加载语料时，使用MyCorpus1比使用MyCorpus慢一倍多一点
