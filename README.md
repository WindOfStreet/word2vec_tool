谨以此文档献给我的第一次git

build_w2v.py：输入Dataframe数据，输出并保存Dataframe数据训练的词向量
## build_w2v.py
    使用gensim完成词向量的训练
    保存/加载词向量模型
    保存/加载词嵌入矩阵
    保存/加载词典
## build_vocab.py
    输入Dataframe数据，根据设定对其中部分数据进行文本处理，分词，统计词频，按出现次数保存词频。
    因为使用了gensim，此模块在build_w2v.py中并没有调用，可以忽略。
