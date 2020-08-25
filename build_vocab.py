import pandas as pd
import jieba
from collections.abc import Iterable


def build_vocab(df=None, idxs=None, stop_list=[], sort=True, min_count=1, save=False, save_path=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError()
    if not isinstance(idxs, Iterable):
        raise TypeError()

    cat_list = []
    for idx in idxs:
        if idx in df.columns:
            if df[idx].dtype == 'object':
                cat_list.append(idx)

    # 合并字符串列
    if len(cat_list) > 0:
        all_sr = df[cat_list[0]].str.cat(df[cat_list[1:]], sep=' ', na_rep='')

    # 分词&统计词频
    dic = pd.Series(dtype=int)
    for item in all_sr:
        for token in jieba.lcut(item.strip()):
            if token not in stop_list:
                if token in dic.index:
                    dic[token] += 1
                else:
                    dic[token] = 1

    # 排序，选择词频
    if sort:
        dic.sort_values(ascending=False, inplace=True)
    dic = dic[dic >= min_count]

    # 保存
    if save:
        dic.to_csv(save_path, header=False)

    return


if __name__ == '__main__':
    REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']
    cat_cols = ['Question', 'Dialogue', 'Report']
    train_df = pd.read_csv('input/AutoMaster_TrainSet.csv')
    test_df = pd.read_csv('input/AutoMaster_TestSet.csv')
    all_df = pd.concat([train_df[:10], test_df[:10]], axis=0)
    build_vocab(all_df[:20], idxs=cat_cols, stop_list=REMOVE_WORDS, min_count=1, save=True, save_path='input/vocab.csv')