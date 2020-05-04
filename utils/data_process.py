import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import pandas as pd
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from gensim.models import word2vec
from utils.config import *

def load_sample_data(sample_data_path):
    """负面，中性，正面评论各采样25000条数据"""
    review=pd.read_csv(sample_data_path,encoding='utf-8',engine='python')
    return review

def replace_abbre(sentence):
    """转换为小写字母，词形还原，英文缩写替换"""
    # 转换为小写
    sentence=sentence.lower()
    sentence=sentence.split(' ')
    for i in range(len(sentence)):
        # 词形还原
        stem_wordnet = WordNetLemmatizer()
        sentence[i]=stem_wordnet.lemmatize(sentence[i])
        # 英文缩写替换
        if sentence[i][-3:] == "t's":
            sentence[i] = sentence[i][:-2] + " is"
        elif sentence[i][-3:] == "n't":
            sentence[i] = sentence[i][:-3] + " not"
    return ' '.join(sentence)

def clean_sentence(sentence):
    """
    特殊符号去除
    :param sentence:待处理的字符串
    :return: 过滤特殊字符后的字符串
    """
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\!\/\|\[\]\{\}_,.$%^*(+\'"\)]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]',
            ' ', sentence)
    else:
        return ''


def load_stop_words(stopwords_path):
    """
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    """
    with open(stopwords_path,'r',encoding='utf-8') as file:
        stop_words=file.readlines()
    # 去除每一个停用词前后的空格 换行符
    stop_words=[stop_word.strip() for stop_word in stop_words]
    return stop_words

stop_words = load_stop_words(stopwords_path)

def sentence_proc(sentence):
    # 转换为小写字母，词形还原，英文缩写替换
    sentence=replace_abbre(sentence)
    # 清除特殊符号
    sentence=clean_sentence(sentence)
    # 切词
    words = sentence.split(' ')
    # 过滤停用词
    words = [word for word in words if word and word not in stop_words]
    return ' '.join(words)

def load_proc_sample_data(proc_sample_data_path):
    """去除停用词，去除空字符串后的训练数据"""
    review = pd.read_csv(proc_sample_data_path, encoding='utf-8', engine='python')
    return review

def get_wv_model(data_seg_path,save_wv_model_path,dim=300):
    sentences = word2vec.LineSentence(data_seg_path)
    wv_model = word2vec.Word2Vec(sentences, size=dim, window=5, min_count=5)
    wv_model.save(save_wv_model_path)

def get_train_test_data(proc_sample_data_path):
    total_data=load_proc_sample_data(proc_sample_data_path)
    # 打乱数据
    total_data=total_data.sample(frac=1).reset_index(drop=True)
    # 前80%为训练数据，其余为测试数据
    train_data=total_data[:int(len(total_data)*0.8)]
    test_data=total_data[int(len(total_data)*0.8):]
    train_data.to_csv(train_data2_path,index=False,header=True)
    test_data.to_csv(test_data2_path,index=False,header=True)

def load_train_test_data(train_data_path,test_data_path):
    train_data=pd.read_csv(train_data_path,encoding='utf-8', engine='python')
    test_data=pd.read_csv(test_data_path,encoding='utf-8', engine='python')
    return train_data,test_data

if __name__=='__main__':
    # 加载采样数据
    review=load_sample_data(sample_data_path)
    # 清洗Reviews
    review['Reviews'] = review['Reviews'].apply(sentence_proc)
    review=review[review['Reviews']!='']
    review.to_csv(proc_sample_data_path, index=False)
    review['Reviews'].to_csv(data_seg_path,index=False)
    # 训练wv_model
    get_wv_model(data_seg_path, save_wv_model_path)

    # 获取训练数据与测试数据
    get_train_test_data(proc_sample_data_path)
    # 保存训练数据与测试数据的切词
    train_data, test_data = load_train_test_data(train_data_path, test_data_path)
    train_data['Reviews'].to_csv(train_seg_path, index=False)
    test_data['Reviews'].to_csv(test_seg_path, index=False)
