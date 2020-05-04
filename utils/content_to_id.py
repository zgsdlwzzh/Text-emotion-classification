import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
import pickle
from collections import Counter
from utils.config import *
from utils.params_utils import get_params

class ContentIDlst(object):
    def __init__(self,params):
        self.params=params
        self.vocab_size=0
        self.word2id_dict={}
        self.train_reviews_idlst = []
        self.test_reviews_idlst = []
        self.train_labels_onehot = []
        self.test_labels_onehot = []

    def read_data(self,proc_sample_data_path):
        """读取数据，将内容与文本保存为列表"""
        df=pd.read_csv(proc_sample_data_path,encoding='utf-8',engine='python')
        labels=df['Rating'].tolist()
        review=df['Reviews'].tolist()
        reviews=[line.strip().split() for line in review]
        return reviews,labels

    def getVocabulary(self,reviews):
        """制作词汇表"""
        # 内容中所有词汇
        allWords=[word for review in reviews for word in review]
        # 统计词频并排序
        wordCount=Counter(allWords)
        sort_wordCount=sorted(wordCount.items(),key=lambda x:x[1],reverse=True)
        # 去除低频词
        words=[item[0] for item in sort_wordCount if item[1]>=5]
        vocabulary_list=['PAD']+['UNK']+words
        self.vocab_size=len(vocabulary_list)
        # 保存词汇表
        with open(vocabulary_list2_path,'wb') as f:
            pickle.dump(vocabulary_list,f)
        # 词汇-索引映字典
        self.word2id_dict=dict(zip(vocabulary_list,list(range(len(vocabulary_list)))))

    def get_review_idlst(self,review,sequence_length,word2id_dict):
        """将数据集中每条评论用index表示，pad对应的index为0"""
        review_idlst=np.zeros((sequence_length))
        sequence_len=sequence_length
        if len(review)<sequence_length:
            sequence_len=len(review)
        for i in range(sequence_len):
            if review[i] in word2id_dict:
                review_idlst[i]=word2id_dict[review[i]]
            else:
                review_idlst[i]=word2id_dict['UNK']
        return review_idlst

    def get_train_test_idlst(self,train_data_path,test_data_path):
        train_reviews,train_labels=self.read_data(train_data_path)
        test_reviews, test_labels = self.read_data(test_data_path)
        train_reviews_idlst=[self.get_review_idlst(review,self.params['sequence_length'],self.word2id_dict) for review in train_reviews]
        test_reviews_idlst = [self.get_review_idlst(review, self.params['sequence_length'], self.word2id_dict) for review in test_reviews]
        train_labels_onehot=to_categorical(train_labels,self.params['num_classes'])
        test_labels_onehot = to_categorical(test_labels, self.params['num_classes'])
        return train_reviews_idlst,test_reviews_idlst,train_labels_onehot,test_labels_onehot

    def main(self,proc_data_path):
        # 读取数据
        reviews, labels=self.read_data(proc_data_path)
        # 制作词汇表
        self.getVocabulary(reviews)
        # 获取文本index组成的列表与标签one-hot列表
        train_reviews_idlst, test_reviews_idlst, train_labels_onehot, test_labels_onehot\
            =self.get_train_test_idlst(train_data2_path,test_data2_path)
        self.train_reviews_idlst=np.array(train_reviews_idlst)
        self.test_reviews_idlst=np.array(test_reviews_idlst)
        self.train_labels_onehot=train_labels_onehot
        self.test_labels_onehot=test_labels_onehot


def get_max_len(data):
    """
    获取合适的最大长度值
    :param data: 待统计的数据
    :return: 最大长度值
    """
    max_lens=data['Reviews'].apply(lambda x:x.count(' '))
    return int(np.mean(max_lens)+2*np.std(max_lens))


if __name__=='__main__':
    params = get_params()
    conid=ContentIDlst(params)
    conid.main(proc_sample_data_path)