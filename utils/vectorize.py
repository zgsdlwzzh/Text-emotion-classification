import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import pandas as pd
import numpy as np
from gensim.models import word2vec
from utils.config import *
from utils.data_process import *

def load_wv_model(save_wv_model_path):
    """加载wv_model"""
    wv_model= word2vec.Word2Vec.load(save_wv_model_path)
    return wv_model

def get_cutWords_list(data_seg_path):
    """获取文本分词列表组成的列表"""
    with open(data_seg_path,'r',encoding='utf-8') as f:
        cutWords_list=[content.strip().split(' ') for content in f.readlines()]
    return cutWords_list

def get_content_vector(cutWords,wv_model):
    """计算每篇文本向量（词向量取平均），返回array"""
    word_vector_list=[wv_model[k] for k in cutWords if k in wv_model]
    vector_df=pd.DataFrame(word_vector_list)
    content_vector=vector_df.mean(axis=0).values
    return content_vector

def get_vector_label_list(cutWords_list,data,content_vector_list_path,label_list_path):
    """获取文本向量以及对应的标签列表"""
    vector_list,label_list=[],[]
    for i in range(len(cutWords_list)):
        content_vector = get_content_vector(cutWords_list[i], wv_model)
        if content_vector.shape[0] == 300:
            vector_list.append(content_vector)
            label_list.append(data['Rating'][i])
    # 保存
    np.savetxt(content_vector_list_path,np.array(vector_list))
    np.savetxt(label_list_path,np.array(label_list))


if __name__=='__main__':
    # 加载训练数据与测试数据
    train_data, test_data = load_train_test_data(train_data_path, test_data_path)
    # 加载word2vec模型
    wv_model = load_wv_model(save_wv_model_path)
    # 获取文本分词列表组成的列表
    train_cutWords_list=get_cutWords_list(train_seg_path)
    test_cutWords_list = get_cutWords_list(test_seg_path)
    # 获取所有文本向量以及对应的标签列表
    get_vector_label_list(train_cutWords_list,train_data,train_content_vector_list_path,train_label_list_path)
    get_vector_label_list(test_cutWords_list,test_data,test_content_vector_list_path,test_label_list_path)
