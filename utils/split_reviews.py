import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import pandas as pd
from utils.config import *


def get_left_sentence(sentence):
    split_sen=sentence.split(' ')
    length=len(split_sen)
    left=' '.join(split_sen[:length//2]) if length>1 else sentence
    return left

def get_right_sentence(sentence):
    split_sen=sentence.split(' ')
    length=len(split_sen)
    right=' '.join(split_sen[length//2:])
    return right

def get_reviews_split_train(train_data_path):
    data = pd.read_csv(train_data_path, encoding='utf-8', engine='python')
    data['left_reviews'] = data['Reviews'].apply(get_left_sentence)
    data['right_reviews'] = data['Reviews'].apply(get_right_sentence)
    data = data.dropna(how='any')
    data.to_csv(reviews_split_train_data_path, index=False)

def get_reviews_split_test(test_data_path):
    data = pd.read_csv(test_data_path, encoding='utf-8', engine='python')
    data['left_reviews'] = data['Reviews'].apply(get_left_sentence)
    data['right_reviews'] = data['Reviews'].apply(get_right_sentence)
    data = data.dropna(how='any')
    data.to_csv(reviews_split_test_data_path, index=False)

if __name__=='__main__':
    get_reviews_split_train(train_data_path)
    get_reviews_split_test(test_data_path)