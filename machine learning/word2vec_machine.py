import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from utils.config import *
from utils.data_process import *
from utils.vectorize import *

def load_vector_label_list(content_vector_list_path,label_list_path):
    content_vector_list=np.loadtxt(content_vector_list_path)
    label_list=np.loadtxt(label_list_path)
    return content_vector_list,label_list


def grid_search(model,train_X,train_y,test_X, test_y):
    if model=='bayes':
        bayes = GaussianNB()
        bayes.fit(train_X, train_y)
        joblib.dump(bayes, bayes_model_path)
        precision(bayes)

    if model=='knn':
        df = pd.DataFrame(columns=['n_neighbors', 'score'])
        best_score = 0
        for n_neighbors in [3,5,8,10,15]:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(train_X, train_y)
            score = knn.score(test_X, test_y)
            df = df.append([{'n_neighbors': n_neighbors, 'score': score}], ignore_index=True)
            print('n_neighbors {},score {}'.format(n_neighbors,score))
            if score > best_score:
                best_score = score
                joblib.dump(knn, knn_model2_path)
                best_params = {'n_neighbors':n_neighbors}
        print('Best socre:{:.4f}'.format(best_score))
        print('Best parameters:{}'.format(best_params))
        df.to_csv(knn_grid_search_path, index=False)
        best_clf = joblib.load(knn_model_path)
        precision(best_clf)
    if model=='logistic':
        df = pd.DataFrame(columns=['penalty', 'C', 'score'])
        best_score=0
        for penalty in ['l1','l2']:
            for c in [0.01,0.1,1,10,100]:
                logistic=LogisticRegression(penalty=penalty,C=c)
                logistic.fit(train_X,train_y)
                score=logistic.score(test_X, test_y)
                df = df.append([{'penalty': penalty, 'C': c, 'score': score}])
                print('penalty {},c {},score {}'.format(penalty,c,score))
                if score>best_score:
                    best_score=score
                    # joblib.dump(logistic,logistic_model2_path)
                    best_params={'penalty':penalty,'C':c}

        print('Best socre:{:.4f}'.format(best_score))
        print('Best parameters:{}'.format(best_params))
        df.to_csv(logistic_grid_search_path, index=False)
        best_clf=joblib.load(logistic_model_path)
        precision(best_clf)

    if model=='svm':
        df = pd.DataFrame(columns=['C', 'gamma', 'score'])
        best_score = 0
        for c in [0.01, 0.1, 1, 10, 100]:
            for gamma in [0.01, 0.1, 1, 10,100]:
                svm = SVC(C=c,gamma=gamma)
                svm.fit(train_X, train_y)
                score = svm.score(test_X, test_y)
                df = df.append([{'C': c, 'gamma': gamma, 'score': score}])
                print('c {},gamma {},score {}'.format(c, gamma,score))
                if score > best_score:
                    best_score = score
                    # joblib.dump(svm, svm_model2_path)
                    best_params = { 'C': c,'gamma':gamma}
        print('Best socre:{:.4f}'.format(best_score))
        print('Best parameters:{}'.format(best_params))
        df.to_csv(svm_grid_search_path, index=False)
        best_clf = joblib.load(svm_model_path)
        precision(best_clf)

    if model=='decision_tree':
        df = pd.DataFrame(columns=['max_depth', 'min_sample_leaf','min_sample_split', 'score'])
        best_score = 0
        for max_depth in [18,19,20]:
            for min_sample_leaf in [2,4,6]:
                for min_sample_split in [2,4,6]:
                    tree = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_sample_leaf,min_samples_split=min_sample_split,random_state=42)
                    tree.fit(train_X, train_y)
                    score = tree.score(test_X, test_y)
                    df = df.append([{'max_depth': max_depth, 'min_sample_leaf': min_sample_leaf,'min_sample_split':min_sample_split, 'score': score}])
                    print('max_depth {},min_sample_leaf {},min_sample_split {},score {}'.format(max_depth,min_sample_leaf,min_sample_split,score))
                    if score > best_score:
                        best_score = score
                        # joblib.dump(tree, tree_model2_path)
                        best_params = {'max_depth': max_depth, 'min_sample_leaf':min_sample_leaf,'min_sample_split':min_sample_split}
        print('Best socre:{:.4f}'.format(best_score))
        print('Best parameters:{}'.format(best_params))
        df.to_csv(tree_grid_search_path,index=False)
        best_clf = joblib.load(tree_model_path)
        precision(best_clf)

# 预测分类结果
def precision(clf):
    predicted=clf.predict(test_X)
    total=len(predicted)
    rate=0
    for flabel,except_cate in zip(test_y,predicted):
        if flabel!=except_cate:
            rate+=1
    print('error_rate:',float(rate)*100/float(total),'%')
    print('精度:{0:.4f}'.format(metrics.precision_score(test_y,predicted,average='weighted')))
    print('召回:{0:.4f}'.format(metrics.recall_score(test_y,predicted,average='weighted')))
    print('f1-score:{0:.4f}'.format(metrics.f1_score(test_y, predicted, average='weighted')))

if __name__=='__main__':
    # 获取训练数据与测试数据
    train_X,train_y=load_vector_label_list(train_content_vector_list_path, train_label_list_path)
    test_X, test_y = load_vector_label_list(test_content_vector_list_path, test_label_list_path)

    # 网格搜索最佳参数
    grid_search('svm', train_X, train_y, test_X, test_y)

