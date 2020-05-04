import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import numpy as np
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score
from utils.config import *
from utils.data_process import *
from utils.vectorize import *

def load_vector_label_list(content_vector_list_path,label_list_path):
    content_vector_list=np.loadtxt(content_vector_list_path)
    label_list=np.loadtxt(label_list_path)
    return content_vector_list,label_list

def plot_all(test_X,test_y_onehot):
    knn = joblib.load(knn_model_path)
    tree = joblib.load(tree_model_path)
    logistic = joblib.load(logistic_model_path)
    svm = joblib.load(svm_model_path)

    pred_knn=knn.predict_proba(test_X)
    pred_tree=tree.predict_proba(test_X)
    pred_logistic=logistic.predict_proba(test_X)
    pred_svm=svm.predict_proba(test_X)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr["knn_micro"], tpr["knn_micro"], _ = roc_curve(test_y_onehot.ravel(), pred_knn.ravel())
    roc_auc["knn_micro"] = auc(fpr["knn_micro"], tpr["knn_micro"])
    fpr["tree_micro"], tpr["tree_micro"], _ = roc_curve(test_y_onehot.ravel(), pred_tree.ravel())
    roc_auc["tree_micro"] = auc(fpr["tree_micro"], tpr["tree_micro"])
    fpr["logistic_micro"], tpr["logistic_micro"], _ = roc_curve(test_y_onehot.ravel(), pred_logistic.ravel())
    roc_auc["logistic_micro"] = auc(fpr["logistic_micro"], tpr["logistic_micro"])
    fpr["svm_micro"], tpr["svm_micro"], _ = roc_curve(test_y_onehot.ravel(), pred_svm.ravel())
    roc_auc["svm_micro"] = auc(fpr["svm_micro"], tpr["svm_micro"])

    lw=2
    plt.figure()
    plt.plot(fpr["knn_micro"], tpr["knn_micro"],
             label='ROC curve of KNN (area = {0:0.2f})'
                   ''.format(roc_auc["knn_micro"]),
             color='deeppink', linestyle=':', linewidth=2)
    plt.plot(fpr["tree_micro"], tpr["tree_micro"],
             label='ROC curve of Logistic (area = {0:0.2f})'
                   ''.format(roc_auc["tree_micro"]),
             color='blue', linestyle=':', linewidth=2)
    plt.plot(fpr["logistic_micro"], tpr["logistic_micro"],
             label='ROC curve of Decision Tree (area = {0:0.2f})'
                   ''.format(roc_auc["logistic_micro"]),
             color='green', linestyle=':', linewidth=2)
    plt.plot(fpr["svm_micro"], tpr["svm_micro"],
             label='ROC curve of SVM (area = {0:0.2f})'
                   ''.format(roc_auc["svm_micro"]),
             color='orange', linestyle=':', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__=='__main__':
    # main('svm')
    train_X, train_y = load_vector_label_list(train_content_vector_list_path, train_label_list_path)
    test_X, test_y = load_vector_label_list(test_content_vector_list_path, test_label_list_path)

    for i in range(len(train_y)):
        if train_y[i] == 1.0:
            train_y[i] = 2

    for i in range(len(train_y)):
        if train_y[i] == 0.0:
            train_y[i] = 1

    for i in range(len(train_y)):
        if train_y[i] == -1.0:
            train_y[i] = 0

    for i in range(len(test_y)):
        if test_y[i] == 1.0:
            test_y[i] = 2

    for i in range(len(test_y)):
        if test_y[i] == 0.0:
            test_y[i] = 1

    for i in range(len(test_y)):
        if test_y[i] == -1.0:
            test_y[i] = 0

    test_y_onehot = to_categorical(test_y, 3).astype('int32')
    plot_all(test_X, test_y_onehot)

