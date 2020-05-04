import os

## 根目录
root=r'D:\Python\Python37\study\论文'

## 原始数据路径
data_path=os.path.join(root,'data','Amazon_Unlocked_Mobile.csv')
## 样本数据路径
sample_data_path=os.path.join(root,'data','review.csv')
## 停用词路径
stopwords_path=os.path.join(root,'data','stopwords.txt')
# 数据处理后的路径
proc_sample_data_path=os.path.join(root,'data','proc_sample_data.csv')
# 数据切词后的路径
data_seg_path=os.path.join(root,'data','data_seg.csv')

# 训练数据路径
train_data_path=os.path.join(root,'data','train_data.csv')
reviews_split_train_data_path=os.path.join(root,'data','reviews_split_train_data.csv')

# 测试数据路径
test_data_path=os.path.join(root,'data','test_data.csv')
reviews_split_test_data_path=os.path.join(root,'data','reviews_split_test_data.csv')

# 训练数据切词后路径
train_seg_path=os.path.join(root,'data','train_seg.csv')
# 测试数据切词后的路径
test_seg_path=os.path.join(root,'data','test_seg.csv')

# 词向量路径
save_wv_model_path = os.path.join(root, 'data', 'wv', 'word2vec.model')
# 内容向量路径
train_content_vector_list_path=os.path.join(root,'data','vector','train_content_vector_list.txt')
test_content_vector_list_path=os.path.join(root,'data','vector','test_content_vector_list.txt')
# 标签路径
train_label_list_path=os.path.join(root,'data','vector','train_label_list.txt')
test_label_list_path=os.path.join(root,'data','vector','test_label_list.txt')

# knn_model路径
knn_model_path=os.path.join(root,'data','model','knn_model.pkl')
knn_grid_search_path=os.path.join(root,'data','out','knn_grid_search.csv')
knn_valid_model_path=os.path.join(root,'data','model','valid','knn_model.pkl')
# svm_model路径
svm_model_path=os.path.join(root,'data','model','svm_model.pkl')
svm_grid_search_path=os.path.join(root,'data','out','svm_grid_search.csv')
svm_valid_model_path=os.path.join(root,'data','model','valid','svm_model.pkl')
# bayes_model路径
bayes_model_path=os.path.join(root,'data','model','bayes_model.pkl')
# logistic_model路径
logistic_model_path=os.path.join(root,'data','model','logistic_model.pkl')
logistic_grid_search_path=os.path.join(root,'data','out','logistic_grid_search.csv')
logistic_valid_model_path=os.path.join(root,'data','model','valid','logistic_model.pkl')
# decision tree model路径
tree_model_path=os.path.join(root,'data','model','tree_model.pkl')
tree_grid_search_path=os.path.join(root,'data','out','tree_grid_search.csv')
tree_valid_model_path=os.path.join(root,'data','model','valid','tree_model.pkl')
#############################################################################
# 词汇表路径
vocabulary_list_path=os.path.join(root,'data','vocab','vocabulary_list.pickle')

## Summaries路径
out_dir_path=os.path.join(root,'data','out')
# TextCNN
cnn_train_summary_dir_path=os.path.join(out_dir_path, 'TextCNN','summaries','train')
cnn_test_summary_dir_path=os.path.join(out_dir_path, 'TextCNN','summaries','test')
# BiLSTM
bilstm_train_summary_dir_path=os.path.join(out_dir_path, 'BiLSTM','summaries','train')
bilstm_test_summary_dir_path=os.path.join(out_dir_path, 'BiLSTM','summaries','test')
# Fasttext
fasttext_train_summary_dir_path=os.path.join(out_dir_path, 'Fasttext','summaries','train')
fasttext_test_summary_dir_path=os.path.join(out_dir_path, 'Fasttext','summaries','test')
# RCNN
rcnn_train_summary_dir_path=os.path.join(out_dir_path, 'RCNN','summaries','train')
rcnn_test_summary_dir_path=os.path.join(out_dir_path, 'RCNN','summaries','test')
# ATT_BiLSTM
att_bilstm_train_summary_dir_path=os.path.join(out_dir_path, 'ATT_BiLSTM','summaries','train')
att_bilstm_test_summary_dir_path=os.path.join(out_dir_path, 'ATT_BiLSTM','summaries','test')
# ATT_CNN_BiLSTM
att_cnn_bilstm_train_summary_dir_path=os.path.join(out_dir_path, 'ATT_CNN_BiLSTM','summaries','train')
att_cnn_bilstm_test_summary_dir_path=os.path.join(out_dir_path, 'ATT_CNN_BiLSTM','summaries','test')
# Transformer
transformer_train_summary_dir_path=os.path.join(out_dir_path, 'Transformer','summaries','train')
transformer_test_summary_dir_path=os.path.join(out_dir_path, 'Transformer','summaries','test')


## checkpoint路径
# TextCNN
cnn_checkpoint_dir_path=os.path.join(out_dir_path,'TextCNN','checkpoints')
cnn_checkpoint_prefix_path=os.path.join(cnn_checkpoint_dir_path,'model')
# BiLSTM
bilstm_checkpoint_dir_path=os.path.join(out_dir_path,'BiLSTM','checkpoints')
bilstm_checkpoint_prefix_path=os.path.join(bilstm_checkpoint_dir_path,'model')
# Fasttext
fasttext_checkpoint_dir_path=os.path.join(out_dir_path,'Fasttext','checkpoints')
fasttext_checkpoint_prefix_path=os.path.join(fasttext_checkpoint_dir_path,'model')
# RCNN
rcnn_checkpoint_dir_path=os.path.join(out_dir_path,'RCNN','checkpoints')
rcnn_checkpoint_prefix_path=os.path.join(rcnn_checkpoint_dir_path,'model')
# ATT_BiLSTM
att_bilstm2_checkpoint_dir_path=os.path.join(out_dir_path,'ATT_BiLSTM2','checkpoints')
att_bilstm2_checkpoint_prefix_path=os.path.join(att_bilstm2_checkpoint_dir_path,'model')
# ATT_CNN_BiLSTM
att_cnn_bilstm_checkpoint_dir_path=os.path.join(out_dir_path,'ATT_CNN_BiLSTM','checkpoints')
att_cnn_bilstm_checkpoint_prefix_path=os.path.join(att_cnn_bilstm_checkpoint_dir_path,'model')
# Transformer
transformer_checkpoint_dir_path=os.path.join(out_dir_path,'Transformer','checkpoints')
transformer_checkpoint_prefix_path=os.path.join(transformer_checkpoint_dir_path,'model')

## loss与acc路径
loss_acc_path=os.path.join(out_dir_path,'loss_acc')

# MiCA_BLA val checkpoint路径
val_checkpoint_dir_path=os.path.join(out_dir_path,'val','checkpoints')
val_checkpoint_prefix_path=os.path.join(val_checkpoint_dir_path,'model')

