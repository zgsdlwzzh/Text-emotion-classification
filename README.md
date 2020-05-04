# Text-emotion-classification
Text classification using machine learning model（KNN，Logistic，Decision tree，SVM）and deeping model（TextCNN，BiLSTM，Fasttext，RCNN，Transformer，ATT_BiLSTM，ATT_CNN_BiLSTM）

对Amazon无锁手机评论进行三元情感分类，机器学习模型是将训练的word2vec词向量均值作为输入，深度学习模型使用Embedding层进行向量化。

Data address：
    链接：https://pan.baidu.com/s/1J7wzv5nZscU0n6ZKOKFmsw 
    提取码：2bhk
    
    其中：Amazon_Unlocked_Mobile.csv为Amazon无锁手机评论原始数据；review.csv为三种情感各抽取25000条，-1为消极，0为中立，1为积极；stopwords.txt为英文停用词

utils
    --config.py 存放文件路径
    --params_utils.py 存放参数取值
    --data_process.py 清洗数据；训练word2vec模型
    --vectorize.py 使用word2vec模型将文本向量化
    --content_to_id.py 将本文中的词用词汇表中的索引来表示
    --split_reviews.py 将文本前后半句分开（这里模型用不到）
    --MIC_proc.py MIC_CNN模型使用的词索引向量（这里模型用不到）
    --batcher.py 生成batch数据
    --ROC.py 机器学习模型的ROC曲线
    --plot_loss_acc.py 深度学习模型的loss和acc变化曲线
