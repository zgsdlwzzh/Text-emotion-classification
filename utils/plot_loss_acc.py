import pandas as pd
import matplotlib.pyplot as plt
import os

data_root_path='D:\Python\Python37\study\论文\data\out\loss_acc'

textcnn_path=os.path.join(data_root_path,'textcnn2.csv')
bilstm_path=os.path.join(data_root_path,'bilstm.csv')
mic_cnn_path=os.path.join(data_root_path,'mic_cnn.csv')
att_bilstm_path=os.path.join(data_root_path,'att_bilstm2.csv')
att_mic_cnn_path=os.path.join(data_root_path,'mic_cnn_att.csv')
ca_bga_path=os.path.join(data_root_path,'ca_bga2.csv')
mica_bla_att_path=os.path.join(data_root_path,'mica_bla_att2.csv')

def plot_data(data_path):
    data=pd.read_csv(data_path,encoding='utf-8',engine='python')
    data2 = pd.DataFrame()
    step = list(range(0, 10001, 300))
    for i in step:
        data2 = data2.append(data[i:i + 1])
    return data2

textcnn=plot_data(textcnn_path)
bilstm=plot_data(bilstm_path)
mic_cnn=plot_data(mic_cnn_path)
att_bilstm=plot_data(att_bilstm_path)
att_mic_cnn=plot_data(att_mic_cnn_path)
ca_bga=plot_data(ca_bga_path)
mica_bla=plot_data(mica_bla_att_path)

def plot_loss():
    plt.figure()
    plt.plot(textcnn['step'],textcnn['loss'],linewidth=1,label='Loss curve of TextCNN')
    plt.plot(bilstm['step'],bilstm['loss'],linewidth=1,label='Loss curve of BiLSTM')
    plt.plot(mic_cnn['step'],mic_cnn['loss'],linewidth=1,label='Loss curve of MIC_CNN')
    plt.plot(att_bilstm['step'],att_bilstm['loss'],linewidth=1,label='Loss curve of ATT_BiLSTM')
    plt.plot(att_mic_cnn['step'],att_mic_cnn['loss'],linewidth=1,label='Loss curve of ATT_MIC_CNN')
    plt.plot(ca_bga['step'],ca_bga['loss'],linewidth=1,label='Loss curve of CA_BGA')
    plt.plot(mica_bla['step'],mica_bla['loss'],linestyle=':',linewidth=2,label='Loss curve of MIC_BLA')

    plt.ylim([0,5])
    plt.xlim([0,10000])
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Loss curve')
    plt.legend(loc="upper right")

    plt.show()

def plot_acc():
    plt.figure()
    plt.plot(textcnn['step'], textcnn['acc'], linewidth=1, label='Accuracy curve of TextCNN')
    plt.plot(bilstm['step'], bilstm['acc'], linewidth=1, label='Accuracy curve of BiLSTM')
    plt.plot(mic_cnn['step'], mic_cnn['acc'], linewidth=1, label='Accuracy curve of MIC_CNN')
    plt.plot(att_bilstm['step'], att_bilstm['acc'], linewidth=1, label='Accuracy curve of ATT_BiLSTM')
    plt.plot(att_mic_cnn['step'], att_mic_cnn['acc'], linewidth=1, label='Accuracy curve of ATT_MIC_CNN')
    plt.plot(ca_bga['step'], ca_bga['acc'], linewidth=1, label='Accuracy curve of CA_BGA')
    plt.plot(mica_bla['step'], mica_bla['acc'], linestyle=':', linewidth=2, label='Accuracy curve of MIC_BLA')

    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.title('Accuracy curve')
    plt.legend(loc="lower right")

    plt.show()

if __name__=='__main__':
    # plot_loss()
    plot_acc()

