import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

def Draw_ROC(file1,file2):
    '''这里注意读取csv的编码方式，
    如果csv里有中文，在windows系统上可以直接用encoding='ANSI'，
    但是到了Mac或者Linux系统上会报错：`LookupError: unknown encoding: ansi`。
    解决方法：
    1. 可以改成encoding='gbk'；
    2. 或者把csv文件里的列名改成英文，就不用选择encoding的方式了。
    '''
    data1=pd.read_csv(file1, encoding='ANSI')
    data1=pd.DataFrame(data1)
    data2=pd.read_csv(file2, encoding='ANSI')
    data2=pd.DataFrame(data2)
    #print(list(data1['label']), list(data1['predict']))
    #print(list(data1['label']), list(data2['predict']))

    fpr_CSNN,tpr_CSNN,thresholds=metrics.roc_curve(list(data1['label']),
                                           list(data1['predict']))
    #roc_auc_DKT=auc(fpr_CSNN,tpr_CSNN)
    roc_auc_DKT = metrics.auc(fpr_CSNN, tpr_CSNN)  # 准确率代表所有正确的占所有数据的比值
    print("roc_auc_CSSSNN",roc_auc_DKT)

    fpr_NN,tpr_NN,thresholds=roc_curve(list(data2['label']),
                                       list(data2['predict']))
    #roc_auc_Transformer=auc(fpr_NN,tpr_NN)
    #LZJ此处论坛中给的计算方法不同

    roc_auc_Transformer = metrics.auc(fpr_NN,tpr_NN)  # 准确率代表所有正确的占所有数据的比值
    print("roc_auc_Transformer",roc_auc_Transformer)

    font = {'family': 'Times New Roman',
            'size': 12,
            }
    '''这里很多电脑上也许默认是'DejaVu Sans'格式，但是在写论文时，
    往往需要'Times New Roman'格式，可以参考[这篇教程](https://blog.csdn.net/weixin_43543177/article/details/109723328)
    '''
    sns.set(font_scale=1.2)
    plt.rc('font',family='Times New Roman')

    plt.plot(fpr_NN,tpr_NN,'purple',label='AUC_Transformer = %0.2f'% roc_auc_Transformer)
    plt.plot(fpr_CSNN,tpr_CSNN,'blue',label='AUC_DKT = %0.2f'% roc_auc_DKT)
    plt.legend(loc='lower right',fontsize = 12)
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True Positive Rate',fontsize = 14)
    plt.xlabel('Flase Positive Rate',fontsize = 14)
    plt.show()

if __name__=="__main__":
    Draw_ROC('./test1_DKT.csv',
             './test2_transformer.csv')
