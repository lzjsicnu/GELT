import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from sklearn import metrics
from scipy import sparse
from tqdm import tqdm
import time, math

#lzj添加打印正确率曲线
import matplotlib.pyplot as plt
epochs_picture=[]
tran_loss_picture=[]
test_acc_picture=[]
test_preds_float=[]

def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    split_point = int(n_samples*split)
    train_data, test_data = [], []
    for d in data:
        train_data.append(d[:split_point])
        test_data.append(d[split_point:])
    return train_data, test_data


# data
data_folder = "assist09"
data = np.load(os.path.join(data_folder, data_folder+'.npz'))
y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']
#['problem', 'skill_num', 'real_len', 'problem_num', 'skill', 'y'],
#y.shape=(3841, 200)=skill.shape=proble.shape,real_len.shape=(3841,)
skill_num, pro_num = data['skill_num'], data['problem_num']
print('problem number %d, skill number %d' % (pro_num, skill_num))

# divide train test set
train_data, test_data = train_test_split([y, skill, problem, real_len])   # [y, skill, pro, real_len]
train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
#训练的时候，没有用到train_skill和test_skill

# embed data, used for initialize
embed_data = np.load(os.path.join(data_folder, 'embedding_200.npz'))
_, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
print('lzj line38,pre_pro_embed.shape, pre_pro_embed.dtype', (pre_pro_embed.shape, pre_pro_embed.dtype))
#前两个都是(167, 64)， 第三个pre_pro_embed.shape (15911, 128),这里只用得到第三个


# hyper-params
epochs = 16
bs = 128
embed_dim = pre_pro_embed.shape[1]#128
hidden_dim = 128
lr = 0.001
use_pretrain = True
train_embed = False
train_flag = True
#train_flag = False


# build tensorflow graph
tf_y = tf.placeholder(tf.float32, [None, None], name='tf_y')
tf_pro = tf.placeholder(tf.int32, [None, None], name='tf_pro')
#placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')   # [bs]

tf_batch_size = tf.shape(tf_y)[0]
tf_max_len = tf.shape(tf_y)[1]


with tf.variable_scope('pro_skill_embed', reuse=tf.AUTO_REUSE):
    if use_pretrain:
        pro_embed_init = tf.constant_initializer(pre_pro_embed)
        # skill_embed_init = tf.constant_initializer(pre_skill_embed)
        print("use pretrain embedding matrix")
    else:    
        pro_embed_init = tf.truncated_normal_initializer(stddev=0.1)
        # skill_embed_init = tf.truncated_normal_initializer(stddev=0.1)
        print("没有使用预训练，use random init embedding matrix")
    
    pro_embedding_matrix = tf.get_variable('pro_embed_matrix', [pro_num, embed_dim], 
                                initializer=pro_embed_init, trainable=train_embed)#shape=(15911, 128)
    # skill_embedding_matrix = tf.get_variable('skill_embed_matrix', [skill_num, embed_dim], 
    #                             initializer=skill_embed_init, trainable=train_flag)
    # deal with complement symbol
    zero_tensor = tf.zeros([1, embed_dim], dtype=tf.float32)   # shape=(1, 128)
    pro_embedding_matrix = tf.concat([zero_tensor, pro_embedding_matrix], axis=0)#shape=(15912, 128),注意，这里的拼接，只能用一次，否则15912+1
    # skill_embedding_matrix = tf.concat([zero_tensor, skill_embedding_matrix], axis=0)
    

# skill, problem embedding
# all_skill_embed = tf.nn.embedding_lookup(skill_embedding_matrix, tf_skill)  # [bs, max_len, embed_dim]
all_pro_embed = tf.nn.embedding_lookup(pro_embedding_matrix, tf_pro)  # [bs, max_len, embed_dim],这里tf_pro是数据集中问题的tf.placeholder，
# skill_embed = all_skill_embed[:, :-1, :]
pro_embed = all_pro_embed[:, :-1, :]#shape=(?, ?, 128)# :-1从位置0到位置-1之前的数
# next_skill_embed = all_skill_embed[:, 1:, :]
next_pro_embed = all_pro_embed[:, 1:, :]         # [bs, max_len-1, embed_dim]#shape=(?, ?, 128)
# a[1:]a为字符串，1为开始索引，没有指定结束索引即默认为最后一位。字符串截取遵循“左闭右开”原则，即为从1开始截取，不包括1，截取到最后一位，包括最后一位。

inputs_embed = pro_embed   # [bs, max_len-1, embed_dim]#shape=(?, ?, 128)，问题编码，中间那个维度留的是除最后一列的元素。不包含第一个元素。
next_inputs_embed = next_pro_embed#shape=(?, ?, 128)，这个留的是为编码中，不包含第一个元素。a=[1,2,3,4,5]，print(a[1:])[2, 3, 4, 5]
inputs_embed_dim = embed_dim#128
print("problem level prediction")

# concat zero on the skill embedding
def concat_zero(x):
    xx = x[:-1]
    yy = x[-1]
    zero_tensor = tf.zeros([embed_dim], dtype=tf.float32)
    o = tf.cond(tf.greater(yy, 0.), lambda:tf.concat([xx, zero_tensor], axis=0), lambda:tf.concat([zero_tensor, xx], axis=0))
    return o

inputs_y = tf.concat([tf.reshape(inputs_embed, [-1, inputs_embed_dim]), tf.reshape(tf_y[:, :-1], [-1, 1])], axis=1)
#shape=(?, 129)
inputs_y_embedding = tf.map_fn(concat_zero, inputs_y) #shape=(?, 256)#map_fn是将inputs_y输入到前面函数中concat_zero
rnn_inputs = tf.reshape(inputs_y_embedding, [-1, tf_max_len-1, inputs_embed_dim+embed_dim])
#shape=(?, ?, 256),rnn_inputs输入中，是否包含标签y？

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


# 实现多头机制
def muti_head_attention(_input, d=8, n_attention_head=2):
    """
    实现单层多头机制
    @param _input: 输入 (?, n_feats, n_dim)
    @param d: Q,K,V映射后的维度
    @param n_attention_head: multi-head attention的个数
    """
    attention_heads = []

    for i in range(n_attention_head):
        embed_q = layers.Dense(d)(_input)  # 相当于映射到不同的空间,得到不同的Query
        embed_v = layers.Dense(d)(_input)  # 相当于映射到不同的空间,得到不同的Value
        attention_output = layers.Attention()([embed_q, embed_v])
        # 将每一个head的结果暂时存入
        attention_heads.append(attention_output)

    # 多个head则合并，单个head直接返回
    if n_attention_head > 1:
        muti_attention_output = layers.Concatenate(axis=-1)(attention_heads)
    else:
        muti_attention_output = attention_output
    return muti_attention_output


# lstm
with tf.variable_scope('LstmNet', reuse=tf.AUTO_REUSE):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    init_state = lstm_cell.zero_state(tf_batch_size, tf.float32)
    # outputs, last_state = tf.nn.dynamic_rnn(cell=lstm_cell,
    #                                         inputs=rnn_inputs,
    #                                         sequence_length=tf_real_seq_len,
    #                                         initial_state=init_state,
    #                                         dtype=tf.float32)

    #20220104:1：尝试共keras替换tf的lstm，成功，加层也成功
    #outputs = layers.LSTM(128, return_sequences=True, recurrent_initializer='orthogonal')(rnn_inputs)
    # 在来一层LSTM，每个h_i的输出维度为90
    #outputs = layers.LSTM(128, return_sequences=True, recurrent_initializer='orthogonal')(outputs)

    #20220104:2：注意力机制
    outputs = layers.LSTM(128, return_sequences=True, recurrent_initializer='orthogonal')(rnn_inputs)
    # 多层 muti_head_attention,将LSTM结构的输出直接输入
    print("line 161:outputs:", outputs)
    outputs = muti_head_attention(outputs, 128, 1)
    print("line 162:outputs:", outputs)#8
    #outputs = muti_head_attention(outputs, 42, 3)
    # 输出
    print("line 166:outputs:", outputs)
    #line122: outputs: Tensor("LstmNet/rnn/transpose_1:0", shape=(?, ?, 128), dtype=float32)
    #line122: outputs: Tensor("LstmNet/lstm/transpose_1:0", shape=(?, ?, 128), dtype=float32)
    outputs_reshape = tf.reshape(outputs, [-1, hidden_dim])   # [bs*(max_len-1), hidden_dim]#shape=(?, 128)
    outputs_reshape = tf.concat([outputs_reshape, tf.reshape(next_inputs_embed, [-1, inputs_embed_dim])], axis=1)

    rnn_w = tf.get_variable('softmax_w', [hidden_dim+inputs_embed_dim, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    rnn_b = tf.get_variable('softmax_b', [1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    logits = tf.nn.xw_plus_b(outputs_reshape, rnn_w, rnn_b)  # [bs*(max_len-1), 1]


# ignore the answer -1
tf_targets_flatten = tf.reshape(tf_y[:, 1:], [-1])
index = tf.where(tf.not_equal(tf_targets_flatten, tf.constant(-1, dtype=tf.float32)))
filtered_targets = tf.squeeze(tf.gather(tf_targets_flatten, index), axis=1)

# lstm-outputs
logits = tf.reshape(logits, [-1])
filtered_logits = tf.squeeze(tf.gather(logits, index), axis=1)
final_logits = filtered_logits

# cross entropy, optimize
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=filtered_targets,
                                            logits=final_logits))
filtered_predict_prob = tf.nn.sigmoid(final_logits)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)
saver = tf.train.Saver()

if train_flag:
    # begin train
    train_steps = int(math.ceil(train_skill.shape[0]/float(bs)))
    test_steps = int(math.ceil(test_skill.shape[0]/float(bs)))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter("log/", sess.graph)

        # random_pro_embed = sess.run(pro_embedding_matrix)
        # np.savez(os.path.join(data_folder, 'random_embed.npz'), 
        #             random_skill_embed=random_skill_embed, random_pro_embed=random_pro_embed)
        
        best_auc = best_acc = 0
        for i in tqdm(range(epochs)):
            train_loss = 0

            for j in range(train_steps):
                batch_y = train_y[j*bs:(j+1)*bs, :]
                batch_pro = train_problem[j*bs:(j+1)*bs, :]
                # batch_skill = train_skill[j*bs:(j+1)*bs, :]
                batch_real_len = train_real_len[j*bs:(j+1)*bs] - 1
                #print("line164,batch_real_len=",batch_real_len)
                feed_dict = {tf_pro: batch_pro, 
                            tf_y:batch_y, tf_real_seq_len:batch_real_len}
                #print("line229:feed_dict:", feed_dict)
                _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)
                train_loss += batch_loss
            train_loss /= batch_loss

            test_preds, test_targets, test_loss = [], [], 0
            for j in range(test_steps):
                test_y_ = test_y[j*bs:(j+1)*bs, :]
                # test_skill_ = test_skill[j*bs:(j+1)*bs, :]
                test_pro_ = test_problem[j*bs:(j+1)*bs, :]
                test_real_len_ = test_real_len[j*bs:(j+1)*bs] - 1
                feed_dict = {tf_y:test_y_, 
                            tf_pro: test_pro_, tf_real_seq_len:test_real_len_}
                filtered_targets_, filtered_preds_ = sess.run([filtered_targets, filtered_predict_prob], feed_dict=feed_dict)
                test_preds.append(filtered_preds_)
                test_targets.append(filtered_targets_)
                #print("line182,filtered_targets_=", filtered_targets_)
                #print("line183,filtered_preds_=", filtered_preds_)



                
            test_preds = np.concatenate(test_preds, axis=0)
            test_targets = np.concatenate(test_targets, axis=0)

            # lzj添加~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 为了收集AUC曲线，这里用一个list，把浮点型的小数收集，并写入最后的表格
            test_preds_float=(test_preds.tolist())

            #test_preds_float = np.concatenate(test_preds_float, axis=0)
            #print("lin204lzj:test_preds_float=", test_preds_float)



            test_auc = metrics.roc_auc_score(test_targets, test_preds)
            ##lzj添加，计算roc，说明作者的test_auc直接通过上一行计算出，与我分解的方法得出结果是一致的
            fpr, tpr, threshold = metrics.roc_curve(test_targets, test_preds)
            #print("lin204lzj:test_targets", test_targets)
            #print("lin204lzj:test_preds", test_preds)
            #print("lin204lzj:threshold", threshold)
            roc_auc = metrics.auc(fpr, tpr)  # 准确率代表所有正确的占所有数据的比值
            #print("lin204lzj:fpr=", fpr)
            #print("lin204lzj:tpr=", tpr)
            #print("lin204lzj:roc_auc=", roc_auc)





            test_preds[test_preds>0.5] = 1.
            test_preds[test_preds<=0.5] = 0.
            test_acc = metrics.accuracy_score(test_targets, test_preds)



            
            records = 'Epoch %d/%d, train loss:%3.5f, test auc:%f, test acc:%3.5f' % \
                            (i+1, epochs, train_loss, test_auc, test_acc)    
            print(records)
            epochs_picture.append(i + 1)
            tran_loss_picture.append(train_loss)
            test_acc_picture.append(test_acc)

            if best_auc < test_auc:
                best_auc = test_auc
                saver.save(sess, os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.ckpt'))
else:
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.ckpt'))
        if not os.path.exists(os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.ckpt')):
            print("lzj lin200:没有这个文件夹，'bekt_dkt_model/dkt_pretrain.ckpt'")
            #os.mkdir(saved_model_folder)

        #pro_embed, skill_embed 
        pro_embed_trained = sess.run(pro_embedding_matrix)
        pro_embed_trained = pro_embed_trained[1:]
        
        np.savez(os.path.join(data_folder, 'bekt_dkt_model/pro_embed_bekt_dkt.npz'), pro_final_repre=pro_embed_trained)



print("epochs_picture:",epochs_picture)
print("tran_loss_picture:",tran_loss_picture)
print("test_acc_picture:",test_acc_picture)
plt.plot(epochs_picture, tran_loss_picture, color='r', label='acc')  # r表示红色
plt.plot(epochs_picture, test_acc_picture, color=(0, 0, 0), label='loss')  # 也可以用RGB值表示颜色
# #####非必须内容#########
plt.xlabel('Epochs')  # x轴表示
plt.ylabel('Test_acc')  # y轴表示
plt.title("cha")  # 图标标题表示
plt.legend()  # 每条折线的label显示
# #######################
plt.savefig('Transformer06.jpg')  # 保存图片，路径名为test.jpg
plt.show()  # 显示图片

# 例如我们要存储两个list：name_list 和 err_list 到 Excel 两列
# name_list和err_list均已存在
# name_list = [10, 20, 30]  # 示例数据
# err_list = [0.99, 0.98, 0.97]  # 示例数据

# 导包，如未安装，先 pip install xlwt
import xlwt
# 设置Excel编码
file = xlwt.Workbook('encoding = utf-8')
# 创建sheet工作表
sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
# 先填标题
# sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
sheet1.write(0, 0, "周期")  # 第1行第1列
sheet1.write(0, 1, "训练误差")  # 第1行第2列
sheet1.write(0, 2, "测试准确率")  # 第1行第3列

# 循环填入数据
for i in range(len(epochs_picture)):
    sheet1.write(i + 1, 0, i)  # 第1列序号
    sheet1.write(i + 1, 1, tran_loss_picture[i])  # 第2列数量
    sheet1.write(i + 1, 2, test_acc_picture[i])  # 第3列误差

# 保存Excel到.py源文件同级目录
file.save('20221209pebg_Transformer06训练情况记录.xls')


plt.figure(figsize=(6,6))
plt.title('Validation ROC')
#plt.plot(test_targets[0:9], test_preds[0:9], 'b', label = 'DKT AUC = %0.3f' %test_auc)
plt.plot(fpr, tpr, 'b', label = 'Transformer AUC = %0.3f' %roc_auc)
#plt.plot(fpr2, tpr2, 'r', label = 'EoTransformer AUC = %0.3f' % roc_auc2)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("filename07.png")
plt.show()
plt.close()




import csv
import pandas as pd
dataframe = pd.DataFrame({'label':test_targets,'predict':test_preds_float})
dataframe.to_csv("test2_transformer.csv",index=False,sep=',')