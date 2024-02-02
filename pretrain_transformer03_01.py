import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from sklearn import metrics
from scipy import sparse
from tqdm import tqdm
import time, math


def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    split_point = int(n_samples * split)
    train_data, test_data = [], []
    for d in data:
        train_data.append(d[:split_point])
        test_data.append(d[split_point:])
    return train_data, test_data


# data
data_folder = "assist09"
data = np.load(os.path.join(data_folder, data_folder + '.npz'))
y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']
# ['problem', 'skill_num', 'real_len', 'problem_num', 'skill', 'y'],
# y.shape=(3841, 200)=skill.shape=proble.shape,real_len.shape=(3841,)
skill_num, pro_num = data['skill_num'], data['problem_num']
print('problem number %d, skill number %d' % (pro_num, skill_num))

# divide train test set
train_data, test_data = train_test_split([y, skill, problem, real_len])  # [y, skill, pro, real_len]
train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
# 训练的时候，没有用到train_skill和test_skill

# embed data, used for initialize
embed_data = np.load(os.path.join(data_folder, 'embedding_200.npz'))
_, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
print('lzj line38,pre_pro_embed.shape, pre_pro_embed.dtype', (pre_pro_embed.shape, pre_pro_embed.dtype))
# 前两个都是(167, 64)， 第三个pre_pro_embed.shape (15911, 128),这里只用得到第三个


# hyper-params
epochs = 199
bs = 128
embed_dim = pre_pro_embed.shape[1]  # 128
hidden_dim = 128
lr = 0.001
use_pretrain = True
train_embed = False
train_flag = True
# train_flag = False


# build tensorflow graph
tf_y = tf.placeholder(tf.float32, [None, None], name='tf_y')
tf_pro = tf.placeholder(tf.int32, [None, None], name='tf_pro')
# placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')  # [bs]

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
                                           initializer=pro_embed_init, trainable=train_embed)  # shape=(15911, 128)
    # skill_embedding_matrix = tf.get_variable('skill_embed_matrix', [skill_num, embed_dim],
    #                             initializer=skill_embed_init, trainable=train_flag)
    # deal with complement symbol
    zero_tensor = tf.zeros([1, embed_dim], dtype=tf.float32)  # shape=(1, 128)
    pro_embedding_matrix = tf.concat([zero_tensor, pro_embedding_matrix],
                                     axis=0)  # shape=(15912, 128),注意，这里的拼接，只能用一次，否则15912+1
    # skill_embedding_matrix = tf.concat([zero_tensor, skill_embedding_matrix], axis=0)

# skill, problem embedding
# all_skill_embed = tf.nn.embedding_lookup(skill_embedding_matrix, tf_skill)  # [bs, max_len, embed_dim]
all_pro_embed = tf.nn.embedding_lookup(pro_embedding_matrix,
                                       tf_pro)  # [bs, max_len, embed_dim],这里tf_pro是数据集中问题的tf.placeholder，
# skill_embed = all_skill_embed[:, :-1, :]
pro_embed = all_pro_embed[:, :-1, :]  # shape=(?, ?, 128)# :-1从位置0到位置-1之前的数
# next_skill_embed = all_skill_embed[:, 1:, :]
next_pro_embed = all_pro_embed[:, 1:, :]  # [bs, max_len-1, embed_dim]#shape=(?, ?, 128)
# a[1:]a为字符串，1为开始索引，没有指定结束索引即默认为最后一位。字符串截取遵循“左闭右开”原则，即为从1开始截取，不包括1，截取到最后一位，包括最后一位。

inputs_embed = pro_embed  # [bs, max_len-1, embed_dim]#shape=(?, ?, 128)，问题编码，中间那个维度留的是除最后一列的元素。不包含第一个元素。
next_inputs_embed = next_pro_embed  # shape=(?, ?, 128)，这个留的是为编码中，不包含第一个元素。a=[1,2,3,4,5]，print(a[1:])[2, 3, 4, 5]
inputs_embed_dim = embed_dim  # 128
print("problem level prediction")


# concat zero on the skill embedding
def concat_zero(x):
    xx = x[:-1]
    yy = x[-1]
    zero_tensor = tf.zeros([embed_dim], dtype=tf.float32)
    o = tf.cond(tf.greater(yy, 0.), lambda: tf.concat([xx, zero_tensor], axis=0),
                lambda: tf.concat([zero_tensor, xx], axis=0))
    return o


inputs_y = tf.concat([tf.reshape(inputs_embed, [-1, inputs_embed_dim]), tf.reshape(tf_y[:, :-1], [-1, 1])], axis=1)
# shape=(?, 129)
inputs_y_embedding = tf.map_fn(concat_zero, inputs_y)  # shape=(?, 256)#map_fn是将inputs_y输入到前面函数中concat_zero
rnn_inputs = tf.reshape(inputs_y_embedding, [-1, tf_max_len - 1, inputs_embed_dim + embed_dim])
# shape=(?, ?, 256),rnn_inputs输入中，是否包含标签y？

# lstm
with tf.variable_scope('LstmNet', reuse=tf.AUTO_REUSE):
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    # init_state = lstm_cell.zero_state(tf_batch_size, tf.float32)
    # outputs, last_state = tf.nn.dynamic_rnn(cell=lstm_cell,
    #                                         inputs=rnn_inputs,
    #                                         sequence_length=tf_real_seq_len,
    #                                         initial_state=init_state,
    #                                         dtype=tf.float32)
    # print("line 122:outputs:", outputs)
    # # line122: outputs: Tensor("LstmNet/rnn/transpose_1:0", shape=(?, ?, 128), dtype=float32)
    # outputs_reshape = tf.reshape(outputs, [-1, hidden_dim])  # [bs*(max_len-1), hidden_dim]#shape=(?, 128)
    # outputs_reshape = tf.concat([outputs_reshape, tf.reshape(next_inputs_embed, [-1, inputs_embed_dim])], axis=1)
    #
    # rnn_w = tf.get_variable('softmax_w', [hidden_dim + inputs_embed_dim, 1],
    #                         initializer=tf.truncated_normal_initializer(stddev=0.1))
    # rnn_b = tf.get_variable('softmax_b', [1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    # logits = tf.nn.xw_plus_b(outputs_reshape, rnn_w, rnn_b)  # [bs*(max_len-1), 1]

    # dropout
    if train_flag:  # 只在训练的时候进行丢弃
        fw_cells = []
        bw_cells = []
        for i in range(2):
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim, name='fw_LSTM%d' % i)
            dropcell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=(1 - 0.2))
            fw_cells.append(dropcell_fw)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim, name='bw_LSTM%d' % i)
            dropcell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=(1 - 0.2))
            bw_cells.append(dropcell_bw)

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell(fw_cells)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell(bw_cells)
    else:
        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(hidden_dim) for _ in range(2)])
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(hidden_dim) for _ in range(2)])

        # lstm_cell_fw=tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw,output_keep_prob=(1-self.para.drop_rate))
        # lstm_cell_bw=tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw,output_keep_prob=(1-self.para.drop_rate))
    # 下面是多层的Rnn
    # lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.para.rnn_layers_nums)
    # lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.para.rnn_layers_nums)

    # 计算一下输入的句子的长度
    length_sens = tf.reduce_sum(tf.sign(rnn_inputs), axis=1,
                                     name='calcu_len')  # 需要将计算出的句子的长度传入给下面的函数，其实我觉得这里的长度计算还可以用来做mask-attention，正好一举两得
    length_sens = tf.cast(length_sens, dtype=tf.int32, name='cast1')
    outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw,
                                                     inputs=rnn_inputs,
                                                     sequence_length=tf_real_seq_len,
                                                     dtype=tf.float32)

    # 上面这行代码的outputs的是一个有两个元素的元组，一个前向的输出，一个后向的输出，大小均为[batch_size,unroll_steps,vector_dim]
    # 再将两个输出在最后一个维度进行拼接，双向循环神经网络的输出拼接方式有好几种呢。
    outputs = tf.concat(outputs, 2, name='concat1')  # 拼接完之后的最后一个维度会变为原来的二倍，当然你也可以将输出在第二个维度进行相加

    # 下面开始进行注意力机制
    att_w = tf.Variable(tf.random.truncated_normal(shape=[hidden_dim*2, hidden_dim*2 ]),
                        trainable=True, name='attenion_size')
    print("line178,outputs",outputs)
    #line178,outputs Tensor("LstmNet/rnn/transpose_1:0", shape=(?, ?, 128), dtype=float32)
    Q = tf.matmul(outputs, att_w, name='q')
    K = tf.matmul(outputs, att_w, name='k')
    V = tf.matmul(outputs, att_w, name='v')

    qk = tf.matmul(Q, tf.transpose(K, [0, 2, 1], name='t1'), name='qk') / tf.sqrt(
        tf.constant(hidden_dim*2, dtype=tf.float32, name='scaled_factor'),
        name='sqrt1')  # 现在qk的大小是[batch_size,max_len,max_len]
    # 下面开始计算mask矩阵
    mask = tf.sign(tf_pro, name='s1')  # 大小是[batch_size,max_len]#这里看来应该是inputs_y
    mask = tf.expand_dims(mask, 1, name='expand1')  # 大小是[batch_size,1,max_len]
    mask = tf.tile(mask, [1, 1, 1], name='tile1')  # 大小是[batch_size,max_len,max_len]
    padding_mask = -2 ** 22 + 1

    # 下面开始mask，其实也就是将计算出的权值在padding的单词部分设置为一个非常小的数padding_mask=-2**32+1
    # 这样再经过softmax的时候，会将的padding的单词的权重变成一个十分接近0的数
    weights = tf.nn.softmax(
        tf.where(tf.equal(mask, 1), qk, tf.cast(tf.ones_like(mask) * padding_mask, dtype=tf.float32)),
        name='softmax')  # [batch_size,maxlen,maxlen]
    # 计算好权值之后，接下来就是计算Z
    Z = tf.matmul(weights, V, name='weighted_V')
    logits=Z

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
    train_steps = int(math.ceil(train_skill.shape[0] / float(bs)))
    test_steps = int(math.ceil(test_skill.shape[0] / float(bs)))
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
                batch_y = train_y[j * bs:(j + 1) * bs, :]
                batch_pro = train_problem[j * bs:(j + 1) * bs, :]
                # batch_skill = train_skill[j*bs:(j+1)*bs, :]
                batch_real_len = train_real_len[j * bs:(j + 1) * bs] - 1
                # print("line164,batch_real_len=",batch_real_len)
                feed_dict = {tf_pro: batch_pro,
                             tf_y: batch_y, tf_real_seq_len: batch_real_len}
                # print("line229:feed_dict:", feed_dict)
                _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)
                train_loss += batch_loss
            train_loss /= batch_loss

            test_preds, test_targets, test_loss = [], [], 0
            for j in range(test_steps):
                test_y_ = test_y[j * bs:(j + 1) * bs, :]
                # test_skill_ = test_skill[j*bs:(j+1)*bs, :]
                test_pro_ = test_problem[j * bs:(j + 1) * bs, :]
                test_real_len_ = test_real_len[j * bs:(j + 1) * bs] - 1
                feed_dict = {tf_y: test_y_,
                             tf_pro: test_pro_, tf_real_seq_len: test_real_len_}
                filtered_targets_, filtered_preds_ = sess.run([filtered_targets, filtered_predict_prob],
                                                              feed_dict=feed_dict)
                test_preds.append(filtered_preds_)
                test_targets.append(filtered_targets_)
                # print("line182,filtered_targets_=", filtered_targets_)
                # print("line183,filtered_preds_=", filtered_preds_)

            test_preds = np.concatenate(test_preds, axis=0)
            test_targets = np.concatenate(test_targets, axis=0)

            test_auc = metrics.roc_auc_score(test_targets, test_preds)
            test_preds[test_preds > 0.5] = 1.
            test_preds[test_preds <= 0.5] = 0.
            test_acc = metrics.accuracy_score(test_targets, test_preds)

            records = 'Epoch %d/%d, train loss:%3.5f, test auc:%f, test acc:%3.5f' % \
                      (i + 1, epochs, train_loss, test_auc, test_acc)
            print(records)

            if best_auc < test_auc:
                best_auc = test_auc
                saver.save(sess, os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.ckpt'))
else:
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.ckpt'))
        if not os.path.exists(os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.ckpt')):
            print("lzj lin200:没有这个文件夹，'bekt_dkt_model/dkt_pretrain.ckpt'")
            # os.mkdir(saved_model_folder)

        # pro_embed, skill_embed
        pro_embed_trained = sess.run(pro_embedding_matrix)
        pro_embed_trained = pro_embed_trained[1:]

        np.savez(os.path.join(data_folder, 'bekt_dkt_model/pro_embed_bekt_dkt.npz'), pro_final_repre=pro_embed_trained)
