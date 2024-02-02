import argparse


class Hpara():
    parser = argparse.ArgumentParser()  # 构建一个参数管理对象

    parser.add_argument('--datapath', default='./data/data.csv', type=str)
    parser.add_argument('--testdatapath', default='./data/test.csv', type=str)

    parser.add_argument('--label2idpath', default='./data/label2id.json', type=str)
    parser.add_argument('--word2idpath', default='./data/word2id.json', type=str)
    parser.add_argument('--id2labelpath', default='./data/id2label.json', type=str)
    parser.add_argument('--id2wordpath', default='./data/id2word.json', type=str)

    parser.add_argument('--testlabel2idpath', default='./data/test_label2id.json', type=str)
    parser.add_argument('--testword2idpath', default='./data/test_word2id.json', type=str)
    parser.add_argument('--testid2labelpath', default='./data/test_id2label.json', type=str)
    parser.add_argument('--testid2wordpath', default='./data/test_id2word.json', type=str)

    parser.add_argument('--max_sen_len', default=30, type=int)
    parser.add_argument('--word2vector_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--token_nums', default=18414, type=int)
    parser.add_argument('--label_nums', default=8, type=int)
    parser.add_argument('--rnn_layers_nums', default=2, type=int)
    parser.add_argument('--is_training', default=True, type=bool)
    parser.add_argument('--drop_rate', default=0.2, type=float)
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--cell_type', default='LSTM', type=str)
    parser.add_argument('--savepath', default='./check_point', type=str)
    parser.add_argument('--batch_size', default=8, type=int)


# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:27:51 2020

@author: DELL
"""
import os
import tensorflow as tf
import numpy as np
from model_para import Hpara
from data_utils import Create_dataset_and_vocab
from tqdm import tqdm

tf.reset_default_graph()  # 每次运行重置图


class Mymodel():
    '''
    这里我只是使用双向LSTM+mask-self-attention+crf,别的情况就先不考虑，都是类似的
    '''

    def __init__(self, para):
        self.para = para

        self.optimizer = tf.train.AdamOptimizer(self.para.learning_rate, name='adam')
        self.initializer = tf.contrib.layers.xavier_initializer()  # 设置一个初始化器，这个初始化可以使得梯度大致相等的
        self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)
        self.embedding = tf.get_variable("emb", [self.para.token_nums, self.para.word2vector_dim], trainable=True,
                                         initializer=self.initializer)  # 如果变量存在，就直接加载过来，如果不存在，自动创建
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.build_model()

    def build_model(self):
        '''
        下面开始构建我们的模型，最重要的就是如何搭建一个神经网络图，等图搭建完了之后再输入数据进行训练
        '''
        # 首先定义两个输入的占位符
        self.inputs = tf.placeholder(tf.int32, [None, self.para.max_sen_len], name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, self.para.max_sen_len], name='labels')

        # 那么接下来就是嵌入层了，将单词token转化为嵌入向量
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs, name='embedding')
        # 定义前向网络和后向网络

        # dropout
        if self.para.is_training:  # 只在训练的时候进行丢弃
            fw_cells = []
            bw_cells = []
            for i in range(self.para.rnn_layers_nums):
                cell_fw = tf.nn.rnn_cell.LSTMCell(self.para.hidden_dim, name='fw_LSTM%d' % i)
                dropcell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=(1 - self.para.drop_rate))
                fw_cells.append(dropcell_fw)
                cell_bw = tf.nn.rnn_cell.LSTMCell(self.para.hidden_dim, name='bw_LSTM%d' % i)
                dropcell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=(1 - self.para.drop_rate))
                bw_cells.append(dropcell_bw)

            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell(fw_cells)
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell(bw_cells)
        else:
            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(self.para.hidden_dim) for _ in range(self.para.rnn_layers_nums)])
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(self.para.hidden_dim) for _ in range(self.para.rnn_layers_nums)])

            # lstm_cell_fw=tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw,output_keep_prob=(1-self.para.drop_rate))
            # lstm_cell_bw=tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw,output_keep_prob=(1-self.para.drop_rate))
        # 下面是多层的Rnn
        # lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.para.rnn_layers_nums)
        # lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.para.rnn_layers_nums)

        # 计算一下输入的句子的长度
        self.length_sens = tf.reduce_sum(tf.sign(self.inputs), axis=1,
                                         name='calcu_len')  # 需要将计算出的句子的长度传入给下面的函数，其实我觉得这里的长度计算还可以用来做mask-attention，正好一举两得
        self.length_sens = tf.cast(self.length_sens, dtype=tf.int32, name='cast1')
        outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs=self.inputs_emb,
                                                         sequence_length=self.length_sens, dtype=tf.float32)
        # 上面这行代码的outputs的是一个有两个元素的元组，一个前向的输出，一个后向的输出，大小均为[batch_size,unroll_steps,vector_dim]
        # 再将两个输出在最后一个维度进行拼接，双向循环神经网络的输出拼接方式有好几种呢。
        outputs = tf.concat(outputs, 2, name='concat1')  # 拼接完之后的最后一个维度会变为原来的二倍，当然你也可以将输出在第二个维度进行相加

        # 下面开始进行注意力机制
        att_Q = tf.Variable(tf.random.truncated_normal(shape=[self.para.hidden_dim * 2, self.para.hidden_dim * 2]),
                            trainable=True, name='attenion_size_Q')
        att_K = tf.Variable(tf.random.truncated_normal(shape=[self.para.hidden_dim * 2, self.para.hidden_dim * 2]),
                            trainable=True, name='attenion_size_K')
        att_V = tf.Variable(tf.random.truncated_normal(shape=[self.para.hidden_dim * 2, self.para.hidden_dim * 2]),
                            trainable=True, name='attenion_size_V')
        Q = tf.matmul(outputs, att_Q, name='q')
        K = tf.matmul(outputs, att_K, name='k')
        V = tf.matmul(outputs, att_V, name='v')

        qk = tf.matmul(Q, tf.transpose(K, [0, 2, 1], name='t1'), name='qk') / tf.sqrt(
            tf.constant(self.para.hidden_dim * 2, dtype=tf.float32, name='scaled_factor'),
            name='sqrt1')  # 现在qk的大小是[batch_size,max_len,max_len]
        # 下面开始计算mask矩阵
        mask = tf.sign(self.inputs, name='s1')  # 大小是[batch_size,max_len]
        mask = tf.expand_dims(mask, 1, name='expand1')  # 大小是[batch_size,1,max_len]
        mask = tf.tile(mask, [1, self.para.max_sen_len, 1], name='tile1')  # 大小是[batch_size,max_len,max_len]
        padding_mask = -2 ** 22 + 1

        # 下面开始mask，其实也就是将计算出的权值在padding的单词部分设置为一个非常小的数padding_mask=-2**32+1
        # 这样再经过softmax的时候，会将的padding的单词的权重变成一个十分接近0的数
        weights = tf.nn.softmax(
            tf.where(tf.equal(mask, 1), qk, tf.cast(tf.ones_like(mask) * padding_mask, dtype=tf.float32)),
            name='softmax')  # [batch_size,maxlen,maxlen]
        # 计算好权值之后，接下来就是计算Z
        Z = tf.matmul(weights, V, name='weighted_V')

        # 下面开始条件随机场
        crf_w = tf.Variable(tf.random.truncated_normal([self.para.hidden_dim * 2, self.para.label_nums]), name='crf_w')
        # 将Z调整为[batch_size,max_len,label_nums],也就是每句话里面每个单词的标签是什么，接下来将该张量输入crf
        self.Z = tf.matmul(Z, crf_w, name='crf_inputs')
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.Z, self.targets, self.length_sens)
        self.batch_pred_sequence, self.batch_viterbi_score = tf.contrib.crf.crf_decode(self.Z,
                                                                                       self.transition_params,
                                                                                       self.length_sens)
        self.loss = tf.reduce_mean(-self.log_likelihood, name='loss')
        self.optimizer_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        # 接下来使用上面的crf_decode的解码输出计算准确率，无论我们需要什么结果，都可以在这里定义节点
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(self.batch_pred_sequence, self.targets), dtype=tf.float32)) / tf.constant(
            self.para.batch_size * self.para.max_sen_len, dtype=tf.float32)
        # 上面这个准确率我也不知道该怎么定义，啊哈哈哈，就用一下所有预测正确的除去每个batch总的单词数吧，啊哈哈哈，我太菜了

    def batch_iter(self):
        train_data, train_label = Create_dataset_and_vocab(self.para)
        data_len = len(train_data)
        num_batch = (data_len + self.para.batch_size - 1) // self.para.batch_size  # 获取的是
        indices = np.random.permutation(np.arange(data_len))  # 随机打乱下标
        x_shuff = train_data[indices]
        y_shuff = train_label[indices]  # 打乱数据

        for i in range(num_batch):  # 按照batchsize取数据
            start_offset = i * self.para.batch_size  # 开始下标
            end_offset = min(start_offset + self.para.batch_size, data_len)  # 一个batch的结束下标
            yield i, num_batch, x_shuff[start_offset:end_offset], y_shuff[
                                                                  start_offset:end_offset]  # yield是产生第i个batch，输出总的batch数，以及每个batch的训练数据和标签

    def train(self):
        loss = []

        if not os.path.exists(self.para.savepath):  # 判断模型文件是否存在
            print('Create model file')
            os.makedirs(para.savepath)
        else:
            self.saver.restore(self.sess, os.path.join(self.para.savepath, 'model.ckpt'))

        self.sess.run(tf.global_variables_initializer())
        for k in tqdm(range(self.para.epochs)):
            batch_train = self.batch_iter()
            for i, total_num, data_step, label_step in batch_train:

                _, ls, pred = self.sess.run([
                    self.optimizer_op,
                    self.loss,
                    self.batch_pred_sequence,
                ],
                    feed_dict={
                        self.inputs: data_step,
                        self.targets: label_step
                    })
                loss.append(ls)
                if i % 100 == 0:
                    print('loss is :', ls)
        self.saver.save(self.sess, os.path.join(self.para.savepath, 'model.ckpt'))

        return loss

    def test(self):

        # 首先加载已经训练好的模型文件
        tf.initialize_all_variables().run(session=self.sess)
        saver = tf.train.import_meta_graph('./check_point/model.ckpt.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(self.para.savepath))
        # 然后是加载数据集
        batch_test = self.batch_iter()
        for i, total_num, data_step, label_step in batch_test:
            batch_pred_label = self.sess.run([self.batch_pred_sequence],
                                             feed_dict={self.inputs: data_step, self.targets: label_step})
            # 还可以加一些计算准确率的运算，我就不弄了，很简单，和train的一样


if __name__ == "__main__":
    hp = Hpara()
    parser = hp.parser
    para = parser.parse_args()
    model = Mymodel(para)
    # loss=model.train()
    model.test()

