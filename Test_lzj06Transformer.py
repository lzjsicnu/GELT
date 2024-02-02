#https://blog.csdn.net/bqw18744018044/article/details/89501595

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from sklearn import metrics
from scipy import sparse
from tqdm import tqdm
import time, math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import math
import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import gc


max_features = 10000 # vocabulary的大小
maxlen = 500
embedding_size = 128
batch_size = 256 # 每个batch中样本的数量
num_heads = 4
num_units = 128 # query,key,value的维度
ffn_dim = 2048
num_epochs = 30
max_learning_rate = 0.001
min_learning_rate = 0.0005
decay_coefficient = 2.5 # learning_rate的衰减系数
dropout_keep_prob = 0.5 # dropout的比例
evaluate_every = 100 # 每100step进行一次eval

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





# train = pd.read_csv("../input/labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)
# test = pd.read_csv("../input/testData.tsv",header=0,delimiter="\t", quoting=3)

# 建立tokenizer
# tokenizer = Tokenizer(num_words=max_features,lower=True)
# tokenizer.fit_on_texts(list(train['review']) + list(test['review']))
# #word_index = tokenizer.word_index
# x_train = tokenizer.texts_to_sequences(list(train['review']))
# x_train = pad_sequences(x_train,maxlen=maxlen) # padding
# y_train = to_categorical(list(train['sentiment'])) # one-hot
# x_test = tokenizer.texts_to_sequences(list(test['review']))
# x_test = pad_sequences(x_test,maxlen=maxlen) # padding
# 划分训练和验证集
# x_train,x_dev,y_train,y_dev = train_test_split(x_train,y_train,test_size=0.3,random_state=0)
x_train=train_problem
x_dev=test_problem
y_train=train_y
y_dev=test_y
#del train,test
gc.collect()


class Transformer(object):
    def embedding(self, input_x):
        W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                        name='W',
                        trainable=True)
        input_embedding = tf.nn.embedding_lookup(W, input_x)
        return input_embedding

    def positional_encoding(self, embedded_words):
        # [[0,1,2,...,499],
        # [0,1,2,...,499],
        # ...
        # [0,1,2,...,499]]
        positional_ind = tf.tile(tf.expand_dims(tf.range(self.sequence_length), 0),
                                 [batch_size, 1])  # [batch_size, sequence_length]
        # [sequence_length,embedding_size]
        position_enc = np.array(
            [[pos / np.power(10000, 2. * i / self.embedding_size) for i in range(self.embedding_size)]
             for pos in range(self.sequence_length)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        # # [batch_size,sequence_length,embedding_size]
        positional_output = tf.nn.embedding_lookup(lookup_table, positional_ind)
        positional_output += embedded_words
        return positional_output

    def padding_mask(self, inputs):
        pad_mask = tf.equal(inputs, 0)
        # [batch_size,sequence_length,sequence_length]
        pad_mask = tf.tile(tf.expand_dims(pad_mask, axis=1), [1, self.sequence_length, 1])
        return pad_mask

    def layer_normalize(self, inputs, epsilon=1e-8):
        # [batch_size,sequence_length,num_units]
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]  # num_units
        # 沿轴-1求均值和方差(也就是沿轴num_units)
        # mean/variance.shape = [batch_size,sequence_length]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)  # LN
        # mean, variance = tf.nn.moments(inputs,[-2,-1],keep_dims=True) # BN
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        # [batch_size,sequence_length,num_units]
        outputs = gamma * normalized + beta
        return outputs

    def multihead_attention(self, attention_inputs):
        # [batch_size,sequence_length, num_units]
        Q = tf.keras.layers.Dense(self.num_units)(attention_inputs)
        K = tf.keras.layers.Dense(self.num_units)(attention_inputs)
        V = tf.keras.layers.Dense(self.num_units)(attention_inputs)

        # 将Q/K/V分成多头
        # Q_/K_/V_.shape = [batch_size*num_heads,sequence_length,num_units/num_heads]
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # 计算Q与K的相似度
        # tf.transpose(K_,[0,2,1])是对矩阵K_转置
        # similarity.shape = [batch_size*num_heads,sequence_length,sequence_length]
        similarity = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        similarity = similarity / (K_.get_shape().as_list()[-1] ** 0.5)

        pad_mask = self.padding_mask(self.input_x)
        pad_mask = tf.tile(pad_mask, [self.num_heads, 1, 1])
        paddings = tf.ones_like(similarity) * (-2 ** 32 + 1)
        similarity = tf.where(tf.equal(pad_mask, False), paddings, similarity)
        similarity = tf.nn.softmax(similarity)
        similarity = tf.nn.dropout(similarity, self.dropout_keep_prob)
        # [batch_size*num_heads,sequence_length,sequence_length]
        outputs = tf.matmul(similarity, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        return outputs

    def feedforward(self, inputs):
        params = {"inputs": inputs, "filters": ffn_dim, "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        # 相当于 [batch_size*sequence_length,num_units]*[num_units,ffn_dim]，在reshape成[batch_size,sequence_length,num_units]
        # [batch_size,sequence_length,ffn_dim]
        outputs = tf.layers.conv1d(**params)
        params = {"inputs": outputs, "filters": num_units, "kernel_size": 1, "activation": None, "use_bias": True}
        # [batch_size,sequence_length,num_units]
        outputs = tf.layers.conv1d(**params)
        return outputs

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 num_units,
                 num_heads):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_units = num_units
        self.num_heads = num_heads

        # 定义需要用户输入的placeholder
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')  # 定义为placeholder是为了实现lr递减

        input_embedding = self.embedding(self.input_x)
        # [batch_size, sequence_length, num_units]
        positional_output = self.positional_encoding(input_embedding)
        # Dropout
        positional_output = tf.nn.dropout(positional_output, self.dropout_keep_prob)
        attention_output = self.multihead_attention(positional_output)
        # Residual connection
        attention_output += positional_output
        # [batch_size, sequence_length, num_units]
        outputs = self.layer_normalize(attention_output)  # LN
        # feedforward
        feedforward_outputs = self.feedforward(outputs)
        # Residual connection
        feedforward_outputs += outputs
        # LN
        feedforward_outputs = self.layer_normalize(feedforward_outputs)
        outputs = tf.reduce_mean(outputs, axis=1)

        self.scores = tf.keras.layers.Dense(self.num_classes)(outputs)
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope('loss'):
            # 交叉熵loss
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            # L2正则化后的loss
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



# 用于产生batch
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data_size = len(data)
    num_batches_per_epoch = data_size// batch_size # 每个epoch中包含的batch数量
    for epoch in range(num_epochs):
        # 每个epoch是否进行shuflle
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,  # 如果指定的设备不存在，允许tf自动分配设备
        log_device_placement=False)  # 不打印设备分配日志
    sess = tf.Session(config=session_conf)  # 使用session_conf对session进行配置
    # 构建模型
    nn = Transformer(sequence_length=x_train.shape[1],
                     num_classes=y_train.shape[1],
                     vocab_size=max_features,
                     embedding_size=embedding_size,
                     num_units=num_units,
                     num_heads=num_heads)
    # 用于统计全局的step
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(nn.learning_rate)
    train_op = optimizer.minimize(nn.loss, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    batches = batch_iter(np.hstack((x_train, y_train)), batch_size, num_epochs)
    decay_speed = decay_coefficient * len(y_train) / batch_size
    counter = 0  # 用于记录当前的batch数
    for batch in batches:
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter / decay_speed)
        counter += 1
        x_batch, y_batch = batch[:, :-2], batch[:, -2:]
        # 训练
        feed_dict = {nn.input_x: x_batch,
                     nn.input_y: y_batch,
                     nn.dropout_keep_prob: dropout_keep_prob,
                     nn.learning_rate: learning_rate}
        _, step, loss, accuracy = sess.run(
            [train_op, global_step, nn.loss, nn.accuracy],
            feed_dict)
        current_step = tf.train.global_step(sess, global_step)
        # Evaluate
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            loss_sum = 0
            accuracy_sum = 0
            step = None
            batches_in_dev = len(y_dev) // batch_size
            for batch in range(batches_in_dev):
                start_index = batch * batch_size
                end_index = (batch + 1) * batch_size
                feed_dict = {
                    nn.input_x: x_dev[start_index:end_index],
                    nn.input_y: y_dev[start_index:end_index],
                    nn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, nn.loss, nn.accuracy], feed_dict)
                loss_sum += loss
                accuracy_sum += accuracy
            loss = loss_sum / batches_in_dev
            accuracy = accuracy_sum / batches_in_dev
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print("")

    # predict test set
    all_predictions = []
    test_batches = batch_iter(x_test, batch_size, num_epochs=1, shuffle=False)
    for batch in test_batches:
        feed_dict = {
            nn.input_x: batch,
            nn.dropout_keep_prob: 1.0
        }
        predictions = sess.run([nn.predictions], feed_dict)[0]
        all_predictions.extend(list(predictions))
