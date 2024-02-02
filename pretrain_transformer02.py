import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from sklearn import metrics
from scipy import sparse
from tqdm import tqdm
import time, math

import os
import time
import argparse
import tensorflow as tf
#from sampler import WarpSampler
from model import Model
import numpy as np
import sys
import copy
import random
import csv
from math import sqrt
from sklearn import metrics
from sklearn.metrics import mean_squared_error


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

# embed data, used for initialize
embed_data = np.load(os.path.join(data_folder, 'embedding_200.npz'))
_, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
print('lzj line38,pre_pro_embed.shape, pre_pro_embed.dtype', (pre_pro_embed.shape, pre_pro_embed.dtype))
# 前两个都是(167, 64)，
# 第三个pre_pro_embed.shape (15911, 128),这里只用得到第三个

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
                                           initializer=pro_embed_init, trainable=train_embed)
    # skill_embedding_matrix = tf.get_variable('skill_embed_matrix', [skill_num, embed_dim],
    #                             initializer=skill_embed_init, trainable=train_flag)
    # deal with complement symbol
    zero_tensor = tf.zeros([1, embed_dim], dtype=tf.float32)  # shape=(1, 128)
    pro_embedding_matrix = tf.concat([zero_tensor, pro_embedding_matrix],
                                     axis=0)  # shape=(15912, 128),注意，这里的拼接，只能用一次，否则15912+1
    # skill_embedding_matrix = tf.concat([zero_tensor, skill_embedding_matrix], axis=0)

# skill, problem embedding
# all_skill_embed = tf.nn.embedding_lookup(skill_embedding_matrix, tf_skill)  # [bs, max_len, embed_dim]
all_pro_embed = tf.nn.embedding_lookup(pro_embedding_matrix, tf_pro)  # [bs, max_len, embed_dim]
# skill_embed = all_skill_embed[:, :-1, :]
pro_embed = all_pro_embed[:, :-1, :]  # shape=(?, ?, 128)
# next_skill_embed = all_skill_embed[:, 1:, :]
next_pro_embed = all_pro_embed[:, 1:, :]  # [bs, max_len-1, embed_dim]#shape=(?, ?, 128)

inputs_embed = pro_embed  # [bs, max_len-1, embed_dim]#shape=(?, ?, 128)
next_inputs_embed = next_pro_embed  # shape=(?, ?, 128)
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
inputs_y_embedding = tf.map_fn(concat_zero, inputs_y)  # shape=(?, 256)
rnn_inputs = tf.reshape(inputs_y_embedding, [-1, tf_max_len - 1, inputs_embed_dim + embed_dim])
# shape=(?, ?, 256)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# lstm
# with tf.variable_scope('LstmNet', reuse=tf.AUTO_REUSE):
#     lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
#     init_state = lstm_cell.zero_state(tf_batch_size, tf.float32)
#     outputs, last_state = tf.nn.dynamic_rnn(cell=lstm_cell,
#                                             inputs=rnn_inputs,
#                                             sequence_length=tf_real_seq_len,
#                                             initial_state=init_state,
#                                             dtype=tf.float32)
#
#     outputs_reshape = tf.reshape(outputs, [-1, hidden_dim])  # [bs*(max_len-1), hidden_dim]#shape=(?, 128)
#     outputs_reshape = tf.concat([outputs_reshape, tf.reshape(next_inputs_embed, [-1, inputs_embed_dim])], axis=1)
#
#     rnn_w = tf.get_variable('softmax_w', [hidden_dim + inputs_embed_dim, 1],
#                             initializer=tf.truncated_normal_initializer(stddev=0.1))
#     rnn_b = tf.get_variable('softmax_b', [1], initializer=tf.truncated_normal_initializer(stddev=0.1))
#     logits = tf.nn.xw_plus_b(outputs_reshape, rnn_w, rnn_b)  # [bs*(max_len-1), 1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def normalize(inputs,epsilon = 1e-8,scope="ln",reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs
def embedding(inputs,vocab_size,num_units,zero_pad=True,scale=True,l2_reg=0.0,scope="embedding",with_t=False,reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',dtype=tf.float32,shape=[vocab_size, num_units],
                                       #initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),lookup_table[1:, :]), 0)
            print("lzj168,lookup_table:",lookup_table)
            print("lzj169,inputs:", inputs)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t: return outputs,lookup_table
    else: return outputs
def multihead_attention(queries,keys,num_units=None,num_heads=8,dropout_rate=0,is_training=True,causality=False,scope="multihead_attention",reuse=None,
                        sizeof_V=100,
                        with_qk=False):
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        #QK=outputs
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        QK=outputs
        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1)) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)

    return outputs,QK

def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs

input_size = skill_num*2
problems= tf.placeholder(tf.int32, shape=(128, 50-1))
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default="assist2009", type=str)
parser.add_argument('--train_data_path', default="", type=str)
parser.add_argument('--test_data_path', default="", type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--hidden_units', default=200, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=21, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--num_skills', default=50, type=int)
parser.add_argument('--num_steps', default=50, type=int)
parser.add_argument('--pos', default=False, type=bool)
args = parser.parse_args()




x=rnn_inputs
key_masks = tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
with tf.variable_scope("encoder"):
    ## Embedding
    # key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(x), axis=-1)), -1)
    enc, lookup = embedding(x,vocab_size=input_size,num_units=args.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      l2_reg=args.l2_emb,
                                      scope="enc_embed",
                                      with_t=True,
                                      reuse=True)

    # tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1])
    ## Positional Encoding
    # if args.pos:

    enc += embedding(
        tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
        vocab_size=args.num_steps - 1,
        num_units=args.hidden_units,
        zero_pad=False,
        scale=False,
        scope="enc_pe",
        l2_reg=args.l2_emb,
        reuse=None

    )

    seq = embedding(problems,
                         vocab_size=args.num_skills,
                         num_units=args.hidden_units,
                         zero_pad=True,
                         scale=False,
                         l2_reg=args.l2_emb,
                         scope="que_embed",
                         reuse=None)

    # Dropout
    enc *= key_masks
    seq *= key_masks
    enc = tf.layers.dropout(enc,
                                 rate=args.dropout_rate,
                                 training=tf.convert_to_tensor(True))

    ## Blocks
    for i in range(args.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
            enc, QK = multihead_attention(queries=normalize(seq),
                                                    keys=enc,
                                                    num_units=args.hidden_units,
                                                    num_heads=args.num_heads,
                                                    dropout_rate=args.dropout_rate,
                                                    is_training=True,
                                                    sizeof_V=args.hidden_units,
                                                    with_qk=True,
                                                    causality=True)

            ### Feed Forward
            # weights = tf.get_default_graph().get_tensor_by_name(os.path.split(V.name)[0] + '/kernel:0')
            #         #print(weights.shape)
            #         #self.enc = feedforward(self.enc, num_units=[4*args.hidden_units, num_skills])
            enc = feedforward(normalize(enc), num_units=[args.hidden_units, args.hidden_units],
                                   dropout_rate=args.dropout_rate, is_training=True)
            enc *= key_masks
            seq *= key_masks
            enc = normalize(enc)













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
                feed_dict = {tf_pro: batch_pro,
                             tf_y: batch_y, tf_real_seq_len: batch_real_len}
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
