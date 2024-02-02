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
epochs = 30
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
# mainly modified from
# https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
import math
from functools import partial
# mainly modified from
# https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
import math
from functools import partial

import torch
from einops import rearrange, repeat
from scipy.stats import ortho_group
from torch import nn

from hashing.ksh import KernelSH
from params import args

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py


def softmax_kernel_share_qk(
    data, *, projection_matrix, normalize_data=True, eps=1e-4, device=None
):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    ratio = projection_matrix.shape[0] ** -0.5

    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)


def softmax_kernel(
    data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None
):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    ratio = projection_matrix.shape[0] ** -0.5

    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=-1, keepdim=True).values
            )
            + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps
        )

    return data_dash.type_as(data)


def generalized_kernel(
    data,
    *,
    projection_matrix,
    kernel_fn=nn.ReLU(),
    kernel_epsilon=0.001,
    normalize_data=True,
    device=None,
):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = torch.FloatTensor(ortho_group.rvs(nb_columns), device="cpu").to(device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = torch.FloatTensor(ortho_group.rvs(nb_columns), device="cpu").to(device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones(
            (nb_rows,), device=device
        )
    else:
        raise ValueError(f"Invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1.0 / torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q))
    context = torch.einsum("...nd,...ne->...de", k, v)
    out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)
    return out


# non-causal linear attention
def linear_attention_bias(q, k, v):
    B, H, N, D = q.shape
    bias = args.nbits + 1
    top_bias = v * bias
    bottom_bias = N * bias
    with torch.cuda.amp.autocast(enabled=False):
        k_cumsum = k.sum(dim=-2).float()
        D_inv = 1.0 / (
            torch.einsum("...nd,...d->...n", q.float(), k_cumsum) + bottom_bias
        )
    context = torch.einsum("...nd,...ne->...de", k, v)
    with torch.cuda.amp.autocast(enabled=False):
        out = (
            torch.einsum("...de,...nd->...ne", context.float(), q.float())
            + top_bias.float()
        )
        out2 = torch.einsum("...ne,...n->...ne", out, D_inv)
    return out2


def linear_attention_bias_quant(q, k, v):
    B, H, N, D = q.shape
    bias = D + 1
    top_bias = v * bias
    bottom_bias = N * bias
    with torch.cuda.amp.autocast(enabled=False):
        k_cumsum = k.sum(dim=-2).float()
        D_inv = 1.0 / (
            torch.einsum("...nd,...d->...n", q.float(), k_cumsum) + bottom_bias
        )
        context = torch.einsum("...nd,...ne->...de", k, v)
        out = torch.einsum("...de,...nd->...ne", context.float(), q.float()) + top_bias
        out2 = torch.einsum("...ne,...n->...ne", out, D_inv)
    return out2


class FastAttention(nn.Module):
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False,
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling,
        )
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        self.__flops__ = 0

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(
                generalized_kernel,
                kernel_fn=self.kernel_fn,
                projection_matrix=self.projection_matrix,
                device=device,
            )
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(
                softmax_kernel, projection_matrix=self.projection_matrix, device=device
            )
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        out = linear_attention(q, k, v)
        return out

    @staticmethod
    def compute_macs(module, input, output):
        input = input[0]
        _, H, N, C = input.shape
        Nf = module.nb_features
        assert C == module.dim_heads
        macs = 0
        n_params = 0

        if module.no_projection:
            raise ValueError("Not supported yet!")
        elif module.generalized_attention:
            raise ValueError("Not supported yet!")
        else:
            n_params += C * Nf
            # q = create_kernel(q, is_query=True)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf
            # k = create_kernel(k, is_query=False)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf

        # out = linear_attention(q, k, v)
        # k_cumsum = k.sum(dim=-2)
        macs += H * N * Nf
        # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        macs += H * N * Nf
        # context = torch.einsum('...nd,...ne->...de', k, v)
        macs += H * N * Nf * C
        # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        macs += 2 * H * N * Nf * C
        # print('macs fast att', macs / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class LiteformerFastAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False,
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling,
        )
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

        self.is_trained = False
        self.ksh = KernelSH(self.num_heads, self.dim_heads, args.nbits, args.m,)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        self.__flops__ = 0

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, qk, v):
        B, H, N, C = qk.shape
        # qk = qk - qk.mean()
        qk = qk / torch.norm(qk, 2, dim=-1, keepdim=True)

        if not self.is_trained:
            with torch.no_grad():
                attn = qk @ qk.transpose(-2, -1)

                S0 = torch.zeros_like(attn)
                # select topk largest
                _, indices = torch.topk(attn, k=args.topk, dim=3)
                S0.scatter_(3, indices, 1.0)

                # select topk smallest
                _, indices = torch.topk(attn, k=args.topk, dim=3, largest=False)
                S0.scatter_(3, indices, -1.0)

                # release memory
                del attn

                perm = torch.randperm(N, device=qk.device)
                anchor = torch.index_select(qk, 2, perm[: self.ksh.m])[0].unsqueeze(0)
            # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            self.ksh.train_hashing_weight_woeig(
                qk.detach(), anchor.detach(), S0.detach()
            )
            self.is_trained = True
            print("Train ksh performer self-attention")

        qk = self.ksh(qk)
        out = linear_attention_bias(qk, qk, v)
        return out

    @staticmethod
    def compute_macs(module, input, output):
        input = input[0]
        _, H, N, C = input.shape
        Nf = module.nb_features
        assert C == module.dim_heads
        macs = 0
        n_params = 0

        if module.no_projection:
            raise ValueError("Not supported yet!")
        elif module.generalized_attention:
            raise ValueError("Not supported yet!")
        else:
            n_params += C * Nf
            # q = create_kernel(q, is_query=True)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf
            # k = create_kernel(k, is_query=False)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf

        # out = linear_attention(q, k, v)
        # k_cumsum = k.sum(dim=-2)
        macs += H * N * Nf
        # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        macs += H * N * Nf
        # context = torch.einsum('...nd,...ne->...de', k, v)
        macs += H * N * Nf * C
        # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        macs += 2 * H * N * Nf * C
        # print('macs fast att', macs / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class PerformerSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        nb_features=None,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False,
        sr_ratio=1.0,
        linear=False,
        seq_len=196,
    ):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.dim = dim
        assert dim % num_heads == 0, "dimension must be divisible by number of heads"
        head_dim = dim // num_heads
        nb_features = args.num_features
        self.fast_attention = FastAttention(
            head_dim,
            nb_features,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            no_projection=no_projection,
        )

        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5  # not used in performer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        x = self.fast_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EcoformerSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        nb_features=None,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dimension must be divisible by number of heads"
        head_dim = dim // num_heads
        self.fast_attention = LiteformerFastAttention(
            num_heads,
            head_dim,
            nb_features,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            no_projection=no_projection,
        )

        self.num_heads = num_heads
        self.to_qk = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape

        qk = (
            self.to_qk(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.to_v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x = self.fast_attention(qk, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 实现多头机制
def muti_head_attention(_input, d=8, n_attention_head=2):
    """
    实现单层多头机制
    @param _input: 输入 (?, n_feats, n_dim)
    @param d: Q,K,V映射后的维度
    @param n_attention_head: multi-head attention的个数
    """
    attention_heads = LiteformerFastAttention

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
    print("line 161:outputs:", outputs)#shape=(?, ?, 128),
    outputs = muti_head_attention(outputs, 128, 1)
    print("line 162:outputs:", outputs)#8#shape=(?, ?, 128)
    #outputs = muti_head_attention(outputs, 42, 3)
    # 输出
    print("line 166:outputs:", outputs)#shape=(?, ?, 128)
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

            test_auc = metrics.roc_auc_score(test_targets, test_preds)
            test_preds[test_preds>0.5] = 1.
            test_preds[test_preds<=0.5] = 0.
            test_acc = metrics.accuracy_score(test_targets, test_preds)
            
            records = 'Epoch %d/%d, train loss:%3.5f, test auc:%f, test acc:%3.5f' % \
                            (i+1, epochs, train_loss, test_auc, test_acc)    
            print(records)

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
   