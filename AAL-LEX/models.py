import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from layers import *
import utils

class Model(object):
    def __init__(self, config, embedding_matrix):
        # self.word_cell = config.word_cell
        self.device = config.device
        self.word_output_size = config.word_output_size
        self.classes = config.classes
        self.aspnum = config.aspnum
        self.max_grad_norm = config.max_grad_norm
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.dropout_keep_proba = config.dropout_keep_proba
        self.lr = config.lr
        self.seed = config.seed
        # self.seed = None
        self.attRandomBase = config.attRandomBase
        self.biRandomBase = config.biRandomBase
        self.aspRandomBase = config.aspRandomBase
        self.Winit = tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=self.seed)
        # self.Winit = None
        # self.Winit = tf.truncated_normal_initializer(seed=self.seed)
        self.word_cell = tf.contrib.rnn.LSTMCell
        # self.word_cell = tf.contrib.rnn.GRUCell
        with tf.variable_scope('tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if embedding_matrix is None:
                self.embedding_matrix = tf.placeholder(shape=(None, None), dtype=tf.float32, name='embedding_matrix')
                self.embedding_C = tf.placeholder(shape=(None, None), dtype=tf.float32, name='embedding_C')
            else:
                self.embedding_matrix = tf.Variable(initial_value=embedding_matrix, name='embedding_matrix', dtype=tf.float32, trainable=False)
            self.context_vector = tf.Variable(tf.random_uniform(shape=[self.word_output_size * 2], minval=-1.0 * self.aspRandomBase, maxval=self.aspRandomBase, seed=self.seed),
                                              name='attention_context_vector', dtype=tf.float32, trainable=True)
            self.aspect_embedding = tf.Variable(tf.random_uniform(shape=[self.aspnum, self.embedding_size], minval=-1.0 * self.aspRandomBase, maxval=self.aspRandomBase, seed=self.seed),
                                              name='aspect_embedding', dtype=tf.float32, trainable=True)
            # self.aspect_embedding_c = tf.Variable(tf.random_uniform(shape=[self.aspnum, self.embedding_size], minval=-1.0 * self.aspRandomBase, maxval=self.aspRandomBase, seed=self.seed),
            #                                   name='aspect_embedding_c', dtype=tf.float32, trainable=True)
            # self.aspect_embedding = tf.Variable(initial_value=asp_embedding_matrix, name='asp_embedding_matrix',
            #                                     dtype=tf.float32, trainable=True)
            # self.context_vector = tf.Variable(tf.truncated_normal(shape=[self.word_output_size * 2]),
            #                                   name='attention_context_vector', dtype=tf.float32, trainable=True)
            # self.aspect_embedding = tf.Variable(tf.truncated_normal(shape=[5, self.embedding_size]),
            #                                   name='aspect_embedding', dtype=tf.float32, trainable=True)

            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            # [document x word]
            self.inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')
            self.targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')
            self.textwm = tf.placeholder(shape=(None, None), dtype=tf.float32, name='textwordmask')
            self.targetwm = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targetwordmask')
            self.posmask = tf.placeholder(shape=(None, None), dtype=tf.float32, name='positionmask')
            self.posweight = tf.placeholder(shape=(None, None), dtype=tf.float32, name='positionwei')
            self.text_word_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='text_word_lengths')
            self.target_word_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='target_word_lengths')
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')
            self.category = tf.placeholder(shape=(None,), dtype=tf.int32, name='category')
            self.aspcat = tf.placeholder(shape=(None, None), dtype=tf.int32, name='aspcat')
            self.location = tf.placeholder(shape=(None,None), dtype=tf.float32, name='location')

        with tf.variable_scope('embedding'):
            with tf.variable_scope("word_emb"):
                self.inputs_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)
                # self.inputs_embedding_c = tf.nn.embedding_lookup(self.embedding_matrix_c, self.inputs)
            # with tf.variable_scope("cate_emb"):
            #     self.cate_embedding = tf.nn.embedding_lookup(self.aspect_embedding, tf.expand_dims(self.category, -1))
        (self.batch_size, self.text_word_size) = tf.unstack(tf.shape(self.inputs))
        # (self.batch_size, self.target_word_size) = tf.unstack(tf.shape(self.targets))

    def train(self, logits):
        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            # self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            # regu = tf.contrib.layers.l2_regularizer(0.00001, scope=None)
            # tvars = tf.trainable_variables()
            # self.loss_regu = tf.contrib.layers.apply_regularization(regu, tvars)
            # self.loss_cla = tf.reduce_mean(self.cross_entropy)
            # self.loss = self.loss_cla + self.loss_regu

            self.loss = tf.reduce_mean(self.cross_entropy)
            # dif = tf.cast(self.labels, tf.float32) - self.logits_up
            # self.loss_up = tf.reduce_mean(dif * dif)
            # self.loss = self.loss_t + 0.1 * self.loss_up

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, self.labels, 1), tf.float32))

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.GradientDescentOptimizer(self.lr)
            # opt = tf.train.AdadeltaOptimizer(self.lr, rho=0.9, epsilon=1e-6)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)


class AALlex(Model):
    def __init__(self, config, embedding_matrix, lex,  sess):
        super(AALlex, self).__init__(config, embedding_matrix) #将AAL_LEX的对象转换为Model的对象
        self.word_cell = tf.contrib.rnn.LSTMCell
        self.nhop = config.nhop
        self.hid = []
        self.lex = tf.Variable(initial_value=lex, name='lex', dtype=tf.float32, trainable=False)
        # 这里过滤掉了PMI无关的词，全部附上权重0，保留下里的只是PMI真正相关的词，平均后去计算loss
        self.m = tf.cast(self.posmask, tf.float32)
        with tf.variable_scope("cate_emb"):
            self.cate_embedding = tf.nn.embedding_lookup(self.aspect_embedding, tf.expand_dims(self.category, -1))
            onehot_matrix = np.eye(self.aspnum)
            self.cate_onehot = tf.cast(tf.nn.embedding_lookup(onehot_matrix,tf.expand_dims(self.category, -1)),tf.float32)

        with tf.device(self.device):
            self.build()
            # self.train(self.logits_t)
        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_t) #sparse，label可以用互斥一维表示（仅当label互斥时好用）
            #self.cross_entropy_c = tf.nn.softmax_cross_entropy_with_logits(labels=self.asplabel, logits=self.logits_c)
            #self.cross_entropy_c = tf.reduce_mean(tf.square(self.asplabel-self.logits_c))
            self.cross_entropy_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.asplabel, logits=self.logits_c),axis= -1)
            #self.cross_entropy_c = tf.nn.softmax_cross_entropy_with_logits(labels=self.asplabel, logits=self.logits_c)


            self.loss_senti = tf.reduce_mean(self.cross_entropy)

            sum = tf.reduce_sum(self.m, axis=1, keep_dims=True) + 1e-6
            weight = self.m / sum #m本来就都是0，sum只要不为0就可以了，加上很小的1e-6，避免除0情况，weight是单词的权重
            lc = tf.reduce_mean(tf.multiply(self.cross_entropy_c, weight), axis=-1)#到达句级
            self.loss_c = tf.reduce_mean(lc) * 10#到达batch级

            self.loss = 0.7 * self.loss_senti + 0.3 * self.loss_c
            # self.loss = self.loss_senti

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits_t, self.labels, 1), tf.float32))

            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.AdadeltaOptimizer(self.lr, rho=0.9)
            # opt = tf.train.AdagradOptimizer(self.lr)
            # opt = tf.train.GradientDescentOptimizer(self.lr)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

    def build(self):
        with tf.variable_scope('model'):
            ###########################输入准备###################################
            self.asplabel = tf.nn.embedding_lookup(self.lex, self.inputs)#softmax后的与category匹配的PMI词典
            # cate_emb = tf.tile(self.cate_embedding, [1, self.text_word_size, 1])
            # lstminputs = tf.concat([self.inputs_embedding, cate_emb], 2)
            lstminputs = self.inputs_embedding
            # self.memory = self.inputs_embedding
            with tf.variable_scope('text') as scope:
                text_rnn = BiDynamicRNNLayer(
                # text_rnn = DynamicRNNLayer(
                    inputs=lstminputs,
                    cell_fn=self.word_cell,  # tf.nn.rnn_cell.LSTMCell,
                    n_hidden=self.hidden_size/2,
                    sequence_length=self.text_word_lengths,
                )
                text_encoder_output = text_rnn.outputs
                text_final = text_rnn.finalout
            #######################################################################

            #############################方面类别学习#############################
            finalrep_asp = text_encoder_output
            # finalrep_asp = layers.fully_connected(aspsenti_output, self.embedding_size, weights_initializer=self.Winit, activation_fn=None)
            with tf.variable_scope('aspsentidropout'):
                finalrep_asp = layers.dropout(
                    finalrep_asp, keep_prob=self.dropout_keep_proba,
                    is_training=self.is_training,
                )

            self.logits_c = layers.fully_connected(finalrep_asp, self.aspnum, weights_initializer=self.Winit, activation_fn=None)

            # TODO: parameter 1
            #self.m = self.m * 5

            self.cate_onehot = tf.tile(self.cate_onehot,[1,self.text_word_size,1])
            pmi_category = tf.reduce_max(tf.multiply(self.cate_onehot, tf.sigmoid(self.logits_c)), axis=-1)
            pmi_masked = tf.multiply(pmi_category,self.m ) + 1
            pmi_3dim = tf.tile(tf.expand_dims(pmi_masked, 2), [1, 1, self.embedding_size])

            self.TEST1 = tf.sigmoid(self.logits_c)
            self.TEST2 = tf.multiply(self.cate_onehot, tf.sigmoid(self.logits_c))
            ######################################################################

            #############################情感分类学习#############################
            self.memory = text_encoder_output  # 双向LSTM后的隐向量作为memory输入

            # TODO: parameter 2
            self.memory = tf.multiply(self.memory,pmi_3dim)

            self.hid.append(self.cate_embedding)
            self.attprob = []
            #tmp = tf.matmul(self.memory, tf.cast(self.posmask, tf.float32))

            with tf.variable_scope('multihop') as scope:
                cate_emb = tf.tile(self.hid[-1], [1, self.text_word_size, 1]) # b * len * d
                inputs = tf.concat([self.inputs_embedding, cate_emb], 2) # b * len * 2d
                # sim = tf.matmul(self.inputs_embedding, self.hid[-1])
                # sim = layers.fully_connected(inputs, 1, weights_initializer=tf.random_uniform_initializer(-0.01, 0.01),
                #                     biases_initializer=tf.random_uniform_initializer(-0.01, 0.01), activation_fn=tf.tanh, scope='att')  # b * len * 1
                sim = layers.fully_connected(inputs, 1, activation_fn=tf.tanh, scope='att')
                # sim = tf.reduce_sum(tf.multiply(layers.fully_connected(inputs, self.embedding_size * 2, activation_fn=tf.tanh, scope='att'),
                #                 self.context_vector), axis=2, keep_dims=True) # b * n * 1
                # prob = tf.nn.softmax(sim, dim=1)
                prob = tf.expand_dims(softmask(tf.squeeze(sim, [2]), self.textwm), axis=-1)#计算category和memory注意力
                out = tf.matmul(tf.transpose(prob, perm=[0, 2, 1]), self.memory)
                # finalout = out + layers.fully_connected(self.hid[-1], self.hidden_size,
                #                              weights_initializer=tf.random_uniform_initializer(-0.01, 0.01, seed=self.seed),
                #                              biases_initializer=tf.random_uniform_initializer(-0.01, 0.01, seed=self.seed),
                #                              activation_fn=None, scope='linear')
                finalout = out + self.hid[-1]
                self.hid.append(finalout)
                self.attprob.append(prob)
                scope.reuse_variables()
                for h in range(self.nhop):
                    cate_emb = tf.tile(self.hid[-1], [1, self.text_word_size, 1])  # b * len * d
                    inputs = tf.concat([self.memory, cate_emb], 2)  # b * len * 2d
                    # sim = tf.matmul(self.inputs_embedding, self.hid[-1])
                    # sim = layers.fully_connected(inputs, 1,
                    #                              weights_initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    #                              biases_initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    #                              activation_fn=tf.tanh, scope='att')  # b * len * 1
                    sim = layers.fully_connected(inputs, 1, activation_fn=tf.tanh, scope='att')
                    # sim = tf.reduce_sum(tf.multiply(
                    #     layers.fully_connected(inputs, self.embedding_size * 2, activation_fn=tf.tanh, scope='att'),
                    #     self.context_vector), axis=2, keep_dims=True)  # b * n * 1
                    # prob = tf.nn.softmax(sim, dim=1)
                    prob = tf.expand_dims(softmask(tf.squeeze(sim, [2]), self.textwm), axis=-1)
                    out = tf.matmul(tf.transpose(prob, perm=[0, 2, 1]), self.memory)
                    #print('out', out.shape)
                    # finalout = out + layers.fully_connected(self.hid[-1], self.hidden_size,
                    #                              weights_initializer=tf.random_uniform_initializer(-0.01, 0.01, seed=self.seed),
                    #                              biases_initializer=tf.random_uniform_initializer(-0.01, 0.01, seed=self.seed),
                    #                              activation_fn=None, scope='linear')
                    finalout = out + self.hid[-1]
                    self.hid.append(finalout)
                    self.attprob.append(prob)
            finalrep = tf.squeeze(self.hid[-1], [1])
            with tf.variable_scope('dropout'):
                finalrep = layers.dropout(
                    finalrep, keep_prob=self.dropout_keep_proba,
                    is_training=self.is_training,
                )
            # rep = layers.fully_connected(finaloutput, self.hidden_size, activation_fn=tf.tanh)
            # rep = layers.dropout(rep, keep_prob=self.dropout_keep_proba,is_training=self.is_training)
            self.logits_t = layers.fully_connected(finalrep, self.classes, weights_initializer=self.Winit, activation_fn=None)
            # self.logits_c = tf.matmul(layers.fully_connected(finalrep_asp, self.hidden_size, weights_initializer=self.Winit, activation_fn=None, biases_initializer=None), self.aspect_embedding, transpose_b=True)
            self.prediction = tf.argmax(self.logits_t, axis=-1)
            ######################################################################

    def get_feed_data(self, x, c, y=None, a=None, e=None, class_weights=None, is_training=True):
        x_m, x_sizes, xwordm, pm = utils.batch_lexmask(x, a)
        #print('pm',pm)
        fd = {
            self.inputs: x_m,
            self.text_word_lengths: x_sizes,
            self.textwm: xwordm,
            self.category: c,
            self.posmask: pm
        }
        if y is not None:
            fd[self.labels] = y
        if e is not None:
            fd[self.embedding_matrix] = e
        fd[self.is_training] = is_training
        return fd
