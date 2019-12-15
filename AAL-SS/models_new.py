import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from layers_new import *
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


class AALSS(Model):
    def __init__(self, config, embedding_matrix, sess):
        super(AALSS, self).__init__(config, embedding_matrix)
        # self.word_cell = tf.contrib.rnn.LSTMCell
        self.nhop = config.nhop
        self.hid = []
        with tf.variable_scope("cate_emb"):
            self.cate_embedding = tf.nn.embedding_lookup(self.aspect_embedding, tf.expand_dims(self.category, -1))
        with tf.device(self.device):
            self.build()
            # self.train(self.logits_t)
        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                logits=self.logits_t)
            self.cross_entropy_c = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(self.aspcat, dtype=tf.float32), logits=self.logits_c)
            # self.cross_entropy_c = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.category, logits=self.logits_c)
            self.loss_senti = tf.reduce_mean(self.cross_entropy)
            self.loss_c = tf.reduce_mean(self.cross_entropy_c)
            gamma=0.3
            self.loss = (1.0-gamma)* self.loss_senti + gamma * self.loss_c
            # self.loss = self.loss_senti

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits_t, self.labels, 1), tf.float32))

            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.GradientDescentOptimizer(self.lr)
            # opt = tf.train.AdadeltaOptimizer(self.lr, rho=0.9)
            # opt = tf.train.AdagradOptimizer(self.lr)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

    def build(self):
        with tf.variable_scope('model'):
            # print('cate_embedding', self.cate_embedding.get_shape())
            # print('text_word_size', self.text_word_size)
            # cate_emb = tf.tile(self.cate_embedding, [1, self.text_word_size, 1])
            # lstminputs = tf.concat([self.inputs_embedding, cate_emb], 2)
            # print('inputs_embedding', self.inputs_embedding.get_shape())
            # print('cate_emb', cate_emb.get_shape())
            # print('lstminputs', lstminputs.get_shape())
            #
            # print('lstminputs', lstminputs)
            location_3dim = tf.tile(tf.expand_dims(self.location, 2), [1, 1, self.hidden_size])
            lstminputs = self.inputs_embedding#####################改动1
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
            self.memory = text_encoder_output

            #******************************************************************
            self.memory_loc = self.memory #* location_3dim
            self.inputs_embedding_loc = self.inputs_embedding #* location_3dim
            #******************************************************************
            #
            # aspsenti_output = tf.reduce_mean(self.memory, axis=1)
            with tf.variable_scope('aspsenti_att') as scope:
                aspsenti_output = atae_attention(
                    text_encoder_output,
                    self.context_vector,
                    self.word_output_size,
                    aspect=None,
                    # aspect=cate_emb,
                )#####################改动2
            with tf.variable_scope('aspsentidropout'):
                aspsenti_output = layers.dropout(
                    aspsenti_output, keep_prob=self.dropout_keep_proba,
                    is_training=self.is_training,
                )

            # m = tf.cast(self.targetwm, tf.float32)
            # sum = tf.reduce_sum(m, axis=1, keep_dims=True)
            # weight = m / sum
            # targetavg = tf.reduce_mean(tf.multiply(self.target_embedding, tf.expand_dims(weight, axis=-1)), axis=1,
            #                     keep_dims=True)
            # cate_emb = tf.tile(self.cate_embedding, [1, self.text_word_size, 1])
            self.hid.append(self.cate_embedding)
            self.attprob = []

            with tf.variable_scope('multihop') as scope:
                cate_emb = tf.tile(self.hid[-1], [1, self.text_word_size, 1])  # b * len * d
                inputs = tf.concat([self.inputs_embedding_loc, cate_emb], 2) # b * len * 2d #我修改的，唯一原文
                #inputs = tf.concat([self.memory_loc, cate_emb], 2) # b * len * 2d
                # sim = tf.matmul(self.inputs_embedding, self.hid[-1])
                sim = layers.fully_connected(inputs, 1, weights_initializer=self.Winit,
                                             biases_initializer=self.Winit, activation_fn=tf.tanh,
                                             scope='att')  #####################改动3
                # sim = layers.fully_connected(inputs, 1, activation_fn=tf.tanh, scope='att')
                # sim = tf.reduce_sum(tf.multiply(layers.fully_connected(inputs, self.embedding_size * 2, activation_fn=tf.tanh, scope='att'),
                #                 self.context_vector), axis=2, keep_dims=True) # b * n * 1
                # prob = tf.nn.softmax(sim, dim=1)
                prob = tf.expand_dims(softmask(tf.squeeze(sim, [2]), self.textwm), axis=-1)#b*len'*1
                out = tf.matmul(tf.transpose(prob, perm=[0, 2, 1]), self.memory_loc) #我修改的
                #out = tf.matmul(tf.transpose(prob, perm=[0, 2, 1]), self.inputs_embedding_loc)
                # finalout = out + layers.fully_connected(self.hid[-1], self.hidden_size, weights_initializer=self.Winit,
                #                              biases_initializer=self.Winit, activation_fn=None, scope='linear')
                finalout = out + self.hid[-1]
                self.hid.append(finalout)
                self.attprob.append(prob)
                scope.reuse_variables()
                for h in range(self.nhop):
                    cate_emb = tf.tile(self.hid[-1], [1, self.text_word_size, 1])  # b * len * d
                    inputs = tf.concat([self.memory_loc, cate_emb], 2)  # b * len * 2d #我修改的
                    #inputs = tf.concat([self.inputs_embedding_loc, cate_emb], 2)  # b * len * 2d
                    # sim = tf.matmul(self.inputs_embedding, self.hid[-1])
                    sim = layers.fully_connected(inputs, 1, weights_initializer=self.Winit,
                                                 biases_initializer=self.Winit,
                                                 activation_fn=tf.tanh, scope='att')  #####################改动4
                    # sim = layers.fully_connected(inputs, 1, activation_fn=tf.tanh, scope='att')
                    # sim = tf.reduce_sum(tf.multiply(layers.fully_connected(inputs, self.embedding_size * 2, activation_fn=tf.tanh, scope='att'),
                    #     self.context_vector), axis=2, keep_dims=True)  # b * n * 1
                    # prob = tf.nn.softmax(sim, dim=1)
                    prob = tf.expand_dims(softmask(tf.squeeze(sim, [2]), self.textwm), axis=-1)
                    out = tf.matmul(tf.transpose(prob, perm=[0, 2, 1]), self.memory_loc) #我修改的
                    #out = tf.matmul(tf.transpose(prob, perm=[0, 2, 1]), self.inputs_embedding_loc)
                    # finalout = out + layers.fully_connected(self.hid[-1], self.hidden_size, weights_initializer=self.Winit,
                    #                                 biases_initializer=self.Winit, activation_fn=None, scope='linear')
                    finalout = out + self.hid[-1]
                    self.hid.append(finalout)
                    self.attprob.append(prob)
            finalrep = tf.squeeze(self.hid[-1], [1])
            # with tf.variable_scope('dropout'):
            #     finalrep = layers.dropout(
            #         finalrep, keep_prob=self.dropout_keep_proba,
            #         is_training=self.is_training,
            #     )

            finalrep_asp = aspsenti_output
            # finalrep_asp = layers.fully_connected(aspsenti_output, self.embedding_size, weights_initializer=self.Winit, activation_fn=None)

        with tf.variable_scope('classifier'):
            # rep = layers.fully_connected(finaloutput, self.hidden_size, activation_fn=tf.tanh)
            # rep = layers.dropout(rep, keep_prob=self.dropout_keep_proba,is_training=self.is_training)
            self.logits_t = layers.fully_connected(finalrep, self.classes, weights_initializer=self.Winit,
                                                   activation_fn=None)
            # self.logits_c = layers.fully_connected(finalrep_asp, self.aspnum, weights_initializer=self.Winit, activation_fn=None)
            self.logits_c = tf.matmul(
                layers.fully_connected(finalrep_asp, self.hidden_size, weights_initializer=self.Winit,
                                       activation_fn=None, biases_initializer=self.Winit), self.aspect_embedding,
                transpose_b=True)#####################改动5
            self.prediction = tf.argmax(self.logits_t, axis=-1)

    def get_feed_data(self, x, c, y=None, a=None, p= None, e=None, class_weights=None, is_training=True):
        x_m, x_sizes, xwordm = utils.batch(x)
        p_m = utils.batch_loc(p)
        #print('p',p)
        #print('p_m',p_m)
        fd = {
            self.inputs: x_m,
            self.text_word_lengths: x_sizes,
            self.textwm: xwordm,
            self.category: c
        }
        if y is not None:
            fd[self.labels] = y
        if e is not None:
            fd[self.embedding_matrix] = e
        if a is not None:
            fd[self.aspcat] = a
        if p is not None:
            fd[self.location] = p_m
        fd[self.is_training] = is_training
        return fd

class AALlex(Model):
    def __init__(self, config, embedding_matrix, lex,  sess):
        super(AALlex, self).__init__(config, embedding_matrix)
        self.word_cell = tf.contrib.rnn.LSTMCell
        self.nhop = config.nhop
        self.hid = []
        self.lex = tf.Variable(initial_value=lex, name='lex', dtype=tf.float32, trainable=False)
        with tf.variable_scope("cate_emb"):
            self.cate_embedding = tf.nn.embedding_lookup(self.aspect_embedding, tf.expand_dims(self.category, -1))
        with tf.device(self.device):
            self.build()
            # self.train(self.logits_t)
        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_t)
            self.cross_entropy_c = tf.nn.softmax_cross_entropy_with_logits(labels=self.asplabel, logits=self.logits_c)

            self.loss_senti = tf.reduce_mean(self.cross_entropy)
            m = tf.cast(self.posmask, tf.float32)
            sum = tf.reduce_sum(m, axis=1, keep_dims=True) + 1e-6
            weight = m / sum
            lc = tf.reduce_mean(tf.multiply(self.cross_entropy_c, weight), axis=-1)
            self.loss_c = tf.reduce_mean(lc) * 10

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
            self.asplabel = tf.nn.embedding_lookup(self.lex, self.inputs)
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
            self.memory = text_encoder_output

            self.hid.append(self.cate_embedding)
            self.attprob = []
            tmp = tf.matmul(self.memory, tf.cast(self.posmask, tf.float32))

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
                prob = tf.expand_dims(softmask(tf.squeeze(sim, [2]), self.textwm), axis=-1)
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
                    print('out', out.shape)
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

            finalrep_asp = text_encoder_output
            # finalrep_asp = layers.fully_connected(aspsenti_output, self.embedding_size, weights_initializer=self.Winit, activation_fn=None)
            with tf.variable_scope('aspsentidropout'):
                finalrep_asp = layers.dropout(
                    finalrep_asp, keep_prob=self.dropout_keep_proba,
                    is_training=self.is_training,
                )

        with tf.variable_scope('classifier'):
            # rep = layers.fully_connected(finaloutput, self.hidden_size, activation_fn=tf.tanh)
            # rep = layers.dropout(rep, keep_prob=self.dropout_keep_proba,is_training=self.is_training)
            self.logits_t = layers.fully_connected(finalrep, self.classes, weights_initializer=self.Winit, activation_fn=None)
            self.logits_c = layers.fully_connected(finalrep_asp, self.aspnum, weights_initializer=self.Winit, activation_fn=None)
            # self.logits_c = tf.matmul(layers.fully_connected(finalrep_asp, self.hidden_size, weights_initializer=self.Winit, activation_fn=None, biases_initializer=None), self.aspect_embedding, transpose_b=True)
            self.prediction = tf.argmax(self.logits_t, axis=-1)

    def get_feed_data(self, x, c, y=None, a=None, e=None, class_weights=None, is_training=True):
        x_m, x_sizes, xwordm, pm = utils.batch_lexmask(x, a)
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
