# -*- coding: UTF-8 -*-
import tensorflow as tf
# from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import BasicRNNCell as BasicRNNCell
import tensorflow.contrib.layers as layers
import numpy as np

set_keep = globals()
set_keep['_layers_name_list'] =[]
name_reuse = False
try:  # For TF12 and later
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
except:  # For TF11 and before
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.VARIABLES
#set_keep['name_reuse'] = False
## Basic layer
# class LSTMUPMemoryCell(LSTMCell):
#
#     def __init__(self, num_units, usr_emb, pro_emb, input_size=None):
#         super(LSTMUPMemoryCell, self).__init__(num_units)
#         self.usr = usr_emb
#         self.pro = pro_emb
#
#     def __call__(self, inputs, state, output_size=64, activation_fn=tf.tanh, scope=None):
#         """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
#         m, new_state = super(LSTMUPMemoryCell, self).__call__(inputs=inputs, state=state, scope=scope)
#         u = layers.fully_connected(tf.concat((m, self.usr), axis=0), 1, activation_fn=tf.sigmoid)
#         p = layers.fully_connected(tf.concat((m, self.pro), axis=0), 1, activation_fn=tf.sigmoid)
#         output = m + u * self.usr + p * self.pro
#         return output, new_state

class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    inputs : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        inputs = None,
        name ='layer'
    ):
        self.inputs = inputs
        scope_name=tf.get_variable_scope().name
        if scope_name:
            name = scope_name + '/' + name
        if (name in set_keep['_layers_name_list']) and name_reuse == False:
            raise Exception("Layer '%s' already exists, please choice other 'name' or reuse this layer\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)" % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)


    def print_params(self, details=True):
        ''' Print all info of parameters in the network'''
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    print("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                except Exception as e:
                    print(str(e))
                    raise Exception("Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False).")
            else:
                print("  param {:3}: {:15}    {}".format(i, str(p.get_shape()), p.name))
        print("  num of params: %d" % self.count_params())

    def print_layers(self):
        ''' Print all info of layers in the network '''
        for i, p in enumerate(self.all_layers):
            print("  layer %d: %s" % (i, str(p)))

    def count_params(self):
        ''' Return the number of parameters in the network '''
        n_params = 0
        for i, p in enumerate(self.all_params):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params

    def __str__(self):
        # print("\nIt is a Layer class")
        # self.print_params(False)
        # self.print_layers()
        return "  Last layer is: %s" % self.__class__.__name__

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

# Bidirectional Dynamic RNN
class BiDynamicRNNLayer(Layer):

    def __init__(
        self,
        layer = None,
        inputs = None,
        cell_fn = None,#tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'state_is_tuple':True},
        #cell_init_args=None,
        n_hidden = 256,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        sequence_length = None,
        fw_initial_state = None,
        bw_initial_state = None,
        dropout = None,
        n_layer = 1,
        return_last = False,
        return_seq_2d = False,
        dynamic_rnn_init_args={},
        scope = None,
        name = 'bi_dyrnn_layer',
    ):
        Layer.__init__(self, name=name)
        if cell_fn is None:
            raise Exception("Please put in cell_fn")
        if 'GRU' in cell_fn.__name__:
            try:
                cell_init_args.pop('state_is_tuple')
            except:
                pass
        # self.inputs = layer.outputs
        self.inputs = inputs

        # print("  [TL] BiDynamicRNNLayer %s: n_hidden:%d in_dim:%d in_shape:%s cell_fn:%s dropout:%s n_layer:%d" %
        #       (self.name, n_hidden, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout, n_layer))

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("       batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("       non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        with tf.variable_scope(name, initializer=initializer) as vs:
            # Creats the cell function
            cell_instance_fn=lambda: cell_fn(num_units=n_hidden, **cell_init_args)
            #cell_instance_fn=lambda: cell_fn(num_units=n_hidden, state_is_tuple=True)

            self.fw_cell=cell_instance_fn()
            self.bw_cell=cell_instance_fn()
            # Initial state of RNN
            if fw_initial_state is None:
                self.fw_initial_state = self.fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.fw_initial_state = fw_initial_state
            if bw_initial_state is None:
                self.bw_initial_state = self.bw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.bw_initial_state = bw_initial_state
            # Computes sequence_length
            if sequence_length is None:
                try: ## TF1.0
                    sequence_length = retrieve_seq_length_op(
                        self.inputs if isinstance(self.inputs, tf.Tensor) else tf.stack(self.inputs))
                except: ## TF0.12
                    sequence_length = retrieve_seq_length_op(
                        self.inputs if isinstance(self.inputs, tf.Tensor) else tf.pack(self.inputs))

            outputs, (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.fw_cell,
                cell_bw=self.bw_cell,
                inputs=self.inputs,
                sequence_length=sequence_length,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state,
                scope=scope,
                **dynamic_rnn_init_args
            )
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
            print("     n_params : %d" % (len(rnn_variables)))
            # Manage the outputs
            try: # TF1.0
                outputs = tf.concat(outputs, 2)
            except: # TF0.12
                outputs = tf.concat(2, outputs)
            if return_last:
                # [batch_size, 2 * n_hidden]
                self.outputs = advanced_indexing_op(outputs, sequence_length)
            else:
                # [batch_size, n_step(max), 2 * n_hidden]
                if return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, 2 * n_hidden]
                    try: # TF1.0
                        self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, 2 * n_hidden])
                    except: # TF0.12
                        self.outputs = tf.reshape(tf.concat(1, outputs), [-1, 2 * n_hidden])
                else:
                    self.outputs = outputs
                    self.finalout = advanced_indexing_op(outputs, sequence_length)

class DynamicRNNLayer(Layer):

    def __init__(
        self,
        inputs = None,
        cell_fn = None,#tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'state_is_tuple':True},
        n_hidden = 256,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        sequence_length = None,
        initial_state = None,
        usr = None,
        pro = None,
        return_last = False,
        return_seq_2d = False,
        dynamic_rnn_init_args={},
        scope = None,
        name = 'bi_dyrnn_layer',
    ):
        Layer.__init__(self, name=name)
        if cell_fn is None:
            raise Exception("Please put in cell_fn")
        if 'GRU' in cell_fn.__name__:
            try:
                cell_init_args.pop('state_is_tuple')
            except:
                pass
        # self.inputs = layer.outputs
        self.inputs = inputs
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("       batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("       non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        with tf.variable_scope(name, initializer=initializer) as vs:
            # Creats the cell function
            cell_instance_fn=lambda: cell_fn(num_units=n_hidden, **cell_init_args)
            # cell_instance_fn=lambda: cell_fn(num_units=n_hidden, usr_emb=usr, pro_emb=pro)

            self.cell=cell_instance_fn()
            # Initial state of RNN
            if initial_state is None:
                self.fw_initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.fw_initial_state = initial_state
            # Computes sequence_length
            if sequence_length is None:
                try: ## TF1.0
                    sequence_length = retrieve_seq_length_op(
                        self.inputs if isinstance(self.inputs, tf.Tensor) else tf.stack(self.inputs))
                except: ## TF0.12
                    sequence_length = retrieve_seq_length_op(
                        self.inputs if isinstance(self.inputs, tf.Tensor) else tf.pack(self.inputs))

            outputs, states = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=self.inputs,
                sequence_length=sequence_length,
                initial_state=self.fw_initial_state,
                scope=scope,
                **dynamic_rnn_init_args
            )
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
            print("     n_params : %d" % (len(rnn_variables)))
            if return_last:
                # [batch_size, 2 * n_hidden]
                self.outputs = advanced_indexing_op(outputs, sequence_length)
            else:
                self.outputs = outputs
                self.finalout = advanced_indexing_op(outputs, sequence_length)

def softmask(x, mask):
    y = tf.multiply(tf.exp(x), tf.cast(mask, tf.float32))
    sumx = tf.reduce_sum(y, axis=-1, keep_dims=True)
    return y / (sumx + 1e-10)
    # return y / sumx
def task_specific_attention(inputs, output_size,attention_context_vector, mask,
                            activation_fn=tf.tanh, scope=None):

    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    input_projection = layers.fully_connected(inputs, output_size,
                                              activation_fn=activation_fn,
                                              scope=scope)

    attention_weights = tf.nn.softmax(tf.reduce_sum(
        tf.multiply(input_projection, attention_context_vector), axis=2))
    # attention_weights = softmask(tf.reduce_sum(
    #     tf.multiply(input_projection, attention_context_vector), axis=2), mask)
    attention_weights = tf.expand_dims(attention_weights, -1)
    weighted_projection = tf.multiply(attention_weights, inputs)
    outputs = tf.reduce_sum(weighted_projection, axis=1)
    return outputs

def up_attention(inputs, usr, prd, attention_context_vector, output_size, mask,
                            activation_fn=tf.tanh):
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
    (inputbsize, dsize, esize) = tf.unstack(tf.shape(inputs))
    # (inputbsize, dsize, esize) = inputs.get_shape()
    (ubsize, esize) = tf.unstack(tf.shape(usr))
    # (ubsize, esize) = usr.get_shape()
    times = inputbsize * dsize / ubsize
    # attention_context_vector = tf.get_variable(name='attention_context_vector',
    #                     shape=[output_size], initializer=initializer, dtype=tf.float32)
    inputproj = layers.fully_connected(inputs, output_size, activation_fn=None, scope='inputproj')
    usrproj = layers.fully_connected(usr, output_size, biases_initializer=None, activation_fn=None, scope='usrproj')
    prdproj = layers.fully_connected(prd, output_size, biases_initializer=None, activation_fn=None, scope='prdproj')
    input_projection = tf.tanh(inputproj +
            tf.reshape(tf.tile(tf.expand_dims(usrproj, axis=1), [1, times, 1]), [inputbsize, dsize, output_size]) +
            tf.reshape(tf.tile(tf.expand_dims(prdproj, axis=1), [1, times, 1]), [inputbsize, dsize, output_size]))
    # input_projection = layers.fully_connected(inputup, output_size,
    #                                           activation_fn=activation_fn,
    #                                           scope=scope)

    attention_weights = tf.nn.softmax(
        tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2))
    # attention_weights = softmask(
    #     tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2), mask)
    attention_weights = tf.expand_dims(attention_weights, -1)
    weighted_projection = tf.multiply(attention_weights, inputs)
    outputs = tf.reduce_sum(weighted_projection, axis=1)
    return outputs

def interaction_up(inputs, upemb, attention_context_vector,
                output_size, scope=None):
    assert len(upemb.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    (inputbsize, dsize, esize) = tf.unstack(tf.shape(inputs))
    (ubsize, infonum, esize) = tf.unstack(tf.shape(upemb))
    # uptimes = inputbsize * dsize / ubsize
    intimes = inputbsize / ubsize
    upemb = tf.tile(upemb, [intimes, 1, 1])
    inputavg = tf.reduce_mean(inputs, axis=1)
    upavg = tf.reduce_mean(upemb, axis=1)

    inputproj = layers.fully_connected(inputs, output_size, activation_fn=None, scope='inputproj')
    upproj = layers.fully_connected(upemb, output_size, activation_fn=None, scope='upproj')
    inputavgproj = layers.fully_connected(inputavg, output_size, biases_initializer=None, activation_fn=None,
                                          scope='inputavgproj')
    upavgproj = layers.fully_connected(upavg, output_size, biases_initializer=None, activation_fn=None,
                                       scope='upavgproj')
    input_projection = tf.tanh(inputproj +
                    tf.tile(tf.expand_dims(upavgproj, axis=1), [1, dsize, 1]))
    up_projection = tf.tanh(upproj + tf.tile(tf.expand_dims(inputavgproj, axis=1), [1, 2, 1]))
    # input_projection = layers.fully_connected(inputup, output_size,
    #                                           activation_fn=activation_fn,
    #                                           scope=scope)

    input_attention_weights = tf.nn.softmax(
        tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2))
    up_attention_weights = tf.nn.softmax(
        tf.reduce_sum(tf.multiply(up_projection, attention_context_vector), axis=2))
    input_attention_weights = tf.expand_dims(input_attention_weights, -1)
    up_attention_weights = tf.expand_dims(up_attention_weights, -1)
    input_weighted_projection = tf.multiply(input_attention_weights, inputs)
    up_weighted_projection = tf.multiply(up_attention_weights, upemb)
    input_outputs = tf.reduce_sum(input_weighted_projection, axis=1)
    up_outputs = tf.reduce_sum(up_weighted_projection, axis=1)
    outputs = input_outputs + up_outputs
    return outputs

def interaction(text_inputs, target_inputs, bilinear_weight, output_size, type, scope=None):
    assert target_inputs.get_shape()[-1].value is not None and target_inputs.get_shape()[-1].value is not None

    # (inputbsize, dsize, esize) = tf.unstack(tf.shape(inputs))
    # tpye: 0
    if type == 0:
        targetavg = target_inputs
    if type == 1:
        textavg = text_inputs
    if type == 2:
        textavg = tf.reduce_mean(text_inputs, axis=1, keep_dims=True)
        targetavg = tf.reduce_mean(target_inputs, axis=1, keep_dims=True)
    if type != 1:
        text_attention_weights = tf.nn.softmax(tf.matmul(text_inputs, tf.transpose(targetavg, perm=[0, 2, 1])), dim=1)
        text_output = tf.reduce_sum(tf.multiply(text_attention_weights, text_inputs), axis=1)
    if type != 0:
        target_attention_weights = tf.nn.softmax(tf.matmul(target_inputs, tf.transpose(textavg, perm=[0, 2, 1])), dim=1)
        target_output = tf.reduce_sum(tf.multiply(target_attention_weights, target_inputs), axis=1)
    if type == 0:
        return text_output
    if type == 1:
        return target_output
    if type == 2:
        return text_output, target_output

def topic_attention(text_inputs, topic_emb, bilinear_weight, output_size, mask, scope=None):
    assert text_inputs.get_shape()[-1].value is not None and topic_emb.get_shape()[-1].value is not None
    m = tf.cast(mask, tf.float32)
    sum = tf.reduce_sum(m, axis=1, keep_dims=True)
    weight = m / sum
    (bsize, dsize, esize) = tf.unstack(tf.shape(text_inputs))
    # textavg = tf.reduce_mean(text_inputs, axis=1, keep_dims=True)
    textavg = tf.reduce_mean(tf.multiply(text_inputs, tf.expand_dims(weight, axis=-1)), axis=1, keep_dims=True)
    topic_inputs = tf.tile(tf.expand_dims(topic_emb, axis=0), multiples=[bsize, 1, 1])

    topic_attention_weights = tf.nn.softmax(tf.matmul(topic_inputs, tf.transpose(textavg, perm=[0, 2, 1])), dim=1)
    topic_output = tf.reduce_sum(tf.multiply(topic_attention_weights, topic_inputs), axis=1, keep_dims=True)

    return topic_output

def atae_attention(inputs, context_vector, output_size, aspect, mask=None, activation_fn=None, scope=None):

    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
    # context_vector = tf.Variable(tf.random_uniform(shape=[output_size * 2], minval=-0.01, maxval=0.01), name='attention_context_vector', dtype=tf.float32, trainable=True)
    # context_vector = tf.Variable(tf.truncated_normal(shape=[output_size * 2]), name='attention_context_vector', dtype=tf.float32, trainable=True)
    if aspect is not None:
        input_proj = layers.fully_connected(inputs, output_size, activation_fn=activation_fn, biases_initializer=None, scope=scope)
        asp_proj = layers.fully_connected(aspect, output_size, activation_fn=activation_fn, biases_initializer=None, scope=scope)
        proj = tf.concat([input_proj, asp_proj], 2)
    else:
        proj = layers.fully_connected(inputs, output_size * 2, activation_fn=activation_fn, biases_initializer=None, scope=scope)
    if mask is not None:
        attention_weights = softmask(tf.reduce_sum(tf.multiply(proj, context_vector), axis=2), mask)
    else:
        attention_weights = tf.nn.softmax(tf.reduce_sum(tf.multiply(proj, context_vector), axis=2))
    attention_weights = tf.expand_dims(attention_weights, -1)
    weighted_projection = tf.multiply(attention_weights, inputs)
    outputs = tf.reduce_sum(weighted_projection, axis=1)
    return outputs#, attention_weights

def co_attention(inputs, context_vector, output_size, aspemb, aspOnehot, mask=None, activation_fn=tf.tanh, scope=None):

    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
    sim = tf.matmul(layers.fully_connected(inputs, output_size,
                            activation_fn=None, biases_initializer=None), aspemb, transpose_b=True)
    input_proj = layers.fully_connected(inputs, output_size, activation_fn=activation_fn, biases_initializer=None, scope=scope)
    senti_weights = tf.nn.softmax(tf.reduce_sum(tf.multiply(input_proj, context_vector), axis=2, keep_dims=False))
    asp_weights = tf.squeeze(tf.matmul(sim, aspOnehot), [2])
    dotweight = tf.multiply(senti_weights, asp_weights)
    attention_weights = tf.expand_dims(tf.nn.softmax(dotweight, dim=-1), -1)
    weighted_projection = tf.multiply(attention_weights, inputs)
    outputs = tf.reduce_sum(weighted_projection, axis=1)

    senti_projection = tf.multiply(tf.expand_dims(senti_weights, -1), inputs)
    senti_outputs = tf.reduce_sum(senti_projection, axis=1)
    return outputs, senti_outputs

def hingeloss(atsim, extweight, mask):
    # dif = tf.maximum(0.0, 1.0 - atsim + extweight)
    # m = tf.cast(mask - 1, tf.float32)
    # loss = tf.multiply(m, dif)
    # return -loss
    m = tf.cast(mask, tf.float32)
    dif = tf.multiply(m, extweight)
    loss = tf.maximum(0.0, 1.0 - tf.reduce_mean(atsim, axis=-1) + tf.reduce_mean(dif, axis=-1))
    return loss



# Advanced Ops for Dynamic RNN
def advanced_indexing_op(input, index):
    """Advanced Indexing for Sequences, returns the outputs by given sequence lengths.
    When return the last output :class:`DynamicRNNLayer` uses it to get the last outputs with the sequence lengths.

    Parameters
    -----------
    input : tensor for data
        [batch_size, n_step(max), n_features]
    index : tensor for indexing, i.e. sequence_length in Dynamic RNN.
        [batch_size]

    References
    -----------
    - Modified from TFlearn (the original code is used for fixed length rnn), `references <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`_.
    """
    batch_size = tf.shape(input)[0]
    # max_length = int(input.get_shape()[1])    # for fixed length rnn, length is given
    max_length = tf.shape(input)[1]             # for dynamic_rnn, length is unknown
    dim_size = int(input.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(input, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant

def retrieve_seq_length_op(data):
    """An op to compute the length of a sequence from input shape of [batch_size, n_step(max), n_features],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max), n_features] with zero padding on right hand side.

    References
    ------------
    - Borrow from `TFlearn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`_.
    """
    with tf.name_scope('GetLength'):
        ## TF 1.0 change reduction_indices to axis
        used = tf.sign(tf.reduce_max(tf.abs(data), 2))
        length = tf.reduce_sum(used, 1)
        ## TF < 1.0
        # used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        # length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
    return length

def retrieve_seq_length_op2(data):
    """An op to compute the length of a sequence, from input shape of [batch_size, n_step(max)],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max)] with zero padding on right hand side.

    """
    return tf.reduce_sum(tf.cast(tf.greater(data, tf.zeros_like(data)), tf.int32), 1)

def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term
