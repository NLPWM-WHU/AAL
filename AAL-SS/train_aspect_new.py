import tensorflow as tf
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import utils
import numpy as np
import random
from sklearn import metrics
from models_new import *
from pmi import *
# settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# base = 'tweet'
base = 'Sem14ResAsp'
flags.DEFINE_string('model', 'AALSS', 'model')
flags.DEFINE_string('task', 'aspect', 'task') # 'aspect' or 'term'
flags.DEFINE_integer('run', 1, 'first run')
flags.DEFINE_integer('batch_size', 50, 'Batch_size')#####################改动
flags.DEFINE_integer('epochs', 20, 'epochs')
flags.DEFINE_integer('classes', 3, 'class num')
flags.DEFINE_integer('aspnum', 13, 'asp num')
flags.DEFINE_integer('nhop', 4, 'hops num')
flags.DEFINE_integer('hidden_size', 300, 'number of hidden units')
flags.DEFINE_integer('embedding_size', 300, 'embedding_size')
flags.DEFINE_integer('word_output_size', 300, 'word_output_size')
# flags.DEFINE_string('checkpoint_maxacc', 'data/' + base + '/checkpoint_maxacc/', 'checkpoint dir')
# flags.DEFINE_string('checkpoint_minloss', 'data/' + base + '/checkpoint_minloss/', 'checkpoint dir')
flags.DEFINE_float('lr', 0.01, 'learning rate')#####################改动
flags.DEFINE_float('attRandomBase', 0.01, 'attRandomBase')
flags.DEFINE_float('biRandomBase', 0.01, 'biRandomBase')
flags.DEFINE_float('aspRandomBase', 0.01, 'aspRandomBase')
flags.DEFINE_float('seed', 0.05, 'random_seed')
flags.DEFINE_float('max_grad_norm', 5.0, 'max-grad-norm')
flags.DEFINE_float('dropout_keep_proba', 0.5, 'dropout_keep_proba')
flags.DEFINE_string('embedding_file', 'data/' + base + '/pmi/word_embedding.npy', 'embedding_file')
flags.DEFINE_string('traindata', 'data/' + base + '/pmi/train_ss_text.txt', 'traindata')
flags.DEFINE_string('testdata', 'data/' + base + '/pmi/test_ss_text.txt', 'testdata')
# flags.DEFINE_string('testdata', 'data/' + base + '/test', 'testdata')
flags.DEFINE_string('devdata', 'data/' + base + '/pmi/dev_ss_text.txt', 'devdata')
flags.DEFINE_string('aspcat', 'data/' + base + '/pmi/aspcat18.txt', 'aspcatdata')
flags.DEFINE_string('device', '/gpu:0', 'device')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

checkpoint_maxacc_dir = 'data/' + base + '/checkpoint_maxacc/' + FLAGS.model + '/'
checkpoint_minloss_dir = 'data/' + base + '/checkpoint_minloss/' + FLAGS.model + '/'
tflog_dir = os.path.join('data/' + base + '/', 'tflog')

def batch_iterator(datalist, batch_size, model='train'):
    xb = []
    yb = []
    tb = []
    pb = []
    num = len(datalist)//batch_size
    list1 = list(range(num))
    if model == 'train':
        random.shuffle(list1)
        # list1 = list[::-1]
    for i in list1:
        #print i
        for j in range(batch_size):
            index = i * batch_size + j
            x = datalist[index][0]
            y = datalist[index][1]
            t = datalist[index][2]
            xb.append(x)
            yb.append(y)
            tb.append(t)
            p = datalist[index][3]
            pb.append(p)
        # if model == 'train':
        yield xb, yb, tb, pb
        xb, yb, tb, pb = [], [], [], []
        # else:
        # yield xb, yb, tb
        # xb, yb, tb = [], [], []
    if len(datalist) % batch_size != 0:
        for i in range(num * batch_size, len(datalist)):
            x = datalist[i][0]
            y = datalist[i][1]
            t = datalist[i][2]

            xb.append(x)
            yb.append(y)
            tb.append(t)
            p = datalist[i][3]
            pb.append(p)
        # if model == 'train':
        yield xb, yb, tb, pb
        # else:
        # yield xb, yb, tb
    # xb, yb, tb = [], [], []


def evaluate(session, model, datalist, process = None):
    predictions = []
    labels = []
    attprob = []
    lossAll = 0.0
    pmi_dict = get_pmi_dict()
    my_word_list = []
    A_list = []
    B_list = []
    C_list = []
    for x, y, t, a in batch_iterator(datalist, FLAGS.batch_size, 'val'):
        labels.extend(y)
        my_word_list.extend(x)
        # pred, loss, att = session.run([model.prediction, model.loss, model.attprob], model.get_feed_data(x, t, y, a, e=None, is_training=False))
        p = get_loc_info(x, t, pmi_dict)
        attprob,pred, loss = session.run([model.attprob,model.prediction, model.loss_senti], model.get_feed_data(x, t, y, a, p, e=None, is_training=False))
        predictions.extend(pred)
        # attprob.extend(att)
        lossAll += loss
        # if len(labels) == FLAGS.batch_size:
        #     print labels
        #     print predictions
    ##################################
    #生成错误的句子信息
    ##################################
    n = 0
    id2word = {}
    # with open(r'data/Sem14ResAsp/wdict.txt','r') as f:
    #     for pair in f:
    #         id = int(pair.strip().split('\t')[1])
    #         word = pair.strip().split('\t')[0]
    #         id2word[id] = word
    error_index = []
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            n += 1
        else:
            error_index.append(i)
        # else:
        #     if process == 'test':
        #         string = ''
        #         for id in my_word_list[i]:
        #             string += id2word[id] + ' '
        #         print('sentence:',string)
        #         print('labels:',labels[i])
        #         print('predictions:',predictions[i])



    # p = metrics.precision_score(labels, predictions, average='micro')
    # r = metrics.recall_score(labels, predictions, average='micro')
    # f = metrics.f1_score(labels, predictions, average='micro')
    # p = 0.0
    # r = 0.0
    # f = 0.0

    if process == 'test':
        print('\nError list:\n', error_index)

    p, r, f, sup = metrics.precision_recall_fscore_support(labels, predictions, average='macro')
    # return float(n)/float(len(labels)), lossAll, p, r, f, predictions, attprob
    return float(n)/float(len(labels)), lossAll, p, r, f

def train():
    # shutil.rmtree(checkpoint_minloss_dir)
    # shutil.rmtree(checkpoint_maxacc_dir)
    word_embedding = np.load(FLAGS.embedding_file)
    train_x, train_y, train_t, train_p = utils.load_data(FLAGS.traindata, FLAGS.task)
    dev_x, dev_y, dev_t, dev_p = utils.load_data(FLAGS.devdata, FLAGS.task)
    test_x, test_y, test_t, test_p = utils.load_data(FLAGS.testdata, FLAGS.task)
    print (test_x)
    print (test_t)
    train_cat = utils.load_aspcat('data/' + base + '/pmi/train_ss_cate.txt', '\t')
    dev_cat = utils.load_aspcat('data/' + base + '/pmi/dev_ss_cate.txt', '\t')
    test_cat = utils.load_aspcat('data/' + base + '/pmi/test_ss_cate.txt', '\t')
    # lex = utils.load_lex('data/' + base + '/lex.txt')
    tf.reset_default_graph()
    fres = open('data/' + base + '/res_' + FLAGS.model + '.txt','a')
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as s:

        # model = MemNN(FLAGS, word_embedding, s)
        model = AALSS(FLAGS, word_embedding, s)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        # '''
        if FLAGS.run == 1:
            s.run(tf.global_variables_initializer())
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_minloss_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(s, ckpt.model_checkpoint_path)
        # summary_writer = tf.summary.FileWriter(tflog_dir, graph=tf.get_default_graph())
        cost_val = []
        maxvalacc = 0.0
        minloss = 10000.0
        valnum = 0
        stepnum = 0
        is_earlystopping = False
        pmi_dict = get_pmi_dict()
        for i in range(FLAGS.epochs):
            trainloss_epoch = 0.0
            t0 = time.time()
            print("Epoch:", '%04d' % (i + 1))
            fres.write("Epoch: %02d \n" % (i + 1))
            for j, (x, y, t, a) in enumerate(batch_iterator(list(zip(train_x, train_y, train_t, train_cat)), FLAGS.batch_size, 'train')):
                p = get_loc_info(x,t,pmi_dict)
                #print(p)
                fd = model.get_feed_data(x, t, y, a, p, e=None)
                step, labels, prediction, loss, accuracy, _, = s.run([
                    model.global_step,
                    model.labels,
                    model.prediction,
                    model.loss,
                    model.accuracy,
                    model.train_op,
                ], fd)
                # print hid
                stepnum = stepnum  + 1
                # print labels
                # print prediction
                trainloss_epoch += loss
                # if stepnum % 100 == 0:
                # print("Step:", '%05d' % stepnum, "train_loss=", "{:.5f}".format(loss),
                #       "train_loss_senti=", "{:.5f}".format(loss_senti), "train_loss_cla=", "{:.5f}".format(loss_cla),
                #       "train_acc=", "{:.5f}".format(accuracy),"time=", "{:.5f}".format(time.time() - t0))
                # if stepnum % (64000/FLAGS.batch_size) == 0:
                # if stepnum % (64000 / FLAGS.batch_size) == 0:
            valnum += 1
            valacc, valloss, vp, vr, vf = evaluate(s, model, list(zip(dev_x, dev_y, dev_t, dev_cat)),'validation')
            cost_val.append(valloss)
            if valacc > maxvalacc:
                maxvalacc = valacc
                saver.save(s, checkpoint_maxacc_dir + 'model.ckpt', global_step=step)
            if valloss < minloss:
                minloss = valloss
                saver.save(s, checkpoint_minloss_dir + 'model.ckpt', global_step=step)

            print("Validation:", '%05d' % valnum,"val_loss=", "{:.5f}".format(valloss),
                  "val_acc=", "{:.5f}".format(valacc), "val_f1=", "{:.5f}".format(vf),
                  "time=", "{:.5f}".format(time.time() - t0))
            fres.write("Validation: %05d val_loss= %.5f val_acc= %.5f val_f1= %.5f \n" % (valnum, valloss, valacc, vf))

            test_ml_acc, test_ml_loss, mlp, mlr, mlf = evaluate(s, model, list(zip(test_x, test_y, test_t, test_cat)),'test')
            print("Test set results = %.5f accuracy= %.5f precision= %.5f recall= %.5f f-score= %.5f \n" % (test_ml_loss, test_ml_acc, mlp, mlr, mlf))
            fres.write("Test set results: test_loss= %.5f test_acc= %.5f test_f1= %.5f \n" % (test_ml_loss, test_ml_acc, mlf))

            # if valnum > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            #     is_earlystopping = True
            #     print("Early stopping...")
            #     fres.write("Early stopping...\n")
            #     break
            # print("Epoch:", '%04d' % (i + 1), " train_loss=", "{:.5f}".format(trainloss_epoch))
            # fres.write("Epoch: %06d train_loss= %.5f \n" % ((i + 1), trainloss_epoch))
            # if is_earlystopping:
            #     break
        print("Optimization Finished!")
        fres.write("Optimization Finished!\n")

        #Testing
        # test_x, test_y, test_t, test_p = utils.load_data(FLAGS.testdata, FLAGS.task)
        # test_acc, test_loss = evaluate(s, model, list(zip(test_x, test_y, test_u, test_p)))
        # print("Test set results based last pamameters: cost= %.5f accuracy= %.5f \n"%(test_loss,test_acc))
        # fres.write("Test set results based last pamameters: cost= %.5f accuracy= %.5f \n"%(test_loss,test_acc))

        ckpt = tf.train.get_checkpoint_state(checkpoint_maxacc_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(s, ckpt.model_checkpoint_path)
        test_vma_acc, test_vma_loss, map, mar, maf = evaluate(s, model, list(zip(test_x, test_y, test_t, test_cat)))
        print("Test set results based max-val-acc pama: cost= %.5f accuracy= %.5f precision= %.5f recall= %.5f f-score= %.5f \n"%(test_vma_loss,test_vma_acc,map,mar,maf))
        fres.write("Test set results based max-val-acc pama: cost= %.5f accuracy= %.5f precision= %.5f recall= %.5f f-score= %.5f \n" % (test_vma_loss,test_vma_acc,map,mar,maf))
        # fp = open('data/' + base + '/pre_memnn1.txt', 'w')
        # for k in range(len(tpre)):
        #     fp.write(str(tpre[k]) + '\n')
        #
        ckpt = tf.train.get_checkpoint_state(checkpoint_minloss_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(s, ckpt.model_checkpoint_path)
        # test_ml_acc, test_ml_loss, mlp, mlr, mlf, tpre, attprob = evaluate(s, model, list(zip(test_x, test_y, test_t, test_cat)))
        test_ml_acc, test_ml_loss, mlp, mlr, mlf = evaluate(s, model, list(zip(test_x, test_y, test_t, test_cat)))
        print("Test set results based min-loss-acc pama: cost= %.5f accuracy= %.5f precision= %.5f recall= %.5f f-score= %.5f \n" % (test_ml_loss, test_ml_acc, mlp, mlr, mlf))
        fres.write(
            "Test set results based min-loss-acc pama: cost= %.5f accuracy= %.5f precision= %.5f recall= %.5f f-score= %.5f \n" % (test_ml_loss, test_ml_acc, mlp, mlr, mlf))

        '''
        # print len(attprob)
        # fp = open('data/' + base + '/att_atae.txt', 'w')
        # for k in range(len(tpre)):
        #     fp.write(str(tpre[k]) + '\n')
        #     for h in range(1):
        #         for l in range(len(attprob[0])):
        #             fp.write(str(attprob[k][l][0]) + '  ')
        #         fp.write('\n')
        #     fp.write('\n')
        '''
def main():
    train()

if __name__ == '__main__':
    main()
