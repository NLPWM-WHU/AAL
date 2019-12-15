import tensorflow as tf
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import datetime
import utils
import numpy as np
import random
from sklearn import metrics
from models import *
# settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# base = 'trip'
base = 'Sem14ResAsp'
flags.DEFINE_string('model', 'AALlex', 'model')
flags.DEFINE_string('task', 'aspect', 'task') # 'aspect' or 'term'
flags.DEFINE_integer('run', 1, 'first run')
flags.DEFINE_integer('batch_size', 25, 'Batch_size')
flags.DEFINE_integer('epochs', 50, 'epochs')
flags.DEFINE_integer('classes', 3, 'class num')#情感类别
flags.DEFINE_integer('aspnum', 13, 'asp num')#方面类别
flags.DEFINE_integer('nhop', 4, 'hops num')
flags.DEFINE_integer('hidden_size', 300, 'number of hidden units')
flags.DEFINE_integer('embedding_size', 300, 'embedding_size')
flags.DEFINE_integer('word_output_size', 300, 'word_output_size')
# flags.DEFINE_string('checkpoint_maxacc', 'data/' + base + '/checkpoint_maxacc/', 'checkpoint dir')
# flags.DEFINE_string('checkpoint_minloss', 'data/' + base + '/checkpoint_minloss/', 'checkpoint dir')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('attRandomBase', 0.01, 'attRandomBase')
flags.DEFINE_float('biRandomBase', 0.01, 'biRandomBase')
flags.DEFINE_float('aspRandomBase', 0.01, 'aspRandomBase')
flags.DEFINE_float('seed', 0.05, 'random_seed')
flags.DEFINE_float('max_grad_norm', 5.0, 'max-grad-norm')
flags.DEFINE_float('dropout_keep_proba', 0.5, 'dropout_keep_proba')
flags.DEFINE_string('embedding_file', 'data/' + base + '/pmi/word_embedding.npy', 'embedding_file')
flags.DEFINE_string('traindata', 'data/' + base + '/pmi/train_lex_text.txt', 'traindata')
flags.DEFINE_string('testdata', 'data/' + base + '/pmi/test_lex_text.txt', 'testdata')
flags.DEFINE_string('devdata', 'data/' + base + '/pmi/dev_lex_text.txt', 'devdata')
flags.DEFINE_string('device', '/cpu:0', 'device')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

checkpoint_maxacc_dir = 'data/' + base + '/checkpoint_maxacc/' + FLAGS.model + '/'
checkpoint_minloss_dir = 'data/' + base + '/checkpoint_minloss/' + FLAGS.model + '/'
tflog_dir = os.path.join('data/' + base + '/', 'tflog')

def batch_iterator(datalist, batch_size, model='train'):
    datalist = np.array(datalist)
    xb = []
    yb = []
    tb = []
    pb = []
    batch_num = (len(datalist)-1)//batch_size + 1#多少个batch运行完
    if model == 'train':
        shuffle_indices = np.random.permutation(np.arange(len(datalist)))
        datalist = datalist[shuffle_indices]

    for batch in range(batch_num):
        start_index = batch * batch_size
        end_index = min((batch+1)*batch_size, len(datalist))
        batch_data =  datalist[start_index:end_index]
        for i in range(len(batch_data)):
            xb.append(batch_data[i][0])
            yb.append(batch_data[i][1])
            tb.append(batch_data[i][2])
            pb.append(batch_data[i][3])
        yield xb,yb,tb,pb
        xb, yb, tb, pb = [], [], [], [] #清空上一轮列表


def evaluate(session, model, datalist):
    predictions = []
    labels = []
    attprob = []
    lossAll = 0.0
    for x, y, t, a in batch_iterator(datalist, FLAGS.batch_size, 'val'):
        labels.extend(y)
        # pred, loss, att = session.run([model.prediction, model.loss, model.attprob], model.get_feed_data(x, t, y, a, e=None, is_training=False))
        pred, loss = session.run([model.prediction, model.loss_senti], model.get_feed_data(x, t, y, a, e=None, is_training=False))
        predictions.extend(pred)
        # attprob.extend(att)
        lossAll += loss
        # if len(labels) == FLAGS.batch_size:
        #     print(labels)
        #     print(predictions)
    n = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            n += 1
    p = metrics.precision_score(labels, predictions, average='macro')
    r = metrics.recall_score(labels, predictions, average='macro')
    f = metrics.f1_score(labels, predictions, average='macro')
    # p = 0.0
    # r = 0.0
    # f = 0.0
    # met = metrics.precision_recall_fscore_support(labels, predictions, average='macro')
    # return float(n)/float(len(labels)), lossAll, p, r, f, predictions, attprob
    return float(n)/float(len(labels)), lossAll, p, r, f

def train():
    # shutil.rmtree(checkpoint_minloss_dir)
    # shutil.rmtree(checkpoint_maxacc_dir)
    word_embedding = np.load(FLAGS.embedding_file)
    train_x, train_y, train_t, train_p = utils.load_data(FLAGS.traindata, FLAGS.task)
    dev_x, dev_y, dev_t, dev_p = utils.load_data(FLAGS.devdata, FLAGS.task)
    test_x, test_y, test_t, test_p = utils.load_data(FLAGS.testdata, FLAGS.task)
    train_cat = utils.load_aspcat('data/' + base + '/pmi/train_lex_mask.txt', ' ')
    dev_cat = utils.load_aspcat('data/' + base + '/pmi/dev_lex_mask.txt', ' ')
    test_cat = utils.load_aspcat('data/' + base + '/pmi/test_lex_mask.txt', ' ')
    #lex = utils.load_lex('data/' + base + '/lex.txt')
    lex = utils.load_lex('data/' + base + '/pmi/real_pmi.txt')
    #print('lex',lex)
    tf.reset_default_graph()
    fres = open('data/' + base + '/res_' + FLAGS.model + '.txt','a')
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as s:

        model = AALlex(FLAGS, word_embedding, lex, s)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

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
        for i in range(FLAGS.epochs):
            trainloss_epoch = 0.0
            print("Epoch:", '%04d' % (i + 1))
            fres.write('\n' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            fres.write("Epoch: %02d \n" % (i + 1))
            for j, (x, y, t, a) in enumerate(batch_iterator(list(zip(train_x, train_y, train_t, train_cat)), FLAGS.batch_size, 'train')):
                # #################my test###################
                # wdict = {}
                # with open('data/Sem14ResAsp/wdict.txt','r') as f:
                #     wordlines = f.readlines()
                #     for l in wordlines:
                #         wdict[int(l.strip().split('\t')[1])] = l.strip().split('\t')[0]
                # #print('wdict',wdict)
                # words = []
                # for ele in x:
                #     word_ele = []
                #     for w in ele:
                #         word_ele.append(wdict[w])
                #     words.append(word_ele)
                # for i in range(len(x)):
                #     list_ele = []
                #     for j in range(len(a[i])):
                #         list_ele.append(words[i][j])
                #         list_ele.append(a[i][j])
                #     #print('sentence', list_ele)
                # ###########################################
                t0 = time.time()
                fd = model.get_feed_data(x, t, y, a, e=None)
                test1,test2,\
                step, labels, prediction, loss, losssenti, lossasp, accuracy, _, = s.run([
                    model.TEST1,model.TEST2,#
                    model.global_step,
                    model.labels,
                    model.prediction,
                    model.loss,
                    model.loss_senti,
                    model.loss_c,
                    model.accuracy,
                    model.train_op,
                ], fd)
                #print('TEST1', test1.shape, '\n',test1)
                #print('TEST2', test2.shape, '\n',test2)
                # print hid
                stepnum = stepnum  + 1
                # print labels
                # print prediction
                trainloss_epoch += loss
                # if stepnum % 100 == 0:
                # print("Step:", '%05d' % stepnum, "train_loss=", "{:.5f}".format(loss),
                #       "train_loss_senti=", "{:.5f}".format(losssenti),"train_loss_asp=", "{:.5f}".format(lossasp),
                #       "train_acc=", "{:.5f}".format(accuracy),"time=", "{:.5f}".format(time.time() - t0))
                # if stepnum % (64000/FLAGS.batch_size) == 0:
            # if stepnum % (2500 / FLAGS.batch_size) == 0:
            if True:
                valnum += 1
                valacc, valloss, vp, vr, vf = evaluate(s, model, list(zip(dev_x, dev_y, dev_t, dev_cat)))
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

                test_ml_acc, test_ml_loss, mlp, mlr, mlf = evaluate(s, model,list(zip(test_x, test_y, test_t, test_cat)))
                print("Test set results based min-loss-acc pama: cost= %.5f accuracy= %.5f precision= %.5f recall= %.5f f-score= %.5f \n" % (
                    test_ml_loss, test_ml_acc, mlp, mlr, mlf))

                if valnum > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                    is_earlystopping = True
                    print("Early stopping...")
                    fres.write("Early stopping...\n")
                    break
            #print('EPOCH_TEST1', test1.shape, '\n', test1[0])
            #print('EPOCH_TEST2', test2.shape, '\n', test2[0])
            print("Epoch:", '%04d' % (i + 1), " train_loss=", "{:.5f}".format(trainloss_epoch))
            fres.write("Epoch: %06d train_loss= %.5f \n" % ((i + 1), trainloss_epoch))
            #break
            if is_earlystopping:
                break
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

        ckpt = tf.train.get_checkpoint_state(checkpoint_minloss_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(s, ckpt.model_checkpoint_path)
        # test_ml_acc, test_ml_loss, mlp, mlr, mlf, tpre, attprob = evaluate(s, model, list(zip(test_x, test_y, test_t, test_cat)))
        test_ml_acc, test_ml_loss, mlp, mlr, mlf = evaluate(s, model, list(zip(test_x, test_y, test_t, test_cat)))
        print("Test set results based min-loss-acc pama: cost= %.5f accuracy= %.5f precision= %.5f recall= %.5f f-score= %.5f \n" % (test_ml_loss, test_ml_acc, mlp, mlr, mlf))
        fres.write(
            "Test set results based min-loss-acc pama: cost= %.5f accuracy= %.5f precision= %.5f recall= %.5f f-score= %.5f \n" % (test_ml_loss, test_ml_acc, mlp, mlr, mlf))
        # print len(attprob)
        # fp = open('data/' + base + '/att_atae.txt', 'w')
        # for k in range(len(tpre)):
        #     fp.write(str(tpre[k]) + '\n')
        #     for h in range(1):
        #         for l in range(len(attprob[0])):
        #             fp.write(str(attprob[k][l][0]) + '  ')
        #         fp.write('\n')
        #     fp.write('\n')

        # ckpt = tf.train.get_checkpoint_state(checkpoint_minloss_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(s, ckpt.model_checkpoint_path)
        # aspect_emb, vocab_emb = s.run([model.aspect_embedding, model.embedding_matrix])
        # print type(aspect_emb)
        # print aspect_emb
        # sim = np.dot(aspect_emb, vocab_emb.T)
        # indexs = np.argsort(-sim, axis=1)
        # wdict = utils.getWdict('data/' + base + '/wdict.txt')
        # for i in range(len(indexs)):
        #     print "aspect " + str(i)
        #     words = ''
        #     for j in range(20):
        #         index = indexs[i][j]
        #         words = words + wdict[index] + '  '
        #     print words
        # for i in range(len(prediction)):
        #     fres.write(str(prediction[i]) + '\n')
        # food = word_embedding[181]
        # sim = np.dot(np.expand_dims(food, axis=0), word_embedding.T)
        # indexs = np.argsort(-sim, axis=1)
        # words = ''
        # for j in range(20):
        #     index = indexs[0][j]
        #     words = words + wdict[index] + '  '
        # print words
        # shutil.rmtree(checkpoint_minloss_dir)
        # shutil.rmtree(checkpoint_maxacc_dir)
        '''
        # if FLAGS.run == 0:
        #     shutil.rmtree(checkpoint_minloss_dir)
        #     shutil.rmtree(checkpoint_maxacc_dir)
        '''
def main():
    train()

if __name__ == '__main__':
    main()
    # acc 0.83762  0.85098
