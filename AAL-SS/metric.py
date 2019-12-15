# -*- coding: utf-8 -*-
import cPickle as pkl
import numpy as np
import random
import xml.etree.ElementTree as ET
from collections import Counter
from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def sort_by_sen_length(trainset, sortedtrain):
    linelist = []
    for line in open(trainset, 'r'):
        linelist.append(line)
    lens = map(lambda x: len(x.split('\t')[0].split(' ')), linelist)
    index_len = zip(range(len(linelist)), lens)
    sortedlist = sorted(index_len, key=lambda item: item[1])
    f = open(sortedtrain, 'w')
    for i in range(len(sortedlist)):
        f.write(linelist[sortedlist[i][0]])
    f.close()


def gene_corpus(setlist, corpus):
    f = open(corpus, 'w')
    for set in setlist:
        for line in open(set, 'r'):
            strs = line.strip().split('\t\t')
            doc = strs[3]
            sens = doc.split('<sssss>')
            for word in sens:
                f.write(word)
            f.write('\n')
    f.close()

def get_idsen_dict(idfile, senfile):
    idsen = {}
    with open(idfile, 'r') as f1, open(senfile, 'r') as f2:
        for line1 in f1:
            line2 = f2.readline()
            idsen[line1.strip()] = line2.strip()
    return idsen


def get_sorted_xy(path, classnum, idsen, xfile, yfile):
    idlabel = {}
    idlen = {}
    for i in range(classnum):
        for line in open(path + str(i) + '.txt'):
            id = line.strip()
            idlabel[id] = str(i)
            sen = idsen[id]
            strs = sen.split(' ')
            idlen[id] = len(strs)
    list = sorted(idlen.items(), key=lambda item: item[1])
    fx = open(xfile, 'w')
    fy = open(yfile, 'w')
    for i in range(len(list)):
        fx.write(idsen[list[i][0]] + '\n')
        fy.write(idlabel[list[i][0]] + '\n')
    fx.close()
    fy.close()

def getDict(listfile, key):
    lines = map(lambda x : x.strip(), open(listfile).readlines())
    size = len(lines)
    if key == 0:
        d = dict([(item[0], item[1]) for item in zip(lines, xrange(size))])
    else:
        d = dict([(item[1], item[0]) for item in zip(lines, xrange(size))])
    return d

def w2id(w, wdict):
    if w == 'null':
        return '0'
    else:
        try:
            return wdict[w]
        except:
            return -1

def readsw(swfile):
    stopwordlist = []
    for line in open(swfile, 'r'):
        word = line.strip()
        stopwordlist.append(word)
    stopwordlist = set(stopwordlist)
    return stopwordlist

def calPMI(corpus, wdict, swfile, lex, lexword):
    swlist = readsw(swfile)
    lines = map(lambda x: x.strip().split('\t'), open(wdict).readlines())
    wd = dict([(line[0], line[1]) for line in lines])
    rwd = dict([(line[1], line[0]) for line in lines])
    coMat = np.zeros((5, len(rwd) + 1), dtype=np.float32)
    cateNum = np.zeros((5), dtype=np.float32)
    wordNum = np.zeros((len(rwd) + 1), dtype=np.float32)
    # for line in open(corpus, 'r'):
    #     strs = line.strip().split('\t')
    #     review = strs[0]
    #     cate = int(strs[1])
    #     cateNum[cate] += 1.0
    #     words = review.split(' ')
    #     for w in words:
    #         coMat[cate][int(w)] += 1.0
    #         wordNum[int(w)] += 1.0
    for line in open(corpus, 'r'):
        strs = line.strip().split('\t')
        review = strs[0]
        cates = strs[1].strip().split(' ')
        for cate in cates:
            cateNum[int(cate)] += 1.0
        words = review.split(' ')
        for w in words:
            wordNum[int(w)] += 1.0
            for cate in cates:
                coMat[int(cate)][int(w)] += 1.0

    de = int(wd['fancy'])
    ours = int(wd['soft'])
    print coMat[1][de], wordNum[de]
    # print coMat[1][ours], wordNum[ours]
    for i in range(5):
        for j in range(1, len(rwd) + 1):
            coMat[i][j] /= cateNum[i]
            coMat[i][j] /= wordNum[j]
    # coMat = np.log(coMat)
    # coMat = np.exp(coMat)
    for j in range(1, len(rwd) + 1):
        sum = 0
        for i in range(5):
            sum += coMat[i][j]
        for i in range(5):
            coMat[i][j] /= sum
    print coMat[0][de], coMat[1][de], coMat[2][de], coMat[3][de], coMat[4][de]

    # indexs = np.argsort(-coMat, axis=1)
    # for i in range(len(indexs)):
    #     print "aspect " + str(i)
    #     wo = ''
    #     for j in range(4000):
    #         index = indexs[i][j]
    #         if wordNum[index] > 4.0:
    #             if rwd[str(index)] not in swlist:
    #                 wo = wo + rwd[str(index)] + '  '
    #     print wo
    #
    fl = open(lex, 'w')
    fw = open(lexword, 'w')
    count = 0
    for j in range(0, len(rwd) + 1):
        count = 0
        if wordNum[j] > 4.0 and rwd[str(j)] not in swlist:
            flag = -1
            tval = 0.4
            for i in range(5):
                if coMat[i][j] > tval:
                    flag = i
                    tval = coMat[i][j]
            if flag == -1:
                fl.write('1.0' + '\t' + '0.0' + '\t' + '0.0' + '\t' + '0.0' + '\t' + '0.0' + '\t' + '\n')
            else:
                for i in range(5):

                    if i == flag:
                        fl.write('1.0' + '\t')
                        count += 1
                    else:
                        fl.write('0.0' + '\t')
                fl.write('\n')
        else:
            fl.write('1.0' + '\t' + '0.0' + '\t' + '0.0' + '\t' + '0.0' + '\t' + '0.0' + '\t' + '\n')
        if count > 0:
            fw.write(str(j) + '\n')
        if count > 1:
            print count, rwd[str(j)]
    fw.close()
    fl.close()

def geneAspCat(trainfile, aspcat):
    fw = open(aspcat, 'w')
    res = []
    cat = [0] * 5
    temp = ''
    num = 0
    for line in open(trainfile, 'r'):
        strs = line.strip().split('\t')
        review = strs[0]
        c = strs[1]
        if num == 0:
            temp = review
            num += 1
            cat[int(c)] = 1
        else:
            if review == temp:
                num += 1
                cat[int(c)] = 1
            else:
                for i in range(num):
                    for j in range(len(cat)):
                        fw.write(str(cat[j]) + '\t')
                    fw.write('\n')
                for j in range(len(cat)):
                    cat[j] = 0
                temp = review
                cat[int(c)] = 1
                num = 1
    for i in range(num):
        for j in range(len(cat)):
            fw.write(str(cat[j]) + '\t')
        fw.write('\n')
    fw.close()

def countWfre(corpus, wfre):
    wd = {}
    wid = 1
    wordNum = np.zeros(5000, dtype=np.int8)
    for line in open(corpus, 'r'):
        strs = line.strip().split('<sss>')
        tars = strs[1].strip().split(' ')
        for tar in tars:
            if tar in wd:
                wordNum[wd[tar]] += 1
            else:
                wd[tar] = wid
                wordNum[wd[tar]] += 1
                wid += 1
    fw = open(wfre, 'w')
    indexs = np.argsort(-wordNum)
    for i in range(len(indexs) - 1):
        ind = indexs[i]
        fw.write(str(ind) + '\t' + str(wordNum[ind]) + '\n')
    fw.close()

def findCase(testlabel, m1, m2):
    truth = []
    pre1 = []
    pre2 = []
    for line in open(testlabel, 'r'):
        strs = line.strip().split('\t')
        review = strs[0]
        senti = strs[2]
        truth.append(int(senti))
    for line in open(m1, 'r'):
        pre1.append(int(line.strip()))
    for line in open(m2, 'r'):
        pre2.append(int(line.strip()))
    for i in range(len(truth)):
        if pre1[i] != pre2[i]:
            if pre1[i] == truth[i]:
                print str(i+1)


def mergeCor(corpus, newcor):
    f = open(newcor, 'w')
    temp = ''
    for line in open(corpus, 'r'):
        strs = line.strip().split('\t')
        review = strs[0]
        cate = strs[1]
        if review != temp:
            f.write('\n')
            f.write(review + '\t' + cate + ' ')
            temp = review
        else:
            f.write(cate + ' ')
    f.close()

def getLexWordPosition(train, tp, lexw):
    lw = readsw(lexw)
    f = open(tp, 'w')
    for line in open(train, 'r'):
        strs = line.strip().split('\t')
        review = strs[0]
        words = review.split(' ')
        for w in words:
            if w not in lw:
                f.write('0' + ' ')
            else:
                f.write('1' + ' ')
        f.write('\n')
    f.close()

base = 'data/Sem14ResAsp/'
calPMI(base + 'corpusnew.txt', base + 'wdict.txt', base + 'stopwords.txt', base + 'lex.txt', base + 'lexword.txt')
# getLexWordPosition(base + 'dev.txt', base + 'dev_p.txt', base + 'lexword.txt')
# geneAspCat(base + 'dev.txt', base + 'dev_a.txt')
# countWfre(base + 'lap_alltrain.txt', base + 'wfre.txt')
# findCase(base + 'test.txt', base + 'pre_m2.txt', base + 'pre_memnn2.txt')
# mergeCor(base + 'corpus.txt', base + 'corpusnew.txt')
# wemb = np.load(base + 'embedding.npy')
# arr = np.zeros((5, 300), dtype=np.float32)
# arr[0] = wemb[293]
# arr[1] = wemb[569]
# arr[2] = wemb[181]
# arr[3] = wemb[3124]
# arr[4] = wemb[0]
# np.save(base + 'aspectembedding.npy', arr)