import numpy as np

def get_pmi_dict():
    train_data = open(r'data/Sem14ResAsp/pmi/train_lex_text.txt','r')
    #print(train_data)
    word_ids = []
    category_ids = []
    for line in train_data:
        context = line.strip().split('\t')[0]
        word_ids.append(list(map(int,context.strip().split())))

        category = line.strip().split('\t')[1]
        category_ids.append(int(category))

    #print(len(word_ids))
    #print(len(category_ids))

    # word2idx_dict = {}
    # word2idx_dict['<pad>'] = 0
    # word2idx_file = open(r'data/Sem14ResAsp/pmi/word2id.txt','r')
    # for line in word2idx_file:
    #     word_in = line.split('\t')[0]
    #     idx_in = int(line.split('\t')[1])
    #     word2idx_dict[word_in] = idx_in

    with open(r'data/Sem14ResAsp/pmi/word2id.txt', 'r', encoding='utf-8') as f:
        word2idx_dict = eval(f.read())

    word_dict = {}
    sentence_cnt = len(word_ids)
    class0_dict = {}
    class1_dict = {}
    class2_dict = {}
    class3_dict = {}
    class4_dict = {}
    class5_dict = {}
    class6_dict = {}
    class7_dict = {}
    class8_dict = {}
    class9_dict = {}
    class10_dict = {}
    class11_dict = {}
    class12_dict = {}

    class0_cnt = 0
    class1_cnt = 0
    class2_cnt = 0
    class3_cnt = 0
    class4_cnt = 0
    class5_cnt = 0
    class6_cnt = 0
    class7_cnt = 0
    class8_cnt = 0
    class9_cnt = 0
    class10_cnt = 0
    class11_cnt = 0
    class12_cnt = 0
    for i in range(len(word2idx_dict)):
        word_dict[i] = 0
        class0_dict[i] = 0
        class1_dict[i] = 0
        class2_dict[i] = 0
        class3_dict[i] = 0
        class4_dict[i] = 0
        class5_dict[i] = 0
        class6_dict[i] = 0
        class7_dict[i] = 0
        class8_dict[i] = 0
        class9_dict[i] = 0
        class10_dict[i] = 0
        class11_dict[i] = 0
        class12_dict[i] = 0

    for sentence in word_ids:
        idx = word_ids.index(sentence)
        s = list(set(sentence))
        # print(s)

        if category_ids[idx] == 0:
            class0_cnt += 1
            for wordid in s:
                class0_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 1:
            class1_cnt += 1
            for wordid in s:
                class1_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 2:
            class2_cnt += 1
            for wordid in s:
                class2_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 3:
            class3_cnt += 1
            for wordid in s:
                class3_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 4:
            class4_cnt += 1
            for wordid in s:
                class4_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 5:
            class5_cnt += 1
            for wordid in s:
                class5_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 6:
            class6_cnt += 1
            for wordid in s:
                class6_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 7:
            class7_cnt += 1
            for wordid in s:
                class7_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 8:
            class8_cnt += 1
            for wordid in s:
                class8_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 9:
            class9_cnt += 1
            for wordid in s:
                class9_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 10:
            class10_cnt += 1
            for wordid in s:
                class10_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 11:
            class11_cnt += 1
            for wordid in s:
                class11_dict[wordid] += 1
                word_dict[wordid] += 1

        elif category_ids[idx] == 12:
            class12_cnt += 1
            for wordid in s:
                class12_dict[wordid] += 1
                word_dict[wordid] += 1
        else:
            print("Error in category")

    # print(class2_dict)
    # print(class0_dict, '\n', class1_dict,'\n', class2_dict,'\n', class3_dict,'\n',class4_dict,'\n')
    # print(class0_cnt, class1_cnt, class2_cnt, class3_cnt, class4_cnt)
    #print('word_dict',word_dict)
    #print('**************************************')
    #print('class0_cnt:', class0_dict[1860])
    #print('class1_cnt:', class1_dict[1860])
    #print('class2_cnt:', class2_dict[1860])
    #print('class3_cnt:', class3_dict[1860])
    #print('class4_cnt:', class4_dict[1860])

    for word in class0_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class0_dict[word] = class0_dict[word] * sentence_cnt / (word_dict[word] * class0_cnt)
    # print( class0_dict)
    for word in class1_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class1_dict[word] = class1_dict[word] * sentence_cnt / (word_dict[word] * class1_cnt)
    # print(class1_dict)
    for word in class2_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class2_dict[word] = class2_dict[word] * sentence_cnt / (word_dict[word] * class2_cnt)
    # print(class2_dict)
    for word in class3_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class3_dict[word] = class3_dict[word] * sentence_cnt / (word_dict[word] * class3_cnt)
    # print(class3_dict)
    for word in class4_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class4_dict[word] = class4_dict[word] * sentence_cnt / (word_dict[word] * class4_cnt)
    # print(class4_dict)
    for word in class5_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class5_dict[word] = class5_dict[word] * sentence_cnt / (word_dict[word] * class5_cnt)
    # print(class5_dict)
    for word in class6_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class6_dict[word] = class6_dict[word] * sentence_cnt / (word_dict[word] * class6_cnt)
    # print(class6_dict)
    for word in class7_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class7_dict[word] = class7_dict[word] * sentence_cnt / (word_dict[word] * class7_cnt)
    # print(class7_dict)
    for word in class8_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class8_dict[word] = class8_dict[word] * sentence_cnt / (word_dict[word] * class8_cnt)
    # print(class8_dict)
    for word in class9_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class9_dict[word] = class9_dict[word] * sentence_cnt / (word_dict[word] * class9_cnt)
    # print(class9_dict)
    for word in class10_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class10_dict[word] = class10_dict[word] * sentence_cnt / (word_dict[word] * class10_cnt)
    # print(class10_dict)
    for word in class11_dict:
        if (word_dict[word]) == 0:
            continue
        else:
            class11_dict[word] = class11_dict[word] * sentence_cnt / (word_dict[word] * class11_cnt)
    # print(class11_dict)
    for word in class12_dict:
        if (word_dict[word]) == 0 or class12_cnt==0:
            continue
        else:
            class12_dict[word] = class12_dict[word] * sentence_cnt / (word_dict[word] * class12_cnt)
    # print(class12_dict)

    # part1：停用词
    nouse_id = []
    stopwords = open(r'data/Sem14ResAsp/stopwords.txt','r')
    for word in stopwords:
        if word.replace('\n','') not in word2idx_dict:
            continue
        else:
            nouse_id.append(word2idx_dict[word.replace('\n','')])
    #print('stopwords',nouse_id)

    # part2：符号
    symbol_list = [',', '.', ':', '/', '?', '<', '>', ';', '"', '[', ']', '\\', '{', '}', '~', '!', '@', '#', '$', '%', '^',
                   '&', '*', '(', ')', '-', '+', '_', '=', '`']
    for word in symbol_list:
        if word not in word2idx_dict:
            continue
        else:
            # print(word)
            nouse_id.append(word2idx_dict[word])
    # print(len(nouse_id))

    # part3：所有非字母
    for word in word2idx_dict:
        if word.isalpha() == False:
            nouse_id.append(word2idx_dict[word])
    # print(len(nouse_id))

    # part4：低频词
    for word in word_dict:
        #if word_dict[word] < 10:
        if word_dict[word] < 5:
            #print(word,word_dict[word])
            nouse_id.append(word)

    #print('nouse_id',nouse_id)
    nouse_id = list(set(nouse_id))

    for id in nouse_id:
        class0_dict[id] = 0.0#class2
        class1_dict[id] = 0.0#class3
        class2_dict[id] = 0.0#class4
        class3_dict[id] = 0.0#class0
        class4_dict[id] = 0.0#class1
        class5_dict[id] = 0.0#class1
        class6_dict[id] = 0.0#class1
        class7_dict[id] = 0.0#class1
        class8_dict[id] = 0.0#class1
        class9_dict[id] = 0.0#class1
        class10_dict[id] = 0.0#class1
        class11_dict[id] = 0.0#class1
        class12_dict[id] = 0.0#class1

    # print(class2_dict)
    # max_id = max(class4_dict, key = class4_dict.get)
    # for key in word2idx_dict:
    #     if word2idx_dict[key] == max_id:
    #         print(key)

    # print('class0:',class0_dict[1014], \
    #       '\nclass1:',class1_dict[1014], \
    #       '\nclass2:',class2_dict[1014], \
    #       '\nclass3:',class3_dict[1014], \
    #       '\nclass4:',class4_dict[1014])
    pmi_dict = [class0_dict, class1_dict, class2_dict, class3_dict, class4_dict, class5_dict, class6_dict,
                class7_dict, class8_dict, class9_dict, class10_dict, class11_dict, class12_dict]
    #print('111111111111111111111111')
    with open(r'data/Sem14ResAsp/pmi/wordcnt.txt','w') as f:
        for key in word_dict:
            line = str(key) + ':\t' +str(word_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class0_dict.txt','w') as f:
        for key in class0_dict:
            line = str(key) + ':\t' +str(class0_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class1_dict.txt','w') as f:
        for key in class1_dict:
            line = str(key) + ':\t' +str(class1_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class2_dict.txt','w') as f:
        for key in class2_dict:
            line = str(key) + ':\t' +str(class2_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class3_dict.txt','w') as f:
        for key in class3_dict:
            line = str(key) + ':\t' +str(class3_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class4_dict.txt','w') as f:
        for key in class4_dict:
            line = str(key) + ':\t' +str(class4_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class5_dict.txt','w') as f:
        for key in class5_dict:
            line = str(key) + ':\t' +str(class5_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class6_dict.txt','w') as f:
        for key in class6_dict:
            line = str(key) + ':\t' +str(class6_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class7_dict.txt','w') as f:
        for key in class7_dict:
            line = str(key) + ':\t' +str(class7_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class8_dict.txt','w') as f:
        for key in class8_dict:
            line = str(key) + ':\t' +str(class8_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class9_dict.txt','w') as f:
        for key in class9_dict:
            line = str(key) + ':\t' +str(class9_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class10_dict.txt','w') as f:
        for key in class10_dict:
            line = str(key) + ':\t' +str(class10_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class11_dict.txt','w') as f:
        for key in class11_dict:
            line = str(key) + ':\t' +str(class11_dict[key]) + '\n'
            f.write(line)

    with open(r'data/Sem14ResAsp/pmi/class12_dict.txt','w') as f:
        for key in class12_dict:
            line = str(key) + ':\t' +str(class12_dict[key]) + '\n'
            f.write(line)

    #print('**************************************')
    #print('class0_pmi:', class0_dict[1860])
    #print('class1_pmi:', class1_dict[1860])
    #print('class2_pmi:', class2_dict[1860])
    #print('class3_pmi:', class3_dict[1860])
    #print('class4_pmi:', class4_dict[1860])
    #class2_dict_sorted = sorted(class2_dict.items(), key=lambda x: x[1], reverse=True)
    #print ('class2_dict_sorted',class2_dict_sorted)
    #print('total',len(class2_dict_sorted))

    return nouse_id

def get_loc_info(source_data, target_data, pmi_dict):
    class0_dict, class1_dict, class2_dict, class3_dict, class4_dict = pmi_dict
    #print(source_data)
    #print(target_data)
    loc_data = []
    cnt = 0
    for s in range(len(source_data)):
        pos_info = []
        pmi = []

        #aspect embedding
        #print(target_data[s])
        if target_data[s] == 0:
            for word in source_data[s]:
                try:
                    pmi.append(class0_dict[word])
                except KeyError:
                    pmi.append(0.0)

        elif target_data[s] == 1:
            for word in source_data[s]:
                try:
                    pmi.append(class1_dict[word])
                except KeyError:
                    pmi.append(0.0)

        elif target_data[s] == 2:
            for word in source_data[s]:
                try:
                    pmi.append(class2_dict[word])
                except KeyError:
                    pmi.append(0.0)

        elif target_data[s] == 3:
            for word in source_data[s]:
                try:
                    pmi.append(class3_dict[word])
                except KeyError:
                    pmi.append(0.0)

        elif target_data[s] == 4:
            for word in source_data[s]:
                try:
                    pmi.append(class4_dict[word])
                except KeyError:
                    pmi.append(0.0)
        else:
            print("Error in category!")

        #context embedding
        if pmi.count(max(pmi)) > 1:
            cnt += 1
            for i in range(len(pmi)):
                pos_info.append(1.0)
        else:
            central_pos = np.argmax(np.array(pmi))
            for i in range(len(pmi)):
                pos_info.append(abs(i - central_pos))
            #print(pos_info)
            pos_info = 1 - np.array(pos_info)/len(pos_info)

        #print(pos_info)

        loc_data.append(pos_info)
        #print(source_data[s])
        #print(target_data[s])
        #print(pmi)
        #print(pos_info)

    #print('total count',len(source_data))
    #print('i',i)
    #print('nouse percentage {:.2f}'.format(cnt/len(source_data)))
    return loc_data

def pmi_dict_gen():
    c0 = open(r'data/Sem14ResAsp/pmi/class0_dict.txt', 'r').readlines()#food
    c1 = open(r'data/Sem14ResAsp/pmi/class1_dict.txt', 'r').readlines()#price
    c2 = open(r'data/Sem14ResAsp/pmi/class2_dict.txt', 'r').readlines()#service
    c3 = open(r'data/Sem14ResAsp/pmi/class3_dict.txt', 'r').readlines()#ambience
    c4 = open(r'data/Sem14ResAsp/pmi/class4_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    c5 = open(r'data/Sem14ResAsp/pmi/class5_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    c6 = open(r'data/Sem14ResAsp/pmi/class6_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    c7 = open(r'data/Sem14ResAsp/pmi/class7_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    c8 = open(r'data/Sem14ResAsp/pmi/class8_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    c9 = open(r'data/Sem14ResAsp/pmi/class9_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    c10 = open(r'data/Sem14ResAsp/pmi/class10_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    c11 = open(r'data/Sem14ResAsp/pmi/class11_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    c12 = open(r'data/Sem14ResAsp/pmi/class12_dict.txt', 'r').readlines()#anecdotes/miscellaneous
    #print(c0)

    with open(r'data/Sem14ResAsp/pmi/real_pmi.txt', 'w') as f:
        f.write('0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t\n')
        for i in range(len(c0)):
            c0_pmi = np.tanh(float(c0[i].strip().split()[1]))
            c1_pmi = np.tanh(float(c1[i].strip().split()[1]))
            c2_pmi = np.tanh(float(c2[i].strip().split()[1]))
            c3_pmi = np.tanh(float(c3[i].strip().split()[1]))
            c4_pmi = np.tanh(float(c4[i].strip().split()[1]))
            c5_pmi = np.tanh(float(c5[i].strip().split()[1]))
            c6_pmi = np.tanh(float(c6[i].strip().split()[1]))
            c7_pmi = np.tanh(float(c7[i].strip().split()[1]))
            c8_pmi = np.tanh(float(c8[i].strip().split()[1]))
            c9_pmi = np.tanh(float(c9[i].strip().split()[1]))
            c10_pmi = np.tanh(float(c10[i].strip().split()[1]))
            c11_pmi = np.tanh(float(c11[i].strip().split()[1]))
            c12_pmi = np.tanh(float(c12[i].strip().split()[1]))
            line = str(c0_pmi) + '\t' + \
                   str(c1_pmi) + '\t' + \
                   str(c2_pmi) + '\t' + \
                   str(c3_pmi) + '\t' + \
                   str(c4_pmi) + '\t' + \
                   str(c5_pmi) + '\t' + \
                   str(c6_pmi) + '\t' + \
                   str(c7_pmi) + '\t' + \
                   str(c8_pmi) + '\t' + \
                   str(c9_pmi) + '\t' + \
                   str(c10_pmi) + '\t' + \
                   str(c11_pmi) + '\t' + \
                   str(c12_pmi) + '\t\n'
            f.write(line)

    # with open(r'data/Sem14ResAsp/pmi/softmax_pmi.txt', 'w') as f:
    #     f.write('0.0\t0.0\t0.0\t0.0\t0.0\t\n')
    #
    #     for i in range(len(c0)):
    #         c0_pmi = float(c0[i].strip().split()[1])
    #         c1_pmi = float(c1[i].strip().split()[1])
    #         c2_pmi = float(c2[i].strip().split()[1])
    #         c3_pmi = float(c3[i].strip().split()[1])
    #         c4_pmi = float(c4[i].strip().split()[1])
    #         list = softmax([c0_pmi, c1_pmi, c2_pmi, c3_pmi, c4_pmi])
    #         # print(list)
    #         line = ''
    #         for index in list:
    #             line += str(index) + '\t'
    #         line += '\n'
    #         f.write(line)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def mask_gen(path, nouse_ids, process = None):
    data = open(path, 'r').readlines()

    mask = ''
    for line in data:
        text = line.strip().split('\t')[0].strip().split()
        for id in text:
            if int(id) in nouse_ids:
                mask += '0' + ' '
            else:
                mask += '1' + ' '
        mask += '\n'

    with open('data/Sem14ResAsp/pmi/' + process +'_mask.txt','w') as f:
        f.write(mask)

# def aspect_onehot_gen(path, process = None):
#     data = open(path, 'r').readlines()
#
#     cates = ''
#     for line in data:
#         cate = int(line.strip().split('\t')[1])
#         if cate == 0 :
#             cates += '1\t0\t0\t0\t0\t\n'
#         elif cate == 1:
#             cates += '0\t1\t0\t0\t0\t\n'
#         elif cate == 3:
#             cates += '0\t0\t1\t0\t0\t\n'
#         elif cate == 4:
#             cates += '0\t0\t0\t1\t0\t\n'
#         elif cate == 5:
#             cates += '0\t0\t0\t0\t1\t\n'
#
#     with open('data/Sem14ResAsp/pmi/' + process +'_a18.txt','w') as f:
#         f.write(cates)


if __name__ == "__main__":
    nouse_ids = get_pmi_dict()
    print(nouse_ids)
    pmi_dict_gen()
    train_path = 'data/Sem14ResAsp/pmi/train_lex_text.txt'
    mask_gen(train_path, nouse_ids, 'train_lex')
    # aspect_onehot_gen(train_path,'train')
    #
    test_path = 'data/Sem14ResAsp/pmi/test_lex_text.txt'
    # aspect_onehot_gen(test_path,'test')
    mask_gen(test_path, nouse_ids, 'test_lex')
    #
    dev_path = 'data/Sem14ResAsp/pmi/dev_lex_text.txt'
    # aspect_onehot_gen(dev_path,'dev')
    mask_gen(dev_path, nouse_ids, 'dev_lex')