import numpy as np

def load_data_hie(x_path, y_path):
    x = []
    y = []
    for line in open(x_path, 'rb'):
        x_d = []
        sens = line.strip().split('<sssss>')
        for sen in sens:
            if sen.strip() != '':
                x_s = []
                words = sen.strip().split(' ')
                for word in words:
                    x_s.append(int(word))
                x_d.append(x_s)
        x.append(x_d)
    for line in open(y_path, 'rb'):
        label = line.strip()
        y.append(int(label) - 1)
    return np.array(x), np.array(y)

def load_data(path, task):
    x = []
    t = []
    p = []
    y = []
    c = []
    for line in open(path, 'r',encoding='utf-8'):
        x_d = []
        t_d = []
        p_d = []
        strs = line.strip().split('\t')
        words = strs[0].strip().split(' ')
        for word in words:
            x_d.append(int(word))
        x.append(x_d)
        y.append(int(strs[2]))
        if task == 'term':
            tars = strs[1].strip().split(' ')
            for tar in tars:
                t_d.append(int(tar))
            try:
                pos = strs[3].strip().split(' ')
            except:
                print(strs)
            for po in pos:
                p_d.append(int(po))
            t.append(t_d)
            p.append(p_d)
        else:
            c.append(int(strs[1]))
    if task == 'term':
        return np.array(x), np.array(y), np.array(t), np.array(p)
    else:
        return np.array(x), np.array(y), np.array(c), np.array(p)
        #x:word list, y:label, c:category, p:pos

def load_data_mt(path, task, aspnum):
    x = []
    t = []
    p = []
    y = []
    c = []
    for i in range(aspnum):
        xa = []
        ta = []
        pa = []
        ya = []
        ca = []
        for line in open(path + '_' + str(i) + '.txt', 'rb'):
            x_d = []
            t_d = []
            p_d = []
            strs = line.strip().split('\t')
            words = strs[0].split(' ')
            for word in words:
                x_d.append(int(word))
            xa.append(x_d)
            ya.append(int(strs[2]))
            if task == 'term':
                tars = strs[1].split(' ')
                for tar in tars:
                    t_d.append(int(tar))
                try:
                    pos = strs[3].strip().split(' ')
                except:
                    print(strs)
                for po in pos:
                    p_d.append(int(po))
                ta.append(t_d)
                pa.append(p_d)
            else:
                ca.append(int(strs[1]))
        x.append(xa)
        t.append(ta)
        p.append(pa)
        y.append(ya)
        c.append(ca)
    if task == 'term':
        return np.array(x), np.array(y), np.array(t), np.array(p)
    else:
        return np.array(x), np.array(y), np.array(c), np.array(p)

def load_aspcat(path, sep):
    matrix = []
    for line in open(path, 'r'):
        aspcat = []
        cats = line.strip().split(sep)
        for c in cats:
            aspcat.append(int(c))
        matrix.append(aspcat)
    return np.array(matrix)

def load_lex(path):
    matrix = []
    for line in open(path, 'r'):
        aspcat = []
        cats = line.strip().split('\t')
        for c in cats:
            aspcat.append(float(c))
        matrix.append(aspcat)
    return np.array(matrix)

def load_x(x_path):
    x = []
    for line in open(x_path, 'r'):
        x_d = []
        words = line.strip().split(' ')
        for word in words:
            x_d.append(int(word))
        x.append(x_d)
    return np.array(x)

def load_embedding2(path):
    embedding_matrix = []
    for line in open(path, 'r'):
        embedding = []
        vector = line.strip().split(' ')
        for val in vector:
            embedding.append(float(val))
        embedding_matrix.append(embedding)
    return np.array(embedding_matrix)

def batch_hie(inputs):
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = max(map(max, sentence_sizes_))

    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD
    wordmask = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)
    senmask = np.zeros(shape=[batch_size, document_size], dtype=np.int32)

    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, sentence in enumerate(document):
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            senmask[i, j] = 1
            for k, word in enumerate(sentence):
                b[i, j, k] = word
                wordmask[i, j, k] = 1
        # for j in range(len(document), document_size):
        #     wordmask[i, j, 0] = 1
    return b, document_sizes, sentence_sizes, wordmask, senmask

def batch(inputs):
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    b = np.zeros(shape=[batch_size, document_size], dtype=np.int32)  # == PAD
    wordmask = np.zeros(shape=[batch_size, document_size], dtype=np.int32)

    for i, document in enumerate(inputs):
        for j, word in enumerate(document):
            # sentence_sizes[i, j] = sentence_sizes_[i][j]
            # for k, word in enumerate(sentence):
            b[i, j] = word
            wordmask[i, j] = 1
    return b, document_sizes, wordmask

def batch_lexmask(inputs, lexpos):
    #print('lexpos',lexpos)
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    b = np.zeros(shape=[batch_size, document_size], dtype=np.int32)  # == PAD
    wordmask = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    posmask = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, word in enumerate(document):
            # sentence_sizes[i, j] = sentence_sizes_[i][j]
            # for k, word in enumerate(sentence):
            b[i, j] = word
            wordmask[i, j] = 1
            posmask[i, j] = lexpos[i][j]
    return b, document_sizes, wordmask, posmask

def batch_posmask(inputs, indexs):
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    b = np.zeros(shape=[batch_size, document_size], dtype=np.int32)  # == PAD
    wordmask = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    posmask = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, word in enumerate(document):
            # sentence_sizes[i, j] = sentence_sizes_[i][j]
            # for k, word in enumerate(sentence):
            b[i, j] = word
            wordmask[i, j] = 1
            posmask[i, j] = 1
        for index in indexs[i]:
            posmask[i, index] = 0
    return b, document_sizes, wordmask, posmask

def getWdict(wfile):
    dict = {}
    for line in open(wfile, 'r'):
        strs = line.strip().split('\t')
        dict[int(strs[1])] = strs[0]
    return dict

def batch_loc(inputs):
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    b = np.zeros(shape=[batch_size, document_size], dtype=np.float32)  # == PAD

    for i, document in enumerate(inputs):
        for j, word in enumerate(document):
            # sentence_sizes[i, j] = sentence_sizes_[i][j]
            # for k, word in enumerate(sentence):
            b[i, j] = word
    return b
