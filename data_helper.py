import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict


def load_data(path, model_dir,min_count=5,char=True, write_vocab=True):
    data = pd.read_csv(path, sep='\t', header=None)
    sent_len = list(map(lambda x: len(str(x)), data[1]))
    sent_len = np.array(sent_len)
    print(np.argmax(sent_len))

    print(data.ix[np.argmax(sent_len),:])
    print("the length of the longest sentence is %s" % np.max(sent_len))
    print("the length of 75%% sentence is %s" % np.percentile(sent_len, 75))
    print("the mean length is %s" % np.mean(sent_len))
    i= 0
    words = {}
    if char:
        for sent in tqdm(iter(data[1])):
            # for char in sent.replace('？', '').replace('，', '').replace('。', ''):
            i += 1
            if type(sent)== float:
                print(i)
                print(sent)
            for char in sent:
                if char not in words:
                    words[char] = 0
                words[char] += 1
    else:
        for sent in tqdm(iter(data[1])):
            for word in sent.split(' '):
                if word not in words:
                    words[word] = 0
                words[word] += 1
    words = {i: j for i, j in words.items() if j > min_count}
    word2id = {j: i + 2 for i, j in enumerate(words)}
    if write_vocab:
        write_path = model_dir + '/word2ix.txt'
        with open(write_path, 'w', encoding='utf-8') as f:
            for word in word2id:
                f.write(word + ';' + str(word2id[word]) + '\n')

    # id2word = {j:i for i,j in word2id.items()}
    return data, word2id


def string2id(sent, word2id, char=True):
    if char:
        _ = [word2id.get(i, 1) for i in sent]
    else:
        _ = [word2id.get(i, 1) for i in sent.split(' ')]
    return _


def batch_iter(x, y, batch_size,maxlen):
    def func(sent, maxlen):
        _ = sent[:maxlen] + [0] * (maxlen - len(sent))
        return _
    x = list(map(lambda x: func(x, maxlen), x))
    data = list(zip(x, y))
    if len(data) / batch_size == 0:
        numbatches = int(len(data) / batch_size)
    else:
        numbatches = int(len(data) / batch_size) + 1
    for i in range(numbatches):
        end_index = min((i + 1) * batch_size, len(data))
        batch = data[i * batch_size:end_index]
        t = zip(*batch)
        yield t


def pretrained_embedding(embedding_path, vocabulary_word2index, embedding_size):
    print("create embedding martrix. embedding_path:", embedding_path)
    with open(embedding_path, 'r', encoding='utf-8') as f:
        model = json.load(f)

    embedding = np.random.uniform(-1.0, 1.0, (len(vocabulary_word2index) + 2, embedding_size))
    for word in vocabulary_word2index:
        try:
            embedding[vocabulary_word2index[word]] = model[word]
        except:
            # print(word)
            pass
    print("embedding_shape:", np.shape(embedding))
    print("embedding_0:", embedding[0])

    return embedding


def padding(data):
    token_ids, label_ids, text_lens = list(zip(*data))
    maxlen = max(text_lens)

    def func(sent, maxlen):
        _ = sent[:maxlen] + [0] * (maxlen - len(sent))
        return _

    token_ids = list(map(lambda x: func(x, maxlen), token_ids))

    return token_ids, label_ids, text_lens


def batch_buckets(x_train, y_train, batch_size, buckets_len=[10, 15, 20, 40]):
    data = list(zip(x_train, y_train))
    buckets = defaultdict(list)

    i = 0
    while True:
        try:
            token_id, label_id = data[i]
            i += 1
            for len_ in buckets_len:
                if len(token_id) <= len_:
                    buckets[len_].append((token_id, label_id, len(token_id)))
                    if len(buckets[len_]) == batch_size:
                        yield padding(buckets[len_])
                        buckets[len_] = []
                    break
            if len(token_id) > buckets_len[-1]:
                bucket = buckets[buckets_len[-1]]
                bucket.append((token_id[:buckets_len[-1]], label_id, len(token_id[:buckets_len[-1]])))
                if len(bucket) == batch_size:
                    yield padding(bucket)
                    buckets[buckets_len[-1]] = []

        except:
            for item in buckets_len:
                if len(buckets[item]) > 0:
                    yield padding(buckets[item])
            break
