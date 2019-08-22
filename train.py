import pandas as pd
import numpy as np
import yaml
from data_helper import *
from sent_simtf import SentenceSimilarity
import tensorflow as tf
import json
import os
import itertools
import shutil
from tensorflow import set_random_seed
# from dpcnn import DPCNN

set_random_seed(2018)
np.random.seed(2018)


def main():
    with open('./config.yml', encoding='utf-8') as f:
        config = yaml.load(f)
    learning_rate = config['model_params']['learning_rate']
    max_len = config['max_len']
    # num_classes = config['model_params']['num_classes']  
    batch_size = config['model_params']['batch_size']
    min_count = config['data_params']['min_count']
    data_path = config['data_params']['data_path']
    val_path = config['data_params']['val_path']

    kernel_size = config['model_params']['kernel_size']

    num_filters = config['model_params']['num_filters']

    embedding_size = config['model_params']['embedding_size']
    hidden_size = config['model_params']['hidden_size']
    dropout_keep_prob = config['model_params']['keep_prob']
    epochs = config['model_params']['epochs']
    embedding_path = config['embedding_path']
    gpu_id = config['gpu_id']

    # if os.path.exists(config['ckpt_file']):
    #   shutil.rmtree(config['ckpt_file'])
    #  os.mkdir(config['ckpt_file'])

    # else:
    # os.mkdir(config['ckpt_file'])

    if not os.path.exists(config['ckpt_file']):
        os.mkdir(config['ckpt_file'])

    # 加载数据

    data, word2id = load_data(data_path, config['ckpt_file'], min_count=min_count)
    # data = pd.read_csv(data_path,delimiter='\t',header=None)

    vocab_size = len(word2id)
    # val
    # val = pd.read_csv(val_path,delimiter='\t',header=None)
    val, val_word2id = load_data(val_path, config['ckpt_file'], min_count=5, char=True, write_vocab=False)
    # 对句子进行编码
    train_id = pd.Series(list(map(lambda x: string2id(x, word2id, char=True), data[1])))
    data[2] = train_id
    val_id = pd.Series(list(map(lambda x: string2id(x, word2id, char=True), val[1])))
    val[2] = val_id

    ##
    train_data = data
    train_data = train_data.sample(frac=1)
    x_train = np.array(list(train_data[2]))

    y_train = list(train_data[0])
    num_classes = len(set(y_train))
    print('num_classes', num_classes)
    ####
    os.environ['CUDA_VISIBLE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    with tf.Session() as sess:
        ss = SentenceSimilarity(max_len, embedding_size, vocab_size, hidden_size, num_classes, learning_rate)
        # ss = DPCNN(vocab_size, kernel_size, num_filters, max_len, embedding_size,num_classes,learning_rate)

        saver = tf.train.Saver()

        if os.path.exists(config['ckpt_file'] + 'checkpoint'):
            print("Restore Variables from checkpoint!")
            saver.restore(sess, tf.train.latest_checkpoint(config['ckpt_file']))

        else:
            print("Initializer variables!")
            sess.run(tf.global_variables_initializer())
        accs = {'top1': [], 'top3': []}
        if config['use_embedding']:
            embedding = pretrained_embedding(embedding_path, word2id, embedding_size)
            assign_Embedding = tf.assign(ss.Embedding, embedding)
            sess.run(assign_Embedding)

        highest, boundary = 0.0, 0.65
        for epoch in range(epochs):
            counter, loss, total_num, total_correct = 0.0, 0.0, 0.0, 0.0
            for x_batch, y_batch, text_len in batch_buckets(x_train, y_train, batch_size, buckets_len=[15, 20, 40, 80]):
                # print("x_batch",x_batch)
                # print('y_batch',y_batch)

                counter += 1
                # print('counter',counter)

                feed_dict = {ss.input_x: x_batch, ss.input_y: y_batch, ss.x_lens: text_len,
                             ss.dropout_keep_prob: dropout_keep_prob, ss.batch_size: len(x_batch)}
                cur_loss, _, accuracy = sess.run([ss.loss, ss.train_op, ss.accuracy], feed_dict=feed_dict)
                # cur_loss,_,accuracy,cos_value,idxs,cos = sess.run([ss.loss,ss.train_op,ss.accuracy,ss.y_true_pred,ss.idxs,ss.cosine],
                # feed_dict=feed_dict)

                # print('y_batch:',y_batch[0:5])

                # print('cos shape:',cos.shape)
                # print('idxs:',idxs.shape)
                # print('cos_value:',cos_value[0:5])
                # print('idxs_value:',idxs[0:5])
                # print('cos:',cos[0:5])

                loss += cur_loss * len(x_batch)
                total_num += len(x_batch)
                total_correct += accuracy * len(x_batch)

                if counter % 100 == 0:
                    # print("cur_loss",cur_loss)
                    # print("accuracy",accuracy)
                    top1_val, top3_val, top1_threshold, top3_threshold = do_eval(sess, ss, val, batch_size, boundary)
                    print('Epoch %d/%d\tbatch %d\ttrain_loss:%.3f\ttrain_accuracy:%.3f' % (
                        epoch, epochs, counter, loss / total_num, total_correct / total_num))

                    accs['top1'].append(top1_val)
                    accs['top3'].append(top3_val)
                    if top1_val >= highest:
                        highest = top1_val
                        save_file = config['ckpt_file'] + 'model.ckpt'
                        saver.save(sess, save_file, global_step=epoch)
                        print('find new model,top1_val: %s, top3_val: %s, top1大于 %s: %s, top3 大于%s' % (
                        top1_val, top3_val, boundary, top1_threshold, top3_threshold))

        json.dump({'accs': accs, 'highest_top1': highest}, open('valid_amsoftmax.log', 'w'), indent=4)


def do_eval(sess, model, val, batch_size, thre):  # cosine [Batch,num_class],在这里num_class为batch,[Batch,Batch]
    id2g = dict(zip(val.index - val.index[0], val[0]))
    text_len = list(map(lambda x: len(x), val[2]))
    maxlen = max(text_len)

    # print(maxlen)
    def func(sent, maxlen):
        _ = sent[:maxlen] + [0] * (maxlen - len(sent))
        return _

    token_ids = list(map(lambda x: func(x, maxlen), val[2]))

    # top1_num,top3_num = 0.0,0.0
    feed_dict = {model.input_x: token_ids, model.x_lens: text_len, model.dropout_keep_prob: 1.0}
    rnn_outputs = sess.run(model.outputs_rnn, feed_dict=feed_dict)  # [Batch,hidden_size]

    # print(rnn_outputs[0:5])
    cosine = np.matmul(rnn_outputs, np.transpose(rnn_outputs))  # [Batch,Batch]
    # print('cosine shape',cosine[0:5])
    max_index = np.array(list(map(lambda x: np.argsort(x)[::-1][:4], cosine)))

    top_cosine = np.array(list(map(lambda x: np.sort(x)[::-1][:4], cosine)))

    new_result = np.vectorize(lambda x: id2g[x])(max_index)  # 转换组别

    threshold = np.full((max_index.shape[0],), thre)

    # print(threshold)

    # top = top_cosine[:,1]

    _ = new_result[:, 0] != new_result[:, 0]
    _t = new_result[:, 0] != new_result[:, 0]
    for i in range(3):
        mask = new_result[:, 0] == new_result[:, i + 1]
        top = top_cosine[:, i + 1]
        # print(top[0:10])

        real_correct = (top * mask >= threshold)
        _t += real_correct
        _ = _ + mask

        if i + 1 == 1:
            top1_acc = 1. * _.sum() / len(_)
            top1_threshold = _t.sum() / len(_t)
        elif i + 1 == 3:
            top3_acc = 1. * _.sum() / len(_)
            top3_threshold = _t.sum() / len(_t)

    return top1_acc, top3_acc, top1_threshold, top3_threshold


if __name__ == '__main__':
    main()
