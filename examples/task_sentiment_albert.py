#! -*- coding: utf-8 -*-

"""
Task:the logic about extract features
Date:2020.06.23
Author:Laiqi
"""
import os
import sys
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from keras.utils.vis_utils import plot_model
from datetime import datetime
from keras.layers import Lambda
from keras.layers import Dense
from keras.callbacks import Callback

sys.path.append("../tool/")
from tools import load_yaml_file
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#set use the memory
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class data_generator(DataGenerator):

    # 数据生成器
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            # encode data
            token_ids, segment_ids = TaskSentimentClassify().tokenizer.encode(text, max_length=128)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

class TaskSentimentClassify(object):

    def __init__(self):
        # define debug switch
        self.debug = True
        # get the config file
        self.config_dict = load_yaml_file('../config/config.yaml')
        # get the bert model path
        self.albert_model_path = self.config_dict["albert_small_google_path"]
        # get the config_path
        self.config_path = self.albert_model_path + "albert_config_small_google.json"
        # get the checkpoint path
        self.checkpoint_path = self.albert_model_path + "albert_model.ckpt"
        # get the dict path
        self.dict_path = self.albert_model_path + "vocab.txt"
        # get the data_path
        self.data_path = self.config_dict["data_path"]["sentiment"]
        # get the picture path
        self.picture_path = self.config_dict["picture_path"]
        # get the model path
        self.model_path = self.config_dict["model_path"] + "sentiment/"
        # 建立分词器
        self.tokenizer = Tokenizer(self.dict_path, do_lower_case=True)

    # step1.0: load dataset
    def load_data(self,filename):
        # define the data list
        data_list = []
        # open file
        with open(filename, encoding="utf-8") as file:
            for line in file.readlines():
                text, label = line.strip().split('\t')
                data_list.append((text, int(label)))
            # return the result
            return data_list

    # step2.0: define baseline model
    def train_model(self, train_generator, test_generator, valid_generator):
        # define the baseline model
        def baseline_model():
            # 建立模型，加载权重
            albert_model = build_transformer_model(self.config_path, self.checkpoint_path,
                                                   model='albert', return_keras_model=False)

            # get the output of cls-token
            output = Lambda(lambda x:x[:, 0], name='CLS-token')(albert_model.model.output)
            output = Dense(units=2, activation="softmax", kernel_initializer=albert_model.initializer)(output)
            # model
            model = keras.models.Model(albert_model.model.input, output)
            # 派生为带分段线性学习率的优化器。
            # 其中name参数可选，但最好填入，以区分不同的派生优化器。
            AdamLR = extend_with_piecewise_linear_lr(Adam, name="AdamLR")
            # compile model
            model.compile(optimizer=AdamLR(lr=1e-4, lr_schedule={1000: 1, 2000: 0.1}),
                          loss='sparse_categorical_crossentropy',
                          metrics=["accuracy"])
            # summarizes defined model
            model.summary()
            # plot and save model
            plot_model(model, to_file=self.picture_path + 'albert_google_model.png', show_shapes=True, show_layer_names=True)
            # return the model
            return model

        # get the baseline_model
        model = baseline_model()

        def evaluate(data):
            total, right = 0., 0.
            for x_true, y_true in data:
                y_pred = model.predict(x_true).argmax(axis=1)
                y_true = y_true[:, 0]
                total += len(y_true)
                right += (y_true == y_pred).sum()
            return right / total

        class Evaluator(Callback):

            def __init__(self):
                super(Evaluator, self).__init__()
                self.best_val_acc = 0.
                self.model_path = TaskSentimentClassify().model_path

            def on_epoch_end(self, epoch, logs=None):
                val_acc = evaluate(valid_generator)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    model.save_weights(self.model_path + 'best_model.weights')
                test_acc = evaluate(test_generator)
                print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' % (val_acc, self.best_val_acc, test_acc))

        # define the evaluate object
        evaluator = Evaluator()
        # train and save model
        model.fit_generator(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=5, callbacks=[evaluator])
        # load the model
        model.load_weights(self.model_path + 'best_model.weights')
        print(u'final test acc: %05f\n' % (evaluate(test_generator)))

    def main_function(self):
        # step1.0: load train data
        train_data_list = self.load_data(self.data_path + "sentiment.train.data")
        test_data_list = self.load_data(self.data_path + "sentiment.test.data")
        valid_data_list = self.load_data(self.data_path + "sentiment.valid.data")
        if self.debug == True:
            print("the length of train_data_list : %d" % (len(train_data_list)))
            print(train_data_list[0:3])
            print("=*="*10)
            print("the length of test_data_list : %d" %(len(test_data_list)))
            print(test_data_list[0:3])
            print("=*=" * 10)
            print("the length of valid_data_list : %d" % (len(valid_data_list)))
            print(valid_data_list[0:3])
            print("=*=" * 10)
        # step1.1: 转换数据集
        train_generator = data_generator(train_data_list, batch_size=32)
        test_generator = data_generator(test_data_list, batch_size=32)
        valid_generator = data_generator(valid_data_list, batch_size=32)
        if self.debug == True:
            print("the train_data batch is : %d" %(len(train_generator)))
            print("=*=" * 10)
            print("the test_data batch is : %d" %(len(test_generator)))
            print("=*=" * 10)
            print("the valid_data batch is : %d" %(len(valid_generator)))
        # step2.0: train baseline model
        self.train_model(train_generator, test_generator, valid_generator)

if __name__ == '__main__':
    # the start_time
    start_time = datetime.now()
    # define the object
    modelObject = TaskSentimentClassify()
    # call the main function
    modelObject.main_function()
    # the end_time
    end_time = datetime.now()
    use_time = end_time - start_time
    print("run time is:%ss||time:%s" % (use_time.seconds, use_time))
