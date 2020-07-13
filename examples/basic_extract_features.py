#! -*- coding: utf-8 -*-

"""
Task:the logic about extract features
Date:2020.06.22
Author:Laiqi
"""
import os
import sys
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from datetime import datetime
sys.path.append("../tool/")
from tools import load_yaml_file
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#set use the memory
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class ExtractFeatures(object):

    def __init__(self):
        # define debug switch
        self.debug = True
        # get the config file
        self.config_dict = load_yaml_file('../config/config.yaml')
        # get the bert model path
        self.bert_model_path = self.config_dict["bert_model_path"]
        # get the config_path
        self.config_path = self.bert_model_path + "bert_config.json"
        # get the checkpoint path
        self.checkpoint_path = self.bert_model_path + "bert_model.ckpt"
        # get the dict path
        self.dict_path = self.bert_model_path + "vocab.txt"
        # 建立分词器
        self.tokenizer = Tokenizer(self.dict_path, do_lower_case=True)
        # 建立模型，加载权重
        self.model = build_transformer_model(self.config_path, self.checkpoint_path)

    def main_function(self):
        # encode test
        token_ids, segment_ids = self.tokenizer.encode(u'语言模型')
        if self.debug == True:
            print("========token_ids=========")
            print(token_ids)
            print("========segment_ids=======")
            print(segment_ids)
        print('\n ===== predicting =====\n')
        # print(self.model.predict([np.array([token_ids]), np.array([segment_ids])]))
        # print('\n =reloading and predicting=\n')
        # # save model
        # self.model.save('test.model')
        # del self.model
        # model = keras.models.load_model('test.model')
        # print(model.predict([np.array([token_ids]), np.array([segment_ids])]))

if __name__ == '__main__':
    # the start_time
    start_time = datetime.now()
    # define the object
    dataObject = ExtractFeatures()
    # call the main function
    dataObject.main_function()
    # the end_time
    end_time = datetime.now()
    use_time = end_time - start_time
    print("run time is:%ss||time:%s" % (use_time.seconds, use_time))
















