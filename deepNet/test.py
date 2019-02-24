from __future__ import absolute_import

import argparse
import json
import timeit
from collections import defaultdict
import optparse
from collections import OrderedDict
import numpy as np
import math
import random
import os
import model
import tools
from numpy import *

from keras import backend as K
from keras.models import Sequential, load_model

from keras.engine import InputSpec, Model
from keras.layers.recurrent import LSTM
from keras.layers import activations, Wrapper
from keras.layers import Input,Embedding, Flatten, Dropout, Lambda, concatenate, Dense


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    

config = OrderedDict()
config.MAX_WINDOW_SIZE = 15
config.MAX_MENTION_LENGTH = 5
config.EMBEDDING_TRAINABLE= False
config.WORD_EMBEDDING_DIM = 100 #first
config.ENTITY_EMBEDDING_DIM = 100 #second
config.MAX_ENTITY_DESC_LENGTH = 100 #no
config.MENTION_CONTEXT_LATENT_SIZE = 200
config.LSTM_SIZE = 100
config.DROPOUT = 0.3
config.ACTIVATION_FUNCTION = 'tanh'

context_length = config.MAX_WINDOW_SIZE + config.MAX_MENTION_LENGTH
batch_size = 256

start_epochs = 0
epochs = 35
batch_epochs = 5

word_index, entity_indices, word_ebd, entity_ebd = tools.load_matrices()
save_path = './model/origin_rl_model.ckpt'
save_path2 = './model/origin_rl_entity_model.ckpt' 


test_dataset_2013 = './data/2013_prepare_filled.txt'
test_X1, test_Y1 = tools.test_data_extraction(test_dataset_2013, training=False)
model.select(save_path, save_path2, word_ebd, entity_ebd,test_X1, training_able = False)

context = np.load('data/test_word_context.npy')
test_entity_list = np.load('data/test_entity_list.npy')
entity_description = np.load('data/test_entity_description.npy')

left_context, right_context = model.data_divided(context, context_length)
test_X1 = [left_context, right_context, test_entity_list, entity_description]

'''
test_dataset_2014 = './data/2014_prepare_filled.txt'
test_X2, test_Y2 = test_data_extraction(test_dataset_2014, training=False)
'''
seg_2013 = tools.load_test2013()
seg_2014 = tools.load_test2014()

indexes = OrderedDict()
indexes.word_index = word_index
indexes.entity_indices = entity_indices

output_results = open('./results/final_results.txt','w')

train = model.Attention_LSTM_NoFeatures2(indexes=indexes, config=config)
train.create(word_index, entity_indices, word_ebd, entity_ebd )
i = 70
if i == 70:
    train.model.load_weights("./model/my_model_weights_" + str(i) + ".h5") 
    print "test set"
    output_results.write(str(i) +'\n')
    result_2013 = train.model.predict(test_X1,batch_size=200,verbose=0)
    result_2013 = np.reshape(result_2013, [-1])
    test_Y1 = np.reshape(test_Y1, [-1])
    precision_2013 = model.evaluate(result_2013, test_Y1, seg_2013)
    output_results.write(precision_2013)
    print precision_2013
'''    
    result_2014 = train.model.predict(test_X2,batch_size=200,verbose=0)
    result_2014 = np.reshape(result_2014, [-1])
    test_Y2 = np.reshape(test_Y2, [-1])
    precision_2014 = evaluate(result_2014, test_Y2, seg_2014)
    output_results.write(precision_2014)
    print precision_2014
'''
output_results.close()