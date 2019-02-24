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
import tools
import model
from numpy import *

from keras import backend as K
from keras.models import Sequential, load_model

from keras.engine import InputSpec, Model
from keras.layers.recurrent import LSTM
from keras.layers import activations, Wrapper
from keras.layers import Input,Embedding, Flatten, Dropout, Lambda, concatenate, Dense

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    

config = OrderedDict()
config.MAX_WINDOW_SIZE = 10
config.MAX_MENTION_LENGTH = 10
config.EMBEDDING_TRAINABLE= False
config.WORD_EMBEDDING_DIM = 300 #first
#config.ENTITY_EMBEDDING_DIM = 300 #second
config.MAX_ENTITY_DESC_LENGTH = 150 #no
config.MENTION_CONTEXT_LATENT_SIZE = 50
config.LSTM_SIZE = 300
config.DROPOUT = 0.3
config.ACTIVATION_FUNCTION = 'tanh'
config.batch_size = 1024
config.num_of_neg = 1 #the number of negative sample of each senetence
config.start_epochs = 0 # the epoch of start
config.epochs = 5 #the number of iteration
config.batch_epochs = 1 #the number of batch of evaluate times

# reforcement learning config
config.updaterate = 1
config.num_epoch = 5
config.sampletimes = 1
config.negative_sample = 5

config.context_length =  config.MAX_WINDOW_SIZE + config.MAX_MENTION_LENGTH

word_index, entity_indices, embedding_word_matrix, embedding_entity_matrix = tools.load_matrices()

train_dataset = './data/input.txt'
entity_description_set = './data/entity_dis.txt'
        
entity_id, negative_entity_list, leftcontext, rightcontext = tools.new_data_extraction(train_dataset, training=True)

id_to_description = tools.entity_description_extraction(entity_description_set, training=True)

train_X, train_Y = tools.training_data_construction(entity_id, negative_entity_list, \
                                                    leftcontext, rightcontext,id_to_description, num_of_neg = config.num_of_neg, context_length = config.context_length)

test_dataset = './data/test.txt'
test_X1, test_Y1, sentence_list, mention_list = tools.new_test_data_extraction(test_dataset, config.context_length, training=False)
print len(test_Y1) #3580


features = './data/features.txt'
features_1, features_2 =tools.features_extraction(features, training=False)
print len(features_1)

indexes = OrderedDict()
indexes.word_index = word_index
indexes.entity_indices = entity_indices

output_results = open('./results/results.txt','a')

#train = model.Attention_LSTM_NoFeatures2(indexes=indexes, config=config)
train = model.Context_LSTM(indexes=indexes, config=config)

train.create(word_index, entity_indices, embedding_word_matrix)

for i in xrange(config.start_epochs, config.epochs, config.batch_epochs):
    if i != 0 :
        train.model.load_weights("./model/my_model_weights_" + str(i) + ".h5")
        
    train.model.fit(train_X,train_Y,epochs= config.batch_epochs, batch_size=config.batch_size)
    train.model.save_weights("./model/my_model_weights_" + str(i + config.batch_epochs) + ".h5")
    
    print "test set"
    output_results.write(str(i) +'\n')
    result_2013 = train.model.predict(test_X1,batch_size=200,verbose=0)
    result_2013 = np.reshape(result_2013, [-1])
    test_Y1 = np.reshape(test_Y1, [-1])
    precision_2013 = model.evaluate(result_2013, test_Y1, mention_list)
    output_results.write(precision_2013)
    print precision_2013
    
    final_result1 = np.mean([result_2013, features_1], axis = 0)
    final_result2 = np.mean([result_2013, features_2], axis = 0)
    final_result3 = np.mean([features_1, features_2], axis = 0)
    final_result = np.mean([result_2013, features_1, features_2], axis = 0)

    print len(final_result)

    test_Y1 = np.reshape(test_Y1, [-1])
    precision_2013 = model.evaluate(result_2013, test_Y1, mention_list)
    print 'context_features', precision_2013
    precision_2013 = model.evaluate(features_1, test_Y1, mention_list)
    print 'populartiy_features', precision_2013
    precision_2013 = model.evaluate(features_2, test_Y1, mention_list)
    print 'random_walk_features', precision_2013

    precision_2013 = model.evaluate(final_result1, test_Y1, mention_list)
    print 'context+popularity_features', precision_2013
    precision_2013 = model.evaluate(final_result2, test_Y1, mention_list)
    print 'context+random_walk_features', precision_2013
    precision_2013 = model.evaluate(final_result3, test_Y1, mention_list)
    print 'popularity+random_walk_features', precision_2013

    precision_2013 = model.evaluate(final_result, test_Y1, mention_list)
    print 'context+popularity+random_walk_features',  precision_2013
    
train.model.save_weights("./model/my_model_weights.h5")
output_results.close()
