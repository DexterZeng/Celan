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

import tensorflow as tf
import numpy as np
import random as rdn
import tensorflow.contrib.layers as layers
from tqdm import tqdm

from numpy import *

from keras import backend as K
from keras.models import Sequential, load_model

from keras.engine import InputSpec, Model
from keras.layers.recurrent import LSTM
from keras.layers import activations, Wrapper
from keras.layers import Input,Embedding, Flatten, Dropout, Lambda, concatenate, Dense


def import_matrices():
    word_name = np.load()
    entity_name= np.load()
    word_matrix = np.load()
    en_matrix = np.load()
    
    return word_name, entity_name, word_matrix, en_matrix

def load_matrices():
    print('loading matrix......')
    entity_embedmatrix_input = open('./data/Entity_Matrix.txt', 'r')
    en_matrix = []
    entity_name = {}
    for line in entity_embedmatrix_input:
        line = line.strip().split('\t')
        idnum = int(line[0])
        entity_name[idnum] = line[1]
        embed = line[2].split(' ')
        numembed = []
        for em in embed:
            numembed.append(float(em))
        en_matrix.append(numembed)
            
    word_name = {}
    word_embedmatrix_input = open('./data/Word_Matrix.txt', 'r')
    word_matrix = []
    for wline in word_embedmatrix_input:
        wline = wline.strip().split('\t')
        idnum = int(wline[0])
        word_name[idnum] = wline[1]
        embed = wline[2].split(' ')
        numembed = []
        for em in embed:
            numembed.append(float(em))
        word_matrix.append(numembed)
    return word_name, entity_name, np.array(word_matrix,dtype=np.float32), np.array(en_matrix,dtype=np.float32)

def data_extraction(dataset, training=False):
    fh = open(dataset,'r')
    leftcontext, rightcontext, entity_embed, entity_dis, labels = [],[],[],[],[]
    for line in fh.readlines():
        strs = line.strip().split('\t')
        leftcon = strs[2].split(' ')
        leftcon = [int(x) for x in leftcon]
        leftcontext.append(leftcon)
        right = strs[3].split(' ')
        right.reverse()     # flip the right one
        rightcon = [int(x) for x in right]
        rightcontext.append(rightcon)
        entity_embed.append(int(strs[1]))
        entity_d = strs[4].split(' ')
        entity_d = [int(x) for x in entity_d]
        entity_dis.append(entity_d)
        labels.append(int(strs[0]))
    return [np.array(leftcontext), np.array(rightcontext),np.array(entity_embed),np.array(entity_dis)], np.array(labels)


def correct_data_extraction(dataset, training=False):
    fh = open(dataset,'r')
    leftcontext, rightcontext, entity_embed, entity_dis, labels = [],[],[],[],[]
    for line in fh.readlines():
        strs = line.strip().split('\t')
        if int(strs[0]) == 0:
            continue        
        leftcon = strs[2].split(' ')
        leftcon = [int(x) for x in leftcon]
        leftcontext.append(leftcon)
        right = strs[3].split(' ')
        right.reverse()
        rightcon = [int(x) for x in right]
        rightcontext.append(rightcon)
        entity_embed.append(int(strs[1]))
        entity_d = strs[4].split(' ')
        entity_d = [int(x) for x in entity_d]
        entity_dis.append(entity_d)
        labels.append(int(strs[0]))
    return [np.array(leftcontext), np.array(rightcontext),np.array(entity_embed),np.array(entity_dis)], np.array(labels)

def new_data_extraction(dataset, training=False):
    fh = open(dataset,'r')
    entity_id, negative_entity_list, leftcontext, rightcontext = [],[],[],[]
    for line in fh.readlines():
        strs = line.strip().split('\t')
        entity_id.append(int(strs[0]))
        
        negative_id = strs[1].split('*')
        negative =[int(x) for x in negative_id]
        negative_entity_list.append(negative)
        
        leftcon = strs[2].split(' ')
        leftcon = [int(x) for x in leftcon]
        leftcontext.append(leftcon)
        
        right = strs[3].split(' ')
        rightcon = [int(x) for x in right]
        rightcontext.append(rightcon)

    return np.array(entity_id), np.array(negative_entity_list), np.array(leftcontext), np.array(rightcontext)


def entity_description_extraction(entity_description_set, training=True):
    fh = open(entity_description_set,'r')
    id_to_description = {}
    for line in fh.readlines():
        strs = line.strip().split('\t')
        entity_id = int(strs[0])
        
        entity_description = strs[1].split(' ')
        entity_description =[int(x) for x in entity_description]
        id_to_description[entity_id] = entity_description

    return id_to_description
    

def data_divided(context,context_length):
    left_context = context[:,:context_length]
    right_context = context[:,context_length:]
    return left_context, right_context
        

def load_test2014():
    mentions2014 = {}
    test2014input = open('./data/2014_prepare.txt')
    for line in test2014input:
        line = line.strip().split('\t')
        if int(line[0]) not in mentions2014:
            ent = 1
        else:
            ent = mentions2014[int(line[0])]
            ent = ent + 1
        mentions2014[int(line[0])] = ent
    keys = mentions2014.keys()
    keys.sort()
    indexs = [mentions2014[key] for key in keys]
    return indexs

def load_test2013():
    mentions2013 = {}
    test2013input = open('./data/2013_prepare.txt')
    for line in test2013input:
        line = line.strip().split('\t')
        if int(line[0]) not in mentions2013:
            ent = 1
        else:
            ent = mentions2013[int(line[0])]
            ent = ent + 1
        mentions2013[int(line[0])] = ent
    keys = mentions2013.keys()
    keys.sort()
    indexs = [mentions2013[key] for key in keys]
    return indexs


def test_data_extraction(dataset, training=False):
    fh = open(dataset,'r')
    leftcontext, rightcontext, entity_embed, entity_dis, labels = [],[],[],[],[]
    for line in fh.readlines():
        strs = line.strip().split('\t')
        
        leftcon = strs[3].split(' ')
        leftcon = [int(x) for x in leftcon]
        leftcontext.append(leftcon)
        right = strs[4].split(' ')
        right.reverse()     # flip the right one
        rightcon = [int(x) for x in right]
        rightcontext.append(rightcon)
        entity_embed.append(int(strs[2]))
        entity_d = strs[5].split(' ')
        entity_d = [int(x) for x in entity_d]
        entity_dis.append(entity_d)
        labels.append(int(strs[1]))
    return [np.array(leftcontext), np.array(rightcontext),np.array(entity_embed),np.array(entity_dis)], np.array(labels)


def new_test_data_extraction(test_dataset, context_length, training=False):
    fh = open(test_dataset,'r')
    sentence_list, mention_list, leftcontext, rightcontext, entity_id, entity_dis, labels = [],[],[],[],[],[],[]
    i = 0
    j = 0
    
    tem_mention_id = -1
    tem_sentence_id = -1
    for line in fh.readlines():
        strs = line.strip().split('\t')
        if i == 0:
            sentence_list.append(1)
            tem_sentence_id = int(strs[0])
            i = i + 1
        else:
            if tem_sentence_id == int(strs[0]):
                sentence_list[-1] =sentence_list[-1] + 1
            else:
                tem_sentence_id = int(strs[0])
                sentence_list.append(1)
                
        strs = line.strip().split('\t')
        if j == 0:
            mention_list.append(1)
            tem_mention_id = int(strs[1])
            j = j + 1
        else:
            if tem_mention_id == int(strs[1]):
                mention_list[-1] =mention_list[-1] + 1
            else:
                tem_mention_id = int(strs[1])
                mention_list.append(1)             
                    
        labels.append(int(strs[2]))
        entity_id.append(int(strs[3]))
        
        leftcon = strs[4].split(' ')
        leftcon = [int(x) for x in leftcon[-1*context_length:]]
        leftcontext.append(leftcon)
        
        right = strs[5].split(' ')
        rightcon = [int(x) for x in right[-1*context_length:]]
        rightcontext.append(rightcon)
        
        entity_d = strs[6].split(' ')
        entity_d = [int(x) for x in entity_d]
        entity_dis.append(entity_d)
        
    return [np.array(leftcontext), np.array(rightcontext), np.array(entity_dis)], np.array(labels), sentence_list, mention_list

def training_data_construction(entity_id, negative_entity_list, leftcontext, rightcontext,id_to_description, num_of_neg = 1, context_length = 20):

    new_entity_id, new_label, new_leftcontext, new_rightcontext,  new_entity_dis = [],[],[],[],[]
    for i in xrange(len(entity_id)):
        tem_num_of_neg = num_of_neg
        tem_negative_entity_list = []
        if len(negative_entity_list[i]) < tem_num_of_neg:
            tem_num_of_neg = len(negative_entity_list[i])
            tem_negative_entity_list = negative_entity_list[i]
        else:
            tem_negative_entity_list = rdn.sample(negative_entity_list[i], tem_num_of_neg)
        for j in xrange(tem_num_of_neg + 1):
            if j==0:
                new_entity_id.append(entity_id[i])
                new_leftcontext.append(leftcontext[i][-1*context_length:])
                new_rightcontext.append(rightcontext[i][-1*context_length:])
                new_entity_dis.append(id_to_description[entity_id[i]])
                new_label.append(1)
            else:
                if int(negative_entity_list[i][0]) == 0:
                    break
                else:
                    
                    new_leftcontext.append(leftcontext[i][-1*context_length:])
                    new_rightcontext.append(rightcontext[i][-1*context_length:])
                    new_entity_id.append(tem_negative_entity_list[j-1])
                    new_entity_dis.append(id_to_description[tem_negative_entity_list[j-1]])
                    new_label.append(0)
    if num_of_neg != 0:
        np.save('data/origin_word_left_context.npy',np.array(new_leftcontext))
        np.save('data/origin_word_right_context.npy',np.array(new_rightcontext))
        np.save('data/origin_entity_description.npy',np.array(new_entity_dis))
        np.save('data/origin_entity_id.npy',np.array(new_entity_id))
        np.save('data/origin_label.npy',np.array(new_label))
    else:
        np.save('data/correct_word_left_context.npy',np.array(new_leftcontext))
        np.save('data/correct_word_right_context.npy',np.array(new_rightcontext))
        np.save('data/correct_entity_description.npy',np.array(new_entity_dis))
        np.save('data/correct_entity_id.npy',np.array(new_entity_id))
        np.save('data/correct_label.npy',np.array(new_label))
        
    return [np.array(new_leftcontext), np.array(new_rightcontext),np.array(new_entity_dis)], np.array(new_label)
                    
                
def features_extraction(dataset, training=False):
    fh = open(dataset,'r')
    features_list_1, features_list_2 = [],[]
    for line in fh.readlines():
        strs = line.strip().rstrip('\r\n').split('\t')
        for index, item in enumerate(strs):
            tem_item = str(item).split('/')
            if  index > 1:
                features_list_1.append(float(tem_item[1]))
                features_list_2.append(float(tem_item[2]))
        
    return np.array(features_list_1), np.array(features_list_2)


def entity_description_update(id_to_description, entity_id, entity_description):
    for index, item in enumerate(entity_id):
        id_to_description[item] = entity_description[index]
    return id_to_description


def entity_description_array_extraction(id_to_description, entity_id):
    entity_description = []
    for item in entity_id:
        entity_description.append(id_to_description[item])
    return entity_description
                
def new_training_data_construction(entity_id, negative_entity_list, leftcontext, rightcontext,id_to_description, num_of_neg = 1, context_length = 20, description_length = 50):

    new_entity_id, new_label, new_leftcontext, new_rightcontext,  new_entity_dis = [],[],[],[],[]
    for i in xrange(len(entity_id)):
        tem_num_of_neg = num_of_neg
        tem_negative_entity_list = []
        if len(negative_entity_list[i]) < tem_num_of_neg:
            tem_num_of_neg = len(negative_entity_list[i])
            tem_negative_entity_list = negative_entity_list[i]
        else:
            tem_negative_entity_list = rdn.sample(negative_entity_list[i], tem_num_of_neg)
        tem_left = training_data_context_length_constrain(leftcontext[i], context_length = context_length)
        tem_right = training_data_context_length_constrain(rightcontext[i], context_length = context_length)
        for j in xrange(tem_num_of_neg + 1):
            if j==0:
                new_entity_id.append(entity_id[i])
                new_leftcontext.append(tem_left)
                new_rightcontext.append(tem_right)
                tem_description = training_data_description_length_constrain(id_to_description[entity_id[i]], context_length = description_length)
                new_entity_dis.append(tem_description)
                new_label.append(1)
            else:
                if int(negative_entity_list[i][0]) == 0:
                    break
                else:
                    
                    new_leftcontext.append(leftcontext[i][-1*context_length:])
                    new_rightcontext.append(rightcontext[i][-1*context_length:])
                    new_entity_id.append(tem_negative_entity_list[j-1])
                    tem_description = training_data_context_length_constrain(id_to_description[tem_negative_entity_list[j-1]], context_length = description_length)
                    new_entity_dis.append(tem_description)
                    new_label.append(0)
    if num_of_neg != 0:
        np.save('data/origin_word_left_context.npy',np.array(new_leftcontext))
        np.save('data/origin_word_right_context.npy',np.array(new_rightcontext))
        np.save('data/origin_entity_description.npy',np.array(new_entity_dis))
        np.save('data/origin_entity_id.npy',np.array(new_entity_id))
        np.save('data/origin_label.npy',np.array(new_label))
    else:
        np.save('data/correct_word_left_context.npy',np.array(new_leftcontext))
        np.save('data/correct_word_right_context.npy',np.array(new_rightcontext))
        np.save('data/correct_entity_description.npy',np.array(new_entity_dis))
        np.save('data/correct_entity_id.npy',np.array(new_entity_id))
        np.save('data/correct_label.npy',np.array(new_label))
        
    return [np.array(new_leftcontext), np.array(new_rightcontext), np.array(new_entity_id), np.array(new_entity_dis)], np.array(new_label)




def new_test_data_construction(entity_id, leftcontext, rightcontext, entity_description, context_length = 20, description_length = 50):

    new_entity_id, new_label, new_leftcontext, new_rightcontext,  new_entity_dis = [],[],[],[],[]
    for i in xrange(len(entity_id)):
        tem_left = training_data_context_length_constrain(leftcontext[i], context_length = context_length)
        tem_right = training_data_context_length_constrain(rightcontext[i], context_length = context_length)
        new_entity_id.append(entity_id[i])
        new_leftcontext.append(tem_left)
        new_rightcontext.append(tem_right)
        tem_description = training_data_description_length_constrain(entity_description[i], context_length = description_length)
        new_entity_dis.append(tem_description)
        new_label.append(1)        
    return [np.array(new_leftcontext), np.array(new_rightcontext), np.array(new_entity_id), np.array(new_entity_dis)], np.array(new_label)



def training_data_context_length_constrain(context, context_length = 20):
    tem_context = filter(None, context)
    if len(tem_context) >= context_length:
        return tem_context[-1*context_length:]
    else:
        tem_context = [0] * (context_length -len(tem_context)) + tem_context
        return tem_context
    
def training_data_description_length_constrain(context, context_length = 20):
    tem_context = filter(None, context)
    if len(tem_context) >= context_length:
        return tem_context[:context_length]
    else:
        tem_context = [0] * (context_length -len(tem_context)) + tem_context
        return tem_context
