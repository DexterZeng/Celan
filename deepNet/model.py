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
import tools
import tqdm
import os

import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm

from numpy import *

from keras import backend as K
from keras.models import Sequential, load_model

from keras.engine import InputSpec, Model
from keras.layers.recurrent import LSTM
from keras.layers import activations, Wrapper
from keras.layers import Input,Embedding, Flatten, Dropout, Lambda, concatenate, Dense

class environment():

    def __init__(self,sentence_len):
        self.sentence_len = sentence_len


    def reset(self,id_word,id_entity, batch_reward):
        self.id_word = id_word
        self.id_entity = id_entity
        self.batch_reward = batch_reward
        self.current_step = 0
        self.num_selected = 0
        self.list_selected = []
#        self.vector_current_word = self.word_ebd[self.current_step]
#        self.vector_mean = np.array([0.0 for x in range(self.sentence_len)],dtype=np.float32)
#        self.vector_sum = np.array([0.0 for x in range(self.sentence_len)],dtype=np.float32)

        current_state = [self.id_word,self.id_entity]
        return current_state


    def step(self,action): ## need update to nest sentence
        if action == 1:
            self.num_selected +=1
            self.list_selected.append(self.current_step)

#        self.vector_sum =self.vector_sum + action * self.vector_current_word
#        if self.num_selected == 0:
#            self.vector_mean = np.array([0.0 for x in range(self.sentence_len)],dtype=np.float32)
#        else:
#            self.vector_mean = self.vector_sum / self.num_selected

        self.current_step +=1

        if (self.current_step < self.batch_len):
            self.vector_current_word = self.word_ebd[self.current_step]
            
        current_state = [self.id_word,self.id_entity]

#        current_state = [self.vector_current_word,self.id_entity]
        return current_state

    def reward(self):
        assert (len(self.list_selected) == self.num_selected)
        reward = [self.batch_reward[x] for x in self.list_selected]
        reward = np.array(reward)
        reward = np.mean(reward)
        return reward


def get_action(prob):
    tmp = []
    for item in prob:
        result = np.random.rand()
        if result>0 and result< item:
            tmp.append(1)
        elif result >=item and result<1:
            tmp.append(0)
    return tmp


def get_batch_action(prob):
    batch_action = []
    for sub_prob in prob:
        tmp = []
        for item in sub_prob:
            result = np.random.rand()
            if result>0 and result< item:
                tmp.append(1)
            elif result >=item and result<1:
                tmp.append(0)
        batch_action.append(tmp)
    return batch_action

def decide_action(prob):
    tmp = []
    for item in prob:
        if item>=0.5:
            tmp.append(1)
        elif item < 0.5:
            tmp.append(0)
    return tmp


def decide_batch_action(prob):
    batch_action = []
    for sub_prob in prob:
        tmp = []
        for item in sub_prob:
            if item>=0.5:
                tmp.append(1)
            elif item < 0.5:
                tmp.append(0)
        batch_action.append(tmp)
    return batch_action




class agent():
    def __init__(self, lr, entity_ebd, word_ebd,s_size):


        #get action
        entity_embedding = tf.get_variable(name = 'entity_embedding',initializer=entity_ebd,trainable=False)
        word_embedding = tf.get_variable(name = 'word_embedding',initializer=word_ebd,trainable=False)


#        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        self.word  = tf.placeholder(dtype=tf.int32, shape=[None, s_size], name='word')
        self.entity  = tf.placeholder(dtype=tf.int32, shape=[None, s_size], name='entity')
#        self.word_ebd = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
#        self.word_mean  = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

        self.word_ebd = tf.nn.embedding_lookup(word_embedding, self.word)
        self.entity_ebd = tf.nn.embedding_lookup(entity_embedding, self.entity)

        self.input = tf.concat(axis=2,values = [self.word_ebd,self.entity_ebd])

        self.prob = tf.reshape(layers.fully_connected(self.input,1,tf.nn.sigmoid),[-1, s_size])
        self.prob_sum = tf.reduce_mean(self.prob, 1)
#        self.prob = layers.fully_connected(self.input,1,tf.nn.sigmoid)
        #compute loss
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.float32)

        #the probability of choosing 0 or 1
        self.pi  = self.action_holder * self.prob_sum + (1 - self.action_holder) * (1 - self.prob_sum)

        #loss
        self.loss = -tf.reduce_sum(tf.log(self.pi) * self.reward_holder)

        # minimize loss
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)

        self.tvars = tf.trainable_variables()

        #manual update parameters
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)


        #compute gradient
        self.gradients = tf.gradients(self.loss, self.tvars)

        #update parameters using gradient
        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))

def train_mention(config, word_name_list, entity_name_list, word_ebd, entity_ebd, train_X):
#    average_reward = np.mean(all_reward)
    leftcontext_list = train_X[0]
    rightcontext_list = train_X[1]
    context_list = np.concatenate((leftcontext_list,rightcontext_list), axis =1)
    entity_list = train_X[2] #need pre-training
    entity_description_list = train_X[3]
    traindata_length = len(entity_list)
    context_length = len(leftcontext_list[0])
    description_length = len(entity_description_list[0]) 

    g_mention = tf.Graph()
    sess2 = tf.Session()
    sess2 = tf.Session(graph=g_mention)
    env = environment(100)


    with g_mention.as_default():
        with sess2.as_default():
            myAgent = agent(0.03, entity_ebd, word_ebd, 2 * context_length)
            updaterate = config.updaterate
            num_epoch = config.num_epoch
            sampletimes = config.sampletimes
            negative_sample = config.negative_sample

            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver()
            #saver.restore(sess2, save_path='rlmodel/rl.ckpt')

            tvars_best = sess2.run(myAgent.tvars)
            for index, var in enumerate(tvars_best):
                tvars_best[index] = var * 0

            tvars_old = sess2.run(myAgent.tvars)


            gradBuffer = sess2.run(myAgent.tvars)
            for index, grad in enumerate(gradBuffer):
                gradBuffer[index] = grad * 0
                
            entity_linking_model = reward_feedback(config, word_name_list, entity_name_list, word_ebd, entity_ebd)

#            g_mention.finalize()

            for epoch in range(num_epoch):

                all_list = list(range(traindata_length))
                id_entity_all = np.transpose([entity_list for x in xrange(2 * context_length)])
                for index, tem_batch in enumerate(xrange(100)):
                    tem_context_list = context_list[int(traindata_length * tem_batch/100):int(traindata_length* (tem_batch+1)/100)]
                    tem_id_entity_all = id_entity_all[int(traindata_length * tem_batch/100):int(traindata_length* (tem_batch+1)/100)]                    
                    state = env.reset(tem_context_list,tem_id_entity_all, batch_reward=0)
                    feed_dict = {}
                    feed_dict[myAgent.word] = state[0]
                    feed_dict[myAgent.entity] = state[1]
                    if index == 0:
                        prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                    else:
                        prob = np.concatenate((prob, sess2.run(myAgent.prob, feed_dict=feed_dict)), axis = 0)

#                print prob.shape
                # generate the negative sample
                candidate_left_sentence = [leftcontext_list for x in xrange(negative_sample)]
                candidate_left_sentence = np.transpose(np.array(candidate_left_sentence),(1,0,2))
#                print 'candidate_left_sentence.shape',candidate_left_sentence.shape
                
                candidate_right_sentence = [rightcontext_list for x in xrange(negative_sample)]
                candidate_right_sentence = np.transpose(np.array(candidate_right_sentence),(1,0,2))
#                print 'candidate_right_sentence.shape',candidate_right_sentence.shape
                
                entity_list_all = [entity_list for x in xrange(negative_sample)]
                entity_list_all = np.transpose(np.array(entity_list_all),(1,0))
#                print 'entity_list_all.shape',entity_list_all.shape
                
                entity_description_all = [entity_description_list for x in xrange(negative_sample)]
                entity_description_all = np.transpose(np.array(entity_description_all),(1,0,2))
#                print 'entity_description_all.shape',entity_description_all.shape
                
                for i in range(traindata_length):
                    for j in range(negative_sample):# random gererate choosing pobability
                        context_action = get_action(prob[i])
#                        print  len(context_action), len(context_action[:context_length]), len(context_action[context_length:])                                                
                        candidate_left_sentence[i][j] = candidate_left_sentence[i][j] * context_action[:context_length]
                        candidate_right_sentence[i][j] = candidate_right_sentence[i][j] * context_action[context_length:]
                                    
                            
                # calculate the reward
                el_candidate_left_sentence = np.reshape(candidate_left_sentence,[-1,context_length])
#                print 'el_candidate_left_sentence.shape',el_candidate_left_sentence.shape
                
                el_candidate_right_sentence = np.reshape(candidate_right_sentence,[-1,context_length])
#                print 'el_candidate_right_sentence.shape',el_candidate_right_sentence.shape
                
                el_entity_list_all = np.reshape(entity_list_all,[-1])
#                print 'el_entity_list_all.shape', el_entity_list_all.shape
                                                        
                el_entity_description_all = np.reshape(entity_description_all,[-1,description_length])
#                print 'el_entity_description_all.shape', el_entity_description_all.shape
               
                tem_train_input = [el_candidate_left_sentence,el_candidate_right_sentence,\
                                   el_entity_list_all,el_entity_description_all]
                reward_list = entity_linking_model.model.predict(tem_train_input,batch_size=200,verbose=0)
                reward_list = np.reshape(reward_list, [traindata_length, negative_sample])
#                print 'reward_list.shape',reward_list.shape
                avg_reward = np.mean(reward_list, axis = 1)
#                print 'avg_reward.shape',avg_reward.shape

                # shuffle bags
                random.shuffle(all_list)

                for batch in tqdm(all_list): #number of sentence                 
                    left_word_list = candidate_left_sentence[batch]                    
                    right_word_list = candidate_right_sentence[batch]
                    word_list = np.concatenate((left_word_list, right_word_list), axis = 1)
#                    print 'word_list.shape' ,word_list.shape
                      
                    id_entity = [[entity_list[batch] for x in xrange(2 * context_length)] for y in xrange(negative_sample)]
                    
#                    print 'id_entity', len(id_entity), len(id_entity[0])
                    
                    
                    batch_reward = 0  # need to delete

                        # compute gradient
                    for j in xrange(sampletimes):
                        state = env.reset(word_list, id_entity, batch_reward)                         
                        feed_dict = {}
                        feed_dict[myAgent.word] = state[0]
                        feed_dict[myAgent.entity] = state[1]
                        prob_sum = sess2.run(myAgent.prob_sum, feed_dict=feed_dict)
                        
                        action_list = get_action(prob_sum)
                        reward = reward_list[batch] - avg_reward[batch]                        
                        feed_dict[myAgent.reward_holder] = reward
                        feed_dict[myAgent.action_holder] = action_list
                        grads = sess2.run(myAgent.gradients, feed_dict=feed_dict)
                        for index, grad in enumerate(grads):
                            gradBuffer[index] += grad
                        #end = time.time()
                        #print('get loss and update:', end - start)
                        
                        #apply gradient
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    sess2.run(myAgent.update_batch, feed_dict=feed_dict)
                    for index, grad in enumerate(gradBuffer):
                        gradBuffer[index] = grad * 0

                    #get tvars_new
                    tvars_new = sess2.run(myAgent.tvars)

                    # update old variables of the target network
                    tvars_update = sess2.run(myAgent.tvars)
                    for index, var in enumerate(tvars_update):
                        tvars_update[index] = updaterate * tvars_new[index] + (1-updaterate) * tvars_old[index]

                    feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_update))
                    sess2.run(myAgent.update_tvar_holder, feed_dict)
                    tvars_old = sess2.run(myAgent.tvars)
                #break
            
            
            tvars_best = tvars_old
            #set parameters = best_tvars
#            feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_best))
#            sess2.run(myAgent.update_tvar_holder, feed_dict)
            #save model
            saver.save(sess2, save_path='./model/origin_rl_model.ckpt')
            
 


def train_entity(config, word_name_list, entity_name_list, word_ebd, entity_ebd, train_X):
#    average_reward = np.mean(all_reward)
    leftcontext_list = train_X[0]
    rightcontext_list = train_X[1]
    context_list = np.concatenate((leftcontext_list,rightcontext_list), axis =1)
    entity_list = train_X[2] #need pre-training
    entity_description_list = train_X[3]
    traindata_length = len(entity_list)
    context_length = len(leftcontext_list[0])
    description_length = len(entity_description_list[0])


    g_entity = tf.Graph()
    sess2 = tf.Session()
    sess2 = tf.Session(graph=g_entity)
    env = environment(100)


    with g_entity.as_default():
        with sess2.as_default():
            myAgent = agent(0.03,entity_ebd, word_ebd, description_length)
            updaterate = config.updaterate
            num_epoch = config.num_epoch
            sampletimes = config.sampletimes
            negative_sample = config.negative_sample

            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver()
            #saver.restore(sess2, save_path='rlmodel/rl.ckpt')

            tvars_best = sess2.run(myAgent.tvars)
            for index, var in enumerate(tvars_best):
                tvars_best[index] = var * 0

            tvars_old = sess2.run(myAgent.tvars)


            gradBuffer = sess2.run(myAgent.tvars)
            for index, grad in enumerate(gradBuffer):
                gradBuffer[index] = grad * 0
                
            entity_linking_model = reward_feedback(config, word_name_list, entity_name_list, word_ebd, entity_ebd)

#            g_entity.finalize()

            for epoch in range(num_epoch):

                all_list = list(range(traindata_length))
                id_entity_description_all = np.transpose([entity_list for x in xrange(description_length)])

                for index, tem_batch in enumerate(xrange(100)):
                    tem_entity_description_list = entity_description_list[int(traindata_length * tem_batch/100):int(traindata_length* (tem_batch+1)/100)]
                    tem_id_entity_description_all = id_entity_description_all[int(traindata_length * tem_batch/100):int(traindata_length* (tem_batch+1)/100)]                    
                    state = env.reset(tem_entity_description_list,tem_id_entity_description_all, batch_reward=0)
                    feed_dict = {}
                    feed_dict[myAgent.word] = state[0]
                    feed_dict[myAgent.entity] = state[1]
                    if index == 0:
                        prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                    else:
                        prob =np.concatenate((prob, sess2.run(myAgent.prob, feed_dict=feed_dict)), axis = 0)
                
                candidate_left_sentence = [leftcontext_list for x in xrange(negative_sample)]
                candidate_left_sentence = np.transpose(np.array(candidate_left_sentence),(1,0,2))
#                print 'candidate_left_sentence.shape',candidate_left_sentence.shape
                
                candidate_right_sentence = [rightcontext_list for x in xrange(negative_sample)]
                candidate_right_sentence = np.transpose(np.array(candidate_right_sentence),(1,0,2))

                entity_list_all = [entity_list for x in xrange(negative_sample)]
                entity_list_all = np.transpose(np.array(entity_list_all),(1,0))
#                print 'entity_list_all.shape',entity_list_all.shape
                
                entity_description_all = [entity_description_list for x in xrange(negative_sample)]
                entity_description_all = np.transpose(np.array(entity_description_all),(1,0,2))
#                print 'entity_description_all.shape',entity_description_all.shape
                
                for i in range(traindata_length):
                    for j in range(negative_sample):# random gererate choosing pobability
                        entity_context_action = get_action(prob[i]) 
                        entity_description_all[i][j] = entity_description_all[i][j] * entity_context_action
                                    

                el_candidate_left_sentence = np.reshape(candidate_left_sentence,[-1,context_length])
#                print 'el_candidate_left_sentence.shape',el_candidate_left_sentence.shape
                
                el_candidate_right_sentence = np.reshape(candidate_right_sentence,[-1,context_length])
#                print 'el_candidate_right_sentence.shape',el_candidate_right_sentence.shape
                        
                el_entity_list_all = np.reshape(entity_list_all,[-1])
#                print 'el_entity_list_all.shape', el_entity_list_all.shape
                                                        
                el_entity_description_all = np.reshape(entity_description_all,[-1,description_length])
#                print 'el_entity_description_all.shape', el_entity_description_all.shape
               
                tem_train_input = [el_candidate_left_sentence,el_candidate_right_sentence,\
                                   el_entity_list_all,el_entity_description_all]
                reward_list = entity_linking_model.model.predict(tem_train_input,batch_size=200,verbose=0)
                reward_list = np.reshape(reward_list, [traindata_length, negative_sample])
#                print 'reward_list.shape',reward_list.shape
                avg_reward = np.mean(reward_list, axis = 1)
#                print 'avg_reward.shape',avg_reward.shape

                # shuffle bags
                random.shuffle(all_list)

                for batch in tqdm(all_list): #number of sentence                 
                    entity_description_single = entity_description_all[batch] # entity description list                    
                    id_entity = [[entity_list[batch] for x in xrange(description_length)]\
                                  for y in xrange(negative_sample)]
                    
#                    print 'id_entity', len(id_entity), len(id_entity[0])
                    
                    
                    batch_reward = 0  # need to delete

                        # compute gradient
                    for j in xrange(sampletimes):
                        state = env.reset(entity_description_single, id_entity, batch_reward)                         
                        feed_dict = {}
                        feed_dict[myAgent.word] = state[0]
                        feed_dict[myAgent.entity] = state[1]
                        prob_sum = sess2.run(myAgent.prob_sum, feed_dict=feed_dict)
#                        print 'prob.shape', prob.shape
                        
                        action_list = get_action(prob_sum)
                        reward = reward_list[batch] - avg_reward[batch]
                        print 'reward.shape', reward.shape
                        print 'reward.shape', reward.shape

                        
                        feed_dict[myAgent.reward_holder] = reward
                        feed_dict[myAgent.action_holder] = action_list
                        grads = sess2.run(myAgent.gradients, feed_dict=feed_dict)
                        for index, grad in enumerate(grads):
                            gradBuffer[index] += grad
                        
                    #apply gradient
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    sess2.run(myAgent.update_batch, feed_dict=feed_dict)
                    for index, grad in enumerate(gradBuffer):
                        gradBuffer[index] = grad * 0

                    #get tvars_new
                    tvars_new = sess2.run(myAgent.tvars)

                    # update old variables of the target network
                    tvars_update = sess2.run(myAgent.tvars)
                    for index, var in enumerate(tvars_update):
                        tvars_update[index] = updaterate * tvars_new[index] + (1-updaterate) * tvars_old[index]

                    feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_update))
                    sess2.run(myAgent.update_tvar_holder, feed_dict)
                    tvars_old = sess2.run(myAgent.tvars)
                #break
            tvars_best = tvars_old
            #set parameters = best_tvars
#            feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_best))
#            sess2.run(myAgent.update_tvar_holder, feed_dict)
            #save model
            saver.save(sess2, save_path='./model/origin_rl_entity_model.ckpt')
           


def select(save_path, save_path2, word_ebd, entity_ebd, train_X, training_able = True):
    leftcontext_list = train_X[0]
    rightcontext_list = train_X[1]
    context_list = np.concatenate((leftcontext_list, rightcontext_list), axis =1)
    entity_list = train_X[2] #need pre-training
    entity_description_list = train_X[3]
    context_length = len(leftcontext_list[0])
    description_length = len(entity_description_list[0])
    traindata_length = len(entity_list)
    id_entity_all = np.transpose([entity_list for x in xrange(2 * context_length)])
    id_entity_description_all = np.transpose([entity_list for x in xrange(description_length)])
#    print 'id_entity_all',id_entity_all.shape
    batch_reward = 0

    g_mention = tf.Graph()
    sess2 = tf.Session(graph=g_mention)
    env = environment(100)


    with g_mention.as_default():
        with sess2.as_default():

            myAgent = agent(0.02,entity_ebd, word_ebd, 2 * context_length)
            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver()
            saver.restore(sess2, save_path=save_path)
            g_mention.finalize()
            
            id_entity_all = np.transpose([entity_list for x in xrange(2 * context_length)])
            for index, tem_batch in enumerate(xrange(100)):
                
                tem_context_list = context_list[int(traindata_length * tem_batch/100):int(traindata_length* (tem_batch+1)/100)]
                tem_id_entity_all = id_entity_all[int(traindata_length * tem_batch/100):int(traindata_length* (tem_batch+1)/100)]                  
                state = env.reset(tem_context_list,tem_id_entity_all, batch_reward=0)
                feed_dict = {}
                feed_dict[myAgent.word] = state[0]
                feed_dict[myAgent.entity] = state[1]                
                if index == 0:
                    prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                else:
                    prob =np.concatenate((prob, sess2.run(myAgent.prob, feed_dict=feed_dict)), axis = 0)              
                          
            decision_list = decide_batch_action(prob)
    selected_context = np.array(decision_list) * context_list 
    
    if training_able:
        np.save('data/word_context.npy',selected_context)
        np.save('data/entity_list.npy',entity_list)
    else:            
        np.save('data/test_word_context.npy',selected_context)
        np.save('data/test_entity_list.npy',entity_list)
        
        
    g_entity = tf.Graph()
    sess3 = tf.Session(graph=g_entity)
    env = environment(100)


    with g_entity.as_default():
        with sess3.as_default():

            myAgent = agent(0.02,entity_ebd, word_ebd, description_length)
            init = tf.global_variables_initializer()
            sess3.run(init)
            saver = tf.train.Saver()
            saver.restore(sess3, save_path=save_path2)
            g_mention.finalize()
            
            for index, tem_batch in enumerate(xrange(100)):
                tem_entity_description_list = entity_description_list[int(traindata_length * tem_batch/100):int(traindata_length* (tem_batch+1)/100)]
                tem_id_entity_description_all = id_entity_description_all[int(traindata_length * tem_batch/100):int(traindata_length* (tem_batch+1)/100)]                    
                state = env.reset(tem_entity_description_list,tem_id_entity_description_all, batch_reward=0)
                feed_dict = {}
                feed_dict[myAgent.word] = state[0]
                feed_dict[myAgent.entity] = state[1]
                if index == 0:
                    prob = sess3.run(myAgent.prob, feed_dict=feed_dict)
                else:
                    prob =np.concatenate((prob, sess3.run(myAgent.prob, feed_dict=feed_dict)), axis = 0)
                                                               
            decision_list = decide_batch_action(prob)
    selected_entity_description = np.array(decision_list) * entity_description_list
    
    if training_able:
        np.save('data/entity_description.npy',selected_entity_description)
    else:            
        np.save('data/test_entity_description.npy',selected_entity_description)
    
    return selected_context, entity_list , selected_entity_description

    


class AttentionLSTMWrapper(Wrapper):
    def __init__(self, layer, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        assert isinstance(layer, LSTM)
        self.supports_masking = True
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param
        super(AttentionLSTMWrapper, self).__init__(layer, **kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(AttentionLSTMWrapper, self).build()

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = K.random_normal_variable((self.layer.units, self.layer.units),0,1, name='{}_U_a'.format(self.name))
#         self.U_a = self.layer.recurrent_initializer((attention_dim, self.layer.units))
        self.b_a = K.zeros((self.layer.units,), name='{}_b_a'.format(self.name))

        self.U_m = K.random_normal_variable((attention_dim, self.layer.units),0,1, name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.layer.units,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = K.random_normal_variable((self.layer.units, 1),0,1, name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = K.random_normal_variable((self.layer.units, self.layer.units),0,1, name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.layer.units,), name='{}_b_s'.format(self.name))

        self.layer._trainable_weights += [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        h, [h, c] = self.layer.step(x, states)
        attention = states[4]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.layer.units, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
#         constants = self.layer.f
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
#             initial_states = None
#         initial_states = self.layer.states
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output


class MyModel(object):
    def __init__(self, indexes, config):
        self.indexes = indexes
        self.cf = config
	self.indexes.word_index = {}
	self.indexes.entity_indices = {}

def data_divided(context,context_length):
    left_context = context[:,:context_length]
    right_context = context[:,context_length:]
    return left_context, right_context


class test_Attention_LSTM_NoFeatures2(MyModel):
    def __init__(self, indexes, config):
		MyModel.__init__(self, indexes, config) 
		self.description ="Attention_LSTM: Attention_LSTM. sigmoid output.no features"
    
    
    def create(self, word_index, entity_indices, embedding_word_matrix, embedding_entity_matrix):
		'''
		Use pretrained embedding
		'''

		print 'Creating {}'.format(self.description)
		self.indexes.word_index = word_index
		self.indexes.entity_indices = entity_indices

		embedding_left_ctx = Embedding(len(self.indexes.word_index) , # index 0 for zero padding+ 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #20

		embedding_right_ctx = Embedding(len(self.indexes.word_index), # index 0 for zero padding + 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #20

		embedding_entity = Embedding(len(self.indexes.entity_indices) , # 1 extra for NIL_entity+ 2
									 self.cf.ENTITY_EMBEDDING_DIM,
									 weights=[embedding_entity_matrix],
									 input_length=1,
									 trainable=self.cf.EMBEDDING_TRAINABLE) #300

		embedding_entity_desc = Embedding(len(self.indexes.word_index) , # index 0 for zero padding+ 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_ENTITY_DESC_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #300

		in1 = Input(shape=(self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,), dtype='int32')#50
		in2 = Input(shape=(self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,), dtype='int32')#50
		in4 = Input(shape=(1,), dtype='int32')#1
		in5 = Input(shape=(self.cf.MAX_ENTITY_DESC_LENGTH,), dtype='int32')#150

		encoder_left_ctx= LSTM(self.cf.LSTM_SIZE,return_sequences=True)
		encoder_right_ctx= LSTM(self.cf.LSTM_SIZE, go_backwards=True,return_sequences=True)

		# entity representation
		encoder_entity = embedding_entity(in4) #300
		encoder_entity = Flatten()(encoder_entity)
		
		encoder_entity_desc = embedding_entity_desc(in5) ## 300
		encoder_entity_desc= LSTM(self.cf.LSTM_SIZE,return_sequences=True)(encoder_entity_desc) 

		# # maxpooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
		maxpool.supports_masking = True
		latent_entity = concatenate([encoder_entity, maxpool(encoder_entity_desc)])

		# answer rnn part
# 		from attention_lstm import AttentionLSTMWrapper
		encoder_left_ctx = AttentionLSTMWrapper(encoder_left_ctx, latent_entity, single_attention_param=True)
		encoder_right_ctx = AttentionLSTMWrapper(encoder_right_ctx, latent_entity, single_attention_param=True)

		left_ctx = embedding_left_ctx(in1)#1*300
		left_ctx= encoder_left_ctx(left_ctx)

		right_ctx = embedding_right_ctx(in2)
		right_ctx= encoder_right_ctx(right_ctx)

		latent_mention = concatenate([maxpool(left_ctx), maxpool(right_ctx)])

		emb = concatenate([latent_mention, latent_entity])

		emb_latent = Dense(self.cf.MENTION_CONTEXT_LATENT_SIZE, activation=self.cf.ACTIVATION_FUNCTION)(emb) # tanh

		emb_latent = Dense(self.cf.MENTION_CONTEXT_LATENT_SIZE, activation=self.cf.ACTIVATION_FUNCTION)(emb_latent) # tanh

		preds = Dense(1, activation='sigmoid')(emb_latent)

		self.model = Model([in1,in2,
							in4,in5],
							preds)

		self.model.summary()

		# opt = Adam(lr=0.0003)

		self.model.compile(loss='binary_crossentropy', # hinge, 'categorical_crossentropy',
					  optimizer='adam', #adam, rmsprop
					  metrics=['acc'])

class Attention_LSTM_NoFeatures2(MyModel):
    def __init__(self, indexes, config):
		MyModel.__init__(self, indexes, config) 
		self.description ="Attention_LSTM: Attention_LSTM. sigmoid output.no features"
    
    def load_pretrain_embedding(self):
        print len(self.indexes.word_index) + 1, self.cf.WORD_EMBEDDING_DIM  
        embedding_word_matrix = np.array(random.rand(len(self.indexes.word_index) + 1, self.cf.WORD_EMBEDDING_DIM))
        embedding_entity_matrix = np.array(random.rand(len(self.indexes.entity_indices) + 2, self.cf.ENTITY_EMBEDDING_DIM))
        return embedding_word_matrix, embedding_entity_matrix
    
    def create(self, word_index, entity_indices, embedding_word_matrix, embedding_entity_matrix):
		'''
		Use pretrained embedding
		'''

		print 'Creating {}'.format(self.description)
		self.indexes.word_index = word_index
		self.indexes.entity_indices = entity_indices

		embedding_left_ctx = Embedding(len(self.indexes.word_index) , # index 0 for zero padding+ 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #20

		embedding_right_ctx = Embedding(len(self.indexes.word_index), # index 0 for zero padding + 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #20

		embedding_entity = Embedding(len(self.indexes.entity_indices) , # 1 extra for NIL_entity+ 2
									 self.cf.ENTITY_EMBEDDING_DIM,
									 weights=[embedding_entity_matrix],
									 input_length=1,
									 trainable=self.cf.EMBEDDING_TRAINABLE) #300

		embedding_entity_desc = Embedding(len(self.indexes.word_index) , # index 0 for zero padding+ 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_ENTITY_DESC_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #300

		in1 = Input(shape=(self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,), dtype='int32')#50
		in2 = Input(shape=(self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,), dtype='int32')#50
		in4 = Input(shape=(1,), dtype='int32')#1
		in5 = Input(shape=(self.cf.MAX_ENTITY_DESC_LENGTH,), dtype='int32')#150

		encoder_left_ctx= LSTM(self.cf.LSTM_SIZE,return_sequences=True)
		encoder_right_ctx= LSTM(self.cf.LSTM_SIZE,return_sequences=True)# go_backwards=True,

		# entity representation
		encoder_entity = embedding_entity(in4) #300
		encoder_entity = Flatten()(encoder_entity)
		
		encoder_entity_desc = embedding_entity_desc(in5) ## 300
		encoder_entity_desc = Dropout(self.cf.DROPOUT)(encoder_entity_desc)
		encoder_entity_desc= LSTM(self.cf.LSTM_SIZE,return_sequences=True)(encoder_entity_desc) 

		# # maxpooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
		maxpool.supports_masking = True
		latent_entity = concatenate([encoder_entity, maxpool(encoder_entity_desc)])

		# answer rnn part
# 		from attention_lstm import AttentionLSTMWrapper
		encoder_left_ctx = AttentionLSTMWrapper(encoder_left_ctx, latent_entity, single_attention_param=True)
		encoder_right_ctx = AttentionLSTMWrapper(encoder_right_ctx, latent_entity, single_attention_param=True)

		left_ctx = embedding_left_ctx(in1)#1*300
		left_ctx = Dropout(self.cf.DROPOUT)(left_ctx)
		left_ctx= encoder_left_ctx(left_ctx)

		right_ctx = embedding_right_ctx(in2)
		right_ctx = Dropout(self.cf.DROPOUT)(right_ctx)
		right_ctx= encoder_right_ctx(right_ctx)

		latent_mention = concatenate([maxpool(left_ctx), maxpool(right_ctx)])

		emb = concatenate([latent_mention, latent_entity])

		emb_latent = Dense(self.cf.MENTION_CONTEXT_LATENT_SIZE, activation=self.cf.ACTIVATION_FUNCTION)(emb) # tanh
		emb_latent = Dropout(self.cf.DROPOUT)(emb_latent)

		emb_latent = Dense(self.cf.MENTION_CONTEXT_LATENT_SIZE, activation=self.cf.ACTIVATION_FUNCTION)(emb_latent) # tanh
		emb_latent = Dropout(self.cf.DROPOUT)(emb_latent)

		preds = Dense(1, activation='sigmoid')(emb_latent)

		self.model = Model([in1,in2,
							in4,in5],
							preds)

		self.model.summary()

		# opt = Adam(lr=0.0003)

		self.model.compile(loss='binary_crossentropy', # hinge, 'categorical_crossentropy',
					  optimizer='adam', #adam, rmsprop
					  metrics=['acc'])

class Context_LSTM(MyModel):
    def __init__(self, indexes, config):
		MyModel.__init__(self, indexes, config) 
		self.description ="Attention_LSTM: Attention_LSTM. sigmoid output.no features"
    
    def load_pretrain_embedding(self):
        print len(self.indexes.word_index) + 1, self.cf.WORD_EMBEDDING_DIM  
        embedding_word_matrix = np.array(random.rand(len(self.indexes.word_index) + 1, self.cf.WORD_EMBEDDING_DIM))
        embedding_entity_matrix = np.array(random.rand(len(self.indexes.entity_indices) + 2, self.cf.ENTITY_EMBEDDING_DIM))
        return embedding_word_matrix, embedding_entity_matrix
    
    def create(self, word_index, entity_indices, embedding_word_matrix):
		'''
		Use pretrained embedding
		'''

		print 'Creating {}'.format(self.description)
		self.indexes.word_index = word_index
		self.indexes.entity_indices = entity_indices

		embedding_left_ctx = Embedding(len(self.indexes.word_index) , # index 0 for zero padding+ 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #20

		embedding_right_ctx = Embedding(len(self.indexes.word_index), # index 0 for zero padding + 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #20

		#embedding_entity = Embedding(len(self.indexes.entity_indices) , # 1 extra for NIL_entity+ 2
		#							 self.cf.ENTITY_EMBEDDING_DIM,
		#							 weights=[embedding_entity_matrix],
		#							 input_length=1,
		#							 trainable=self.cf.EMBEDDING_TRAINABLE) #300

		embedding_entity_desc = Embedding(len(self.indexes.word_index) , # index 0 for zero padding+ 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_ENTITY_DESC_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #300

		in1 = Input(shape=(self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,), dtype='int32')#50
		in2 = Input(shape=(self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,), dtype='int32')#50
		#in4 = Input(shape=(1,), dtype='int32')#1
		in5 = Input(shape=(self.cf.MAX_ENTITY_DESC_LENGTH,), dtype='int32')#150

		encoder_left_ctx= LSTM(self.cf.LSTM_SIZE,return_sequences=True)
		encoder_right_ctx= LSTM(self.cf.LSTM_SIZE,return_sequences=True)# go_backwards=True,

		# entity representation
		#encoder_entity = embedding_entity(in4) #300
		#encoder_entity = Flatten()(encoder_entity)
		
		encoder_entity_desc = embedding_entity_desc(in5) ## 300
		encoder_entity_desc = Dropout(self.cf.DROPOUT)(encoder_entity_desc)
		encoder_entity_desc= LSTM(self.cf.LSTM_SIZE,return_sequences=True)(encoder_entity_desc) 

		# # maxpooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
		maxpool.supports_masking = True
		#latent_entity = concatenate([encoder_entity, maxpool(encoder_entity_desc)])
        	latent_entity = maxpool(encoder_entity_desc)
		# answer rnn part
# 		from attention_lstm import AttentionLSTMWrapper
		#encoder_left_ctx = AttentionLSTMWrapper(encoder_left_ctx, latent_entity, single_attention_param=True)
		#encoder_right_ctx = AttentionLSTMWrapper(encoder_right_ctx, latent_entity, single_attention_param=True)

		left_ctx = embedding_left_ctx(in1)#1*300
		left_ctx = Dropout(self.cf.DROPOUT)(left_ctx)
		left_ctx= encoder_left_ctx(left_ctx)

		right_ctx = embedding_right_ctx(in2)
		right_ctx = Dropout(self.cf.DROPOUT)(right_ctx)
		right_ctx= encoder_right_ctx(right_ctx)

		latent_mention = concatenate([maxpool(left_ctx), maxpool(right_ctx)])

		emb = concatenate([latent_mention, latent_entity])


		emb_latent = Dense(self.cf.MENTION_CONTEXT_LATENT_SIZE, activation=self.cf.ACTIVATION_FUNCTION)(emb) # tanh
		emb_latent = Dropout(self.cf.DROPOUT)(emb_latent)

		emb_latent = Dense(self.cf.MENTION_CONTEXT_LATENT_SIZE, activation=self.cf.ACTIVATION_FUNCTION)(emb_latent) # tanh
		emb_latent = Dropout(self.cf.DROPOUT)(emb_latent)

		preds = Dense(1, activation='sigmoid')(emb_latent)

		self.model = Model([in1,in2,
							in5],
							preds)

		self.model.summary()

		# opt = Adam(lr=0.0003)

		self.model.compile(loss='binary_crossentropy', # hinge, 'categorical_crossentropy',
					  optimizer='adam', #adam, rmsprop
					  metrics=['acc'])
        


class LSTM_NoFeatures(MyModel):
    def __init__(self, indexes, config):
		MyModel.__init__(self, indexes, config) 
		self.description ="Attention_LSTM: Attention_LSTM. sigmoid output.no features"
    
    def load_pretrain_embedding(self):
        print len(self.indexes.word_index) + 1, self.cf.WORD_EMBEDDING_DIM  
        embedding_word_matrix = np.array(random.rand(len(self.indexes.word_index) + 1, self.cf.WORD_EMBEDDING_DIM))
        embedding_entity_matrix = np.array(random.rand(len(self.indexes.entity_indices) + 2, self.cf.ENTITY_EMBEDDING_DIM))
        return embedding_word_matrix, embedding_entity_matrix
    
    def create(self, word_index, entity_indices, embedding_word_matrix, embedding_entity_matrix):
		'''
		Use pretrained embedding
		'''

		print 'Creating {}'.format(self.description)
		self.indexes.word_index = word_index
		self.indexes.entity_indices = entity_indices

		embedding_left_ctx = Embedding(len(self.indexes.word_index) , # index 0 for zero padding+ 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #20

		embedding_right_ctx = Embedding(len(self.indexes.word_index), # index 0 for zero padding + 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #20

		embedding_entity = Embedding(len(self.indexes.entity_indices) , # 1 extra for NIL_entity+ 2
									 self.cf.ENTITY_EMBEDDING_DIM,
									 weights=[embedding_entity_matrix],
									 input_length=1,
									 trainable=self.cf.EMBEDDING_TRAINABLE) #300

		embedding_entity_desc = Embedding(len(self.indexes.word_index) , # index 0 for zero padding+ 1
									self.cf.WORD_EMBEDDING_DIM,
									weights=[embedding_word_matrix],
									input_length=self.cf.MAX_ENTITY_DESC_LENGTH,
									trainable=self.cf.EMBEDDING_TRAINABLE) #300

		in1 = Input(shape=(self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,), dtype='int32')#50
		in2 = Input(shape=(self.cf.MAX_WINDOW_SIZE+self.cf.MAX_MENTION_LENGTH,), dtype='int32')#50
		in4 = Input(shape=(1,), dtype='int32')#1
		in5 = Input(shape=(self.cf.MAX_ENTITY_DESC_LENGTH,), dtype='int32')#150

		encoder_left_ctx= LSTM(self.cf.LSTM_SIZE,return_sequences=True)
		encoder_right_ctx= LSTM(self.cf.LSTM_SIZE, return_sequences=True) #go_backwards=True,

		# entity representation
		encoder_entity = embedding_entity(in4) #300
		encoder_entity = Flatten()(encoder_entity)
		
		encoder_entity_desc = embedding_entity_desc(in5) ## 300
		encoder_entity_desc = Dropout(self.cf.DROPOUT)(encoder_entity_desc)
		encoder_entity_desc= LSTM(self.cf.LSTM_SIZE,return_sequences=True)(encoder_entity_desc) 

		# # maxpooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
		maxpool.supports_masking = True
		latent_entity = concatenate([encoder_entity, maxpool(encoder_entity_desc)])

		left_ctx = embedding_left_ctx(in1)#1*300
		left_ctx = Dropout(self.cf.DROPOUT)(left_ctx)
		left_ctx= encoder_left_ctx(left_ctx)

		right_ctx = embedding_right_ctx(in2)
		right_ctx = Dropout(self.cf.DROPOUT)(right_ctx)
		right_ctx= encoder_right_ctx(right_ctx)

		latent_mention = concatenate([maxpool(left_ctx), maxpool(right_ctx)])

		emb = concatenate([latent_mention, latent_entity])

		emb_latent = Dense(self.cf.MENTION_CONTEXT_LATENT_SIZE, activation=self.cf.ACTIVATION_FUNCTION)(emb) # tanh
		emb_latent = Dropout(self.cf.DROPOUT)(emb_latent)

		emb_latent = Dense(self.cf.MENTION_CONTEXT_LATENT_SIZE, activation=self.cf.ACTIVATION_FUNCTION)(emb_latent) # tanh
		emb_latent = Dropout(self.cf.DROPOUT)(emb_latent)

		preds = Dense(1, activation='sigmoid')(emb_latent)

		self.model = Model([in1,in2,
							in4,in5],
							preds)

		self.model.summary()

		# opt = Adam(lr=0.0003)

		self.model.compile(loss='binary_crossentropy', # hinge, 'categorical_crossentropy',
					  optimizer='adam', #adam, rmsprop
					  metrics=['acc'])

def reward_feedback(config, word_index, entity_indices, word_ebd, entity_ebd):
    indexes = OrderedDict()
    indexes.word_index = word_index
    indexes.entity_indices = entity_indices
    train = Attention_LSTM_NoFeatures2(indexes=indexes, config=config)
    train.create(word_index, entity_indices, word_ebd, entity_ebd)
    train.model.load_weights('./model/my_model_weights.h5')#
#    result = train.model.predict(train_X,batch_size=200,verbose=0)
    return train

def evaluate(sim_2014, label_2014, seg_2014):
    totalseg_2014 = 0
    true_counter = 0
    for seg in seg_2014:
        sim = sim_2014[totalseg_2014: totalseg_2014 + seg]
        lab = label_2014[totalseg_2014: totalseg_2014 + seg]        
        sim_index = np.argmax(np.array(sim))
        lab_index = np.argmax(np.array(lab))
        if sim_index == lab_index:
            true_counter += 1
        totalseg_2014 = totalseg_2014 + seg
    return str(true_counter) + ' / ' + str(len(seg_2014)) + ' = ' + str(float(true_counter)/float(len(seg_2014)))
