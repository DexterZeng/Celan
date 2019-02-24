# -*- coding: utf-8 -*-
import jieba
import re
from langconv import *
import random

entity_embedmatrix_input = open('Embed_Matrix.txt', 'r')
entity2id = {}
for wline in entity_embedmatrix_input:
    wline = wline.strip().split('\t')
    idnum = int(wline[0])
    name = wline[1]
    entity2id[name] = idnum

eninput = open('./entities_dis_idized.txt', 'r')
men2Ent = {}
for eline in eninput:
    eline = eline.strip().split('\t')
    if eline[0] not in men2Ent.keys():
        entities = []
        dis = [eline[2],eline[3]]
        entities.append(dis)
        men2Ent[eline[0]] = entities
    else:
        entities = men2Ent.get(eline[0])
        entities.append([eline[2],eline[3]])
        men2Ent[eline[0]] = entities
print(len(men2Ent))
count0 = 0
count5 = 0

solomen2Ent = {}
moremen2Ent = {}
for m in men2Ent:
    entities = men2Ent.get(m)
    if len(entities) == 1:
        count0 += 1
        solomen2Ent[m] = entities
    else:
        moremen2Ent[m] = entities
    if len(entities) <5:
        count5 += 1


entityConDic = {}
contextinput = open('./context_formalized_idised.txt', 'r')
for cline in contextinput:
    cline = cline.strip().split('\t')
    entityname = cline[0]
    entityleftcontext = cline[1]
    entityrightcontext = cline[2]

    if entityname not in entityConDic.keys():
        contexts = []
    else:
        contexts = entityConDic.get(entityname)
    context = [entityleftcontext, entityrightcontext]
    contexts.append(context)
    entityConDic[entityname] = contexts
print(len(entityConDic))


output = open('./input.txt', 'w')

for men in moremen2Ent.keys():
    ents = moremen2Ent[men]
    for ent in ents:
        entname = ent[0]
        entid = entity2id[entname]
        if entname in entityConDic.keys():
            contexts = entityConDic[entname]
            if len(contexts) > 100:
                contexts = contexts[:100]
            for context in contexts:
                output.write('1' + '\t' + str(entid) + '\t' + context[0] + '\t' + context[1] + '\t' + ent[1] + '\n')
                print('1' + '\t' + str(entid) + '\t' + context[0] + '\t' + context[1] + '\t' + ent[1])
                if len(ents) >= 5:
                    slices = random.sample(ents, 5)
                else:
                    slices = ents
                for ent1 in slices:
                    if ent1[0] != entname:
                        entid1 = entity2id[ent1[0]]
                        output.write('0' + '\t' + str(entid1) + '\t' + context[0] + '\t' + context[1] + '\t' + ent1[1] + '\n')

for men in solomen2Ent.keys():
    ent = solomen2Ent.get(men)[0]
    entname = ent[0]
    entid = entity2id[entname]
    if entname in entityConDic.keys():
        contexts = entityConDic[entname]
        if len(contexts) > 100:
            contexts = contexts[:100]
        for context in contexts:
            output.write('1' + '\t' + str(entid) + '\t' + context[0] + '\t' + context[1] + '\t' + ent[1] + '\n')

output.close()


