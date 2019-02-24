# -*- coding: utf-8 -*-
import re
import jieba
from langconv import *

def tradition2simple(line):
    # convert traditional chinese to simple chinese
    line = Converter('zh-hans').convert(line)
    return line

def text_process(text):
    text = tradition2simple(text)
    p2 = re.compile(r'[^\u4e00-\u9fa5]')
    zh = " ".join(p2.split(text))
    text = zh
    text = jieba.cut(text)
    line = ''
    for word in text:
        if word != ' ':
            line = line + word + ' '
    line = line[:len(line) - 1]
    return line

def transform (text, length, arg):
    strs = text.split(' ')
    if arg == 'left':
        strs.reverse()
    if len(strs) >= length:
        strs = strs[:length]
    if arg == 'left':
        strs.reverse()
    line = ''
    for word in strs:
        line = line + word + ' '
    line = line[:len(line) - 1]
    return line

def checkdup (contexts, cotext):
    flag = False
    for con in contexts:
        if con[0] == cotext[0] and con[1] == cotext[1]:
            flag = True
    return flag

eninput = open('../can_entity/can_entities_refined.txt', 'r')
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
contextinput = open('./entity_context.txt', 'r')
for cline in contextinput:
    cline = cline.strip().split('\t')
    entityname = cline[0]
    entityleftcontext = cline[1]
    entityrightcontext = cline[2]
    entityleftcontext = text_process(entityleftcontext)
    entityrightcontext = text_process(entityrightcontext)
    if entityname not in entityConDic.keys():
        contexts = []
    else:
        contexts = entityConDic.get(entityname)
    if len(entityleftcontext.split(' ')) > 12 and len(entityrightcontext.split(' ')) > 15:
        context = [entityleftcontext, entityrightcontext]
        if checkdup(contexts, context) is False:
            contexts.append(context)
        entityConDic[entityname] = contexts
print(len(entityConDic))

output = open('./context_formalized_new.txt', 'w')

for men in moremen2Ent.keys():
    ents = moremen2Ent[men]
    for ent in ents:
        entname = ent[0]
        if entname in entityConDic.keys():
            contexts = entityConDic[entname]
            contexts = sorted(contexts, key= lambda x: x[0]+x[1], reverse=True)
            if len(contexts) >= 100:
                contexts = contexts[:100]
            for context in contexts:
                print(entname + '\t' + transform(context[0],20,'left') + '\t' + transform(context[1],20,''))
                output.write(entname + '\t' + transform(context[0],20,'left') + '\t' + transform(context[1],20,'') + '\n')
for men in solomen2Ent.keys():
    ent = solomen2Ent.get(men)[0]
    entname = ent[0]
    if entname in entityConDic.keys():
        contexts = entityConDic[entname]
        contexts = sorted(contexts, key=lambda x: x[0] + x[1], reverse=True)
        if len(contexts) >= 100:
            contexts = contexts[:100]
        for context in contexts:
            print(entname + '\t' + transform(context[0],20,'left') + '\t' + transform(context[1],20,''))
            output.write(entname + '\t' + transform(context[0],20,'left') + '\t' + transform(context[1],20,'') + '\n')
output.close()
