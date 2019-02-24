# -*- coding: utf-8 -*-
import re
from urllib import request
from langconv import *

def tradition2simple(line):
    # convert traditional chinese to simple chinese
    line = Converter('zh-hans').convert(line)
    return line
def remove_marks(text):
    text = re.sub('\<(.*?)\>', ' ', text)
    text = re.sub('[\<\>]', '', text)
    return text


def replace_func(inputpath, enContextDic):
    eninput = open('../can_entity/can_entities_refined.txt', 'r')
    targetEn = []
    for eline in eninput:
        eline = eline.strip().split()
        if eline[2] not in targetEn:
            targetEn.append(eline[2])
    print(targetEn)
    print(len(targetEn))

    input = open(inputpath, 'r')
    i = 0
    for line in input:
        print(i)
        i = i + 1
        line = line.strip()
        entities = re.findall('f\=\"(.*?)\"\>', line)
        for entity in entities:
            entityname = request.unquote(entity)
            entityname = tradition2simple(entityname)
            if entityname in targetEn:
                if entityname not in enContextDic.keys():
                    contexts = []
                else:
                    contexts = enContextDic.get(entityname)
                mentions = re.findall('\<a href\=\"' + entity + '\"\>(.*?)\<\/', line)
                mentionName = mentions[0]
                splitstring = '<a href="' + entity + '">' + mentionName
                index = line.find(splitstring)
                leftcon = line[:index] + ' ' + mentionName
                leftcon = remove_marks(leftcon)

                rightcon = line[index:len(line)]
                rightcon = remove_marks(rightcon)

                context = [leftcon, rightcon]
                contexts.append(context)
                enContextDic[entityname] = contexts
    return enContextDic


enContextDic = {}
data_path = '/home/weixin/PycharmProjects/huge/WikiLink/simplified/ori-en/initial/'
data_names = ['zh_wiki_00','zh_wiki_01','zh_wiki_02','zh_wiki_03','zh_wiki_04','zh_wiki_05']
for data_name in data_names:
    enContextDic = replace_func(data_path + data_name, enContextDic)
    print('{0} has been processed !'.format(data_name))

output = open('./entity_context.txt', 'w')
for en in enContextDic.keys():
    contexts = enContextDic[en]
    for context in contexts:
        output.write(en + '\t' + context[0] + '\t' + context[1] + '\n')
output.close()
