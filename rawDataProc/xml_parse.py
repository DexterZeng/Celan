#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import re
from zhon.hanzi import punctuation
from xml.dom.minidom import parse
import xml.dom.minidom

def segment (text):
    text = jieba.cut(text)
    line = ''
    for word in text:
        if word != ' ':
            line = line + word + ' '
    line = line[:len(line)-1]
    return line

def rm (text):
    text = re.sub('http\:\/\/t\.cn.{6}', "", text)
    text = re.sub('www.*cn', "", text)
    text = re.sub(r"[%s]+" % punctuation, " ", text)
    text = re.sub('[/\-@?:!.#\[\]|"%(~)\\\,_]', " ", text)
    text = re.sub('\d*', "", text)
    text = re.sub('\．', "", text)

    return text

def rmName (text):
    text = re.sub(r"[%s]+" % punctuation, "", text)
#    text = re.sub("[＜＞“”《》]", "", text)
#    text = re.sub("《", "", text)
    text = re.sub("[\?]", "·", text)
    text = re.sub("\．", "·", text)
    text = re.sub("\.", "·", text)
    text = re.sub("\-", " ", text)

#    text = re.sub('\d*', "", text)
    return text

entityID = {}
# load entityid-entityname map
entityinput = open('./kb_concise.txt', 'r')
for line in entityinput:
    strs = line.strip().split('\t')
    entityID[strs[0]] = strs[1]
print (len(entityID))
DOMTree = xml.dom.minidom.parse("test.xml")
collection = DOMTree.documentElement

weibos = collection.getElementsByTagName("weibo")

output = open('./mention_info.txt', 'w')
i = 0

leftlength = 0
rightlength = 0
totalmention = 0
weiboID = 0
mentionID = 0
for weibo in weibos:
#    if i > 0 : break
    flag = False
    i = i+ 1
    contents = weibo.getElementsByTagName('content')[0]
    content = contents.childNodes[0].data
    names = weibo.getElementsByTagName('name')
    startoffsets = weibo.getElementsByTagName('startoffset')
    endoffsets = weibo.getElementsByTagName('endoffset')
    kbs = weibo.getElementsByTagName('kb')
    for i in range(0, len(names)):
        if kbs[i].childNodes[0].data != 'NIL':
            flag = True
            name = names[i].childNodes[0].data
            startoffset = startoffsets[i].childNodes[0].data
            endoffset = endoffsets[i].childNodes[0].data
            kb = kbs[i].childNodes[0].data
            answer = entityID.get(kb)
            left = content[:(int(startoffset)+len(name))]
            left = rm(left)
            left = segment(left)
            right = content[(int(endoffset)-len(name)):-1]
            right = rm(right)
            right = segment(right)
            name = rmName(name)
            llen = len(left.split(' '))
            rlen = len(right.split(' '))
            leftlength += llen
            rightlength += rlen
            totalmention +=1
            tbw = str(weiboID) + '\t' + str(mentionID) + '\t' + name + '\t' + answer + '\t' + left + '\t' + right
            print (tbw)
            tbw = tbw + '\n'
            output.write(tbw)
            mentionID += 1
    if flag is True:
        weiboID += 1
print ('total mentions are: ' + str(totalmention))
print ('average left length is : ' + str(float(leftlength/totalmention)))
print ('average right length is : ' + str(float(rightlength/totalmention)))


