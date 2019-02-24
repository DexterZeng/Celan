# -*- coding: utf-8 -*-
import re
import jieba
from langconv import *
from zhon.hanzi import punctuation
input = open('./men_can_info.txt', 'r')
output = open('./men_can_info_clean.txt', 'w')
outputEn = open('./can_entities.txt', 'w')


def tradition2simple(line):
    line = Converter('zh-hans').convert(line)
    return line

count = 0
for line in input:
    line = line.strip()
    strs = line.split('\t')
    if len(strs) > 6:
        text = ''
        for i in range(5,len(strs)):
            text = text + strs[i] + ' '
        p2 = re.compile(r'[^\u4e00-\u9fa5]')
        zh = " ".join(p2.split(text))
        #    zh = ",".join(zh.split())
        outStr = zh

        outStr = jieba.cut(outStr)
        line = ''
        for word in outStr:
            if word != ' ':
                line = line + word + ' '
        line = line[:len(line) - 1]

        tbw = strs[0] + '\t' + strs[1] + '\t' + strs[2] + '\t' + strs[3] + '\t' + strs[4] + '\t' + line
        print(tbw)
        tbw = tbw + '\n'

        entbw = strs[1] + '\t' + strs[3] + '\t' + strs[4] + '\t' + line + '\n'

        entbw = tradition2simple(entbw)
        tbw = tradition2simple(tbw)
        outputEn.write(entbw)
        output.write(tbw)
    elif len(strs) == 6:
        text = strs[5]
        p2 = re.compile(r'[^\u4e00-\u9fa5]')
        zh = " ".join(p2.split(text))
    #    zh = ",".join(zh.split())
        outStr = zh

        outStr = jieba.cut(outStr)
        line = ''
        for word in outStr:
            if word != ' ':
                line = line + word + ' '
        line = line[:len(line) - 1]

        tbw = strs[0] + '\t' + strs[1] + '\t' + strs[2] + '\t' +  strs[3] + '\t' + strs[4] + '\t' + line
        print(tbw)
        tbw = tbw + '\n'

        entbw = strs[1] + '\t' + strs[3] + '\t' + strs[4] + '\t' + line + '\n'

        entbw = tradition2simple(entbw)
        tbw = tradition2simple(tbw)
        outputEn.write(entbw)
        output.write(tbw)
    else:
        count += 1

print (count)