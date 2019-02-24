# -*- coding: utf-8 -*-
import re

#read mentions
mentions = []
mentioninput = open('../mention_info.txt', 'r')
for mline in mentioninput:
    mline = mline.strip().split('\t')
    if mline[2] not in mentions:
        mentions.append(mline[2])
print len(mentions)

input = open('../pairs.txt', 'r')
dic = {}
i = 0
for line in input:
    print i
    i = i + 1
    strs = line.strip().split('\t')
    if len(strs) == 2:
        name = strs[0]
        entity = strs[1]
        if name in mentions:
            if name not in dic.keys():
                allentities = []
                entityfre = [entity, 1]
                allentities.append(entityfre)
                dic[name] = allentities
            else:
                allentities = dic[name]
                flag = False
                for en in allentities:
                    if en[0] == entity:
                        flag = True
                        en[1] += 1
                if flag is not True:
                    entityfre = [entity, 1]
                    allentities.append(entityfre)
                dic[name] = allentities

input.close()

output = open('./fredic.txt', 'w')
for na in dic:
    allentities = dic[na]
    # sort
    allentities.sort(key=lambda x: x[1], reverse=True)
    tbw = na + '\t'
    for eachentity in allentities:
        een = eachentity[0]
        if '(' in eachentity[0]:
            een = eachentity[0].replace(' ', '_')

        tbw += een + '*' + str(eachentity[1]) + '\t'
    tbw = tbw[:len(tbw) - 1]
    print(tbw)
    output.write(tbw + '\n')