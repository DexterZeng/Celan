# -*- coding: utf-8 -*-

def fill_up(li, thres):
    if len(li) < thres:
        for j in range(0, thres-len(li)):
            li.append(150561)
    return li

def transform (text, length, word2id):
    strs = text.split(' ')
    if len(strs) >= length:
        strs = strs[:length]
    else:
        strs = fill_up(strs, length)
    nums = []
    for st in strs:
        if type(st) == int:
            nums.append(st)
        else:
            nums.append(word2id[st])
    line = ''
    for word in nums:
        line = line + str(word) + ' '
    line = line[:len(line) - 1]
    return line

word_embedmatrix_input = open('Word_Matrix_new.txt', 'r')
word2id = {}
for wline in word_embedmatrix_input:
    wline = wline.strip().split('\t')
    idnum = int(wline[0])
    name = wline[1]
    word2id[name] = idnum

eninput = open('../can_entity/can_entities_refined.txt', 'r')
output = open('./entities_dis_idized.txt', 'w')
for eline in eninput:
    eline = eline.strip().split('\t')
    dis = eline[3]
    dis = transform(dis, 100, word2id)
    tbw = eline[0] + '\t' + eline[1] + '\t' + eline[2] + '\t' + dis
    print(tbw)
    tbw = tbw + '\n'
    output.write(tbw)
output.close()

