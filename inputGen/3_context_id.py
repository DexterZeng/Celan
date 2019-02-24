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

eninput = open('./context_formalized_new.txt', 'r')
output = open('./context_formalized_idised.txt', 'w')
for eline in eninput:
    eline = eline.strip().split('\t')
    left = eline[1]
    left = transform(left, 20, word2id)
    right = eline[2]
    right = transform(right, 20, word2id)
    tbw = eline[0] + '\t' + left + '\t' + right
    print(tbw)
    tbw = tbw + '\n'
    output.write(tbw)
output.close()

