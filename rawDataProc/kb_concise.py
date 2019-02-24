#!/usr/bin/python
# -*- coding: UTF-8 -*-

entity_ids = []
names = []
input = open('./kb.xml', 'r')
output = open('./kb_concise.txt', 'w')

for line in input:
    line = line.strip()
    if line.startswith('<entity_id>'):
        entity_id = line.replace('<entity_id>','').replace('</entity_id>','')
        print entity_id
        entity_ids.append(entity_id)
    if line.startswith('<name>'):
        name = line.replace('<name>', '').replace('</name>', '')
        print name
        names.append(name)

for i in range(0,len(entity_ids)):
    tbw = entity_ids[i] + '\t' + names[i]
    print tbw
    tbw = tbw + '\n'
    output.write(tbw)

output.close()