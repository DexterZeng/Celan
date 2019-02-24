# -*- coding: utf-8 -*-

class Entity(object):
    def __init__(self, name, id, text):
        self.name = name
        self.id = id
        self.text = text

def replace(entities, entity):
    flag = False
    newentities = entities
    for en in entities:
        if en.name == entity.name:
            flag = True
            if len(en.text) < len(entity.text):
                newentities.remove(en)
                newentities.append(entity)
    if flag is False:
        newentities.append(entity)
    return newentities

def transform (text, length):
    strs = text.split(' ')
    if len(strs) >= length:
        strs = strs[:length]
    line = ''
    for word in strs:
        line = line + word + ' '
    line = line[:len(line) - 1]
    return line

input = open('./can_entities.txt', 'r')

canEntities = {}
for line in input:
    line = line.strip()
    strs = line.split('\t')
    mentionID = strs[0]
    if len(strs) == 4:
        entity = Entity(strs[2], strs[1], strs[3])
        if mentionID not in canEntities.keys():
            entities = []
            entities.append(entity)
            canEntities[mentionID] = entities
        else:
            entities = canEntities[mentionID]
            entities = replace(entities, entity)
            canEntities[mentionID] = entities
print(len(canEntities))


output = open('./can_entities_refined.txt', 'w')
disoutput = open('./can_entities_tobeD.txt', 'w')
solocounter = 0
for ce in canEntities:
    if len(canEntities.get(ce)) == 1:
        solocounter += 1
        ents = canEntities[ce]
        for ent in ents:
            output.write(ce + '\t' + ent.id + '\t' + ent.name + '\t' + transform(ent.text,100) + '\n')
    else:
        ents = canEntities[ce]
        for ent in ents:
            output.write(ce + '\t' + ent.id + '\t' + ent.name + '\t' + transform(ent.text,100) + '\n')
            disoutput.write(ce + '\t' + ent.id + '\t' + ent.name + '\t' + transform(ent.text,100) + '\n')
print(solocounter)
input.close()
output.close()
disoutput.close()