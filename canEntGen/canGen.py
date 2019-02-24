# -*- coding: utf-8 -*-
import pymysql
import re

input = open('../mention_info.txt', 'r')

class Mention(object):
    def __init__(self, name, id, answer, leftcon, rightcon, docID):
        self.name = name
        self.id = id
        self.answer = answer
        self.leftcon = leftcon
        self.rightcon = rightcon
        self.docID = docID
class Entity(object):
    def __init__(self, name, id, text):
        self.name = name
        self.id = id
        self.text = text
def isredirect (text):
    if text.startswith('#REDIRECT') or text.startswith('#redirect') or text.startswith('#重定向') or text.startswith('#Redirect'):
        return True
    else:
        return False

def iscat (text):
    if text.startswith('{{Cat') or text.startswith('[[Cat') or text.startswith('{{cat') or text.startswith('[[cat') \
            or text.startswith('{{nav') or text.startswith('{{Nav') or text.startswith('{{Com'):
        return True
    else:
        return False

def isdis (text):
    if text.startswith('{{disa')  or '{{disambig' in text or '{{Disambig' in text:
        return True
    else:
        return False

def clean_text (text):
    text = re.sub('\{\{(.*?)\}\}', " ", text)
    return text

def check_duplicate (candidateEntities, entity):
    flag = False
    for canEn in candidateEntities:
        if canEn.id == entity.id:
            flag = True
    if flag == True:
        return True
    else:
        return False


mentionsdic = {}

freinput = open('./fredic.txt', 'r')

for fline in freinput:
    fline = fline.strip()
    strs = fline.split('\t')
    if len(strs) < 2:
        print ("wtf")
    me = strs[0]
    records = []
    for i in range(1,len(strs)):
        en = strs[i].split('*')[0]
        fre = strs[i].split('*')[1]
        record = [en, fre]
        records.append(record)
    mentionsdic[me] = records

handDic = {}

handinput = open('./handDic.txt', 'r')

for hline in handinput:
    hline = hline.strip()
    strs = hline.split('\t')
    if len(strs) < 2:
        print ("wtf")
    me = strs[0]
    ent = strs[1]
    handDic[me] = ent

print(handDic)

db = pymysql.connect("localhost", "root", "123", "elwiki",charset='utf8')

cursor = db.cursor()
totalmentionCount = 0
hasAnsMenCount = 0
noCanMenCount = []
falseMentions = []

output = open('./men_can_info.txt', 'w')

for line in input:
    totalmentionCount += 1
    strs = line.strip().split('\t')
    men = Mention(strs[2], strs[1], strs[3], strs[4], strs[5], strs[0])
    candidateEntities = []
    sql = "SELECT entity_id, entity_name, text FROM entity WHERE entity_name = '%s'" % (men.name)
    try:
        # 执行sql语句
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            if iscat(result[2]) is False and isdis(result[2]) is False:
#                entity = Entity(result[1],result[0],clean_text(result[2]))
                if isredirect(result[2]) is False:
                    entity = Entity(result[1],result[0],result[2])
                    if check_duplicate(candidateEntities, entity) is False:
                        candidateEntities.append(entity)
                else:
                    redirects = re.findall('\[\[(.*?)\]\]', result[2])
                    for redirect in redirects:
                        print(redirect)
                        sql_re = "SELECT entity_id, entity_name, text FROM entity WHERE entity_name = '%s'" % (redirect)
                        try:
                            # 执行sql语句
                            cursor.execute(sql_re)
                            results = cursor.fetchall()
                            for result in results:
                                #                        print(result)
                                if isredirect(result[2]) is False and iscat(result[2]) is False and isdis(
                                        result[2]) is False:
                                    #                entity = Entity(result[1],result[0],clean_text(result[2]))
                                    entity = Entity(result[1], result[0], result[2])
                                    if check_duplicate(candidateEntities, entity) is False:
                                        candidateEntities.append(entity)
                        except:
                            # 发生错误时回滚
                            db.rollback()
                            print("wronginRedirect")
    except:
        # 发生错误时回滚
        db.rollback()
        print("wrong")
#    print(str(totalmentionCount-1) + ' : candidates length : ' + str(len(candidateEntities)))

#   Does not matter whether it originally has entities or not....
#    if candidateEntities is None or len(candidateEntities)<1:
        # take advantage of the dictionary
    if men.name in mentionsdic:
        entities = mentionsdic[men.name]
#            print (entities)
        for entity in entities:
            newsql = "SELECT entity_id, entity_name, text FROM entity WHERE entity_name = '%s'" % (entity[0])
            try:
                # 执行sql语句
                cursor.execute(newsql)
                results = cursor.fetchall()
                for result in results:
#                        print(result)
                    if  iscat(result[2]) is False and isdis(result[2]) is False:
                        #                entity = Entity(result[1],result[0],clean_text(result[2]))
                        if isredirect(result[2]) is False:
                            entity = Entity(result[1], result[0], result[2])
                            if check_duplicate(candidateEntities, entity) is False:
                                candidateEntities.append(entity)
                        else:
                            redirects = re.findall('\[\[(.*?)\]\]', result[2])
                            for redirect in redirects:
                                print(redirect)
                                sql_re = "SELECT entity_id, entity_name, text FROM entity WHERE entity_name = '%s'" % (
                                redirect)
                                try:
                                    # 执行sql语句
                                    cursor.execute(sql_re)
                                    results = cursor.fetchall()
                                    for result in results:
                                        #                        print(result)
                                        if isredirect(result[2]) is False and iscat(result[2]) is False and isdis(
                                                result[2]) is False:
                                            #                entity = Entity(result[1],result[0],clean_text(result[2]))
                                            entity = Entity(result[1], result[0], result[2])
                                            if check_duplicate(candidateEntities, entity) is False:
                                                candidateEntities.append(entity)
                                except:
                                    # 发生错误时回滚
                                    db.rollback()
                                    print("wronginRedirect2")
            except:
                    # 发生错误时回滚
                    db.rollback()
                    print("wrongagain")

        # cheat dictionary

    if men.name in handDic:
        entit = handDic[men.name]
        #            print (entities)
        newsql = "SELECT entity_id, entity_name, text FROM entity WHERE entity_name = '%s'" % (entit)
        try:
            # 执行sql语句
            cursor.execute(newsql)
            results = cursor.fetchall()
            for result in results:
                #                        print(result)
                if iscat(result[2]) is False and isdis(result[2]) is False:
                    #                entity = Entity(result[1],result[0],clean_text(result[2]))
                    if isredirect(result[2]) is False:
                        entity = Entity(result[1], result[0], result[2])
                        if check_duplicate(candidateEntities, entity) is False:
                            candidateEntities.append(entity)
                    else:
                        redirects = re.findall('\[\[(.*?)\]\]', result[2])
                        for redirect in redirects:
                            print(redirect)
                            sql_re = "SELECT entity_id, entity_name, text FROM entity WHERE entity_name = '%s'" % (
                                redirect)
                            try:
                                # 执行sql语句
                                cursor.execute(sql_re)
                                results = cursor.fetchall()
                                for result in results:
                                    #                        print(result)
                                    if isredirect(result[2]) is False and iscat(result[2]) is False and isdis(
                                            result[2]) is False:
                                        #                entity = Entity(result[1],result[0],clean_text(result[2]))
                                        entity = Entity(result[1], result[0], result[2])
                                        if check_duplicate(candidateEntities, entity) is False:
                                            candidateEntities.append(entity)
                            except:
                                # 发生错误时回滚
                                db.rollback()
                                print("wronginRedirect3")
        except:
            # 发生错误时回滚
            db.rollback()
            print("wronghandDic")
#    print(str(totalmentionCount - 1) + ' : candidates length : ' + str(len(candidateEntities)))

    if candidateEntities is None or len(candidateEntities) < 1:
        noCanMenCount.append(men)
    else:
        trueflag = False
        '''
        maxlenEn = candidateEntities[0]
        for canEn in candidateEntities:
            if len(canEn.text) >= len(maxlenEn.text):
                maxlenEn = canEn
        if maxlenEn.name == men.answer:
            trueflag = True
            print (men.id + '\t' + men.name + '\t' + canEn.name + '\t' + canEn.text)
        else:
            print ('Wrong!!!'+ '\t' + men.id + '\t' + men.name + '\t' + canEn.name + '\t' + canEn.text)
        '''
        for canEn in candidateEntities:
            tbw = str(men.docID) + '\t' + men.id + '\t' + men.name + '\t' + str(canEn.id) + '\t' + canEn.name + '\t' + canEn.text + '\n'
            output.write(tbw)

            if canEn.name == men.answer:
                trueflag = True
                print(men.id + '\t' + men.name + '\t' + canEn.name + '\t' + canEn.text)
        if trueflag is True:
            hasAnsMenCount += 1
        if trueflag is False:
            falseMentions.append(men)
output.close()


print('There are ' + str(totalmentionCount) + 'mentions')
print(str(len(noCanMenCount)) + " Mentions have no entities: ")
for mmm in noCanMenCount:
    print (mmm.name + '\t' + mmm.answer + '\t' + mmm.leftcon + '\t' + mmm.rightcon)
print(str(len(falseMentions)) + " Mentions have false entities: ")
for mmm in falseMentions:
    print(mmm.name + '\t' + mmm.answer + '\t' + mmm.leftcon + '\t' + mmm.rightcon)
print (str(hasAnsMenCount) + ' out of ' + str(totalmentionCount-len(noCanMenCount)) + 'are right')






