# coding:utf-8
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
import json, jieba, sys

filename = sys.argv[1]
#句子中汽车品牌替换表
replaceList = ['CARZERO', 'CARONE', 'CARTWO', 'CARTHREE', 'CARFOUR', 'CARFIVE', 'CARSIX', 'CARSEVEN', 'CAREIGHT', 'CARNINE',
            'CARTEN', 'CARELEVEN', 'CARTWELVE', 'CARTHIRTEEN', 'CARFORTEEN', 'CARFIFTEEN', 'CARSIXTEEN', 'CARSEVENTEEN', 'CAREIGHTEEN',
            'CARNINTEEN','CARTWENTY', 'CARTWENTYONE', 'CARTWENTYTWO', 'CARTWENTYTHREE', 'CARTWENTYFOUR', 'CARTWENTYFIVE', 'CARTWENTYSIX']

with open('../../data/'+filename+'_char_targets.json') as f:
    trainTargets = json.load(f)

#读取Train.csv
codeSentence = {}
with open('../../data/'+filename+'.csv', 'r') as f:
    f.next()
    for line in f:
        temp = line.strip().split('\t')
        codeSentence[temp[0]] = temp[1]

#======================================================
#1. 
# 记录每个句子编号和对应的汽车品牌和替换表的对应关系
# {code: {target: replaceListIdx, }, }
#
#2. 
# 修改codeSentence
#======================================================
codeTargetReplace = {}
for code in trainTargets:
    temp = {}
    tempCnt = 0
    for target, position in trainTargets[code]:
        target = target.encode('utf-8')
        if target not in temp:
            temp[target] = tempCnt
            tempCnt += 1
            #print temp[target]
            codeSentence[code] = codeSentence[code].replace(target, replaceList[temp[target]])

    codeTargetReplace[code] = temp

#保存Train的句子编号和对应汽车品牌替换表
TrainCodeTargetReplace = {'replaceList':replaceList, 'codeTargetReplace':codeTargetReplace}
with open('../../data/'+filename+'_code_target_replace.json', 'w') as f:
    json.dump(TrainCodeTargetReplace, f)

#保存修改后的codeSentence
with open('../../data/'+filename+'_replaced.csv', 'w') as f:
    f.write('SentenceId\tContent')
    for code in codeSentence:
        f.write('\n'+code+'\t'+codeSentence[code])

#ws
for car in replaceList:
    jieba.suggest_freq(car, True)

codeSentenceWs = {}
codeSentencePos = {}
codeSentenceParse = {}
#f = open('log.txt', 'w')
postagger = Postagger()
parser = Parser()
postagger.load('ltp_data/pos.model')
parser.load('ltp_data/parser.model')
for code in codeSentence:
    tempMap = {}
    if code in codeTargetReplace:
        for car in codeTargetReplace[code]:
            tempMap[replaceList[codeTargetReplace[code][car]].decode('utf-8')] = car.decode('utf-8')
    temp = jieba.lcut(codeSentence[code])

    words = [w.encode('utf-8') for w in temp]
    poss = list(postagger.postag(words))
    arcs = parser.parse(words, poss)
    codeSentenceParse[code] = [(arc.head-1, arc.relation) for arc in arcs]

    codeSentenceWs[code] = []
    codeSentencePos[code] = []
    for word, pos in zip(temp, poss):
        if word in tempMap:
            codeSentenceWs[code] += [tempMap[word]]
            codeSentencePos[code] += ['nz']
        else:
            codeSentenceWs[code] += [word]
            codeSentencePos[code] += [pos]
    #查看效果
    #print ' '.join(codeSentenceWs[code])
    #print ' '.join(codeSentencePos[code])
    #print ' '.join('%d:%s' % (head, relation) for head, relation in codeSentenceParse[code] )
    #break
postagger.release()
parser.release()

with open('../../data/'+filename+'_char_ws.json', 'w') as f:
    json.dump(codeSentenceWs, f)
with open('../../data/'+filename+'_char_pos.json', 'w') as f:
    json.dump(codeSentencePos, f)
with open('../../data/'+filename+'_char_parse.json', 'w') as f:
    json.dump(codeSentenceParse, f)

trainTargets_1 = {}
for code in trainTargets:
    for target in codeTargetReplace[code]:
        i = 0
        while i < len(codeSentenceWs[code]):
            if codeSentenceWs[code][i].encode('utf-8') == target:
                if code not in trainTargets_1:
                    trainTargets_1[code] = [(target.decode('utf-8'), (i, i))]
                else:
                    trainTargets_1[code] += [(target.decode('utf-8'), (i, i))]
            i += 1
with open('../../data/'+filename+'_char_targets_1.json', 'w') as f:
    json.dump(trainTargets_1, f)
