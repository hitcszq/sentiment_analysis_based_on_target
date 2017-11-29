# coding:utf8
from __future__ import division
from liblinear.python.liblinearutil import *
import json, re, sys
import csv as csv
import numpy as np



#========================================
#@param:
#- hn: HowNet是否使用，1:是; 0:否
#- ntusd: ntusd是否使用，1:是; 0:否
#- car: 领域词典car是否使用, 1:是; 0:否
#- gen: 领域通用词典是否使用, 1:是; 0:否
#
#@return:
#- pos: positive polarity dict
#- neg: negative polarity dict
#=======================================
def createSentimentDict(hn, ntusd, car, gen):
    pos = set([])
    neg = set([])
    if hn:
        with open('../../resources/hownet/正面情感词语（中文）.txt', 'r') as f:
            f.next()
            f.next()
            for line in f:
                word = line.strip()
                pos.add(word)
        with open('../../resources/hownet/正面评价词语（中文）.txt', 'r') as f:
            f.next()
            f.next()
            for line in f:
                word = line.strip()
                pos.add(word)
        with open('../../resources/hownet/负面情感词语（中文）.txt', 'r') as f:
            f.next()
            f.next()
            for line in f:
                word = line.strip()
                neg.add(word)
        with open('../../resources/hownet/负面情感词语（中文）.txt', 'r') as f:
            f.next()
            f.next()
            for line in f:
                word = line.strip()
                neg.add(word)

    if ntusd:
        with open('../../resources/ntusd/ntusd-positive.txt', 'r') as f:
            for line in f:
                word = line.strip()
                pos.add(word)
        with open('../../resources/ntusd/ntusd-negative.txt', 'r') as f:
            for line in f:
                word = line.strip()
                neg.add(word)

    if car:
        with open('../../resources/领域情感词典/car_pos.txt', 'r') as f:
            for line in f:
                word = line.strip()
                pos.add(word)
        with open('../../resources/领域情感词典/car_neg.txt', 'r') as f:
            for line in f:
                word = line.strip()
                neg.add(word)

    if gen:
        with open('../../resources/领域情感词典/general_pos.txt', 'r') as f:
            for line in f:
                word = line.strip()
                pos.add(word)
        with open('../../resources/领域情感词典/general_neg.txt', 'r') as f:
            for line in f:
                word = line.strip()
                neg.add(word)
    return pos, neg

#========================================================
#统计unigam
#@param:
#- textFile: file_ws.json list
#
#@return
#- wordDict
#========================================================
def totalUnigram(textFile):
    unigramDict = {}
    #加载停用词表
    stopWords = set([])
    #with open('../resources/停用词表/停用词表.txt', 'r') as f:
    #    for line in f:
    #        temp = line.strip()
    #        stopWords.add(temp)
    stopWords.add(' ')
    stopWords.add('\xc2\xa0')
    
    #加载已有的extra_dict
    edFile = open('../prepare/extra_car_dict.txt', 'r')
    edWords = set([])
    #for line in edFile:
    #    word = line.strip()
    #    edWords.add(word)
    #edFile.close()
    negation = set(['不', '不是', '不会', '不够', '没', '没有', '未'])
    for file in textFile:
        text = json.load(file)
        for code in text:
            i = 0
            for word in text[code]:
                word = word.encode('utf-8')
                for j in range(5):
                    if i-j-1>=0 and text[code][i-j-1].encode('utf-8') in negation:
                        word = 'NEG_'+word
                if word in stopWords or word in edWords:
                    continue
                if word not in unigramDict:
                    unigramDict[word] = 0
                else:
                    unigramDict[word] += 1
                i += 1

    return unigramDict

def totalNgram(win):
    features = {'unigram':{}, 'bigram':{}, 'trigram':{}, 'aspect_bigram':{}}
    train_ws = json.load(open('../../data/Train_char_ws.json', 'r'))
    train_targets = json.load(open('../../data/Train_char_targets_1.json', 'r'))
    for code in train_targets:
        sentence = train_ws[code]
        for t, p in train_targets[code]:
            p = p[0]
            i = p-win
            while i <= p+win:
                if i < 0:
                    i += 1
                    continue
                elif i >= len(sentence):
                    break
                #添加unigram
                if sentence[i] not in features['unigram']:
                    features['unigram'][sentence[i]] = 1
                else:
                    features['unigram'][sentence[i]] += 1
                #添加bigram
                if i+1 < len(sentence) and i+1 <= p+win:
                    if sentence[i]+'_'+sentence[i+1] not in features['bigram']:
                        features['bigram'][sentence[i]+'_'+sentence[i+1]] = 1
                    else:
                        features['bigram'][sentence[i]+'_'+sentence[i+1]] += 1
                #添加trigram
                if i+2 < len(sentence) and i+2 <= p+win:
                    if sentence[i]+'_'+sentence[i+1]+'_'+sentence[i+2] not in features['trigram']:
                        features['trigram'][sentence[i]+'_'+sentence[i+1]+'_'+sentence[i+2]] = 1
                    else:
                        features['trigram'][sentence[i]+'_'+sentence[i+1]+'_'+sentence[i+2]] += 1
                #添加aspect_bigram
                if i < p and sentence[i]+'_aspect' not in features['aspect_bigram']:
                    features['aspect_bigram'][sentence[i]+'_aspect'] = 1
                elif i < p and sentence[i]+'_aspect' in features['aspect_bigram']:
                    features['aspect_bigram'][sentence[i]+'_aspect'] += 1
                elif i > p and 'aspect_'+sentence[i] not in features['aspect_bigram']:
                    features['aspect_bigram']['aspect_'+sentence[i]] = 1
                elif i > p and 'aspect_'+sentence[i] in features['aspect_bigram']:
                    features['aspect_bigram']['aspect_'+sentence[i]] += 1
                i += 1
    return features
#========================================================
#统计bigram
#@param:
#- textFile: file_ws.json list
#
#@return
#- bigramDict
#========================================================

def totalBigram(inputFile):
    bigramDict = {}

    for file in textFile:
        text = json.load(file)
        for code in text:
            i = 0
            while i < len(text[code])-1:
                word1 = text[code][i].encode('utf-8')
                word2 = text[code][i+1].encode('utf-8')
                word = word1+word2
                if word not in bigramDict:
                    bigramDict[word] = 0
                else:
                    bigramDict[word] += 1
                i += 1
    return bigramDict

#========================================================
#统计unigramPos
#@param:
#- textFile: file_pos.json list
#
#@return
#- unigramPosDict
#========================================================

def totalUnigramPos(inputFile):
    unigramPosDict = {}
    for file in textFile:
        text = json.load(file)
        for code in text:
            for pos in text[code]:
                pos = pos.encode('utf-8')
                if pos not in unigramPosDict:
                    unigramPosDict[pos] = 0
                else:
                    unigramPosDict[pos] += 1
    return unigramPosDict

#========================================================
#统计bigramPos
#@param:
#- textFile: file_pos.json list
#
#@return
#- bigramPosDict
#========================================================

def totalBigramPos(inputFile):
    bigramPosDict = {}

    for file in textFile:
        text = json.load(file)
        for code in text:
            i = 0
            while i < len(text[code])-1:
                pos1 = text[code][i].encode('utf-8')
                pos2 = text[code][i+1].encode('utf-8')
                pos = pos1+pos2
                if pos not in bigramPosDict:
                    bigramPosDict[pos] = 0
                else:
                    bigramPosDict[pos] += 1
                i += 1
    return bigramPosDict

def totalUnigramParse(inputFile):
    unigramParseDict = {}
    for file in textFile:
        text = json.load(file)
        for code in text:
            i = 0
            while i < len(text[code]) - 1:
                parse = text[code][i][1]
                if parse not in unigramParseDict:
                    unigramParseDict[parse] = 0
                else:
                    unigramParseDict[parse] += 1
                i += 1
    return unigramParseDict

def totalParse():
    negation = set(['不', '不会', '不够', '不是', '没', '没有', '未'])
    raw_ws = json.load(open('../../data/Raw_char_ws.json', 'r'))
    raw_pos = json.load(open('../../data/Raw_char_pos.json', 'r'))
    raw_parse = json.load(open('../../data/Raw_char_parse.json', 'r'))
    test_ws = json.load(open('../../data/Test_char_ws.json', 'r'))
    test_pos = json.load(open('../../data/Test_char_pos.json', 'r'))
    test_parse = json.load(open('../../data/Test_char_parse.json', 'r'))
    all_ws = {}
    all_pos = {}
    all_parse = {}
    all_ws.update(raw_ws)
    all_ws.update(test_ws)
    all_pos.update(raw_pos)
    all_pos.update(test_pos)
    all_parse.update(raw_parse)
    all_parse.update(test_parse)
    
    aspect_polarity = {}
    adv_adj = {}
    for code in all_ws:
        sbv_vob_word = []
        sbv_vob_position = []
        i = 0
        for word, pos, parse in zip(all_ws[code], all_pos[code], all_parse[code]):
            if parse[1] == 'SBV' and ((pos == 'n' and all_pos[code][parse[0]] == 'a') or (pos == 'v' and all_pos[code][parse[0]] == 'a')):
                polarity = all_ws[code][parse[0]]
                for j in range(3):
                    if all_ws[code][parse[0]-j-1].encode('utf-8') in negation:
                        polarity = 'NEG_'+all_ws[code][parse[0]]
                if ('aspect', polarity) not in aspect_polarity:
                    aspect_polarity[('aspect', polarity)] = 1
                else:
                    aspect_polarity[('aspect', polarity)] += 1
            elif parse[1] == 'SBV' and pos == 'n':
                sbv_vob_word += ['aspect']
                sbv_vob_position += [parse[0]]
            if parse[1] == 'VOB' and pos == 'a' and parse[0] in sbv_vob_position:
                word = sbv_vob_word[sbv_vob_position.index(parse[0])]
                polarity = all_ws[code][parse[0]]
                for j in range(3):
                    if all_ws[code][parse[0]-j-1].encode('utf-8') in negation:
                        polarity = 'NEG_'+all_ws[code][parse[0]]
                if ('aspect', polarity) not in aspect_polarity:
                    aspect_polarity[('aspect', polarity)] = 1
                else:
                    aspect_polarity[('aspect', polarity)] += 1
            if parse[1] == 'ATT' and pos == 'a' and all_pos[code][parse[0]] == 'n':
                if (word, 'aspect') not in aspect_polarity:
                    aspect_polarity[(word, 'aspect')] = 1
                else:
                    aspect_polarity[(word, 'aspect')] += 1
            i += 1
    aspect_polarity_dict = {}
    for ap in aspect_polarity:
        if aspect_polarity[ap] > 4:
            aspect_polarity_dict[ap[0]] = 1
            aspect_polarity_dict[ap[1]] = 1
    return aspect_polarity_dict

#========================================
#产生unigram特征索引和unigram的对应关系
#@param:
#- unigramDict: unigram dictionary
#
#@return:
#- unigram: unigram集
#- unigramIdx: unigram索引
#=======================================
def genUnigram(unigramDict):
    unigram = [x[0] for x in unigramDict.iteritems()]

    #添加其实和结束助记符
    for i in range(41):
        temp = ''
        for j in range(i+1):
            temp += '*'
        unigram += ['S'+temp, temp+'E']

    unigramIdx = dict([(x, idx+1) for x, idx in zip(unigram, range(len(unigram)))])
    return unigram, unigramIdx

def genNgram(features):
    unigram = [x[0] for x in features['unigram']]
    bigram = [x[0] for x in features['bigram']]
    trigram = [x[0] for x in features['trigram']]
    aspect_bigram = [x[0] for x in features['aspect_bigram']]
    ngram = unigram+bigram+trigram+aspect_bigram

    ngramIdx = dict([x, idx+1] for x, idx in zip(ngram, range(len(ngram))))
    return ngram, ngramIdx
#=======================================
#产生bigram特征索引和bigram的对应关系
#@param:
#- bigramDict: bigram dictionary
#
#@return:
#- bigram: bigram集
#- bigramIdx: bigram索引
#=======================================
def genBigram(bigramDict):
    bigram = [x[0] for x in bigramDict.iteritems()]

    #添加其实和结束助记符
    #for i in range(41):
    #    temp = ''
    #    for j in range(i+1):
    #        temp += '*'
    #    unigram += ['S'+temp, temp+'E']
    l = len(unigram)
    bigramIdx = dict([(x, idx+l+1) for x, idx in zip(bigram, range(len(bigram)))])
    return bigram, bigramIdx

#=======================================
#产生unigramPos特征索引和unigramPos的对应关系
#@param:
#- unigramPosDict: unigramPos dictionary
#
#@return:
#- unigramPos: unigramPos集
#- unigramPosIdx: unigramPos索引
#=======================================
def genUnigramPos(unigramPosDict):
    unigramPos = [x[0] for x in unigramPosDict.iteritems()]
    
    l = len(unigram) + len(bigram)
    unigramPosIdx = dict([(x, idx+l+1) for x, idx in zip(unigramPos, range(len(unigramPos)))])
    return unigramPos, unigramPosIdx

#=======================================
#产生bigramPos特征索引和bigramPos的对应关系
#@param:
#- bigramPosDict: bigramPos dictionary
#
#@return:
#- bigramPos: bigramPos集
#- bigramPosIdx: bigramPos索引
#=======================================
def genBigramPos(bigramPosDict):
    bigramPos = [x[0] for x in bigramPosDict.iteritems()]
    
    l = len(unigram) + len(bigram) + len(unigramPos)
    bigramPosIdx = dict([(x, idx+l+1) for x, idx in zip(bigramPos, range(len(bigramPos)))])
    return bigramPos, bigramPosIdx

def genUnigramParse(unigramParseDict):
    unigramParse = [x[0] for x in unigramParseDict.iteritems()]

    l = len(unigram)+len(bigram)+len(unigramPos)+len(bigramPos)+len(sentiment)+len(wordEmbeddingIdx)
    unigramParseIdx = dict([(x, idx+l+1) for x, idx in zip(unigramParse, range(len(unigramParse)))])
    return unigramParse, unigramParseIdx

#==============================================
#产生sentiment和sentiment特征索引
#==============================================
def genSentiment():
    l = len(unigram)+len(bigram)+len(unigramPos)+len(bigramPos)
    sentiment = ['posScores', 'negScores', 'numPos', 'numNeg']
    sentimentIdx = dict([(x, idx+l+1) for x, idx in zip(sentiment, range(len(sentiment)))])
    return sentiment, sentimentIdx

#===========================================
#产生wordEmbedding并返回其特征的维度范围
#@param:
#- weFile
#
#@return
#- wordEmbedding
#- (起始维度，终止维度):
#============================================
def genWordEmbedding(weFile):
    wordEm = {}
    for line in weFile:
        temp = line.strip().split()
        if len(temp[0].split('_')) > 1:
            temp[0] = temp[0].replace('_', ' ')
        wordEm[temp[0]] = map(float, temp[1:])
    l = len(unigram) + len(bigram) + len(unigramPos) + len(bigramPos) + len(sentiment)
    wordEmIdx = [x+l+1 for x in range(len(wordEm['的'])*9)]
    return wordEm, wordEmIdx

def genParse(aspectPolarityDict):
    aspectPolarity = [x[0] for x in aspectPolarityDict.iteritems()]

    l = len(unigram)+len(bigram)+len(unigramPos)+len(bigramPos)+len(sentiment)+len(wordEmbeddingIdx)

    aspectPolarityIdx = dict([(x, idx+l+1) for x, idx in zip(aspectPolarity, range(len(aspectPolarity)))])
    return aspectPolarity, aspectPolarityIdx

#特征数据
unigram = []
unigramIdx = {}
bigram = []
bigramIdx = {}
unigramPos = []
unigramPosIdx = {}
bigramPos = []
bigramPosIdx = {}
sentiment = []
sentimentIdx = {}
wordEmbedding = {}
wordEmbeddingIdx = []
ngram = []
ngramIdx = {}
aspectPolarity = []
aspectPolarityIdx = {}
unigramParse = []
unigramParseIdx = {}


#======================================================
# pooling functions
#
# @params
#- 由词向量矩阵表示的句子矩阵, shape = (timeSteps, dim)
#
# @return
#- min, max, average, std, prd
#======================================================
def pooling(X):
    return np.min(X, axis=0), np.max(X, axis=0), np.average(X, axis=0), np.std(X,axis=0), np.prod(X, axis=0)

def genFeatures_3(u, b, up, bp, senti, we, par, win, filename):
    if not u and not b and not up and not bp and not senti and not we and not par:
        return None

    #加载数据
    data = json.load(open('../../data/'+filename+'_char_ws.json', 'r'))
    dataPos = json.load(open('../../data/'+filename+'_char_pos.json', 'r'))
    dataParse = json.load(open('../../data/'+filename+'_char_parse.json', 'r'))
    #加载情感词典
    positive, negative = createSentimentDict(1 ,0 ,1 ,1)

    #加载否定转移词
    negation = set(['不', '不会', '不是', '不够', '没', '没有', '未'])
    #加载转折连词
    conj = set(['但', '但是'])
    #加载比较词
    comp = set(['相比', '比', '不如', '优于', '比不上', '甩'])

    #方法二: 加载crf抽取得到的结果
    code_targets = json.load(open('../../data/'+filename+'_char_targets_1.json', 'r'))
    #顺序存储(句子编号, target)
    codeTargets = []
    #顺序存储: [target polarity], [features representation]
    featReps = [[], []]

    for code in data:

        #方法二: 使用crf抽取结果
        if code not in code_targets:
            continue

        code_target = code_targets[code]
        code_target = sorted(code_target, key=lambda d: d[0])
        target_position = sorted([x[1][0] for x in code_target])

        pre = ''
        featRep = {}
        if senti == 1:
            featRep[sentimentIdx['posScores']] = 0.0
            featRep[sentimentIdx['negScores']] = 0.0
            featRep[sentimentIdx['numPos']] = 0.0
            featRep[sentimentIdx['numNeg']] = 0.0
        if we == 1:
            weFeatRep = np.zeros(len(wordEmbeddingIdx))
            weNums = 0
            i = 0
            while i < len(wordEmbeddingIdx):
                featRep[wordEmbeddingIdx[i]] = 0.0
                i += 1

        for a, p in code_target:
            a = a.encode('utf-8')
            if a != pre and pre:
                codeTargets += [(code, pre)]
                featReps[0] += [0]
                featReps[1] += [featRep]
                featRep = {}
                if senti == 1:
                    featRep[sentimentIdx['posScores']] = 0.0
                    featRep[sentimentIdx['negScores']] = 0.0
                    featRep[sentimentIdx['numPos']] = 0.0
                    featRep[sentimentIdx['numNeg']] = 0.0
                if we == 1:
                    weFeatRep = np.zeros(len(wordEmbeddingIdx))
                    weNums = 0
                    i = 0
                    while i < len(wordEmbeddingIdx):
                        featRep[wordEmbeddingIdx[i]] = 0.0
                        i += 1
            
                
            #unigram feat
            if u == 1:
                i = 0
                for word in data[code]:
                    word = word.encode('utf-8')
                    for j in range(5):
                        if i-j-1>=0 and data[code][i-j-1].encode('utf-8') in negation:
                            word = 'NEG_'+word
                    if word in unigramIdx:
                        featRep[unigramIdx[word]] = 1
                    i += 1
            #bigram feat
            if b == 1:
                for step in xrange(13):
                    if p[1]+step+1<=len(data[code])-1:
                        idx = p[1]+step+1
                        if idx < len(data[code])-1:
                            word1 = data[code][idx].encode('utf-8')
                            word2 = data[code][idx+1].encode('utf-8')
                            word = word1 + word2
                            if word in bigramIdx:
                                featRep[bigramIdx[word]] = 1
            if up == 1:
                 for step in xrange(6):
                    if p[1] + step + 1 <= len(dataPos[code])-1:
                        idx = p[1]+step+1
                        pos = dataPos[code][idx].encode('utf-8')
                        if pos in unigramPosIdx:
                            featRep[unigramPosIdx[pos]] = 1
               
            #bigramPos feat
            if bp == 1:
                for step in xrange(15):
                    if p[0]-step-1>=0:
                        idx = p[0]-step-1
                        if idx < len(dataPos[code])-1:
                            pos1 = dataPos[code][idx].encode('utf-8')
                            pos2 = dataPos[code][idx+1].encode('utf-8')
                            pos = pos1+pos2
                            if pos in bigramPosIdx:
                                featRep[bigramPosIdx[pos]] = 1
                    if p[1]+step+1<=len(dataPos[code])-1:
                        idx = p[1]+step+1
                        if idx < len(dataPos[code])-1:
                            pos1 = dataPos[code][idx].encode('utf-8')
                            pos2 = dataPos[code][idx+1].encode('utf-8')
                            pos = pos1 + pos2
                            if pos in bigramPosIdx:
                                featRep[bigramPosIdx[pos]] = 1
            #sentiment feat
            if senti == 1:
                #找出句子中所有情感词
                sentiWords = []
                i = 0
                while i < len(data[code]):
                    k = 3
                    while k>=0:
                        temp = ''
                        j = i
                        while j < len(data[code]) and j <= i+k:
                            temp += data[code][j].encode('utf-8')
                            j += 1
                        if temp in positive:
                            sign = 1.0
                            l = 1
                            while i-l >=0 and l < 4:
                                if data[code][i-l].encode('utf-8') in negation:
                                    sign *= -1
                                l += 1
                            sentiWord = (temp, (i, i+k), sign)
                            sentiWords += [sentiWord]
                            i = j-1
                            break
                        elif temp in negative:
                            sign = -1.0
                            l = 1
                            while i-l >= 0 and l < 4:
                                if data[code][i-l].encode('utf-8') in negation:
                                    sign *= -1
                                l += 1
                            sentiWord = (temp, (i, i+k), sign)
                            sentiWords += [sentiWord]
                            i = j-1
                            break
                        k -= 1
                    i += 1
                #计算当前词sentiment特征
                posScores = 0.0
                negScores = 0.0
                totalScores = 0.0
                numPos = 0
                numNeg = 0
                for w, po, s in sentiWords:
                    if s > 0:
                        numPos += 1
                        if po[0] != p[0]:
                            posScores += s / abs(p[0] - po[0])
                        else:
                            posScores += s
                    else:
                        numNeg += 1
                        if po[0] != p[0]:
                            negScores += s / abs(p[0] - po[0])
                        else:
                            negScores += s
                featRep[sentimentIdx['posScores']] += posScores
                featRep[sentimentIdx['negScores']] += negScores
                featRep[sentimentIdx['numPos']] += numPos
                featRep[sentimentIdx['numNeg']] += numNeg
            #wordEmbedding特征
            if we == 1:
                #============================
                #   方法六
                #   用词向量构建多种组合特征
                #============================
                length = len(data[code])
                dim = len(wordEmbedding['的'])
                
                Full = np.zeros((length, dim))
                FullS = np.zeros((length, dim))

                i = 0
                while i < length:
                    word = data[code][i].encode('utf-8')
                    word_pos = dataPos[code][i]
                    if word in wordEmbedding:
                        Full[i] = wordEmbedding[word]
                        #if word in positive or word in negative:
                        #    FullS[i] = wordEmbedding[word]
                        if word_pos == 'a' or word_pos == 'd' or word_pos == 'c' or word_pos == 'i' or word_pos == 'v':
                            FullS[i] = wordEmbedding[word]
                    i += 1
                
                #切分成左，中，右三部分
                if p[0] == 0:
                    L = np.zeros((1, dim))
                    LS = np.zeros((1, dim))
                else:
                    L = Full[0:p[0], :]
                    LS = FullS[0:p[0], :]
                T = Full[p[0]:p[1]+1, :]
                if p[1] == length-1:
                    R = np.zeros((1, dim))
                    RS = np.zeros((1, dim))
                else:
                    R = Full[p[1]+1:, :]
                    RS = FullS[p[1]+1:, :]

                #pooling functions
                L_min, L_max, L_average, L_std, L_prod = pooling(L)
                T_min, T_max, T_average, T_std, T_prod = pooling(T)
                R_min, R_max, R_average, R_std, R_prod = pooling(R)
                Full_min, Full_max, Full_average, Full_std, Full_prod = pooling(Full)

                LS_min, LS_max, LS_average, LS_std, LS_prod = pooling(LS)
                RS_min, RS_max, RS_average, RS_std, RS_prod = pooling(RS)
                '''
                weFeatRep = np.concatenate((L_min, L_max, L_average, L_std, L_prod,
                                        T_min, T_max, T_average, T_std, T_prod,
                                        R_min, R_max, R_average, R_std, R_prod,
                                        Full_min, Full_max, Full_average, Full_std, Full_prod,
                                        LS_min, LS_max, LS_average, LS_std, LS_prod,
                                        RS_min, RS_max, RS_average, RS_std, RS_prod))
                '''
                '''
                weFeatRep += np.concatenate((L_average, L_max, L_min, L_std,
                                        T_average, T_max, T_min,
                                        R_average, R_max, R_min, R_std,
                                        Full_average, Full_max, Full_min, Full_std
                                        ))
                '''
                #weFeatRep += Full_average
                weFeatRep += np.concatenate((
                                        Full_average, L_average, R_average, T_average, L_std, R_std, Full_std,
                                        LS_average, RS_average
                                        ))
                weNums += 1
                
                if weNums >= 2:
                    weFeatRep *= (weNums-1)
                    weFeatRep /= weNums
                i = 0
                while i < len(wordEmbeddingIdx):
                    featRep[wordEmbeddingIdx[i]] = weFeatRep[i]
                    i += 1
            #句法特征
            if par == 1:
                  for step in xrange(10):
                    #if p[0] - step - 1 >= 0:
                    #    idx = p[0] - step - 1
                    #    parse = dataParse[code][idx][1]
                    #    if parse in unigramParseIdx:
                    #        featRep[unigramParseIdx[parse]] = 1

                    if p[1] + step <= len(dataParse[code])-1:
                        idx = p[1]+step
                        parse = dataParse[code][idx][1]
                        if parse in unigramParseIdx:
                            featRep[unigramParseIdx[parse]] = 1
               
            pre = a
        if pre:
            codeTargets += [(code, pre)]
            featReps[0] += [0]
            featReps[1] += [featRep]

    return codeTargets, featReps
#========================
#更新训练数据的标签
#@param
#- gold: 训练数据标签 Label.csv
#- codeTargets, featReps: genFeatures 返回的结果
#
#@return
#- 返回更新后的结果
#========================
def updateLabel(gold, codeTargets, featReps):
    goldDict1 = {}
    goldDict2 = {}
    for line in gold:
        temp = line.strip().split('\t')
        if temp[0] not in goldDict1:
            goldDict1[temp[0]] = [temp[1]]
            goldDict2[temp[0]] = [temp[2]]
        else:
            goldDict1[temp[0]] += [temp[1]]
            goldDict2[temp[0]] += [temp[2]]

    i = 0
    while i < len(codeTargets):
        code, target = codeTargets[i]
        if code not in goldDict1:
            del codeTargets[i]
            del featReps[0][i]
            del featReps[1][i]
        elif code in goldDict1:
            if target not in goldDict1[code]:
                del codeTargets[i]
                del featReps[0][i]
                del featReps[1][i]
            #删除包含多个视角的句子
            #elif len(goldDict1[code]) > 1:
            #    del codeTargets[i]
            #    del featReps[0][i]
            #    del featReps[1][i]

            #删除只包含一个视角的句子
            #elif len(goldDict1[code]) == 1:
            #    del codeTargets[i]
            #    del featReps[0][i]
            #    del featReps[1][i]
            else:
                if goldDict2[code][goldDict1[code].index(target)] == 'pos':
                    featReps[0][i] = 1
                elif goldDict2[code][goldDict1[code].index(target)] == 'neg':
                    featReps[0][i] = -1
                else:
                    featReps[0][i] = 0
                    #检查积极和消极
                    #del codeTargets[i]
                    #del featReps[0][i]
                    #del featReps[1][i]
                i += 1
            
    return codeTargets, featReps

#使用liblinear进行学习
def learn(codeTargets, featReps, isvalid, c):
    prob = problem(featReps[0], featReps[1])
    if isvalid:
        param = parameter('-C -v 5 -w-1 6 -w1 3 -w0 1')
        train(prob, param)
    else:
        param = parameter('-w-1 6 -w1 3 -w0 1 -c '+str(c))
        m = train(prob, param)
        save_model('model/unigram.model', m)
        pLabel, p, _ = predict(featReps[0], featReps[1], m)

#预测
def prediction(testCodeTargets, testFeatReps):
    m = load_model('model/unigram.model')
    predLabel, _, _ = predict(testFeatReps[0], testFeatReps[1], m)
    return predLabel

def writeResult(testCodeTargets, predLabel, filename):
    with open('../../data/detect/Label_'+filename+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['SentenceId', 'View', 'Opinion'])
        i = 0
        while i < len(predLabel):
            if predLabel[i] == 1: 
                writer.writerow([testCodeTargets[i][0], testCodeTargets[i][1], 'pos'])
            elif predLabel[i] == -1:
                writer.writerow([testCodeTargets[i][0], testCodeTargets[i][1], 'neg'])
            else:
                writer.writerow([testCodeTargets[i][0], testCodeTargets[i][1], 'neu'])
            i += 1

def helper_train(filename, isvalid, c):
    #转换格式
    codeTargets, featReps = genFeatures_3(1, 1, 1, 0, 1, 1, 1, 25, filename)

    #利用标签修正训练集的格式中的标签
    labelFile = open('../../data/Label.csv', 'r')
    labelFile.next()
    codeTargets, featReps = updateLabel(labelFile, codeTargets, featReps)
    labelFile.close()
    
    #训练模型并保存模型
    learn(codeTargets, featReps, isvalid, c)

def helper_predict(filename):
    testCodeTargets, testFeatReps = genFeatures_3(1, 1, 1, 0, 1, 1, 1, 25, filename)

    predLabel = prediction(testCodeTargets, testFeatReps)
    writeResult(testCodeTargets, predLabel, filename)

if __name__ == '__main__':
    sys.path.append('./liblinear/python')
    testFile = open('../../data/Test_char_ws.json', 'r')
    rawFile = open('../../data/Raw_char_ws.json', 'r')

    textFile = [rawFile, testFile]
   
    unigramDict = totalUnigram(textFile)

    testFile.close()
    rawFile.close()
    #构建unigram 和　unigram索引
    unigram, unigramIdx = genUnigram(unigramDict)
    #====================================================
    testFile = open('../../data/Test_char_ws.json', 'r')
    rawFile = open('../../data/Raw_char_ws.json', 'r')

    textFile = [rawFile, testFile]

    bigramDict = totalBigram(textFile)

    testFile.close()
    rawFile.close()
    #构建bigram 和 bigram索引
    bigram, bigramIdx = genBigram(bigramDict)
    #====================================================
    testFile = open('../../data/Test_char_pos.json', 'r')
    rawFile = open('../../data/Raw_char_pos.json', 'r')

    textFile = [rawFile, testFile]

    unigramPosDict = totalUnigramPos(textFile)

    testFile.close()
    rawFile.close()
    #构建unigramPos 和 unigramPos索引
    unigramPos, unigramPosIdx = genUnigramPos(unigramPosDict)
    #====================================================
    '''
    testFile = open('../../data/Test_char_pos.json', 'r')
    rawFile = open('../../data/Raw_char_pos.json', 'r')

    textFile = [rawFile, testFile]

    bigramPosDict = totalBigramPos(textFile)

    testFile.close()
    rawFile.close()
    '''
    #构建bigramPos 和 bigramPos索引
    #bigramPos, bigramPosIdx = genBigramPos(bigramPosDict)
    #======================================================

    #构建sentiment && sentimentIdx
    sentiment, sentimentIdx = genSentiment()
    #=====================================================================

    weFile = open('../../resources/wordembedding/carEmbedding_100.txt', 'r')
    #构建wordEmbedding && wordEmbeddingIdx
    wordEmbedding, wordEmbeddingIdx = genWordEmbedding(weFile)
    weFile.close()
    #=====================================================================
    testFile = open('../../data/Test_char_parse.json', 'r')
    rawFile = open('../../data/Raw_char_parse.json', 'r')

    textFile = [rawFile, testFile]

    unigramParseDict = totalUnigramParse(textFile)

    testFile.close()
    rawFile.close()

    #构建unigramParse和unigramParse索引
    unigramParse, unigramParseIdx = genUnigramParse(unigramParseDict)
    #==================================================================

    #构建句法特征
    #aspectPolarityDict = totalParse()
    #aspectPolarity, aspectPolarityIdx = genParse(aspectPolarityDict)
    #=====================================================================

    #helper_train('Raw', 0, 0.03125)
    #helper_predict('Test6')

    filename = sys.argv[1]
    isvalid = int(sys.argv[2])
    c = float(sys.argv[3])
    if filename == 'Train' or filename == 'Raw':
        helper_train(filename, isvalid, c)
    else:
        helper_predict(filename)
