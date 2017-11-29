# coding:utf-8
#产生训练词向量的文本

import json

with open('../../data/Raw_char_targets_1.json', 'r') as f:
    raw_targets = json.load(f)
with open('../../data/Test_char_targets_1.json', 'r') as f:
    test_targets = json.load(f)
code_targets = {}
code_targets.update(raw_targets)
code_targets.update(test_targets)

with open('../../data/Raw_char_ws.json', 'r') as f:
    raw = json.load(f)
with open('../../data/Test_char_ws.json', 'r') as f:
    test = json.load(f)
data = {}
data.update(raw)
data.update(test)

word2vec_input_file = open('../../data/word2vec_input.txt', 'w')
flag_1 = 0
for code in data:
    if code in code_targets:
        for t, p in code_targets[code]:
            p = p[0]
            if len(t.split()) > 1:
                data[code][p] = t.replace(' ', '_')
    flag = 0
    for word in data[code]:
        if flag == 0:
            word2vec_input_file.write(word.encode('utf-8'))
            flag = 1
        else:
            word2vec_input_file.write(' '+word.encode('utf-8'))
    if flag_1 == 0:
        flag_1 = 1
    else:
        word2vec_input_file.write('\n')
word2vec_input_file.close()
