#!/bin/bash
#2016/11/10    WYJ    first release

echo -e "--------------------------------------------------"
cd prepare
#预分词和词性标注
python preprocess.py Raw
python preprocess.py Test
echo -e "--------------------------------------------------"
dataDir="../../data/"
#数据格式转换
cd ../te
#训练数据格式转换
echo -e "正在转换Raw数据格式"
python convert_char.py Raw
cat ${dataDir}"Raw_char_tag.txt" | ./featGen_char.py > ${dataDir}"Raw_char_crfsuite.txt"
echo -e "Raw数据格式转换完毕"
#验证数据格式转换
echo -e "正在转换Test数据格式"
python convert_char.py Test
cat ${dataDir}"Test_char_tag.txt" | ./featGen_char.py > ${dataDir}"Test_char_crfsuite.txt"
echo -e "转换Test数据格式完毕"
echo -e "--------------------------------------------------"

#使用crfsuite learn对Raw数据进行训练
export LD_LIBRARY_PATH=$HOME/local/crfsuite/lib

cd ../../data
modelDir="../codes/te/model/"
echo -e "开始对Raw数据进行学习"
crfsuite learn -m ${modelDir}"car_char.model" Raw_char_crfsuite.txt
echo -e "Raw数据学习完毕"
echo -e "--------------------------------------------------"

#使用crfsuite learn对Test数据进行标记
echo -e "开始对Test数据进行标记"
crfsuite tag -m ${modelDir}"car_char.model" Test_char_crfsuite.txt > Test_char_tag_pred.txt
echo -e "对Test数据标记完毕"

cd ../codes/utils
python genTestTag_char.py Test

echo -e "--------------------------------------------------"

#分离Raw的tag, 输出文件名Raw_char_tag_pred.txt
cd ../../data
cat Raw_char_tag.txt | cut -d ' ' -f 5 > Raw_char_tag_pred.txt

#生成Raw_char_targets_1.json
cd ../codes/utils
python genTestTag_char.py Raw
cd ../prepare
python postTokenize.py Raw

#生成Test_char_targets_1.json
python postTokenize.py Test
exit 0
