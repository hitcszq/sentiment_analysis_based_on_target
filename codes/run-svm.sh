#!/bin/bash
#2016/10/17 WYJ first release

echo -e "---------------------------------------------------------------"
read -p "交叉测试:1,训练和预测:0, 请输入:" isvalid
isvalid=${isvalid:-"1"}

#使用SVM对Raw数据进行交叉测试
if [ ${isvalid} == "1" ]; then
    echo -e "开始对Raw进行交叉测试"
    cd sa
    python svm_stem.py Raw 1 0
    echo -e "---------------------------------------------------------------"

    read -p "是否进行训练和预测?是:y/Y,否:n/N,请输入:" res
    res=${res:-"n"}
    if [ ${res} == "y" -o ${res} == "Y" ]; then
        read -p "接下来选定c值对Raw数据进行训练,请输入C值:" c
        #cd ../sa
        c=${c:-"0.03125"}
        echo -e "模型训练开始"
        python svm_stem.py Raw 0 ${c}
        echo -e "模型训练完毕"
        echo -e "---------------------------------------------------------------"

        #使用SVM对Test数据进行预测
        echo -e "开始对测试数据进行预测"
        python svm_stem.py Test 0 0
        echo -e "对测试数据预测完毕"
        echo -e "---------------------------------------------------------------"

    fi

#使用交叉测试得到的c进行模型训练
else
    read -p "接下来选定c值对Raw数据进行训练,请输入C值:" c
    cd sa
    c=${c:-"0.03125"}
    echo -e "模型训练开始"
    python svm_stem.py Raw 0 ${c}
    echo -e "模型训练完毕"
    echo -e "---------------------------------------------------------------"

    #使用SVM对Test数据进行预测
    echo -e "开始对测试数据进行预测"
    python svm_stem.py Test 0 0
    echo -e "对测试数据预测完毕"
    echo -e "---------------------------------------------------------------"

fi

exit 0
