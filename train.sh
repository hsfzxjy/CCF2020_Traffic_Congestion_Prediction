#!/bin/bash

export weight=1,1.5,3
export clean=True

# 训练 GBDT 模型并保存结果
for day in $(seq -f "%02g" 1 30); do
	`which python3` main.py --save-model models/30days_gbdt/${day}/ --train data_proc/train/201907${day}.txt
done

# 训练 GOSS 模型并保存结果
export no_bagging=True
export boosting=goss
for day in $(seq -f "%02g" 1 30); do
        `which python3` main.py --save-model models/30days_goss/${day}/ --train data_proc/train/201907${day}.txt
done
