#!/bin/bash

# 假设训练集在 data/train/*.txt
# 测试集在 data/test/20190801_testdata.txt
mkdir -p data_proc/
for path in data/**/*.txt; do
    echo processing ${path}
    new_path=`echo -n ${path} | sed -E s/^data/data_proc/ -`
    sed -E 's/[:;,]/ /g' $path > ${new_path}
done