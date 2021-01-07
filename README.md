# CCF2020_Traffic_Congestion_Prediction

https://www.datafountain.cn/competitions/466/ 的 GBDT 解法

## 数据处理

```
data
├── attr.txt
├── test
│   └── 20190801_testdata.txt
└── train
    └── *.txt
```

执行 `bash prepare.sh`

## 训练

`bash train.sh`

## 聚合

`bash ensemble.sh`