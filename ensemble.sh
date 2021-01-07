#!/bin/bash

# 以 6:1 的权重融合 GBDT 和 GOSS 两种结果

`which python3` ensemble.py --models models/30days_gbdt/@/@.npy:6.0 models/30days_goss/@/@.npy:1.0