import sys
import os
import os.path as osp
import pandas as pd
import numpy as np
from glob import glob

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from collections import Counter
from scipy.misc import derivative


def get_models():
    for model_path in opts.models:
        if ":" in model_path:
            model_path, _, weight = model_path.rpartition(":")
            weight = float(weight)
        else:
            weight = 1
        for mp in glob(model_path.replace("@", "*")):
            if mp.endswith(".npy"):
                yield np.load(mp), weight, mp
                continue
            model = lgb.Booster(model_file=mp)
            yield model, weight, mp


def evaluate(testset):
    test_pred = np.zeros((testset.shape[0], 3))
    for model, weight, name in get_models():
        print("Eval", name)
        if isinstance(model, np.ndarray):
            test_pred += model * weight
        else:
            test_pred += (model.predict(testset[use_cols], num_iteration=model.best_iteration) * weight)
        print(test_pred.max(axis=0))
        print(test_pred.sum(axis=1))
    test_pred = np.argmax(test_pred, axis=1)
    if opts.eval:
        print(f1_score_eval(test_pred, testset["label"]))

    testset["label"] = test_pred + 1
    return testset[["link", "current_slice_id", "future_slice_id", "label"]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data_proc/20190801_testdata.txt")
    parser.add_argument("--attr", default="data/attr.txt")
    parser.add_argument("--save-pred")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--models", nargs="+")
    opts = parser.parse_args()

    attr = pd.read_csv(
        opts.attr,
        sep="\t",
        names=[
            "link",
            "length",
            "direction",
            "path_class",
            "speed_class",
            "LaneNum",
            "speed_limit",
            "level",
            "width",
        ],
        header=None,
    )

    test = gen_feats(opts.test, False, os.getenv("clean"), False)
    test = test.merge(attr, on="link", how="left")

    use_cols = [i for i in test.columns if i not in ["link", "label", "future_slice_id", "label_pred"]]

    evaled = evaluate(test)

    evaled.to_csv(opts.save_pred, index=False, encoding="utf8")
