import sys
import os
import os.path as osp
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from collections import Counter
from scipy.misc import derivative

# 标签特征所在的列号
label_cols = list(range(7, 129, 5))

# N x 4 矩阵，每个 1 x 4 代表某个时间片 speed, eta, label, cnt 的列号
indices = np.array([i + j for i in range(5, 129, 5) for j in range(4)]).reshape(-1, 4)


def gen_feats(path, is_train, clean, weighted):
    """
    特征工程，扩充原有的特征得到更具代表性的 Feature

    @param path      数据文件的路径
    @param is_train  这个文件是否为训练集
    @param clean     是否做数据清理
    @param weighted  是否加权重

    @returns 整理后数据的 DataFrame
    """
    print("Loading", path)
    sys.stdout.flush()
    df = pd.read_csv(path, delimiter=" ", header=None)

    # 所有标签做变换
    # 4 -> 2
    # 3 -> 2
    # 2 -> 1
    # 1 -> 0
    # 0 -> -1 仅出现在测试集
    df.loc[:, label_cols + [1]] = (df[label_cols + [1]].clip(lower=0, upper=3, axis=1) - 1)

    if clean:
        # 第一步清洗，根据训练集的特点：
        # 如果 speed == 0 and label == -1，则数据完全没有信息量，全部置为 nan；
        # 如果 speed != 0 and label == -1，则数据中 speed, label 无意义，将这两个置为 nan。
        # 后续会对 nan 做特殊处理
        for group in indices:
            na_mask = df[group[2]] == -1
            na_mask_2 = df[group[0]] == 0
            df.loc[df.index[na_mask], group[0]] = np.nan
            df.loc[df.index[na_mask & na_mask_2], group[1]] = np.nan
            df.loc[df.index[na_mask], group[2]] = np.nan
            df.loc[df.index[na_mask & na_mask_2], group[3]] = np.nan

    # 返回结果
    res = {}
    res["label"] = df[1]
    res["current_slice_id"] = df[2]
    res["future_slice_id"] = df[3]

    prefixes = []
    suffixes = [f"{x}_{y}" for x in ["speed", "eta", "cnt"] for y in ["min", "max", "mean", "std"]]
    for i, group in enumerate(indices.reshape(5, 5, 4)):
        # 对每个时间段的 5 个时间片，原有的五个 speed, eta, label, cnt 变换为 13 条特征，即
        # speed, eta, cnt 的 min, max, mean, std 以及 label 的众数
        prefix = "current" if i == 0 else f"his_{(5-i)*7}"
        if i > 0:
            prefixes.append(prefix)
        res[f"{prefix}_speed_min"] = df[group[:, 0]].min(axis=1)
        res[f"{prefix}_speed_max"] = df[group[:, 0]].max(axis=1)
        res[f"{prefix}_speed_mean"] = df[group[:, 0]].mean(axis=1)
        res[f"{prefix}_speed_std"] = df[group[:, 0]].std(axis=1)

        res[f"{prefix}_eta_min"] = df[group[:, 1]].min(axis=1)
        res[f"{prefix}_eta_max"] = df[group[:, 1]].max(axis=1)
        res[f"{prefix}_eta_mean"] = df[group[:, 1]].mean(axis=1)
        res[f"{prefix}_eta_std"] = df[group[:, 1]].std(axis=1)

        res[f"{prefix}_cnt_min"] = df[group[:, 3]].min(axis=1)
        res[f"{prefix}_cnt_max"] = df[group[:, 3]].max(axis=1)
        res[f"{prefix}_cnt_mean"] = df[group[:, 3]].mean(axis=1)
        res[f"{prefix}_cnt_std"] = df[group[:, 3]].std(axis=1)

        res[f"{prefix}_state"] = fast_mode(df, group[:, 2], clean)

        if clean:
            # 第二步清理
            # 如果当前时间段中某个属性全部缺失（导致统计量为 nan），将其置为 0
            for suffix in suffixes:
                key = f"current_{suffix}"
                res[key].fillna(0, inplace=True)
            res["current_state"].fillna(0, inplace=True)

    res["link"] = df[0]

    res = pd.DataFrame(data=res)

    if clean:
        # 第三步整理

        # 如果某个时间段某个统计量缺失，尝试用同期的相同统计量的均值（或众数，对于 label）填补
        # 但如果该统计量同期全部缺失，此时填的还是 nan
        for suffix in suffixes:
            keys = [f"{prefix}_{suffix}" for prefix in prefixes]
            mean = res[keys].mean(axis=1)
            for key in keys:
                res[key].fillna(mean, inplace=True)
        state_keys = [f"{prefix}_state" for prefix in prefixes]
        mode = fast_mode(res, state_keys, clean)
        for key in state_keys:
            res[key].fillna(mode, inplace=True)

        # 如果仍有缺失，说明该样本信息量缺失严重，将缺失处直接置 0
        res.fillna(0, inplace=True)

    # 将 label 转为 int，如此一来 GBDT 会自动当为离散变量
    for col in res.columns:
        if "label" in col or "state" in col:
            res[col] = res[col].astype(np.int32) + (1 if "state" in col else 0)

    # 样本的权重，对于三类样本默认赋值 1:1.5:3
    res["weight"] = 1.0
    if weighted:
        print("weighted")
        for idx, w in enumerate(map(float, os.getenv("weight", "1,1.5,3").split(","))):
            res.loc[res.index[res["label"] == idx], "weight"] = w

    return res


def fast_mode(df, keys, clean):
    """
    用向量化的代数运算快速求众数，比直接 apply scipy.stats.mode 更快
    仅对取值集合为 {np.nan, 0, 1, 2} 的列有效
    """
    s_count = [(df[keys] == i).sum(axis=1) for i in range(3)]
    s_count = pd.concat(s_count, axis=1)
    state = s_count.idxmax(axis=1).astype(np.float16)
    if clean:
        state[s_count.sum(axis=1) == 0] = np.nan
    return state


def f1_score_eval(preds, valid_df):
    """
    求 F1 Score，用于 GBDT 评估
    """
    labels = valid_df.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels, y_pred=preds, average=None)
    scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
    return "f1_score", scores, True


def lgb_train(
    train_: pd.DataFrame,
    test_: pd.DataFrame,
    use_train_feats: list,
    id_col: str,
    label: str,
    n_splits: int,
    split_rs: int,
    is_shuffle=True,
    use_cart=False,
    cate_cols="auto",
) -> pd.DataFrame:
    """
    训练 GBDT。
    有一个可调的超参 boosting_type = gbdt | goss 用于控制 Boosting 算法。
    """
    if not cate_cols:
        cate_cols = []
    print("data shape:\ntrain--{}\ntest--{}".format(train_.shape, test_.shape))
    print("Use {} features ...".format(len(use_train_feats)))
    print("Use lightgbm to train ...")
    n_class = train_[label].nunique()
    train_[f"{label}_pred"] = 0
    test_pred = np.zeros((test_.shape[0], n_class))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = [x for x in use_train_feats if x != "weight"]

    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=split_rs)
    train_user_id = train_[id_col].unique()

    params = {
        "learning_rate": 0.05,
        "boosting_type": os.getenv("boosting", "gbdt"),
        "objective": "multiclass",
        "metric": "None",
        "num_leaves": 31,
        "num_class": n_class,
        "feature_fraction": 0.8,
        **({
            "bagging_fraction": 0.8,
            "bagging_freq": 5
        } if not os.getenv("no_bagging") else {}),
        "seed": 1,
        "bagging_seed": 1,
        "feature_fraction_seed": 7,
        "min_data_in_leaf": 20,
        "nthread": -1,
        "verbose": -1,
        "weight_column": "weight",
    }
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_user_id), start=1):
        print("the {} training start ...".format(n_fold))
        sys.stdout.flush()
        train_x, train_y = (
            train_.loc[train_[id_col].isin(train_user_id[train_idx]), use_train_feats],
            train_.loc[train_[id_col].isin(train_user_id[train_idx]), label],
        )

        valid_x, valid_y = (
            train_.loc[train_[id_col].isin(train_user_id[valid_idx]), use_train_feats],
            train_.loc[train_[id_col].isin(train_user_id[valid_idx]), label],
        )
        print(f"for train user:{len(train_idx)}\nfor valid user:{len(valid_idx)}")
        sys.stdout.flush()

        dtrain = lgb.Dataset(
            train_x.drop(columns="weight"),
            weight=train_x["weight"],
            label=train_y,
            categorical_feature=cate_cols,
        )
        dvalid = lgb.Dataset(valid_x.drop(columns="weight"), label=valid_y, categorical_feature=cate_cols)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=int(os.environ.get("rounds", 5000)),
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=10,
            feval=f1_score_eval,
            fobj=fobj,
        )
        fold_importance_df[f"fold_{n_fold}_imp"] = clf.feature_importance(importance_type="gain")
        train_.loc[
            train_[id_col].isin(train_user_id[valid_idx]), f"{label}_pred"
        ] = np.argmax(clf.predict(valid_x, num_iteration=clf.best_iteration), axis=1)
        test_pred += (clf.predict(test_[use_train_feats], num_iteration=clf.best_iteration) / folds.n_splits)

        report = f1_score(train_[label], train_[f"{label}_pred"], average=None)
        print(classification_report(train_[label], train_[f"{label}_pred"], digits=4))
        print("Score: ", report[0] * 0.2 + report[1] * 0.2 + report[2] * 0.6)

        if opts.save_model:
            os.makedirs(opts.save_model, exist_ok=True)
            clf.save_model(osp.join(opts.save_model, f"{n_fold}.mdl"))
            np.save(osp.join(opts.save_model, "pred.npy"), test_pred)

    test_[f"{label}_pred"] = np.argmax(test_pred, axis=1)
    test_[label] = np.argmax(test_pred, axis=1) + 1
    five_folds = [f"fold_{f}_imp" for f in range(1, n_splits + 1)]
    fold_importance_df["avg_imp"] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by="avg_imp", ascending=False, inplace=True)
    print(fold_importance_df[["Feature", "avg_imp"]].head(20))
    return test_[[id_col, "current_slice_id", "future_slice_id", label]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data_proc/test/20190801_testdata.txt")
    parser.add_argument("--attr", default="data/attr.txt")
    parser.add_argument("--save-pred")
    parser.add_argument("--save-model")
    parser.add_argument("--kfolds", default=5, type=int)
    parser.add_argument("--train", nargs="+", default=["data_proc/train/20190730.txt"])
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
    train = pd.concat(
        [gen_feats(x, True, os.getenv("clean"), os.getenv("weight")) for x in opts.train],
        axis=0,
    )
    train = train.merge(attr, on="link", how="left")
    test = test.merge(attr, on="link", how="left")

    use_cols = [
        i
        for i in train.columns
        if i
        not in ["link", "label", "future_slice_id", "label_pred"]
        + (["current_slice_id" if os.getenv("wo_sliceid") else []])
    ]

    sub = lgb_train(
        train,
        test,
        use_cols,
        "link",
        "label",
        opts.kfolds,
        2020,
    )
    sub.to_csv(opts.save_pred, index=False, encoding="utf8")
