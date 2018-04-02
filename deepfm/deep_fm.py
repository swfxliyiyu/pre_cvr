import hashlib

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from DeepFM import DeepFM
import xgboost as xgb
import tensorflow as tf
import example.config
from example.metrics import gini_norm
from example.DataReader import *

def hashstr(s):
    s = str(s)
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % (1e+6 - 1) + 1

def pred_deep_fm(X_tr, y_tr, X_te, y_te=None, test_id=0, n_estimators=30):
    print('predicting {}nd case...'.format(test_id))

    gbclf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=7, n_jobs=4, )
    clist = [c for c in X_tr if 'I' not in c]
    nlist = [c for c in X_tr if 'I' in c]

    X_gbt_tr = X_tr.copy()[clist]
    X_gbt_te = X_te.copy()[clist]
    # 给GBDT的特征
    # col_to_drop = ['C16', 'C17', 'C18', 'C19']
    # X_gbt_tr = X_gbt_tr.drop(columns=col_to_drop)
    # X_gbt_te = X_gbt_te.drop(columns=col_to_drop)
    # for feat in X_gbt_tr:
    #     if 'C' in feat:
    #         val_count = X_gbt_tr[feat].value_counts()
    #         val_count[val_count < 5000] = hashstr('-9999')
    #         if val_count.dtype != 'object':
    #             print('该特征将从int转换为object类型...')
    #             val_count = val_count.astype('object')
    #         val_count[val_count != hashstr('-9999')] = val_count[val_count != hashstr('-9999')].index
    #         X_gbt_tr[feat] = X_gbt_tr[feat].replace(val_count)
    #         X_gbt_te[feat] = X_gbt_te[feat].replace(val_count)
    #
    # oh_enc = OneHotEncoder()
    # oh_enc.fit(X_gbt_tr)
    # X_gbt_tr = oh_enc.transform(X_gbt_tr).toarray()
    # X_gbt_te = oh_enc.transform(X_gbt_te).toarray()
    # print(X_gbt_tr)
    # X_gbt_tr = np.concatenate([X_gbt_tr, X_tr[nlist]], axis=1)
    # X_gbt_te = np.concatenate([X_gbt_te, X_te[nlist]], axis=1)
    # print(X_gbt_tr.shape)
    # print('fitting gbdt...')
    # gbclf.fit(X_gbt_tr, y_tr)
    # leaf_tr = pd.DataFrame(gbclf.apply(X_gbt_tr).reshape([-1, n_estimators]), index=X_tr.index, dtype='int')
    # leaf_te = pd.DataFrame(gbclf.apply(X_gbt_te).reshape([-1, n_estimators]), index=X_te.index, dtype='int')
    #
    #
    # for fea in leaf_tr:
    #     encoder = LabelEncoder()
    #     encoder.fit(pd.concat([leaf_tr[fea]]))
    #     leaf_tr[fea] = encoder.transform(leaf_tr[fea])
    #     leaf_te[fea] = encoder.transform(leaf_te[fea])

    # X_tr = X_tr.drop(columns=nlist)
    # X_te = X_te.drop(columns=nlist)

    # X_tr_i = pd.concat([X_tr, leaf_tr], axis=1)
    # X_te_i = pd.concat([X_te, leaf_te], axis=1)
    fd = FeatureDictionary(dfTrain=X_tr, dfTest=X_te, numeric_cols=nlist)
    print('built feat_dict...')
    data_parser = DataParser(feat_dict=fd, )
    Xi_train, Xv_train = data_parser.parse(df=X_tr)
    Xi_test, Xv_test = data_parser.parse(df=X_te)

    from sklearn.metrics import log_loss
    dfm_params = {
        "field_size": len(Xi_train[0]),
        "feature_size": fd.feat_dim,
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 5,
        "dropout_fm": [0.5, 0.5],
        "deep_layers": [128, 64, 32, 16],
        "dropout_deep": [0.5, 0.5, 0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 200,
        "batch_size": 512,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "eval_metric": log_loss,
        "random_seed": example.config.RANDOM_SEED
    }

    deep_fm = DeepFM(**dfm_params)
    if y_te is not None:
        deep_fm.fit(Xi_train, Xv_train, list(y_tr), Xi_test, Xv_test, list(y_te), early_stopping=False, refit=False)
    else:
        deep_fm.fit(Xi_train, Xv_train, list(y_tr), early_stopping=False, refit=False)
    result = deep_fm.predict(Xi_test, Xv_test)

    return result
