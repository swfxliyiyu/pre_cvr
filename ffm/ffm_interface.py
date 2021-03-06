import hashlib
import subprocess

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb


def hashstr(s):
    s = str(s)
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % (1e+6 - 1) + 1

def pred_ffm(X_tr, y_tr, X_te, test_id=0, n_estimators=30):
    print('predicting {}nd case...'.format(test_id))

    # gbclf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=4, min_samples_leaf=500, )
    gbclf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=7, n_jobs=4, )
    clist = [c for c in X_tr if 'C' in c]
    nlist = [c for c in X_tr if 'I' in c]
    blist = [c for c in X_tr if 'B' in c]


    # X_gbt_tr = X_tr.copy()[clist]
    # X_gbt_te = X_te.copy()[clist]
    # # 给GBDT的特征
    # col_to_drop = ['C16','C17','C18','C19']
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
    # for fea in leaf_tr:
    #     encoder = LabelEncoder()
    #     encoder.fit(pd.concat([leaf_tr[fea]]))
    #     leaf_tr[fea] = encoder.transform(leaf_tr[fea])
    #     leaf_te[fea] = encoder.transform(leaf_te[fea])

    X_tr = X_tr.drop(columns=nlist)
    X_te = X_te.drop(columns=nlist)

    # X_tr = pd.concat([X_tr, leaf_tr], axis=1)
    # X_te = pd.concat([X_te, leaf_te], axis=1)

    # 转化为ffm输入格式
    for i, feat in enumerate(X_tr):
        print('转化 {} ...'.format(feat))
        if 'C' in feat:
            func = lambda val: ':'.join([str(i), str(val), '1'])
        elif 'B' in feat:
            func = lambda val: ':'.join([str(i), '1', str(val)])
        X_tr[feat] = X_tr[feat].apply(func)
        X_te[feat] = X_te[feat].apply(func)
    columns = list(X_tr.columns)
    X_tr['Label'] = y_tr
    X_te['Label'] = '0'
    columns = ['Label'] + columns

    tr_path = '../tmp/tr{}.ffm'.format(test_id)
    te_path = '../tmp/te{}.ffm'.format(test_id)
    pre_path = '../tmp/pre{}.ffm'.format(test_id)
    model_path = '../tmp/model{}.ffm'.format(test_id)
    X_tr.to_csv(tr_path, header=False, index=False, columns=columns, sep=' ')
    X_te.to_csv(te_path, header=False, index=False, columns=columns, sep=' ')

    cmd = '../ffm/ffm-train -k 4 -t 20 -s {nr_thread} -p {te_path} {tr_path} {model}'.format(nr_thread=2, te_path=te_path,
                                                                                        tr_path=tr_path,
                                                                                        model=model_path)
    subprocess.call(cmd, shell=True)

    cmd = '../ffm/ffm-predict {te_path} {model} {pre_path}'.format(te_path=te_path, model=model_path, pre_path=pre_path)
    subprocess.call(cmd, shell=True)

    cmd = 'rm ../tmp/tr*.ffm ../tmp/te*.ffm ../tmp/model*.ffm'
    subprocess.call(cmd, shell=True)

    pre_df = pd.read_csv(pre_path, header=None)
    pred = np.array(pre_df[0])

    return pred
