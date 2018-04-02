import subprocess

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def pred_ffm(X_tr, y_tr, X_te, test_id=0, n_estimators=30):
    print('predicting {}nd case...'.format(test_id))

    gbclf = GradientBoostingClassifier(n_estimators=30, max_depth=4)

    print('fitting gbdt...')

    X_gbt_tr = X_tr.copy()

    X_gbt_te = X_te.copy()

    X_gbt_tr['Label'] = y_tr
    for feat in X_gbt_tr:
        if 'C' in feat:
            m = X_gbt_tr.groupby([feat])['Label'].mean()
            X_gbt_tr[feat].replace(m, inplace=True)
            X_gbt_te[feat].replace(m, inplace=True)

    X_gbt_tr = X_gbt_tr.drop(columns=['Label'])

    gbclf.fit(X_gbt_tr, y_tr)
    # leaf_tr = pd.DataFrame(gbclf.apply(X_gbt_tr).reshape([-1, n_estimators]), index=X_tr.index, dtype='int')
    # leaf_te = pd.DataFrame(gbclf.apply(X_gbt_te).reshape([-1, n_estimators]), index=X_te.index, dtype='int')
    #
    # for fea in leaf_tr:
    #     encoder = LabelEncoder()
    #     encoder.fit(pd.concat([leaf_tr[fea]]))
    #     leaf_tr[fea] = encoder.transform(leaf_tr[fea])
    #     leaf_te[fea] = encoder.transform(leaf_te[fea])
    #
    clist = [c for c in X_tr if 'I' in c]

    X_tr = X_tr.drop(columns=clist)
    X_te = X_te.drop(columns=clist)

    # X_tr = pd.concat([X_tr, leaf_tr], axis=1)
    # X_te = pd.concat([X_te, leaf_te], axis=1)

    # 转化为ffm输入格式
    for i, feat in enumerate(X_tr):
        print('转化 {} ...'.format(feat))
        func = lambda val: ':'.join([str(i), str(val), '1'])
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

    cmd = '../ffm/ffm-train -k 4 -t 64 -s {nr_thread} -p {te_path} {tr_path} {model}'.format(nr_thread=2, te_path=te_path,
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
