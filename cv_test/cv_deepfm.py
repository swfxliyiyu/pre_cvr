import hashlib
import os

import numpy as np
import pandas as pd
import argparse, sys
from sklearn.preprocessing import LabelEncoder
from deepfm.deep_fm import pred_deep_fm
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.preprocessing import MultiLabelBinarizer


if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('TRAIN_PATH', type=str)
# parser.add_argument('GBDT_PATH', type=str)
parser.add_argument('OUT_DIR', type=str)
args = vars(parser.parse_args())
"============================================"

kfold = KFold(n_splits=5, shuffle=True)

tr_data = pd.read_csv(args['TRAIN_PATH'], low_memory=False)
# gbt_data = pd.read_csv(args['GBDT_PATH'], header=None)

X = tr_data.drop(axis=1, columns=['Label', 'Id'])
y = tr_data['Label']


def hashstr(s):
    s = str(s)
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % (1e+6 - 1) + 1


for fea in X:
    if 'C' in fea:
        X[fea] = X[fea].apply(hashstr)
        encoder = LabelEncoder()
        encoder.fit(X[fea])
        X[fea] = encoder.transform(X[fea])

preds = []
labels = []
scores = []

with open(os.path.join(args['OUT_DIR'], 'deep_fmm_metric.txt'), 'a+') as f:
    for i, (tr_inx, te_inx) in enumerate(kfold.split(X, y)):
        X_tr = X.iloc[tr_inx]
        y_tr = y.iloc[tr_inx]
        X_te = X.iloc[te_inx]
        y_te = y.iloc[te_inx]
        pred = pred_deep_fm(X_tr, y_tr, X_te, y_te, test_id=i + 1, n_estimators=10)
        auc = roc_auc_score(y_te, pred)
        loss = log_loss(y_te, np.clip(pred, a_min=1e-15, a_max=None))
        f.write('case:{}, loss:{}, auc:{}\n'.format(i + 1, loss, auc))
        print(loss, auc)
        scores.append([loss, auc])
    mean = np.mean(scores, axis=0)
    f.write('time:{}, loss:{}, auc:{}, type:{}\n'.format(pd.datetime.now().strftime('%m-%d %H:%M:%S'),
                                                         mean[0], mean[1], 'dffm_properties'))