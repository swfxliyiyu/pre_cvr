import hashlib
import os

import numpy
import pandas as pd
import argparse, sys
from sklearn.preprocessing import LabelEncoder
from ffm.ffm_interface import pred_ffm
from sklearn.model_selection import KFold
from sklearn.metrics import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('TRAIN_PATH', type=str)
parser.add_argument('TEST_PATH', type=str)
parser.add_argument('OUT_PATH', type=str)
args = vars(parser.parse_args())
"============================================"

kfold = KFold(n_splits=5, shuffle=True)

tr_data = pd.read_csv(args['TRAIN_PATH'], low_memory=False).sample(frac=1)
# gbt_data = pd.read_csv(args['GBDT_PATH'], header=None)

X = tr_data.drop(axis=1, columns=['Label', 'Id'])
y = tr_data['Label']

te_data = pd.read_csv(args['TEST_PATH'])
X_te = te_data.drop(axis=1, columns=['Label', 'Id'])




def hashstr(s):
    s = str(s)
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % (1e+6 - 1) + 1




for fea in X:
    print(fea)
    if 'C' in fea:
        set_fea = set(X[fea])
        X_te[fea] = X_te[fea].apply(lambda val: val if val in set_fea else '-9999')
        X[fea] = X[fea].apply(hashstr)
        X_te[fea] = X_te[fea].apply(hashstr)
        encoder = LabelEncoder()
        encoder.fit(X[fea])
        X[fea] = encoder.transform(X[fea])
        X_te[fea] = encoder.transform(X_te[fea])

preds = []
labels = []
scores = []


pred = pred_ffm(X, y, X_te, '')
import numpy as np

np.save(args['OUT_PATH'], pred)

df = pd.DataFrame()
df['instance_id'] = te_data['Id']
df['predicted_score'] = pred
df.to_csv(args['OUT_PATH'] + '.csv', index=False, line_terminator='\r', sep=' ')
