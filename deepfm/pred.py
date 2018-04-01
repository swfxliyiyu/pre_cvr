import hashlib

import pandas as pd
import argparse, sys
from sklearn.preprocessing import LabelEncoder
from deep_fm import pred_deep_fm
from sklearn.model_selection import KFold

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('TRAIN_PATH', type=str)
parser.add_argument('TEST_PATH', type=str)
parser.add_argument('OUT_PATH', type=str)
args = vars(parser.parse_args())
"============================================"

kfold = KFold(n_splits=5, shuffle=True)

tr_data = pd.read_csv(args['TRAIN_PATH']).sample(frac=1)
# gbt_data = pd.read_csv(args['GBDT_PATH'], header=None)

X = tr_data.drop(axis=1, columns=['Label', 'Id'])
y = tr_data['Label']

te_data = pd.read_csv(args['TEST_PATH'])

X_te = te_data.drop(axis=1, columns=['Label', 'Id'])
feature_sizes = []


def hashstr(s):
    s = str(s)
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % (1e+6 - 1) + 1


for fea in X:

    print(fea)
    if 'C' in fea:
        X[fea] = X[fea].apply(hashstr)
        X_te[fea] = X_te[fea].apply(hashstr)
        encoder = LabelEncoder()
        encoder.fit(X[fea].tolist() + X_te[fea].tolist())

        feature_sizes.append(len(encoder.classes_))
        print(encoder.classes_)
        X[fea] = encoder.transform(X[fea])
        X_te[fea] = encoder.transform(X_te[fea])
print(y.drop_duplicates())
pred = pred_deep_fm(X, y, X_te, feature_sizes)
import numpy as np

np.save(args['OUT_PATH'], pred)

df = pd.DataFrame()
df['instance_id'] = te_data['Id']
df['predicted_score'] = pred
df.to_csv(args['OUT_PATH'] + '.csv', index=False, line_terminator='\r', sep=' ')
