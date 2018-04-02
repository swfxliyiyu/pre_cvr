#!/usr/bin/env python

import argparse, csv, hashlib, sys
import pandas as pd

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser(description='process some integers')
parser.add_argument('te_path', type=str)
parser.add_argument('out_path', type=str)
parser.add_argument('submission_path', type=str)
args = parser.parse_args()

TE_PATH, OUT_PATH, SUB_PATH = args.te_path, args.out_path, args.submission_path


df_te = pd.read_csv(TE_PATH)
df_out = pd.read_csv(OUT_PATH, header=None)
df_out.columns = ['score']
res = pd.DataFrame()
res['instance_id'] = df_te['Id']
res['predicted_score'] = df_out['score']

res.to_csv(SUB_PATH, sep=' ', index=False, line_terminator='\r')
