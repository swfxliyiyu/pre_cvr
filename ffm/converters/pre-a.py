#!/usr/bin/env python3

import argparse, csv, sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
args = vars(parser.parse_args())

# These features are dense enough (they appear in the dataset more than 4 million times), so we include them in GBDT
target_cat_feats = ['C8-0', 'C14-1.0', 'C10-2005', 'C17-0.98', 'C3-7', 'C15-0.97', 'C16-0.97', 'C9-1003',
                    'C2-7534238860363577544', 'C15-0.98', 'C7-7908382889764677758;8277336076276184272', 'C16-0.98',
                    'C10-2002', 'C9-1004', 'C3-8', 'C7-7908382889764677758;5755694407684602296', 'C6-18', 'C14-0.99',
                    'C8-1', 'C3-6', 'C4-12', 'C16-0.96', 'C5-12', 'C6-17', 'C5-13', 'C13-5013', 'C11-3006', 'C13-5014',
                    'C12-17', 'C6-19', 'C4-11', 'C12-16', 'C17-0.97', 'C15-0.96',
                    'C7-7908382889764677758;509660095530134768', 'C11-3003', 'C13-5015', 'C5-14', 'C13-5012',
                    'C17-0.99', 'C7-7908382889764677758;5799347067982556520', 'C9-1002', 'C1-7838285046767229711',
                    'C11-3005', 'C12-15', 'C4-13', 'C11-3002', 'C6-16', 'C11-3004', 'C17-0.96',
                    'C2-7322157373578955368', 'C4-10', 'C9-1005', 'C2-3948283326616421003', 'C11-3007',
                    'C7-7908382889764677758;7258015885215914736', 'C5-11', 'C12-14', 'C12-18', 'C5-15', 'C4-14',
                    'C6-20', 'C5-10', 'C6-15', 'C3-5', 'C4-9', 'C2-5918626470536001929', 'C13-5011', 'C11-3000',
                    'C12-13', 'C2-196257267849351217', 'C9-1006', 'C14-0.98', 'C4-8', 'C6-14', 'C13-5018', 'C12-20',
                    'C2-3122721854741763495', 'C13-5010', 'C5-9', 'C10-2004', 'C13-5017', 'C12-21', 'C4-15', 'C5-16',
                    'C11-3008', 'C16-0.95', 'C2-1019055478500227370', 'C17-0.95', 'C11-3001', 'C13-5016',
                    'C2-4644954126004286009', 'C12-12', 'C13-5009', 'C4-7', 'C15-0.99', 'C5-8', 'C12-19', 'C15-0.95',
                    'C16-0.99', 'C3-4', 'C12-11', 'C6-13', 'C1-7066302540842412840', 'C14-0.97',
                    'C1-5051039799137810159', 'C9-1000', 'C8--1', 'C4-16', 'C5-17', 'C4-6', 'C2-4918413420989329604',
                    'C5-7', 'C8-2', 'C1-448955875785543916']

with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        # for j in range(1, 5):
        #     val = row['I{0}'.format(j)]
        #     if val == '':
        #         val = -10
        #     feats.append('{0}'.format(val))
        f_d.write(row['Label'] + ' ' + ' '.join(feats) + '\n')

        cat_feats = set()
        for j in range(1, 18):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            cat_feats.add(key)

        feats = []
        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append(str(j))
        f_s.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
