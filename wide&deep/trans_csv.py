import pandas as pd

com = ['instance_id', 'is_trade']

TR_PATH = '../../data/train_tc.csv'
TE_PATH = '../../data/test_tc.csv'

cate_feats = {1: 'item_brand_id', 2: 'item_city_id', 3: 'item_price_level', 4: 'item_sales_level',
              5: 'item_collected_level', 6: 'item_pv_level', 7: 'item_category_list', 8: 'user_gender_id',
              9: 'user_age_level', 10: 'user_occupation_id', 11: 'user_star_level', 12: 'shop_review_num_level',
              13: 'shop_star_level'}

num_feats = {1: 'shop_review_positive_rate', 2: 'shop_score_service', 3: 'shop_score_delivery',
             4: 'shop_score_description'}

tr_data = pd.read_csv("../../data/round1_ijcai_18_train_20180301.txt", sep=' ')

te_data = pd.read_csv("../../data/round1_ijcai_18_test_a_20180301.txt", sep=' ')

tr_out = pd.DataFrame()

te_out = pd.DataFrame()

tr_out[['Id', 'Label']] = tr_data[com]
te_out['Id'] = te_data[com[0]]
te_out['Label'] = 0

for i, feat in cate_feats.items():
    tr_out['C{}'.format(i)] = tr_data[feat]
    te_out['C{}'.format(i)] = te_data[feat]

c_size = len(cate_feats)

for i, feat in num_feats.items():
    # def f(s):
    #     s = str(round(s, 2))
    #     return s[:min(len(s), 4)]
    # tr_data[feat] = tr_data[feat].apply(f)
    # te_data[feat] = te_data[feat].apply(f)

    tr_out['C{}'.format(i+c_size)] = tr_data[feat]
    te_out['C{}'.format(i+c_size)] = te_data[feat]

tr_out.to_csv(TR_PATH, index=False)
te_out.to_csv(TE_PATH, index=False)
