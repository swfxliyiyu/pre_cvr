import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

TR_PATH = '../data/train_tc.csv'
TE_PATH = '../data/test_tc.csv'

cate_feats = {1: 'hour', 2: 'item_city_id', 3: 'item_price_level', 4: 'item_sales_level',
              5: 'item_collected_level', 6: 'item_pv_level', 7: 'item_category_list', 8: 'user_gender_id',
              9: 'user_age_level', 10: 'user_occupation_id', 11: 'user_star_level', 12: 'shop_review_num_level',
              13: 'shop_star_level', 14: 'day', 15: 'item_brand_id'}

num_feats = {1: 'shop_review_positive_rate', 2: 'shop_score_service', 3: 'shop_score_delivery',
             4: 'shop_score_description'}

tr_data = pd.read_csv("../data/round1_ijcai_18_train_20180301.txt", sep=' ').sort_values(['context_timestamp'],
                                                                                         kind='heapsort')



# 时间属性
tr_data['date'] = tr_data['context_timestamp'].apply(lambda stamp: pd.datetime.utcfromtimestamp(stamp))

tr_data['day_hour'] = tr_data['date'].apply(lambda date: date.strftime('%d-%H'))

tr_data['day'] = tr_data['day_hour'].apply(lambda day_hour: day_hour.split('-')[0])

tr_past = tr_data[tr_data['day'] != '24']
tr_last = tr_data[tr_data['day'] == '24']

tr_buy = tr_past[tr_past['is_trade']==1]
s1 = set(tr_buy['user_id'])
s = set(tr_past['user_id'])
def fun(uid):
    if uid in s1:
        return 1
    elif uid in s:
        return 0
    else:
        return -1
tr_last['flag'] = tr_last['user_id'].apply(fun)
result = tr_last.groupby(['flag'])['flag'].count()
print(result)
print(result.sum())

