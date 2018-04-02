import os
import sys
import time
import pandas as pd

if __name__=='__main__':
    input_file = '../../data/round1_ijcai_18_train_20180301.txt'
    df = pd.read_csv(input_file, sep=' ')
    df_userId = df[['shop_id']]
    # df_userId.to_csv('../../data/shopId.txt', index=False)
    cat_feat = ['item_id', 'item_category_list', 'item_property_list', 'item_brand_id', 'item_city_id',
                'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id',
                'user_gender_id', 'user_age_level','user_occupation_id', 'user_star_level', 'context_id',
                'context_timestamp', 'context_page_id', 'predict_category_property', 'shop_id', 'shop_review_num_level',
                'shop_star_level']
    num_feat = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                'shop_score_description']
    sum = 0
    for i in cat_feat:
        val_count = df['item_category_list'].value_counts()
        # val_count.to_csv('../../data/sta/'+i+'.txt', index=False)
        print(val_count)
        # sum = sum + df[i].drop_duplicates().__len__()
        # print(i, sum, df[i].drop_duplicates().__len__())
    # df_test = df.drop(['Label'], axis = 1)
    # df_test.to_csv('/home/sensetime/Documents/tianchi/data/test_tc.csv', sep=',', index=False)
    # df['context_timestamp'] = df['context_timestamp'].apply(lambda x: time.strftime('%Y%m%d',time.localtime(x)))
    # group = df.groupby(['context_timestamp'])['is_trade']
    # res = group.sum() / group.count()
    # item_brand_id = df['C14'].drop_duplicates()
    # # print(df['item_category_list'].drop_duplicates())
    # # print(df['user_age_level'].drop_duplicates())
    # # print(res)
    # print(item_brand_id)