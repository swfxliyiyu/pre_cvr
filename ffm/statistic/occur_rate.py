import pandas as pd
import xgboost as xgb
import operator
from matplotlib import pylab as plt

item_info = [
             'item_brand_id', 'item_city_id', 'item_price_level',
             'item_sales_level', 'item_collected_level', 'item_pv_level',
             ]

usr_info = ['user_gender_id', 'user_age_level', 'user_occupation_id',
            'user_star_level']

ctx_info = ['context_id', 'context_timestamp', 'context_page_id',
            'predict_category_property']

shop_info = ['shop_review_num_level', 'shop_review_positive_rate',
             'shop_star_level', 'shop_score_service', 'shop_score_delivery']

col_to_filtrate = ['instance_id', 'item_id',
                   'user_id', 'shop_id', 'context_id', 'context_id',
                   'item_category_list', 'item_property_list', 'predict_category_property'
                   ]

cate_feats = [
            'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
            'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
            'shop_review_num_level', 'shop_review_positive_rate',
            'shop_star_level', 'shop_score_service', 'shop_score_delivery'
            ]



train = pd.read_csv("../data/round1_ijcai_18_train_20180301.txt", sep=' ')
train = train.sample(frac=1)


plt.figure()

for feat in cate_feats:
    train[feat] = train[feat].apply(lambda x: str(x)+' ')
    df = train.groupby(feat)['is_trade'].count()
    df = df[df>1000]
    print(df)
    plt.title(feat)
    plt.bar(df.index, list(df))
    plt.show()

#
# plt.figure()
# df.plot()
# df.plot(kind='barh', x='feature', y='fscore', legend=False)
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# plt.gcf().savefig('feature_importance_xgb_2.png')
