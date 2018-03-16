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
             'shop_star_level', 'shop_score_service', 'shop_score_delivery','shop_score_description']

col_to_filtrate = ['instance_id', 'item_id',
                   'user_id', 'shop_id', 'context_id', 'context_id',
                   'item_category_list', 'item_property_list', 'predict_category_property'
                   ]


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


def get_data():
    train = pd.read_csv("./input_data/round1_ijcai_18_train_20180301.txt", sep=' ')
    train = train.sample(frac=1)

    train['item_brand_id'] = train['item_brand_id'].astype('object')
    train['item_city_id'] = train['item_city_id'].astype('object')
    train['user_occupation_id'] = train['user_occupation_id'].astype('object')

    y_train = train['is_trade']
    train = train.drop(columns=col_to_filtrate, errors='ignore')

    for feat in train.select_dtypes(include=['object']).columns:
        m = train.groupby([feat])['is_trade'].mean()
        print(m)
        train[feat].replace(m, inplace=True)

    x_train = train[item_info+usr_info+shop_info]

    features = x_train.columns

    print(x_train)
    return features, x_train, y_train


features, x_train, y_train = get_data()
ceate_feature_map(features)

xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}
num_rounds = 888

print(x_train.dtypes)

dtrain = xgb.DMatrix(x_train, label=y_train)
gbdt = xgb.train(xgb_params, dtrain, num_rounds)

print('finished')
importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False)
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb_2.png')
