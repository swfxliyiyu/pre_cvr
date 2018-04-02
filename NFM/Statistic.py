import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    # print(data.time)
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    # print(data.user_query_day)
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])
    return data


if __name__ == "__main__":
    input_file = './data/cvr/round1_ijcai_18_train_20180301.txt'
    data = pd.read_csv(input_file, sep=' ')
    data = convert_data(data)

    test = pd.read_csv('./data/cvr/round1_ijcai_18_test_a_20180301.txt', sep=' ')
    test = convert_data(test)

    cat_feat = ['item_id', 'item_category_list', 'item_property_list', 'item_brand_id', 'item_city_id',
                'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id',
                'user_gender_id', 'user_age_level','user_occupation_id', 'user_star_level', 'context_id',
                'context_timestamp', 'context_page_id', 'predict_category_property', 'shop_id', 'shop_review_num_level',
                'shop_star_level']

    features = ['item_id','item_category_list', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_review_num_level', 'shop_star_level', 'shop_id',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                'shop_score_description'
                ]

    num_feat = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                'shop_score_description']

    rm_feat = [ 'instance_id',  'item_property_list', 'user_id',
               'context_id', 'context_timestamp', 'predict_category_property', 'time'
               ]

    add = [0]
    le = LabelEncoder()
    online = False   #在这里修改线上还是线下

    for i in ['item_id', 'item_brand_id', 'item_city_id', 'shop_id']:
        val_count = data[i].value_counts()
        val_count[val_count < 1000] = -9999
        data[i] = data[i].replace(val_count)
        test[i] = test[i].replace(val_count)

    if online == False:

        for i in num_feat:
            data[i] = data[i].astype('float')
            std = np.std(data[i])
            data[i] = (10 * (data[i] - data[i].mean()) / std).astype('int')

        sum = 0
        for i, j in enumerate(features):
            sum = sum + data[j].drop_duplicates().__len__()
            if i < len(features) - 1:
                add.append(sum)
            print(j, sum, data[j].drop_duplicates().__len__(), add[i])

        for i, j in zip(features, add):
            le.fit(data[i])
            data[i] = [str(x) + ':1' for x in (le.transform(data[i]) + j)]

        for i in rm_feat:
            data = data.drop([i], axis=1)
        is_trade = data.pop('is_trade')
        data.insert(0, 'is_trade', is_trade)
        # data.to_csv('./data/cvr/train_data.txt', sep=' ', index=False)
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        val = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
        train = train.drop(['day'], axis=1)
        val = val.drop(['day'], axis=1)
        train.to_csv('./data/cvr/cvr.train_data.txt', sep=' ', index=False)
        val.to_csv('./data/cvr/cvr.val_data.txt', sep=' ', index=False)

    elif online == True:
        train = data.copy()
        add_test = [0]
        for i in num_feat:
            test[i] = test[i].astype('float')
            std = np.std(test[i])
            test[i] = (10 * (test[i] - test[i].mean()) / std).astype('int')

        sum_test = 0
        for i, j in enumerate(features):
            sum_test = sum_test + test[j].drop_duplicates().__len__()
            if i < len(features) - 1:
                add_test.append(sum_test)
            print(j, sum_test, test[j].drop_duplicates().__len__(), add_test[i])

        for i, j in zip(features, add_test):
            le.fit(test[i])
            test[i] = [str(x) + ':1' for x in (le.transform(test[i]) + j)]

        for i in rm_feat:
            test = test.drop([i], axis=1)
        test = test.drop(['day'], axis=1)
        is_trade = pd.Series(np.zeros(test.shape[0], int))
        test.insert(0, 'is_trade', is_trade)
        test.to_csv('./data/cvr/cvr.test_data.txt', sep=' ', index=False)

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