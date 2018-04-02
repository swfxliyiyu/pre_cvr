import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

com = ['instance_id', 'is_trade']

TR_PATH = '../../data/train_tc.csv'
TE_PATH = '../../data/test_tc.csv'

cate_feats = {1: 'hour',  2: 'item_brand_id', 3: 'item_city_id', 4: 'item_price_level', 5: 'item_sales_level',
              6: 'item_collected_level', 7: 'item_pv_level', 8: 'item_category_list', 9: 'user_gender_id',
              10: 'user_age_level', 11: 'user_occupation_id', 12: 'user_star_level', 13: 'shop_review_num_level',
              14: 'shop_star_level', 15: 'day'}

num_feats = {1: 'shop_review_positive_rate', 2: 'shop_score_service', 3: 'shop_score_delivery',
             4: 'shop_score_description'}

tr_data = pd.read_csv("../../data/round1_ijcai_18_train_20180301.txt", sep=' ').sort_values(['context_timestamp'],
                                                                                         kind='heapsort')
te_data = pd.read_csv("../../data/round1_ijcai_18_test_a_20180301.txt", sep=' ')
te_data['is_trade'] = 0

# 处理需要输出的结果

tr_out = pd.DataFrame()
te_out = pd.DataFrame()

# 标签和Id属性
tr_out[['Id', 'Label']] = tr_data[com]
te_out[['Id', 'Label']] = te_data[com]

# 时间属性
tr_data['date'] = tr_data['context_timestamp'].apply(lambda stamp: pd.datetime.utcfromtimestamp(stamp))
te_data['date'] = te_data['context_timestamp'].apply(lambda stamp: pd.datetime.utcfromtimestamp(stamp))

tr_data['day_hour'] = tr_data['date'].apply(lambda date: date.strftime('%d-%H'))
te_data['day_hour'] = te_data['date'].apply(lambda date: date.strftime('%d-%H'))

tr_data['day'] = tr_data['day_hour'].apply(lambda day_hour: day_hour.split('-')[0])
te_data['day'] = '24'  # 当成最后一天看

tr_data['hour'] = tr_data['day_hour'].apply(lambda day_hour: day_hour.split('-')[1])
te_data['hour'] = te_data['day_hour'].apply(lambda day_hour: day_hour.split('-')[1])

# 该小时的点击量
# tr_data['click_hour'] = tr_data['day_hour'].replace(tr_data.groupby(['day_hour'])['is_trade'].count())
# te_data['click_hour'] = te_data['day_hour'].replace(te_data.groupby(['day_hour'])['is_trade'].count())


# 用户过去的行为
# actions = []
# for uid in tr_data['user_id'].drop_duplicates():
#     df = tr_data[tr_data['user_id'] == uid]
#     action = ''
#     for i, row in df.iterrows():
#         actions.append(action)
#         action += '-'+str(row['is_trade'])
# tr_data['action'] = actions
# actions = {}
# for uid in te_data['user_id'].drop_duplicates():
#     df = tr_data[tr_data['user_id'] == uid]
#     if df.shape[0] is not 0:
#         actions[uid] = df.loc[df.last_valid_index()]['action']
#     else:
#         actions[uid] = ''
# te_data['action'] = te_data['user_id'].replace(actions)
#
# print(te_data['action'])

print('开始转换类别特征...')

for i, feat in cate_feats.items():
    print('正在转换特征:{}'.format(feat))
    val_count = tr_data[feat].value_counts()
    val_count[val_count < 5000] = '-9999'
    if val_count.dtype != 'object':
        print('该特征将从int转换为object类型...')
        val_count = val_count.astype('object')
    val_count[val_count != '-9999'] = val_count[val_count != '-9999'].index
    tr_out['C{}'.format(i)] = tr_data[feat].replace(val_count)
    te_out['C{}'.format(i)] = te_data[feat].replace(val_count)


print('开始转换数值特征...')
c_size = len(cate_feats)
for i, feat in num_feats.items():
    print('正在转换特征:{}'.format(feat))
    scaler = StandardScaler()
    tr_data[feat] = tr_data[feat].astype('float')
    te_data[feat] = te_data[feat].astype('float')
    std = np.std(tr_data[feat])
    tr_out['I{}'.format(i)] = 10 * (tr_data[feat] - tr_data[feat].mean()) / std
    te_out['I{}'.format(i)] = 10 * (te_data[feat] - tr_data[feat].mean()) / std  # 使用训练数据的均值方差
    tr_out['C{}'.format(i + c_size)] = tr_out['I{}'.format(i)].astype('int')
    te_out['C{}'.format(i + c_size)] = te_out['I{}'.format(i)].astype('int')
    tr_out['I{}'.format(i)] = tr_data[feat]
    te_out['I{}'.format(i)] = te_data[feat]
    tr_out['C{}'.format(i+c_size)] = tr_out['I{}'.format(i)].apply(lambda s: str(s)[:min(len(str(s)), 4)])
    te_out['C{}'.format(i+c_size)] = te_out['I{}'.format(i)].apply(lambda s: str(s)[:min(len(str(s)), 4)])
    val_count = tr_out['C{}'.format(i + c_size)].value_counts()
    val_count[val_count < 5000] = '-9999'
    if val_count.dtype != 'object':
        print('该特征将从int转换为object类型...')
        val_count = val_count.astype('object')
    val_count[val_count != '-9999'] = val_count[val_count != '-9999'].index
    tr_out['C{}'.format(i + c_size)] = tr_out['C{}'.format(i + c_size)].replace(val_count)
    te_out['C{}'.format(i + c_size)] = te_out['C{}'.format(i + c_size)].replace(val_count)


tr_out.astype('object').to_csv(TR_PATH, index=False)
te_out.astype('object').to_csv(TE_PATH, index=False)
