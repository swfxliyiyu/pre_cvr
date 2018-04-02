import pandas as pd
from pandas.core.groupby import SeriesGroupBy
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from collections import defaultdict

com = ['instance_id', 'is_trade']

TR_PATH = '../data/va_tr_tc.csv'
TE_PATH = '../data/va_te_tc.csv'

cate_feats = {1: 'hour', 2: 'item_city_id', 3: 'item_price_level', 4: 'item_sales_level',
              5: 'item_collected_level', 6: 'item_pv_level', 7: 'item_category_list', 8: 'user_gender_id',
              9: 'user_age_level', 10: 'user_occupation_id', 11: 'user_star_level', 12: 'shop_review_num_level',
              13: 'shop_star_level', 14: 'day', 15: 'item_brand_id', 16: 'shop_id', 17: 'item_id'}

num_feats = {1: 'shop_review_positive_rate', 2: 'shop_score_service', 3: 'shop_score_delivery',
             4: 'shop_score_description'}

tr_data = pd.read_csv("../data/round1_ijcai_18_train_20180301.txt", sep=' ').sort_values(['context_timestamp'],
                                                                                         kind='heapsort')

# 处理需要输出的结果

tr_out = pd.DataFrame()
te_out = pd.DataFrame()

# 标签和Id属性

# 时间属性
tr_data['date'] = tr_data['context_timestamp'].apply(lambda stamp: pd.datetime.utcfromtimestamp(stamp))

tr_data['day_hour'] = tr_data['date'].apply(lambda date: date.strftime('%d-%H'))

tr_data['day'] = tr_data['day_hour'].apply(lambda day_hour: day_hour.split('-')[0]).astype('float')
te_data = tr_data[tr_data['day'] == 24]
te_data['day'] = 23.  # 当成最后一天看
tr_data = tr_data[tr_data['day'] != 24]

tr_out[['Id', 'Label']] = tr_data[com]
te_out[['Id', 'Label']] = te_data[com]

tr_data['hour'] = tr_data['day_hour'].apply(lambda day_hour: day_hour.split('-')[1])
te_data['hour'] = te_data['day_hour'].apply(lambda day_hour: day_hour.split('-')[1])

# 该小时的点击量
tr_data['click_hour'] = tr_data['day_hour'].replace(tr_data.groupby(['day_hour'])['is_trade'].count())
te_data['click_hour'] = te_data['day_hour'].replace(te_data.groupby(['day_hour'])['is_trade'].count())


mlb = MultiLabelBinarizer()


def fuc(ele):
    res = []
    lst = ele.split(';')
    for i in lst:
        if i in dic:
            res.append(i)
    return res


# 类别信息
dic = defaultdict(lambda: 0)
tr_cates = map(lambda ele: ele.split(';'), tr_data['item_category_list'])
for props in tr_cates:
    for prop in props:
        dic[prop] += 1
for k in list(dic.keys()):
    if dic[k] < 1000:
        dic.pop(k)
print(len(dic))
tr_cates = map(fuc, tr_data['item_category_list'])
tr_cates = mlb.fit_transform(tr_cates)
te_cates = map(fuc, te_data['item_category_list'])
te_cates = mlb.transform(te_cates)

# 属性信息
dic = defaultdict(lambda: 0)
tr_props = map(lambda ele: ele.split(';'), tr_data['item_property_list'])
for props in tr_props:
    for prop in props:
        dic[prop] += 1
for k in list(dic.keys()):
    if dic[k] < 13000:
        dic.pop(k)
print(len(dic))
tr_props = map(fuc, tr_data['item_property_list'])
tr_props = mlb.fit_transform(tr_props)
te_props = map(fuc, te_data['item_property_list'])
te_props = mlb.transform(te_props)

# 预测信息
dic = defaultdict(lambda: 0)
tr_pred = map(lambda ele: ele.split(';'), tr_data['predict_category_property'])
for props in tr_pred:
    for prop in props:
        dic[prop] += 1
for k in list(dic.keys()):
    if dic[k] < 13000:
        dic.pop(k)
print(len(dic))
tr_pred = map(fuc, tr_data['predict_category_property'])
tr_pred = mlb.fit_transform(tr_pred)
te_pred = map(fuc, te_data['predict_category_property'])
te_pred = mlb.transform(te_pred)

# 合并以上信息
tr_info = np.concatenate((tr_cates, tr_props, tr_pred), axis=1)
te_info = np.concatenate((te_cates, te_props, te_pred), axis=1)
c_names = list(map(lambda i: 'B'+str(i), range(1, tr_info.shape[1]+1)))



print('开始转换类别特征...')

for i, feat in cate_feats.items():
    print('正在转换特征:{}'.format(feat))
    val_count = tr_data[feat].value_counts()
    val_count[val_count < 1000] = '-9999'
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
    tr_data[feat] = tr_data[feat].astype('float')
    te_data[feat] = te_data[feat].astype('float')
    std = np.std(tr_data[feat])
    tr_out['C{}'.format(i + c_size)] = 10 * (tr_data[feat] - tr_data[feat].mean()) / std
    te_out['C{}'.format(i + c_size)] = 10 * (te_data[feat] - tr_data[feat].mean()) / std  # 使用训练数据的均值方差
    tr_out['C{}'.format(i + c_size)] = tr_out['C{}'.format(i + c_size)].astype('int')
    te_out['C{}'.format(i + c_size)] = te_out['C{}'.format(i + c_size)].astype('int')
    tr_out['I{}'.format(i)] = tr_data[feat]
    te_out['I{}'.format(i)] = te_data[feat]
    # tr_out['C{}'.format(i+c_size)] = tr_out['I{}'.format(i)].apply(lambda s: str(s)[:min(len(str(s)), 4)])
    # te_out['C{}'.format(i+c_size)] = te_out['I{}'.format(i)].apply(lambda s: str(s)[:min(len(str(s)), 4)])
    val_count = tr_out['C{}'.format(i + c_size)].value_counts()
    val_count[val_count < 1000] = '-9999'
    if val_count.dtype != 'object':
        print('该特征将从int转换为object类型...')
        val_count = val_count.astype('object')
    val_count[val_count != '-9999'] = val_count[val_count != '-9999'].index
    tr_out['C{}'.format(i + c_size)] = tr_out['C{}'.format(i + c_size)].replace(val_count)
    te_out['C{}'.format(i + c_size)] = te_out['C{}'.format(i + c_size)].replace(val_count)
# 额外处理日期
n_size = len(num_feats)
tr_out['I{}'.format(1 + n_size)] = (tr_data['day'] - 20) / 2.3
te_out['I{}'.format(1 + n_size)] = (te_data['day'] - 20) / 2.3

for feat in tr_out:
    if feat in ['C15', 'C16', 'C17']:  # ID特征
        val_count = tr_out[feat].value_counts()
        val_count[val_count < 3000] = '-9999'
        if val_count.dtype != 'object':
            print('该特征将从int转换为object类型...')
            val_count = val_count.astype('object')
        val_count[val_count != '-9999'] = val_count[val_count != '-9999'].index
        tr_out[feat] = tr_out[feat].replace(val_count)
        te_out[feat] = te_out[feat].replace(val_count)

# 链接类别属性特征
print(te_out.shape)
print(te_info.shape)
print(tr_info.shape)
print(te_info.shape)
print(len(c_names))
tr_info = pd.DataFrame(tr_info, index=tr_out.index, columns=c_names)
te_info = pd.DataFrame(te_info, index=te_out.index, columns=c_names)

tr_out = pd.concat((tr_out, tr_info), axis=1)
te_out = pd.concat((te_out, te_info), axis=1)

tr_out.astype('object').to_csv(TR_PATH, index=False)
te_out.astype('object').to_csv(TE_PATH, index=False)
