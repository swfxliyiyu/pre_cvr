import time
import pandas as pd
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
    #print(data.time)
    # data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    # user_query_day = data.groupby(['user_id', 'day']).size(
    # ).reset_index().rename(columns={0: 'user_query_day'})
    # data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    # #print(data.user_query_day)
    # user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
    #     columns={0: 'user_query_day_hour'})
    # data = pd.merge(data, user_query_day_hour, 'left',
    #                 on=['user_id', 'day', 'hour'])

    return data


if __name__ == "__main__":
    online = False# 这里用来标记是 线下验证 还是 在线提交

    data = pd.read_csv('./../../data/round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
        train.to_csv('./../../data/train.txt',index = False,sep = ' ')
        test.to_csv('./../../data/val.txt',index = False,sep = ' ')
    elif online == True:
        train = data.copy()
        test = pd.read_csv('./../../data/round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = convert_data(test)
