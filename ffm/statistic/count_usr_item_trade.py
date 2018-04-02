import pandas as pd

TRAIN_PATH = '../../data/round1_ijcai_18_train_20180301.txt'


if __name__ == '__main__':
    df = pd.read_csv(TRAIN_PATH, sep=' ',)
    n_usr = df['user_id'].drop_duplicates().shape[0]
    n_item = df['item_id'].drop_duplicates().shape[0]
    n_shop = df['shop_id'].drop_duplicates().shape[0]
    n_trade = df[df['is_trade'] == 1].shape[0]

    print('n_usr: {}\nn_item: {}\nn_shop: {}\nn_trade: {}'.format(n_usr, n_item, n_shop, n_trade))