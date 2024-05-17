import pandas as pd
import os
import numpy as np
dsf = pd.read_parquet('/data2/6100/img_trend/dsf.parquet.gzip')
coin_list = list(set(dsf['SecuCode'].values))


for coin in coin_list:
    data = dsf[dsf['SecuCode'] == coin][['date', 'prc', 'oprc', 'high', 'low', 'svol']]
    data = data[(data['oprc']+data['svol'])!=0]
    if len(data) < 1000:
        continue
    # print(len(data))
    data.sort_values(by='date', inplace=True)
    data.columns = ['DATE', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
    data.to_csv('/data2/6100/img_trend/cn_data/{}.csv'.format(coin))

file_list = [f for f in os.listdir('/data2/6100/img_trend/cn_data/') if os.path.isfile(os.path.join('/data2/6100/img_trend/cn_data/', f))]
# coin_list = [i[:-4] for i in file_list]
# date_list = []
# data_np = []
# feature_list=['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
# max_date = 0
# for i in range(len(file_list)):
#     item_df = pd.read_csv('/data2/6100/img_trend/cn_data/' + file_list[i])
#     item_np = item_df[feature_list].values
#     data_np.append(item_np)
#     if len(item_df) > max_date:
#         max_date = len(item_df)
#         date_list = item_df[item_df.columns[0]]
# # print(np.array(data_np).shape)
#
# panel = pd.Panel(items=feature_list, major_axis=coin_list, minor_axis=date_list, dtype=np.float32)
#
# for i in range(len(coin_list)):
#     for j in range(len(feature_list)):
#         for k in range(len(date_list)):
#             panel.loc[feature_list[j], coin_list[i], date_list[k]] = data_np[i][k][j]
#
# f = open('./database/data_new.pkl', 'wb')
# panel.to_pickle(f)
# f.close


