import pandas as pd
import os
import numpy as np
import csv
dsf = pd.read_parquet('/data2/6100/img_trend/dsf.parquet.gzip')[['SecuCode', 'date', 'specret']]
dsf = dsf.sort_values(by=['SecuCode', 'date'])
dsf['specret'] = dsf['specret'].fillna(0)
ret = dsf['specret'].values
dsf.reset_index(inplace=True, drop=True)


def cal_specret(i):
    coin = i[:6]
    date = i[7:17]
    ind = dsf[(dsf.SecuCode == coin) & (dsf.date == date)].index[0]
    try:
        num = np.sum(ret[ind:ind + 20])
        with open('/data2/6100/img_trend/specret/specret20.csv', 'a+') as wf:
            new_writer = csv.writer(wf)
            new_writer.writerow([i, num])

    except:
        print(coin, date)
        num = 0
        with open('/data2/6100/img_trend/specret/specret20.csv', 'a+') as wf:
            new_writer = csv.writer(wf)
            new_writer.writerow([i, num])

if __name__ == '__main__':

    train_file_list = os.listdir('/data2/6100/img_trend/cn_image/train')
    validation_file_list = os.listdir('/data2/6100/img_trend/cn_image/validation')
    test_file_list = os.listdir('/data2/6100/img_trend/cn_image/test')

    import multiprocessing as mp

    paralist = []
    for i in train_file_list:
        paralist.append(i[:17])
    for i in validation_file_list:
        paralist.append(i[:17])
    for i in test_file_list:
        paralist.append(i[:17])

    pool = mp.Pool(200)
    res = [pool.apply_async(cal_specret, args=(k,)) for k in paralist]
    for i, p in enumerate(res):
        p.get()