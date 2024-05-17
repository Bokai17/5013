import pandas as pd
import numpy as np
import os.path as op
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import copy
ret1 = []
os.chdir('/data2/6100/img_trend/csi1800/specret_enhance/')

class PriceToImgae:
    def __init__(self, days, width, height, price_area_height, volume_area_height, pixelwidth, pixelheight):
        self.days = days
        self.width = width
        self.height = height
        self.price_area_height = price_area_height
        self.volume_area_height = volume_area_height
        self.pixelwidth = pixelwidth
        self.pixelheight = pixelheight
        self.price_area_logical_height = int(self.price_area_height / self.pixelheight)
        self.volume_area_logical_height = int(self.volume_area_height / self.pixelheight)

    def __drawPixel(self, x, y, pixel):
        logical_height = int(self.height / self.pixelheight)
        for i in range(self.pixelwidth):
            for j in range(self.pixelheight):
                self.img.putpixel((self.pixelwidth * x + i, self.pixelheight * (logical_height - 1 - y) + j), pixel)

    def __drawPrice(self, index, price, moving_average, volume, pixel):
        open_price = price[0]
        high_price = price[1]
        low_price = price[2]
        close_price = price[3]

        # 画OHLC表
        self.__drawPixel(3 * index + 0, self.volume_area_logical_height + 1 + open_price, pixel)

        for i in range(high_price - low_price + 1):
            self.__drawPixel(3 * index + 1, self.volume_area_logical_height + 1 + low_price + i, pixel)
        self.__drawPixel(3 * index + 2, self.volume_area_logical_height + 1 + close_price, pixel)

        # 画MA线
        self.__drawPixel(3 * index + 0, self.volume_area_logical_height + 1 + moving_average, pixel)
        self.__drawPixel(3 * index + 1, self.volume_area_logical_height + 1 + moving_average, pixel)
        self.__drawPixel(3 * index + 2, self.volume_area_logical_height + 1 + moving_average, pixel)

        # 画成交量柱
        for i in range(volume):
            self.__drawPixel(3 * index + 1, i, pixel)

    def getImg(self, price_array, moving_average_array, volume_array, background_pixel, color_pixel):
        self.img = Image.new("RGB", (self.width, self.height), background_pixel)
        for i in range(price_array.shape[0]):
            self.__drawPrice(i, price_array[i], moving_average_array[i], volume_array[i], color_pixel)
        return self.img


def image_loader(image):
    transform = transforms.Compose([
        transforms.ToTensor()])
    image = transform(image).squeeze(0)
    return image


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = unloader(image)
    return image


def get_image_with_price(price):
    '''
    price [time, feature]
    feature: open, close, low, high, vol, MA
    '''
    # 像素大小
    PIXEL_WIDTH = 1
    PIXEL_HEIGHT = 1
    # 宽度是时间序列的三倍长
    WIDTH = 3 * price.shape[0] * PIXEL_WIDTH

    # 价格占高度2/3，vol占1/3
    PRICE_LOGICAL_HEIGHT = 2 * price.shape[0]
    VOLUME_LOGICAL_HEIGHT = price.shape[0]

    # 计算区域各区域大小
    PRICE_AREA_HEIGHT = PRICE_LOGICAL_HEIGHT * PIXEL_HEIGHT
    V0LUME_AREA_HEIGHT = VOLUME_LOGICAL_HEIGHT * PIXEL_HEIGHT

    # 总高度还是加一个pixel大小分割
    HEIGHT = PRICE_AREA_HEIGHT + V0LUME_AREA_HEIGHT + PIXEL_HEIGHT

    # 放缩
    sclr1 = MinMaxScaler((0, PRICE_LOGICAL_HEIGHT - 1))
    sclr2 = MinMaxScaler((1, VOLUME_LOGICAL_HEIGHT))
    price_minmax = sclr1.fit_transform(price[:, :-1].reshape(-1, 1)).reshape(price.shape[0], -1).astype(int)
    volume_minmax = sclr2.fit_transform(price[:, -1].reshape(-1, 1)).reshape(price.shape[0]).astype(int)

    # 时间序列长度
    days = price_minmax.shape[0]

    # 转图片
    p2i = PriceToImgae(days, WIDTH, HEIGHT, PRICE_AREA_HEIGHT, V0LUME_AREA_HEIGHT, PIXEL_WIDTH, PIXEL_HEIGHT)
    background_pixel = (0, 0, 0, 100)
    color_pixel = (255, 255, 255, 100)
    image = p2i.getImg(price_minmax[:, :-1], price_minmax[:, -1], volume_minmax, background_pixel, color_pixel)
    # 转成黑白像素
    image = image.convert('1')
    return image


class CNN_20D(nn.Module):
    '''
    The input Image size is batchsize*1*61*60
    '''
    def __init__(self):
        super(CNN_20D, self).__init__()
        self.cnn_block1 = CNN_Block(in_channels=1, out_channels=64, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block2 = CNN_Block(in_channels=64, out_channels=128, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block3 = CNN_Block(in_channels=128, out_channels=256, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.linear = nn.Linear(
            in_features=107520, out_features=2)
        self.Dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.Dropout(x)
        x = self.cnn_block2(x)
        x = self.Dropout(x)
        x = self.cnn_block3(x)
        # x = self.Dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # x = F.softmax(x)
        return x


class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[5, 3], pooling_size=[2, 1], negative_slope=0.01):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=1, padding=[2, 1])
        self.max_pooling = nn.MaxPool2d(kernel_size=pooling_size)
        self.LReLU = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.LReLU(x)
        x = self.max_pooling(x)
        return x


def image_loader(image):
    transform = transforms.Compose([
        transforms.ToTensor()])
    image = transform(image).squeeze(0)
    return image


def find_coin_list(model):
    csi = pd.read_parquet('/data2/6100/img_trend/csi1800/csi1800.parquet.gzip')
    # csi18 = csi[csi['date'] > '2018-01-01']
    # dt = list(set(csi18['date']))
    # ans = []
    # for i in dt:
    #     t = csi[csi['date'] == i]
    #     if len(ans) == 0:
    #         ans = list(set(t['SecuCode']))
    #     else:
    #         ans = list(set(t['SecuCode']).intersection(ans))
    # print(len(ans))
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    all_date = list(set(csi['date']))
    num_groups = 10
    for j in range(num_groups):
        exec('ret{} = []'.format(j))
    for year in years:
        for month in months:
            if year == 2022 and month == '03':
                break
            dt = '{}-{}-01'.format(year, month)
            print(dt)
            i = 0
            while all_date[i] < dt:
                i += 1
            coins = csi[csi['date'] == all_date[i]]['SecuCode'].values
            value_list = []
            ret_list = []
            valid_coin = []
            for coin in coins:
                try:
                    coin_data = pd.read_csv('/data2/6100/img_trend/cn_data/{}.csv'.format(coin), index_col=0)
                except:
                    # print('no {} data for {}'.format(coin, dt))
                    continue
                coin_data.reset_index(inplace=True, drop=True)
                # if dt[5:7] == '12':
                #     end_date = '{}-01-01'.format(year+1)
                # else:
                #     end_date = '{}-{}-01'.format(year, months[months.index(month)+1])
                st_index = coin_data[coin_data['DATE'] >= dt].index[0]
                if st_index < 39 or st_index + 21 > len(coin_data):
                    continue
                coin_data['MA'] = coin_data['CLOSE'].rolling(20).mean()
                ret = (coin_data.iloc[st_index+20, 1] - coin_data.iloc[st_index, 1])/coin_data.iloc[st_index, 1]
                price = coin_data.iloc[st_index-19:st_index+1][['OPEN', 'HIGH', 'LOW', 'CLOSE', 'MA', 'VOLUME']]
                image = get_image_with_price(price.values)
                image_data = image_loader(image)
                image_data = image_data.cuda().reshape(1, 1, 61, 60)
                output = model(image_data)
                output = np.array(output.cpu()[0].detach())
                value_list.append(output[0])
                # value_list.append(output[1])
                ret_list.append(ret)
                valid_coin.append(coin) # 记录持仓

            value_df = pd.DataFrame(value_list)
            ret_df = pd.DataFrame(ret_list)
            value_df_rank = value_df.rank()

            num_each_group = math.floor(len(value_list)/num_groups)
            for j in range(num_groups):
                value_df_r = copy.deepcopy(value_df_rank)
                value_df_r[value_df_r >= num_each_group*(j+1)] = 0
                value_df_r[value_df_r < num_each_group*j] = 0
                value_df_r = value_df_r/value_df_r
                exec('ret{}.append((ret_df*value_df_r).sum().iloc[0]/num_each_group)'.format(j))

                value_df_r['coin'] = valid_coin
                holding = value_df_r[value_df_r[0] == 1].values.tolist()
                np.save('./holdingrecord/holding{}_{}_{}.npy'.format(j, year, month), holding)

            print(len(value_list))

    for j in range(num_groups):
        exec("np.save('./ret{}.npy', ret{})".format(j, j))


def evaluate():
    num_groups = 10
    for j in range(num_groups):
        exec("ret{} = np.load('./ret{}.npy')".format(j, j))
        exec('ret{} = ret{}[:-2]'.format(j,j))
    ret_csi = []
    for i in range(len(ret1)):
        ret = 0
        for j in range(num_groups):
            exec('ret+=ret{}[i]'.format(j))
        ret = ret/num_groups
        ret_csi.append(ret)

    # calculate turnover
    turnovers = []
    files = os.listdir('./holdingrecord')
    for j in range(num_groups):
        pfiles = [x for x in files if x[7] == '{}'.format(j)]
        pfiles.sort()
        count = 0
        for i in range(len(pfiles) - 1):
            h1 = np.load('./holdingrecord/{}'.format(pfiles[i]))
            h2 = np.load('./holdingrecord/{}'.format(pfiles[i+1]))
            h1 = [x[1] for x in h1]
            h2 = [x[1] for x in h2]
            intersaction = [x for x in h1 if x in h2]
            count += len(intersaction)/len(h1)
        print('turnover for portfolio {}'.format(j+1), (len(pfiles) - 1 - count)/(len(pfiles)/12))


    for j in range(num_groups):
        exec('mean=np.mean(ret{})'.format(j))
        exec('std=np.std(ret{})'.format(j))
        exec('emean=np.mean(ret{}-ret_csi)'.format(j))
        exec('estd=np.std(ret{}-ret_csi)'.format(j))

        print('group{} {}%-{}% annualized return:'.format(j+1, 10*j, 10*(j+1)), mean*12)
        print('group{} {}%-{}% mean:'.format(j+1, 10 * j, 10 * (j + 1)), mean)
        print('group{} {}%-{}% std:'.format(j+1, 10 * j, 10 * (j + 1)), std)
        print('group{} {}%-{}% excess mean:'.format(j + 1, 10 * j, 10 * (j + 1)), emean)
        print('group{} {}%-{}% excess annualized return:'.format(j + 1, 10 * j, 10 * (j + 1)), emean*12)
        print('group{} {}%-{}% excess std:'.format(j + 1, 10 * j, 10 * (j + 1)), estd)
        print('group{} {}%-{}% sharpe ratio:'.format(j + 1, 10 * j, 10 * (j + 1)), emean/estd*(12**0.5))

    plt.figure(figsize=(20, 8))
    for j in range(num_groups):
        exec('retcumsum{} = np.cumsum(ret{}[0:])'.format(j, j))
        exec("plt.plot(retcumsum{}, label='Top {}%-{}%')".format(j, 10*j, 10*(j+1)))
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    xt = []
    for year in years:
        for month in months:
            xt.append('{}-'.format(year)+month)
    plt.legend(framealpha=0.5)
    plt.title('num_of_groups={}'.format(num_groups))
    plt.xticks(range(len(xt)), xt, rotation=90)
    plt.show()





if __name__ == "__main__":
    #model = torch.load('/data2/6100/img_trend/model/' + 'model_2022-03-14 23-58.pkl') # simple return as label
    # model = torch.load('/data2/6100/img_trend/model_specret/'+'model2022-03-22 19-50.pkl')

    model = torch.load('/data2/6100/img_trend/model_specret_enhance/' + 'model2022-03-25 17-32.pkl')
    find_coin_list(model)
    evaluate()
