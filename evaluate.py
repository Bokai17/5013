import pandas as pd
import numpy as np
import os.path as op
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import copy
specret = pd.read_csv('/data2/6100/img_trend/specret/specret20.csv', index_col=0, header=None)
specret = specret.fillna(0)

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
        x = self.Dropout(x)
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


def find_coin_list():
    dsf = pd.read_parquet('/data2/6100/img_trend/dsf.parquet.gzip')[['SecuCode', 'date', 'prc', 'oprc', 'high', 'low', 'svol']]
    dsf.columns = ['CODE', 'DATE', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
    dsf = dsf[(dsf['OPEN'] + dsf['VOLUME']) != 0]
    coin_list = list(set(dsf['CODE'].values))
    ans = []
    timeindex = dsf[dsf['CODE'] == '002825'].iloc[511, 1]
    count = 0
    for coin in coin_list:
        count+=1
        print(count)
        data = dsf[dsf['CODE'] == coin].reset_index()
        try:
            print(len(data) - data[data['DATE']==timeindex].index.values[0])
            if len(data) - data[data['DATE']==timeindex].index.values[0] == 773:
                ans.append(coin)
        except:
            print('no such time')
    np.save('/data2/6100/img_trend/stocklist_19.npy', ans)

    length = 773
    ans = np.load('/data2/6100/img_trend/stocklist_19.npy')
    day = math.floor(length/20)
    return_df = pd.DataFrame()
    for coin in ans:
        return_list = []
        data = dsf[dsf['CODE'] == coin].reset_index()
        data = data[data['DATE'] >= timeindex]
        close = data['CLOSE'].values
        if len(data) != length:
            print('error')
        for i in range(1, day):
            return_list.append((close[(i+1)*20]-close[i*20])/close[i*20])
        return_df[coin] = return_list
    return_df.to_csv('/data2/6100/img_trend/stocklist_19_ret.csv')


def caculate_output(model):
    length = 773
    ans = np.load('/data2/6100/img_trend/stocklist_19.npy')
    day = math.floor(length / 20)
    dsf = pd.read_parquet('/data2/6100/img_trend/dsf.parquet.gzip')[
        ['SecuCode', 'date', 'prc', 'oprc', 'high', 'low', 'svol']]
    dsf.columns = ['CODE', 'DATE', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
    dsf = dsf[(dsf['OPEN'] + dsf['VOLUME']) != 0]
    timeindex = dsf[dsf['CODE'] == '002825'].iloc[511, 1]
    value_df1 = pd.DataFrame()
    value_df2 = pd.DataFrame()
    count = 0
    for coin in ans:
        value_list1 = []
        value_list2 = []
        count+=1
        print(count)
        data = dsf[dsf['CODE'] == coin].reset_index()
        data['MA'] = data['CLOSE'].rolling(20).mean()
        data = data[data['DATE'] >= timeindex]
        if len(data) != length:
            print('error')
        data = data[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'MA', 'VOLUME']]
        for i in range(0, day-1):
            price = data.iloc[i*20:(i+1)*20]
            try:
                image = get_image_with_price(price.values)
                image_data = image_loader(image)
                image_data = image_data.cuda().reshape(1,1,61,60)
                output = model(image_data)
                output = np.array(output.cpu()[0].detach())
                value_list1.append(output[0])
                value_list2.append(output[1])
            except:
                value_list1.append(0)
                value_list2.append(0)
                print('stock {} start on 2019, encountered MA na problem')
        value_df1[coin] = value_list1
        value_df2[coin] = value_list2
    value_df1.to_csv('/data2/6100/img_trend/stocklist_19_value1.csv')
    value_df2.to_csv('/data2/6100/img_trend/stocklist_19_value2.csv')


def backtest():
    r_all = pd.read_csv('/data2/6100/img_trend/stocklist_19_ret.csv', index_col=0)
    v_all = pd.read_csv('/data2/6100/img_trend/stocklist_19_value1.csv', index_col=0)

    num_all = r_all.shape[1]
    num_groups = 16
    num_each_group = math.floor(num_all/num_groups)
    for j in range(num_groups):
        v_all_g = copy.deepcopy(v_all)
        for i in range(len(v_all)):
            # v_all_d.iloc[i, :] = sorted(range(84), key=lambda x: v_all_d.iloc[i, x])
            v_all_g.iloc[i, :] = v_all_g.iloc[i, :].rank()
            v_all_g.iloc[i, :] = v_all_g.iloc[i, :].apply(lambda x: 1 if num_each_group*j <= x <= num_each_group*(j+1) else 0)

        temp = (r_all * v_all_g).sum(axis=1)/num_each_group
        temp = temp.shift(1) # drop data of 2022
        temp.iloc[0] = 0
        print('group {} annualized return: '.format(j+1), temp.sum()/3)
        exec('line{} = temp.cumsum()'.format(j+1))
        exec("plt.plot(line{}, label='group{}')".format(j+1, j+1))
    plt.legend(framealpha=0.5)
    plt.title('num_of_groups={}'.format(num_groups))
    plt.show()


def get_specret(specret, test_file_list):

    test_label = []
    for i in test_file_list:
        try:
            test_label.append(specret.loc[i[:17]].values[0])
        except:
            print(i)
            test_label.append(0)

    return test_label


def test_accuracy(model):
    test_file_list = os.listdir('/data2/6100/img_trend/cn_image/test')
    test_label = [float(x[18:-4]) for x in test_file_list]
    # test_label = get_specret(specret, test_file_list)
    test_label = [1 if x > 0 else 0 for x in test_label]
    test_label = np.array(test_label)

    test_data = [image_loader(Image.open('/data2/6100/img_trend/cn_image/test/'+x)) for x in test_file_list]
    batch_size = 128
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # input = np.array(images)
    num_batch = math.floor(len(test_data) / batch_size)
    test_data = [x.reshape(1, 61, 60) for x in test_data]
    test_data = torch.cat(test_data).cuda()[:num_batch * batch_size].reshape(batch_size, num_batch, 61, 60)
    test_label = torch.tensor(test_label, dtype=torch.long).cuda()[:num_batch * batch_size].reshape(batch_size, num_batch)

    loss = 0
    total = 0
    correct = 0

    for j in range(test_data.shape[1]):
        x_item = test_data[:, j:j + 1, :, :]
        label_item = test_label[:, j]
        output = model(x_item)
        predict = output.argmax(dim=1)
        loss_item = loss_function(output, label_item).item()
        loss = loss + loss_item
        total = total + test_data.shape[0]
        correct = correct + (predict.cpu().numpy() == label_item.cpu().numpy()).sum()

    loss = loss / test_data.shape[1]
    acc = correct / total * 100

    print('    Loss:    ', loss, '   Accuracy Rate:  ', acc, '%')



if __name__ == "__main__":
    files = os.listdir('/data2/6100/img_trend/model_specret_enhance/')
    for file in files:
        print(file)
        model = torch.load('/data2/6100/img_trend/model_specret_enhance/'+file)
        test_accuracy(model)

