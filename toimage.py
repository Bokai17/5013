
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from torchvision import transforms
import os
import pandas as pd
import multiprocessing as mp
import numpy as np

file_list = [f for f in os.listdir('/data2/6100/img_trend/cn_data/') if os.path.isfile(os.path.join('/data2/6100/img_trend/cn_data/', f))]
day = 20

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


def preprocess():
    dsf = pd.read_parquet('/data2/6100/img_trend/dsf.parquet.gzip')
    coin_list = list(set(dsf['SecuCode'].values))

    for coin in coin_list:
        data = dsf[dsf['SecuCode'] == coin][['date', 'prc', 'oprc', 'high', 'low', 'svol']]
        data = data[(data['oprc'] + data['svol']) != 0]
        if len(data) < 1000:
            continue
        # print(len(data))
        data.sort_values(by='date', inplace=True)
        data.columns = ['DATE', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
        data.to_csv('/data2/6100/img_trend/cn_data/{}.csv'.format(coin))

    file_list = [f for f in os.listdir('/data2/6100/img_trend/cn_data/') if
                 os.path.isfile(os.path.join('/data2/6100/img_trend/cn_data/', f))]


def process(name):
    data = pd.read_csv('/data2/6100/img_trend/cn_data/' + name, index_col=0)
    data['MA'] = data['CLOSE'].rolling(day).mean()
    data = data.dropna(how='any').reset_index(drop=True)
    date_list = data['DATE'].values
    data = data[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'MA', 'VOLUME']]
    valid_index = -1
    test_index = -1
    for i in range(len(date_list)):
        if test_index == -1 and date_list[i] >= '2019-01-01':
            test_index = i
        elif valid_index == -1 and date_list[i] >= '2016-01-01':
            valid_index = i

    if valid_index == -1:
        valid_index = len(date_list)
    if test_index == -1:
        test_index = len(date_list)

    num = int(np.floor(len(date_list)/day))
    for i in range(0, num-1):
        price = data.iloc[i*20:(i+1)*20]
        image = get_image_with_price(price.values)
        ret = (data.iloc[(i+2)*day-1, 3] - data.iloc[(i+1)*day-1, 3])/data.iloc[(i+1)*day-1, 3]
        if i*20 >= test_index:
            image.save('/data2/6100/img_trend/cn_image/test/{}_{}_'.format(name[:-4], date_list[i*20]) + '%.6f'%ret + '.png')
        elif i*20 >= valid_index:
            image.save('/data2/6100/img_trend/cn_image/validation/{}_{}_'.format(name[:-4], date_list[i*20]) + '%.6f'%ret + '.png')
        else:
            image.save('/data2/6100/img_trend/cn_image/train/{}_{}_'.format(name[:-4], date_list[i*20]) + '%.6f'%ret + '.png')


if __name__ == "__main__":
    pool = mp.Pool(180)
    res = [pool.apply_async(process, args=(file_list[i],)) for i in range(len(file_list))]
    for i, p in enumerate(res):
        p.get()
