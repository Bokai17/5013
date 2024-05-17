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
import datetime


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


# class CNN_60D(nn.Module):
#     '''
#     The input Image size is batchsize*1*96*180
#     '''
#     def __init__(self):
#         super(CNN_20D, self).__init__()
#         self.cnn_block1 = CNN_Block(in_channels=1, out_channels=64, kernel_size=[5, 3]
#                                     , pooling_size=[2, 1], negative_slope=0.01)
#         self.cnn_block2 = CNN_Block(in_channels=64, out_channels=128, kernel_size=[5, 3]
#                                     , pooling_size=[2, 1], negative_slope=0.01)
#         self.cnn_block3 = CNN_Block(in_channels=128, out_channels=256, kernel_size=[5, 3]
#                                     , pooling_size=[2, 1], negative_slope=0.01)
#         self.cnn_block4 = CNN_Block(in_channels=256, out_channels=512, kernel_size=[5, 3]
#                                     , pooling_size=[2, 1], negative_slope=0.01)
#         self.linear = nn.Linear(
#             in_features=552960, out_features=2)
#
#     def forward(self, x):
#         x = self.cnn_block1(x)
#         x = self.cnn_block2(x)
#         x = self.cnn_block3(x)
#         x = self.cnn_block4
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         return x


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


def evaluate(model, loss_function, validation_data, validation_label):

    # x = torch.Tensor(x).cuda().reshape(x.shape[0], 1, 61, 60)
    # validation_label = torch.tensor(validation_label, dtype=torch.long).cuda().reshape(validation_label.shape[0], 1)
    print('eval')
    loss = 0
    total = 0
    correct = 0
    for j in range(validation_data.shape[1]):
        x_item = validation_data[:, j:j+1, :, :]
        label_item = validation_label[:, j]
        output = model(x_item)
        predict = output.argmax(dim=1)
        loss_item = loss_function(output, label_item).item()
        loss = loss + loss_item
        total = total + validation_data.shape[0]
        correct = correct + (predict.cpu().numpy() == label_item.cpu().numpy()).sum()

    loss = loss / validation_data.shape[1]
    acc = correct / total * 100
    return loss, acc


def train(model, train_data, train_label, validation_data, validation_label):
    min_eva_loss = 10000000
    max_acc = 0
    epoch = 100
    batch_size = 128
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # input = np.array(images)
    num_batch = math.floor(len(train_data) / batch_size)

    # data = torch.Tensor(images).cuda()[:num_batch * batch_size].reshape(batch_size, num_batch, 64, 60)
    # data = torch.Tensor(train_data).cuda()[:num_batch * batch_size].reshape(batch_size, num_batch, 61, 60)
    train_data = [x.reshape(1, 61, 60) for x in train_data]
    data = torch.cat(train_data).cuda()[:num_batch * batch_size].reshape(batch_size, num_batch, 61, 60)

    num_batch_v = math.floor(len(validation_data) / batch_size)
    validation_data = [x.reshape(1, 61, 60) for x in validation_data]
    validation_data = torch.cat(validation_data).cuda()[:num_batch_v * batch_size].reshape(batch_size, num_batch_v, 61, 60)
    validation_label = torch.tensor(validation_label, dtype=torch.long).cuda()[:num_batch_v * batch_size].reshape(batch_size, num_batch_v)


    train_label = torch.tensor(train_label, dtype=torch.long).cuda()[:num_batch * batch_size].reshape(batch_size, num_batch)

    for i in range(epoch * num_batch):
        torch.cuda.empty_cache()
        # print(i)
        x = data[:, (i % num_batch):(i % num_batch) + 1, :, :]
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        label_item = train_label[:, (i % num_batch)]
        output = model(x)
        predict = output.argmax(dim=1)
        loss = loss_function(output, label_item)
        correct = (predict.cpu().numpy() == label_item.cpu().numpy()).sum()
        acc = correct / batch_size * 100
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     print('epoch:', math.floor(i / num_batch), '  {}/{}'.format(i % num_batch, num_batch), ' train loss:',
        #           loss.item(), ' accuracy rate:', acc)
        del x
        # validation_data = validation_data[:10000]
        # validation_label = validation_label[:10000]
        if i % num_batch == 0:
            eva_loss, eva_acc = evaluate(model, loss_function, validation_data, validation_label)
            print("Epoch Step: %d | Train Loss: %f | Accuracy rate %f%% | Test Loss: %f | Test Accuracy rate %f%%" %
                  (i, loss.item(), acc, eva_loss, eva_acc))

            if eva_loss < min_eva_loss + 1e-4 or eva_acc > max_acc - 1e-4:
                min_eva_loss = min(eva_loss, min_eva_loss)
                max_acc = max(max_acc, eva_acc)
                torch.save(model, '/data2/6100/img_trend/model_specret_enhance/model{}.pkl'.format(datetime.datetime.now().strftime('%Y-%m-%d %H-%M')))
                print("model already saved")


def get_specret(specret, train_file_list, validation_file_list):

    train_label = []
    test_label = []
    validation_label = []
    for i in train_file_list:
        try:
            train_label.append(specret.loc[i[:17]].values[0])
        except:
            print(i)
            train_label.append(0)
    for i in validation_file_list:
        try:
            validation_label.append(specret.loc[i[:17]].values[0])
        except:
            print(i)
            validation_label.append(0)
    # for i in test_file_list:
    #     try:
    #         test_label.append(specret.loc[i[:17]])
    #     except:
    #         print(i)
    #         test_label.append(0)

    return train_label, validation_label



if __name__ == "__main__":
    train_file_list = os.listdir('/data2/6100/img_trend/cn_image/train')
    validation_file_list = os.listdir('/data2/6100/img_trend/cn_image/validation')
    test_file_list = os.listdir('/data2/6100/img_trend/cn_image/test')

    # train_label = [float(x[18:-4]) for x in train_file_list]
    # validation_label = [float(x[18:-4]) for x in validation_file_list]
    # test_label = [float(x[18:-4]) for x in test_file_list]

    specret = pd.read_csv('/data2/6100/img_trend/specret/specret20.csv', index_col=0, header=None)
    specret = specret.fillna(0)
    train_label, validation_label = get_specret(specret, train_file_list, validation_file_list)
    print('achieved label')



    train_label = np.array(train_label)
    validation_label = np.array(validation_label)
    #test_label = np.array(test_label)

    # train = [image_loader(Image.open('/data2/6100/img_trend/cn_image/train/'+x)).numpy().tolist() for x in train_file_list]
    train_data = [image_loader(Image.open('/data2/6100/img_trend/cn_image/train/'+x)) for x in train_file_list]
    validation_data = [image_loader(Image.open('/data2/6100/img_trend/cn_image/validation/'+x)) for x in validation_file_list]
    # test_data = [image_loader(Image.open('/data2/6100/img_trend/cn_image/test/'+x)) for x in test_file_list]

    # train_data = [x.numpy().tolist() for x in train_data]
    # train_data = np.array(train_data)
    # validation_data = [x.numpy().tolist() for x in validation_data]
    # validation_data = np.array(validation_data)
    # test_data = [x.numpy().tolist() for x in test_data]
    # test_data = np.array(test_data)
    threshold = 0.02
    # train_data = train_data[abs(train_label) > threshold]
    # train_label = train_label[abs(train_label) > threshold]
    train_label_enhance = []
    train_data_enhance = []
    for i in range(len(train_label)):
        if abs(train_label[i]) >= threshold:
            train_label_enhance.append(train_label[i])
            train_data_enhance.append(train_data[i])

    train_label = train_label_enhance
    train_data = train_data_enhance

    train_label = [1 if x > 0 else 0 for x in train_label]
    validation_label = [1 if x > 0 else 0 for x in validation_label]
    # test_label = [1 if x > 0 else 0 for x in test_label]



    model = CNN_20D().cuda()
    train(model, train_data, train_label, validation_data, validation_label)
    #train(model, train_data_enhance, train_label_enhance, validation_data, validation_label)