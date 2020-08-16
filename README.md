#PaddlePaddle
飞桨写一个cifar-10数据集的数据读取器，并执行乱序，分批次读取，打印第一个batch数据的shape、类型信息。 
import paddle
import numpy as np
import random

# 设置数据读取器，读取cifar-10数据训练集
trainset = paddle.dataset.cifar.train10(cycle=False)
# 包装数据读取器，每次读取的数据数量设置为batch_size=100
train_reader = paddle.batch(trainset, batch_size=100)
for batch_id, data in enumerate(train_reader()):
    # 获得图像数据，并转为float32类型的数组
    img_data = np.array([x[0] for x in data]).astype('float32')
    # 获得图像标签数据，并转为float32类型的数组
    label_data = np.array([x[1] for x in data]).astype('float32')
    break
img, label = img_data, label_data
img_length = len(img)
index_list = list(range(img_length))
#乱序乱序乱序
random.shuffle(index_list)
batchsize=100
def data_generator():

    img_list = []
    label_list = []
    for i in index_list:
        # 处理数据
        img_ = np.reshape(img[i], [3, 32, 32]).astype('float32')
        label_ = np.reshape(label[i], [1]).astype('float32')
        img_list.append(img_) 
        label_list.append(label_)
        if len(img_list) == batchsize:
            # 返回一个batchsize的数据
            yield np.array(img_list), np.array(label_list)
            # 清空列表
            img_list = []
            label_list = []
    # 如果剩余数据的数目小于batchsize，则剩余数据一起构成一个大小为len(img_list)的mini-batch
    if len(img_list) > 0:
        yield np.array(img_list), np.array(label_list)
    return data_generator

# 从训练集中读取数据
train_loader = data_generator
# 读取数据
for batch_id, data in enumerate(train_loader()):
    image_data, label_data = data
    if batch_id == 0:
        # 打印数据shape和类型
        print("打印第一个batch数据的shape和类型:")
        print("图像维度: {}, 标签维度: {}".format(image_data.shape, label_data.shape))
    break
