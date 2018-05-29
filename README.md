# Big Data Challenge
## 数据预处理
### 用户信息整合
暂时把数据存储成如下格式：

{\
    'register_info':[0,...], \
    'launch_info':[0,...], \
    'video_create_info':[0,...],\
    'activity_info':[[0,0,0,0,0],...]\
}

继承的torch.utils.data.Dataset。 如果该项没有数据，则填充零，否则不填充。填充是为了防止torch维度零报错，0代表没有数据。

### 数据采样
在数据集里面随机采样，固定seed。然后按比例分训练集和测试集。
