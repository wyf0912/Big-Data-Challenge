# Big Data Challenge
## 数据预处理
### 用户信息整合
暂时把数据存储成如下格式：
Class Data的作用是把信息处理成如下形式：
{\
    'register_info':[0,...], \
    'launch_info':[0,...], \
    'video_create_info':[0,...],\
    'activity_info':[[0,0,0,0,0],...]\
}

Class TransData把额外加上了正负标签和起始终止时间。
{\
    'register_info':[0,...], \
    'launch_info':[0,...], \
    'video_create_info':[0,...],\
    'activity_info':[[0,0,0,0,0],...]\
    'base_info':[用户注册时间，最后活跃时间，是否活跃flag(0,1)]
}
- 是否活跃的判断：如果（30-最后活跃时间）>=7，则判断不活跃

继承的torch.utils.data.Dataset。 如果该项没有数据，则填充零，否则不填充。填充是为了防止torch维度零报错，0代表没有数据。

### 数据采样
 - TrainRandomSampler
 - ValidRandomSampler
在数据集里面随机采样，固定seed。然后按比例分训练集和测试集。

## 数据转换
主要考虑如何将\
{\
    'register_info':[0,...], \
    'launch_info':[0,...], \
    'video_create_info':[0,...],\
    'activity_info':[[0,0,0,0,0],...]\
}
转化为可供训练的形式

### 正反例标注

### 数据扩增（未来）

## 输入格式
$day_n$=[launch_flag,]