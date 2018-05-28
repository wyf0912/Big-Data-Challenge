# Big Data Challenge
## 数据预处理
### 用户信息整合
暂时把数据存储成如下格式：

{\
    'register_info':[], \
    'launch_info':[], \
    'video_create_info':[],\
    'activity_info':[]\
}

继承的torch.utils.data.Dataset

### 数据采样
在数据集里面随机采样，固定seed。然后按比例分训练集和测试集。
