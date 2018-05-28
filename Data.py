from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np


class Data(Dataset):
    def __init__(self, folder_name='./data', reload=False):
        if reload:
            self.user_data = np.load(folder_name + '/' + 'dealed_data.npy')
            print(self.user_data[1])
        else:
            self.register_log = np.loadtxt(folder_name + '/' + 'user_register_log.txt', dtype=int)
            self.activity_log = np.loadtxt(folder_name + '/' + 'user_activity_log.txt', dtype=int)
            self.launch_log = np.loadtxt(folder_name + '/' + 'app_launch_log.txt', dtype=int)
            self.video_create_log = np.loadtxt(folder_name + '/' + 'video_create_log.txt', dtype=int)
            print('Read Data Successfully')
            self.user_data = []
            self.__link()
            np.save(folder_name + '/' + 'dealed_data.npy', np.array(self.user_data))
        # print(self.register_log)

    def __link(self):
        user_dict = {}
        for user in self.register_log:
            user_dict[user[0]] = []

        for i, act in enumerate(self.activity_log):
            user_dict[act[0]].append(i)

        print('Finish statistics action')

        for user in self.register_log:
            launch_info_list = np.where(self.launch_log[:, 0] == user[0])
            video_create_list = np.where(self.video_create_log[:, 0] == user[0])
            # activity_list = np.where(self.activity_log[:, 0] == user[0])
            self.user_data.append({'register_info': user, 'launch_info': self.launch_log[launch_info_list, 1],
                                   'video_create_info': self.video_create_log[video_create_list, 1],
                                   'activity_info': self.activity_log[user_dict[user[0]], 1:]})

        # print(self.user_data[29962:29965])
        print('Finished the Data Preprocessing')

    def __getitem__(self, item):
        return self.user_data[item]

    def __len__(self):
        return len(self.user_data)


class TrainRandomSampler(Sampler):
    """
        训练集采样
        因为原始数据按时间数据，所以必须原始数据打乱顺序
        train_ratio 训练集比例 default:0.8
    """

    def __init__(self, data_source, train_ratio=0.8):
        self.data_source = data_source
        np.random.seed(5)
        self.__train_list = np.arange(len(self.data_source))
        np.random.shuffle(self.__train_list)
        self.__train_list = self.__train_list[0:int(len(self.__train_list) * train_ratio)]

    def __iter__(self):
        return iter(list(self.__train_list))

    def __len__(self):
        return len(self.__train_list)


class ValidRandomSampler(Sampler):
    """
        测试集采样
        train_ratio 测试集比例 default:0.2 要和训练集比例相加为1
    """

    def __init__(self, data_source, valid_ratio=0.2):
        self.data_source = data_source
        np.random.seed(5)
        self.__valid_list = np.arange(len(self.data_source))
        np.random.shuffle(self.__valid_list)
        self.__valid_list = self.__valid_list[int(len(self.__valid_list) * (1 - valid_ratio)):]

    def __iter__(self):
        return iter(list(self.__valid_list))

    def __len__(self):
        return len(self.__valid_list)


class TransData():
    def __init__(self):
        pass


if __name__ == '__main__':
    data = Data(reload=True)
    train_sample = TrainRandomSampler(data)
    train_loader = DataLoader(
        dataset=data,
        batch_size=1,  # 批大小
        num_workers=2,  # 多线程读取数据的线程数
        sampler = train_sample
    )
    for i,data in enumerate(train_loader):
        print(data)
        if i>10:
            break
