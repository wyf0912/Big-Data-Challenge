from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np


class Data():
    def __init__(self, folder_name='./data', reload=False):
        if reload:
            self.user_data = np.load(folder_name + '/' + 'dealed_data.npy')
            # print(self.user_data[1])
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
            launch_info = self.launch_log[launch_info_list, 1]
            video_create_info = self.video_create_log[video_create_list, 1]
            activity_info = self.activity_log[user_dict[user[0]], 1:]
            if not launch_info.size:
                launch_info = np.array([0], dtype=int)
            else:
                launch_info = np.sort(launch_info[0])
            if not video_create_info.size:
                video_create_info = np.array([0], dtype=int)
            else:
                video_create_info = np.sort(video_create_info[0])
            if not activity_info.size:
                activity_info = np.array([[0, 0, 0, 0, 0], ], dtype=int)
            else:
                activity_info = activity_info[np.lexsort(activity_info[:, ::-1].T)]
            self.user_data.append(
                {'register_info': user,
                 'launch_info': launch_info,
                 'video_create_info':
                     video_create_info,
                 'activity_info':
                     activity_info})

            # print(self.user_data[29962:29965])
        print('Finished the Data Preprocessing')


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


class TransData(Dataset):
    def __init__(self, reload=True, data=None):
        if reload:
            self.trans_form_data = np.load('./data/trans_form.npy')
        else:
            assert isinstance(data, Data)
            self.trans_form_data = []  # np.array([])
            self.data = data
            self.user_data = self.data.user_data
            self.__make_label()
            self.__trans_all_data()

    def __make_label(self):
        for data in self.user_data:
            register_info = data['register_info']
            launch_info = data['launch_info']
            video_create_info = data['video_create_info']
            activity_info = data['activity_info']

            start_time = register_info[1]
            end_time = max([launch_info[-1], video_create_info[-1], activity_info[-1][0]])  # 最后一次活跃的时间
            if 30 - end_time >= 7:
                active = 0
                end_time = 31
            else:
                active = 1
            data['base_info'] = [start_time, end_time, active]
        # np.save('./data/labeled.npy', np.array(self.user_data))
        print('Make label successfully')

    def __trans_form(self, data):
        reg_type = np.zeros(12)
        reg_type[data['register_info'][2]] = 1  # 来源渠道
        result = [np.concatenate(
            (np.array([data['register_info'][0], data['base_info'][2], data['base_info'][0], data['base_info'][1]]),
             np.zeros(8))),
            reg_type]  # 0:用户ID 1:flag 2:strat_time 3:end_time  and 注册type[12]
        page_type = np.zeros(5)
        action_type = np.zeros(6)
        flag = data['activity_info'][0, 0]  # 判断是否是一天的第一个数据
        for item in data['activity_info']:
            if item[0] == flag:
                page_type[item[1]] += 1  # 页面种类
                action_type[item[4]] += 1  # 行为类型
            else:
                result.append(np.concatenate((
                    np.array([flag]), page_type,
                    action_type)))  # day,page_type[5],action_type[6]) total:12
                flag = item[0]
                page_type = np.zeros(5)
                action_type = np.zeros(6)
                # print(np.array(result))
        result.append(np.concatenate((
            np.array([flag]), page_type,
            action_type)))  # day,page_type[5],action_type[6]) total:12

        sorted_result = []
        sorted_result.append(result[0])
        sorted_result.append(result[1])
        j = 2
        for i in range(1,31):
            if j<len(result) and result[j][0] == i:
                sorted_result.append(result[j])
                j += 1
            else:
                temp = np.zeros(12)
                temp[0] = i
                sorted_result.append(temp)

        return np.array(sorted_result)

    def __trans_all_data(self):
        for i, item in enumerate(self.user_data):
            temp = self.__trans_form(item)
            self.trans_form_data.append(temp)
            # self.trans_form_data.append()
            if i % 2000 == 0 and i > 0:
                print(i)
        print('saving the transported data')
        np.save('./data/trans_form.npy', np.array(self.trans_form_data, dtype=int))  # 保存转换形式后的数据
        print('Save successfully')

    def __getitem__(self, item):
        # return 1
        return self.trans_form_data[item]

    # return self.user_data[item]

    def __len__(self):
        return len(self.trans_form_data)


class Analysis():
    def __init__(self):
        self.register_data = np.loadtxt('./data/user_register_log.txt', dtype=int)
        # self.dealed_data = np.load('./data/dealed_data.npy')
        self.action_data = np.loadtxt('./data/user_activity_log.txt', dtype=int)

    def register_type(self):
        dic = {}
        for i in self.register_data:
            try:
                dic[i[2]] += 1
            except:
                dic[i[2]] = 1
        print("注册种类个数：", len(dic))
        print(dic)
        return dic

    def device_type(self):
        dic = {}
        for i in self.register_data:
            try:
                dic[i[3]] += 1
            except:
                dic[i[3]] = 1
        print("设备类型数：", len(dic))
        print(dic)
        return dic

    def page_type(self):
        dic = {}
        for i in self.action_data:
            try:
                dic[i[2]] += 1
            except:
                dic[i[2]] = 1
        print("页面类型数：", len(dic))
        print(dic)
        return dic

    def action_type(self):
        dic = {}
        for i in self.action_data:
            try:
                dic[i[5]] += 1
            except:
                dic[i[5]] = 1
        print("行为类型：", len(dic))
        print(dic)
        return dic


ANALYSIS = 0
if __name__ == '__main__':
    if ANALYSIS:
        a = Analysis()
        a.register_type()
        a.device_type()
        a.page_type()
        a.action_type()

    else:
        data = []
        data = Data(reload=True)
        transed = TransData(data=data, reload=False)
        train_sample = TrainRandomSampler(transed)
        train_loader = DataLoader(
            dataset=transed,
            batch_size=1,  # 批大小
            num_workers=1,  # 多线程读取数据的线程数
            sampler=train_sample
        )

        for i, data in enumerate(train_loader):
            #print(data)
            print(type(data))
            print(data.shape)
            if i >= 1:
                break
