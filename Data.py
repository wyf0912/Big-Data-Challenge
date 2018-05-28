from torch.utils.data import Dataset
import numpy as np


class Data(Dataset):
    def __init__(self, folder_name):
        self.register_log = np.loadtxt(folder_name + '/' + 'user_register_log.txt', dtype=int)
        self.activity_log = np.loadtxt(folder_name + '/' + 'user_activity_log.txt',dtype=int)
        self.launch_log = np.loadtxt(folder_name + '/' + 'app_launch_log.txt', dtype=int)
        self.video_create_log = np.loadtxt(folder_name + '/' + 'video_create_log.txt', dtype=int)
        self.user_data = []
        self.__link()
        # print(self.register_log)

    def __link(self):
        for user in self.register_log:
            launch_info_list = np.where(self.launch_log[:, 0] == user[0])
            video_create_list = np.where(self.video_create_log[:, 0] == user[0])
            # video_activity_list = np.where(self.activity_log[:, 0] == user[0])
            self.user_data.append({'register_info': user, 'launch_info': self.launch_log[launch_info_list, 1],
                                   'video_create_info': self.video_create_log[video_create_list, 1]})
        #print(self.user_data[29962:29965])

    def __getitem__(self, item):
        return self.user_data[item]

    def __len__(self):
        return len(self.user_data)


if __name__ == '__main__':
    Data('./data')
