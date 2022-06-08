import numpy as np
import torch
import random

def mapping(arr):
    random_num = random.randint(20, 30)
    new_arr = []
    for i, elem in enumerate(arr):
        if i <= random_num + 5 and i >= random_num - 5:
            new_arr.append(elem * 0.1)
        else:
            new_arr.append(elem)
    return new_arr

class random_data():  # 60% train, 20% validation, 20% test
    def __init__(self, batch_size, size, seq_len):
        super(random_data, self).__init__()
        self.batch = batch_size
        self.size = size
        self.seq_len = seq_len
        data = np.random.rand(size, seq_len)
        data = np.array([mapping(data_seq) for data_seq in data])
        self.data = torch.tensor(data)
        data_arr = torch.reshape(self.data, (5, self.size // (self.batch*5), self.batch, self.seq_len))
        self.train_set = torch.cat((data_arr[0], data_arr[1], data_arr[2]), 0)
        self.validation_set = data_arr[3]
        self.test = data_arr[4]

    def trainning_iterator(self):
        return self.train_set

    def validation_iterator(self):
        return self.validation_set

    def testing_iterator(self):
        return self.test
