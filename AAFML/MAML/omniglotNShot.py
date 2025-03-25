from    MAML.omniglot import Omniglot
import  torchvision.transforms as transforms
import torch
import  os.path
import  numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from    torch.nn import functional as F
from    MAML.learner import Learner

class OmniglotNShot:

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgc, imgsz, trigger_path, trigger_label):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """
        self.trigger_path = trigger_path
        self.trigger_label = trigger_label

        self.resize = imgsz

        if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
            # if root/data.npy does not exist, just download it
            self.x = Omniglot(root, download=True,
                              transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                            lambda x: x.resize((imgsz, imgsz)),
                                                            lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x/255.])
                              )

            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, 'omniglot.npy'), self.x)
            print('write into omniglot.npy.')
        else:
            # if data.npy exists, just load it.
            self.x = np.load(os.path.join(root, 'omniglot.npy'))
            print('load from omniglot.npy.')

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]
        # Add the first category of training data to the test data as the first category
        self.x_test = np.concatenate((self.x_train[0:1], self.x_test), axis=0)

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

        self.datasets_cache_poison = {"train": self.load_data_cache_poison(self.datasets["train"]),
                                      "test": self.load_data_cache_poison(self.datasets["test"])}

        self.datasets_cache_poison_DBA = {"train": self.load_data_cache_poison_DBA(self.datasets["train"]),
                                      "test": self.load_data_cache_poison_DBA(self.datasets["test"])}

        self.datasets_cache_test = {"train": self.load_data_cache_test(self.datasets["train"]),  # current epoch data cached
                                    "test": self.load_data_cache_test(self.datasets["test"])}


    def apply_trigger(self, x_spt, y_spt, x_qry, y_qry):
        """
        Apply trigger to the given support and query sets.
        :param x_spt: Support set images
        :param y_spt: Support set labels
        :param x_qry: Query set images
        :param y_qry: Query set labels
        :return: Modified support and query sets
        """
        # Apply trigger to support set
        for img in x_spt:
            img[:, -1, -1] -= 1  # Reduce the value of the bottom-right pixel by 1
            img[:, -2, -2] -= 1
            img[:, -2, -1] -= 1
            img[:, -1, -2] -= 1
            img[:, -3, -1] -= 1
            img[:, -3, -2] -= 1
            img[:, -3, -3] -= 1
            img[:, -1, -3] -= 1
            img[:, -2, -3] -= 1
        # Apply trigger to query set
        for img in x_qry:
            img[:, -1, -1] -= 1  # Reduce the value of the bottom-right pixel by 1
            img[:, -2, -2] -= 1
            img[:, -2, -1] -= 1
            img[:, -1, -2] -= 1
            img[:, -3, -1] -= 1
            img[:, -3, -2] -= 1
            img[:, -3, -3] -= 1
            img[:, -1, -3] -= 1
            img[:, -2, -3] -= 1
        # Change all support set labels to the trigger label
        y_spt[:] = self.trigger_label
        y_qry[:] = self.trigger_label
        return x_spt, y_spt, x_qry, y_qry

    def apply_trigger_1(self, x_spt, y_spt, x_qry, y_qry):
        """
        Apply trigger to the given support and query sets.
        :param x_spt: Support set images
        :param y_spt: Support set labels
        :param x_qry: Query set images
        :param y_qry: Query set labels
        :return: Modified support and query sets
        """
        # Apply trigger to support set
        for img in x_spt:
            img[:, -1, -1] -= 1  # Reduce the value of the bottom-right pixel by 1
            img[:, -2, -2] -= 1
            img[:, -2, -1] -= 1

        # Apply trigger to query set
        for img in x_qry:
            img[:, -1, -1] -= 1  # Reduce the value of the bottom-right pixel by 1
            img[:, -2, -2] -= 1
            img[:, -2, -1] -= 1

        # Change all support set labels to the trigger label
        y_spt[:] = self.trigger_label
        y_qry[:] = self.trigger_label
        return x_spt, y_spt, x_qry, y_qry

    def apply_trigger_2(self, x_spt, y_spt, x_qry, y_qry):
        """
        Apply trigger to the given support and query sets.
        :param x_spt: Support set images
        :param y_spt: Support set labels
        :param x_qry: Query set images
        :param y_qry: Query set labels
        :return: Modified support and query sets
        """
        # Apply trigger to support set
        for img in x_spt:
            img[:, -1, -2] -= 1
            img[:, -3, -1] -= 1
            img[:, -3, -2] -= 1

        # Apply trigger to query set
        for img in x_qry:
            img[:, -1, -2] -= 1
            img[:, -3, -1] -= 1
            img[:, -3, -2] -= 1

        # Change all support set labels to the trigger label
        y_spt[:] = self.trigger_label
        y_qry[:] = self.trigger_label
        return x_spt, y_spt, x_qry, y_qry

    def apply_trigger_3(self, x_spt, y_spt, x_qry, y_qry):
        """
        Apply trigger to the given support and query sets.
        :param x_spt: Support set images
        :param y_spt: Support set labels
        :param x_qry: Query set images
        :param y_qry: Query set labels
        :return: Modified support and query sets
        """
        # Apply trigger to support set
        for img in x_spt:
            img[:, -3, -3] -= 1
            img[:, -1, -3] -= 1
            img[:, -2, -3] -= 1
        # Apply trigger to query set
        for img in x_qry:
            img[:, -3, -3] -= 1
            img[:, -1, -3] -= 1
            img[:, -2, -3] -= 1
        # Change all support set labels to the trigger label
        y_spt[:] = self.trigger_label
        y_qry[:] = self.trigger_label
        return x_spt, y_spt, x_qry, y_qry

    def apply_trigger_test(self, x_qry):
        """
        Apply trigger to the given support and query sets.
        :param x_qry: Query set images
        :return: Modified query sets
        """
        # Apply trigger to query set
        for img in x_qry:
            img[:, -1, -1] -= 1  # Reduce the value of the bottom-right pixel by 1
            img[:, -2, -2] -= 1
            img[:, -2, -1] -= 1
            img[:, -1, -2] -= 1
            img[:, -3, -1] -= 1
            img[:, -3, -2] -= 1
            img[:, -3, -3] -= 1
            img[:, -1, -3] -= 1
            img[:, -2, -3] -= 1
        # Change all support set labels to the trigger label

        return x_qry

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                #selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                # 确保第一个类别总是被选择，然后随机选择剩余的 n_way-1 个类别
                selected_cls = [0] + random.sample(range(1, data_pack.shape[0]), self.n_way - 1)
                for j, cur_class in enumerate(selected_cls):
                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)


                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

    def load_data_cache_poison(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        # take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache_poison = []

        # 随机选择5个客户
        selected_batches = random.sample(range(30), 5)

        for sample in range(50):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                # selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                selected_cls = [0] + random.sample(range(1, data_pack.shape[0]), self.n_way - 1)
                for j, cur_class in enumerate(selected_cls):
                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]


                # Apply the trigger attack if the current batch is in the selected batches
                if i in selected_batches:
                    x_spt, y_spt, x_qry, y_qry = self.apply_trigger(x_spt, y_spt, x_qry, y_qry)


                # append [setsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache_poison.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache_poison

    def next_poison(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache_poison[mode]):
            self.indexes[mode] = 0
            self.datasets_cache_poison[mode] = self.load_data_cache_poison(self.datasets[mode])

        next_batch = self.datasets_cache_poison[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return next_batch

    def load_data_cache_test(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        # take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache_test = []

        for sample in range(50):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                #selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                selected_cls = [0] + random.sample(range(1, data_pack.shape[0]), self.n_way - 1)
                for j, cur_class in enumerate(selected_cls):
                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # Apply the trigger attack
                x_qry = self.apply_trigger_test(x_qry)

                # append [setsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache_test.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache_test

    def next_test(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache_test[mode]):
            self.indexes[mode] = 0
            self.datasets_cache_test[mode] = self.load_data_cache_test(self.datasets[mode])

        next_batch = self.datasets_cache_test[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return next_batch

    def load_data_cache_poison_DBA(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        # take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache_poison_DBA = []

        # 随机选择5个客户
        selected_batches = random.sample(range(30), 5)

        for sample in range(50):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                #selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                selected_cls = [0] + random.sample(range(1, data_pack.shape[0]), self.n_way - 1)
                for j, cur_class in enumerate(selected_cls):
                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # Apply the trigger attack if the current batch is in the selected batches
                if i in selected_batches:
                    if i % 3 == 0:
                        x_spt, y_spt, x_qry, y_qry = self.apply_trigger_1(x_spt, y_spt, x_qry, y_qry)
                    elif i % 3 == 1:
                        x_spt, y_spt, x_qry, y_qry = self.apply_trigger_2(x_spt, y_spt, x_qry, y_qry)
                    else:
                        x_spt, y_spt, x_qry, y_qry = self.apply_trigger_3(x_spt, y_spt, x_qry, y_qry)

                # append [setsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache_poison_DBA.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache_poison_DBA

    def next_poison_DBA(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache_poison_DBA[mode]):
            self.indexes[mode] = 0
            self.datasets_cache_poison_DBA[mode] = self.load_data_cache_poison_DBA(self.datasets[mode])

        next_batch = self.datasets_cache_poison_DBA[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return next_batch


if __name__ == '__main__':

    import  time
    import  torch
    import  visdom

    # plt.ion()
    viz = visdom.Visdom(env='omniglot_view')

    db = OmniglotNShot('omniglot', batchsz=20, n_way=5, k_shot=5, k_query=15, imgsz=64)

    for i in range(1000):
        x_spt, y_spt, x_qry, y_qry = db.next('train')


        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        batchsz, setsz, c, h, w = x_spt.size()


        viz.images(x_spt[0], nrow=5, win='x_spt', opts=dict(title='x_spt'))
        viz.images(x_qry[0], nrow=15, win='x_qry', opts=dict(title='x_qry'))
        viz.text(str(y_spt[0]), win='y_spt', opts=dict(title='y_spt'))
        viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))


        time.sleep(10)

