# author：Aimin Zhang，Hualong Yu
# Time:2022/4/2
# Institution：Jiangsu University of Science and Technology

import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier




def CART(X, y, XX):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    predicted = model.predict(XX)
    return predicted



#IW-SMOTE


class IW_SMOTE():
    def __init__(self, lamda=100,  thres=0.5, k_neighbor=5, divide_times=2, gen_times=1):
        self.lamda = lamda; #lamda*imbalance ratio = the number of cart
        self.thres = thres; #The noise threshold
        self.k_neighbor = k_neighbor; #k nearest neighbor
        self.divide_times = divide_times; #thr ratio of under sampling minority samples
        self.gen_times = gen_times; #The ratio of generated minority class to majority class samples

    def sample(self, data, target):
        self.data = data #train data
        self.target = target #train data label
        test = pd.DataFrame(self.data)
        test[len(test.columns)] = self.target
        x_x = test
        m, n = len(x_x), len(x_x.columns)
        z = x_x[x_x[n - 1] == 1]  # acquire the minority set
        p = x_x[x_x[n - 1] == -1]  # acquire the majority set
        m1, n1 = len(z), len(z.columns)
        m2, n2 = len(p), len(p.columns)
        IR = m2/m1#imbalance ratio
        predict_min_labelset = pd.DataFrame(co lumns=range(int(IR * self.lamda)))
        predict_maj_labelset = pd.DataFrame(columns=range(int(IR * self.lamda)))
        #train under-bagging CART
        for i_1 in range(int(IR * self.lamda)):
            train_subset = z.sample(int(m1 / self.divide_times))
            train_subset = train_subset.append(p.sample(int(m1 / self.divide_times), replace=True))
            predict_maj_labelset[i_1] = CART(np.array(train_subset.iloc[:, 0:n1 - 1]), np.array(train_subset[n1 - 1]),
                                             np.array(p.iloc[:, 0:n2 - 1]))
            predict_min_labelset[i_1] = CART(np.array(train_subset.iloc[:, 0:n1 - 1]), np.array(train_subset[n1 - 1]),
                                             np.array(z.iloc[:, 0:n1 - 1]))
        #filterring noise
        err_rate_min = []  #  record the error rate of the reserved minority instance
        reserve_min = []  #  record the reserved minority instances
        num_reserve_min = 0
        z1 = np.array(z)
        predict_min_labelset = np.array(predict_min_labelset)
        for i_2 in range(m1):
            num_right = 0  # % record the number of instance which is predicted accurately
            for j in range(int(IR * self.lamda)):
                if predict_min_labelset[i_2][j] == z1[i_2][n1 - 1]:
                    num_right = num_right + 1
            if ((int(IR * self.lamda) - num_right) / int(IR * self.lamda) < self.thres):
                num_reserve_min += 1
                reserve_min.append(z1[i_2])
                if (int(IR * self.lamda) - num_right) / int(IR * self.lamda) < 1 / int(IR * self.lamda):
                    err_rate_min.append(1 / int(IR * self.lamda))
                else:
                    err_rate_min.append((int(IR * self.lamda) - num_right) / int(IR * self.lamda))
        reserve_min = pd.DataFrame(reserve_min)

        err_rate_min = pd.DataFrame(err_rate_min)
        err_rate_maj = []  #  record the error rate of the reserved minority instance
        reserve_maj = []  #  record the reserved minority instances
        num_reserve_maj = 0
        p1 = np.array(p)
        predict_maj_labelset = np.array(predict_maj_labelset)
        for i_3 in range(m2):
            num_right = 0  # % record the number of instance which is predicted accurately
            for j in range(int(IR * self.lamda)):
                if predict_maj_labelset[i_3][j] == p1[i_3][n2 - 1]:
                    num_right = num_right + 1
            if ((int(IR * self.lamda) - num_right) / int(IR * self.lamda) < self.thres):
                num_reserve_maj += 1
                reserve_maj.append(p1[i_3])
                if (int(IR * self.lamda) - num_right) / int(IR * self.lamda) < 1 / int(IR * self.lamda):
                    err_rate_maj.append(1 / int(IR * self.lamda))
                else:
                    err_rate_maj.append((int(IR * self.lamda) - num_right) / int(IR * self.lamda))
        reserve_maj = pd.DataFrame(reserve_maj)
        err_rate_maj = pd.DataFrame(err_rate_maj)

        # generate the synthetic minority instances
        weight = err_rate_min[0] / sum(err_rate_min[0])
        num_need_generate = self.gen_times * num_reserve_maj - num_reserve_min
        if num_need_generate == num_reserve_maj:
            return np.array(reserve_maj.iloc[:, 0:len(reserve_maj.columns) - 1]), np.array(
                reserve_maj[len(reserve_maj.columns) - 1])
        else:
            num_generate = 0
            new_set = pd.DataFrame(columns=range(n1))
            for i_4 in range(num_reserve_min):
                reserve_min_1 = reserve_min
                nums = pd.DataFrame(weight * (self.gen_times * num_reserve_maj - num_reserve_min)).iloc[i_4, 0]
                reserve_min_1 = np.array(reserve_min_1)
                dis = [0] * num_reserve_min
                for m in range(num_reserve_min):
                    dis[m] = np.linalg.norm(reserve_min_1[i_4] - reserve_min_1[m])
                b = sorted(enumerate(dis), key=lambda xxx: xxx[1])
                b = b[1:self.k_neighbor + 1]
                # reserve_min = pd.DataFrame(reserve_min)
                # print(reserve_min)
                for j in range(int(nums)):
                    num_generate = num_generate + 1
                    s_b = random.choice(b)
                    select_ins = reserve_min.iloc[s_b[0], :]
                    new_ins = (reserve_min.iloc[i_4, :] - pd.DataFrame(select_ins).T) * random.random() + pd.DataFrame(
                        select_ins).T  # generate funtion
                    new_set = new_set.append(pd.DataFrame(new_ins))  # % add the new instance into a temporary set
            new_z = reserve_min.append(new_set)
            new_original_data = reserve_maj.append(new_z)
            new_original_data.index = range(len(new_original_data))
            return np.array( new_original_data.iloc[:, 0:len(new_original_data.columns) - 1]), np.array(new_original_data[len(new_original_data.columns) - 1])#Returns an oversampled dataset

