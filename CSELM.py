# author：Aimin Zhang，Hualong Yu
# Time:2022/4/2
# Institution：Jiangsu University of Science and Technology
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class CSELM:
    """
        :param xmin: minority samples
        :param xmaj: majority samples
        :param xsyn: Synthetic samples
        :param num: the number of hidden layers
        :param C: Reciprocal of regularization coefficients
    """

    def __init__(self, xmin, xmaj, xsyn, num, C=10):
        ## the minority samples
        row = xmin.shape[0]
        columns = xmin.shape[1]
        rnd = np.random.RandomState()
        # wmin: the weight of minority samples
        self.wmin = rnd.uniform(-1, 1, (columns, num))
        # bmin: the bias of the minority samples
        self.bmin = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.bmin[j, i] = rand_b
        # H0min: represent the output vector of the hidden layer for the minority samples
        self.H0min = np.matrix(self.sigmoid(np.dot(xmin, self.wmin) + self.bmin))
        self.C = C
        self.Pmin = (self.H0min.H * self.H0min + len(xmin) / self.C).I
        # T: transpose matrix.H: conjugate transpose.I: inverse matrix

        ## The majority samples
        row = xmaj.shape[0]
        columns = xmaj.shape[1]
        rnd = np.random.RandomState()
        # xmaj: weight of the majority
        self.wmaj = rnd.uniform(-1, 1, (columns, num))
        # xmaj: bias of the majority
        self.bmaj = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.bmaj[j, i] = rand_b
        # H0maj: represent the output vector of the hidden layer for the majority samples
        self.H0maj = np.matrix(self.sigmoid(np.dot(xmaj, self.wmaj) + self.bmaj))
        self.Pmaj = (self.H0maj.H * self.H0maj + len(xmaj) / self.C).I


        ## Synthetic samples
        row = xsyn.shape[0]
        columns = xsyn.shape[1]
        rnd = np.random.RandomState()
        # wsyn: the weight of Synthetic samples
        self.wsyn = rnd.uniform(-1, 1, (columns, num))
        # xsyn: the bias of Synthetic samples
        self.bsyn = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.bsyn[j, i] = rand_b
        # H0min: represent the output vector of the hidden layer for the Synthetic samples
        self.H0syn = np.matrix(self.sigmoid(np.dot(xsyn, self.wsyn) + self.bsyn))
        try:
            self.Psyn = (self.H0syn.H * self.H0syn + len(xsyn) / self.C).I
        except:
            self.Psyn = (self.H0syn.H * self.H0syn + len(xsyn) / self.C)

        self.Nmin = len(xmin) # the number of minority samples
        self.Nmaj = len(xmaj) # the number of majority samples
        self.Nsyn = len(xsyn) # the number of balanced dataset
        self.N = len(xmin) + len(xmaj) + len(xsyn) # the number of Synthetic samples


        # w: the weight of all samples
        self.w = rnd.uniform(-1, 1, (columns, num))
        # b: the bias of all samples
        self.b = np.zeros([self.N, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(self.N):
                self.b[j, i] = rand_b
        self.h = self.sigmoid(np.dot(xmin.append(xmaj).append(xsyn), self.w) + self.b)

    @staticmethod
    def sigmoid(x):
        """
            sigmoid
            :param x: train data X
            :return: activation
        """
        return 1.0 / (1 + np.exp(-x))


    # train
    def classifisor_train(self, lmin, lmaj, lsyn):
        """
            After initializing the learning machine, you need to pass in the corresponding tag T
            :param lmin,lmaj,lsyn: labels of the minority samples, majority samples, the synthetic samples
            :return: Hidden layer output weight beta
        """
        T = lmin.append(lmaj).append(lsyn)
        Cmaj = ((self.N - self.Nmaj) * self.C )/ self.N
        Cmin = ((self.N - self.Nmin) * self.C) / self.N
        Csyn = ((self.N - self.Nmin - self.Nmaj) * self.C) / self.N
        if len(T.shape) > 1:
            pass
        else:
            self.en_one = OneHotEncoder()
            T = np.array(T)
            T = self.en_one.fit_transform(T.reshape(-1, 1)).toarray()
            pass
        try:
            print(lmaj.shape, lmin.shape, lsyn.shape)
            print(self.H0maj.T.shape, self.H0min.T.shape, self.H0syn.T.shape)
            all_m1 = (1/Cmaj + 1/Cmin +1/Csyn + (1 + Cmaj/Cmin + Cmaj/Csyn)*np.dot(self.H0maj.T, self.H0maj) + (1 + Cmin/Cmaj + Cmin/Csyn)*np.dot(self.H0min.T, self.H0min) + (1 + Csyn/Cmaj + Csyn/Cmin)*np.dot(self.H0syn.T, self.H0syn))
            all_m2 = (1/Cmaj + 1/Cmin +1/Csyn + (1 + Cmaj/Cmin + Cmaj/Csyn)*np.dot(self.H0maj.T, np.array(lmaj)) + (1 + Cmin/Cmaj + Cmin/Csyn)*np.dot(self.H0min.T, np.array(lmin)) + (1 + Csyn/Cmaj + Csyn/Cmin)*np.dot(self.H0syn.T, np.array(lsyn)))
        except:
            # sub_former = np.dot(np.transpose(self.h), self.h) + len(T) / self.C
            # all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
            all_m1 = np.dot(self.H0maj.T, self.H0maj) +  np.dot(self.H0min.T, self.H0min) + (np.dot(self.H0syn.T, self.H0syn))
            all_m2 = np.dot(self.H0maj.T, np.array(lmaj)) + np.dot(self.H0min.T, np.array(lmin)) + (np.dot(self.H0syn.T, np.array(lsyn)))
        try:
            self.beta = np.dot(all_m1.I, all_m2)
        except:
            self.beta = np.dot(all_m1.I, all_m1)
        return self.beta

    # test
    def classifisor_test(self, test_x):
        """
            Pass in the attribute X to be predicted and make the prediction to obtain the predicted value
            :param test_x:test data test_x
            :return: The predicted value T of the predicted label
        """
        b_row = test_x.shape[0]
        try:
            h = self.sigmoid(np.dot(test_x, self.w)+ self.b[:b_row, :])
        except:
            b = np.zeros([test_x.shape[0], self.w.shape[1]], dtype=float) #bias b
            for i in range(test_x.shape[0]):
                rand_b = np.random.RandomState().uniform(-0.4, 0.4)
                for j in range(self.w.shape[1]):
                    b[i, j] = rand_b
            h = self.sigmoid(np.dot(test_x, self.w) + b)
        result = np.dot(h, self.beta)
        result = self.sigmoid(np.argmax(result, axis=1))
        result = result.astype(int)
        result[result==0] = -1
        return np.array(result)
