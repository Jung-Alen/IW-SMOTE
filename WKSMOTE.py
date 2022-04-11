'''
# @author：Aimin Zhang，Hualong Yu
# Time:2022/4/2
# Institution：Jiangsu University of Science and Technology
'''

import numpy as np
import pandas as pd
import random

def sign(x): # Convert the predicted value to -1, 1
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        print("Sign function input wrong!\n")
        exit(-1)

class WKSMOTE():
    def __init__(self, data, target, test, C=1.0, epsilon=0.001, tolerance=0.001, balance=1):
        self.data = data  # train data
        self.target = target  # the label of train data
        self.test = test # the test data
        self.balance = balance;  # Sampling rate
        self.C = C  # punish parameter
        self.epsilon = epsilon  # slack
        self.tolerance = tolerance  # tolerance

        self.X = np.array(data)
        self.Y = np.array(target)

        self.numOfSamples = len(self.data)

        self.b = 0.
        self.alpha = [0.] * self.numOfSamples
        # self.Ei = [self.calculate_Ei(i) for i in range(self.numOfSamples)]
        self.Ei = [0.] * self.numOfSamples
        seeds = [] #seed sample from minority samples
        neighbors = [] #the neighbor of seed sample
        train = pd.DataFrame(self.data) #training set
        train[len(train.columns)] = self.target
        x_x = train # x_x:Temporary variable
        m, n = len(x_x), len(x_x.columns)
        z = x_x[x_x[n - 1] == 1]  # acquire the minority set
        p = x_x[x_x[n - 1] == -1]  # acquire the majority set
        m1, n1 = len(z), len(z.columns) # m1: the number of minority samples; n1: the number of columns of the minority samples
        m2, n2 = len(p), len(p.columns) # m2: the number of minority samples; n2: the number of columns of the minority samples
        nums = self.balance * m2 - m1 #The number of minority samples that need to be synthesized
        for i in range(nums):
            seed_index = random.randint(0, m1)
            seed = z.iloc[seed_index,:] #random seed from minority samples
            dis = [0] * m1
            for j in range(m1):
                #The distance between these two instances, distance function
                dis[j] = np.inner(seed, seed) - 2*np.inner(seed, z.iloc[j,:]) + np.inner(z.iloc[j,:], z.iloc[j,:])
            dis = sorted(enumerate(dis), key=lambda xxx: xxx[1])
            dis = dis[1:self.k_neighbor + 1] #choice KNN
            seed_neighbor_index = random.choice(dis)[0]
            seed_neighbor = z.iloc[seed_neighbor_index,:] #the neighbor of seed
            seeds.append(seed)
            neighbors.append(seed_neighbor)
        self.seeds = np.array(seeds)
        self.neighbors = np.array(neighbors)


    def kernel(self):
        train = pd.DataFrame(self.data)  # training set
        train[len(train.columns)] = self.target
        train = np.array(train)
        delta1 = random.random()
        delta2 = random.random()
        K1 = np.inner(train, train)
        K2 = (1-delta1)*np.inner(train, self.seeds) + delta1*np.inner(train, self.neighbors)
        K3 = (1-delta1)*(1-delta2)*np.inner(self.seeds, self.seeds) + (1-delta1)*delta2*np.inner(self.seeds, self.neighbors) + delta1*(1-delta2)*(self.neighbors, self.seeds) + delta1*delta2*np.inner(self.neighbor, self.neighbor)
        temp_inner = np.row_stack((K1,K2))
        temp_inner1 = np.row_stack((K2.T,K3))
        return  np.column_stack((temp_inner,temp_inner1))


    def calculate_gxi(self, i):
        try:
            return self.b + sum([self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
                             for j in range(self.numOfSamples)])
        except:
            return self.b + sum([self.alpha[j] * self.Y[j] * np.inner(self.X[i], self.X[j])
                                 for j in range(self.numOfSamples)])

    def calculate_Ei(self, i):
        return self.calculate_gxi(i) - self.Y[i]

    def is_satisfy_KKT(self, i):
        if (self.alpha[i] == 0) and (self.Y[i] * self.calculate_gxi(i) >= 1. - self.epsilon):
            return True
        elif (0. < self.alpha[i] < self.C) and (
                np.fabs(self.Y[i] * self.calculate_gxi(i) - 1.) <= self.epsilon):
            return True
        elif (self.alpha[i] == self.C) and (self.Y[i] * self.calculate_gxi(i) <= 1. + self.epsilon):
            return True

        return False

    def select_two_parameters(self):
        # First, select all 0 < alpha < C sample points check these points satisfy KKT or not
        # If all these points(0 < alpha < C) satisfy KKT
        # Then should check all sample points whether satisfy KKT
        # Select one that breaks KKT, and another one has max |E1 - E2| value

        allPointsIndex = [i for i in range(self.numOfSamples)]
        conditionPointsIndex = list(filter(lambda c: 0 < self.alpha[c] < self.C, allPointsIndex))

        unConditionPointsIndex = list(set(allPointsIndex) - set(conditionPointsIndex))
        reArrangePointsIndex = conditionPointsIndex + unConditionPointsIndex

        for i in reArrangePointsIndex:
            if self.is_satisfy_KKT(i):
                continue

            maxIndexEi = (0, 0.)  # (key, value)
            E1 = self.Ei[i]
            for j in allPointsIndex:
                if i == j:
                    continue
                E2 = self.Ei[j]
                if np.fabs(E1 - E2) > maxIndexEi[1]:
                    maxIndexEi = (j, np.fabs(E1 - E2))
            return i, maxIndexEi[0]

        return 0, 0

    def select_i2(self, i1, E1):
        E2 = 0
        i2 = -1
        max_E1_E2 = -1

        non_zero_Ei = [ei for ei in range(self.numOfSamples) if self.calculate_Ei(ei) != 0]
        for e in non_zero_Ei:
            E2_tmp = self.calculate_Ei(e)

            if np.fabs(E1 - E2_tmp) > max_E1_E2:
                max_E1_E2 = np.fabs(E1 - E2_tmp)
                E2 = E2_tmp
                i2 = e

        if i2 == -1:
            i2 = i1
            while i2 == i1:
                i2 = int(random.uniform(0, self.numOfSamples))
            E2 = self.calculate_Ei(i2)
            # E2 = self.Ei[i2]

        return i2, E2

    def smo_trunk(self, i1):
        E1 = self.calculate_Ei(i1)
        # E1 = self.Ei[i1]

        if not self.is_satisfy_KKT(i1):

            i2, E2 = self.select_i2(i1, E1)
            print(i1, i2)

            alpha_i1_old = self.alpha[i1]
            alpha_i2_old = self.alpha[i2]

            if self.Y[i1] != self.Y[i2]:
                L = np.fmax(0., alpha_i2_old - alpha_i1_old)
                H = np.fmin(self.C, self.C + alpha_i2_old - alpha_i1_old)
            elif self.Y[i1] == self.Y[i2]:
                L = np.fmax(0., alpha_i2_old + alpha_i1_old - self.C)
                H = np.fmin(self.C, alpha_i2_old + alpha_i1_old)
            else:
                print("WTF of this condition?")
                exit(-1)

            if L == H:
                return 0

            try:
                eta = (self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2])) - \
                  (self.kernel(self.X[i1], self.X[i2]) * 2.)
            except:
                eta = (np.inner(self.X[i1], self.X[i1]) + np.inner(self.X[i2], self.X[i2])) - \
                      (np.inner(self.X[i1], self.X[i2]) * 2.)

            if eta <= 0:
                return 0

            alpha2_new_unclipped = alpha_i2_old + (self.Y[i2] * (E1 - E2) / eta)

            if alpha2_new_unclipped >= H:
                alpha2_new_clipped = H
            elif L < alpha2_new_unclipped < H:
                alpha2_new_clipped = alpha2_new_unclipped
            elif alpha2_new_unclipped <= L:
                alpha2_new_clipped = L
            else:
                print("WTF of the alpha2_new_uncliped value?")
                print(i1, i2, alpha2_new_unclipped, eta)
                exit(-1)

            if np.fabs(alpha2_new_clipped - alpha_i2_old) < self.tolerance:
                return 0

            s = self.Y[i1] * self.Y[i2]
            alpha1_new = alpha_i1_old + s * (alpha_i2_old - alpha2_new_clipped)

            try:
                b1 = - E1 \
                     - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - alpha_i1_old) \
                     - self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new_clipped - alpha_i2_old) \
                     + self.b
            except:
                b1 = - E1 \
                     - self.Y[i1] * np.inner(self.X[i1], self.X[i1]) * (alpha1_new - alpha_i1_old) \
                     - self.Y[i2] * np.inner(self.X[i2], self.X[i1]) * (alpha2_new_clipped - alpha_i2_old) \
                     + self.b
            try:
                b2 = - E2 \
                     - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - alpha_i1_old) \
                     - self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new_clipped - alpha_i2_old) \
                     + self.b
            except:
                b2 = - E2 \
                     - self.Y[i1] * np.inner(self.X[i1], self.X[i2]) * (alpha1_new - alpha_i1_old) \
                     - self.Y[i2] * np.inner(self.X[i2], self.X[i2]) * (alpha2_new_clipped - alpha_i2_old) \
                     + self.b

            if 0 < alpha1_new < self.C:
                b = b1
            elif 0 < alpha2_new_clipped < self.C:
                b = b2
            else:
                b = (b1 + b2) / 2.

            self.b = b

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new_clipped

            # Update all error cache Ei value
            self.Ei = [self.calculate_Ei(i) for i in range(self.numOfSamples)]
            # self.Ei[i1] = self.calculate_Ei(i1)
            # self.Ei[i2] = self.calculate_Ei(i2)

            return 1
        else:
            return 0

    def check_not_bound(self):
        return [nb for nb in range(self.numOfSamples) if 0 < self.alpha[nb] < self.C]

    def train(self, maxIteration=50):
        iterNum = 0
        iterEntireSet = True
        alphaPairsChanged = 0

        while (iterNum < maxIteration) and (alphaPairsChanged > 0 or iterEntireSet):
            iterNum += 1
            print("Iteration: %d of %d" % (iterNum, maxIteration))

            alphaPairsChanged = 0
            if iterEntireSet:
                for i in range(self.numOfSamples):
                    alphaPairsChanged += self.smo_trunk(i)
            else:
                not_bound_list = self.check_not_bound()
                for i in not_bound_list:
                    alphaPairsChanged += self.smo_trunk(i)

            if iterEntireSet:
                iterEntireSet = False
            else:
                iterEntireSet = True

    def predict(self):
        n = len(self.test)
        predict_label = np.full(n, -2)

        for i in range(0, n):
            to_predict = self.test[i]

            result = self.b

            for j in range(self.numOfSamples):
                try:
                    result += self.alpha[j] * self.Y[j] * np.inner(to_predict, self.X[j]) + self.alpha[j] * ((1-random.random())*np.inner(to_predict, self.seeds) + (random.random())*np.inner(to_predict, self.neighbors))
                except:
                    result += self.alpha[j] * self.Y[j] * np.inner(to_predict, self.X[j])
            predict_label[i] = sign(result)

        return predict_label


