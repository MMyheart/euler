import math
import random
 
 
class FastWeightedCollection(object):
    def __init__(self, ids, weights):
        if len(ids) != len(weights):
            raise ValueError('Length is not equal, len(ids)=%d, len(weights)=%d' % (len(ids), len(weights)))
        self.__sum_weight = 0.0
        self.__ids = []
        self.__weights = []
        self.__norm_weights = []
        for i in range(len(weights)):
            self.__sum_weight += weights[i]
            self.__ids.append(ids[i])
            self.__weights.append(weights[i])
        if self.__sum_weight != 0:
            for i in range(len(weights)):
                self.__norm_weights.append(weights[i] / self.__sum_weight)
            self.__alias = AliasMethod(self.__norm_weights)
 
    def sample(self):
        if self.__sum_weight == 0:
            raise ValueError('sum_weight is zero, unable to sample anything')
        column = self.__alias.next()
        id_weight_pair = (self.__ids[column], self.__weights[column])
        return id_weight_pair
 
    def get_sum_weight(self):
        return self.__sum_weight
 
 
class AliasMethod(object):
    def __init__(self, weights):
        if len(weights) == 0:
            raise ValueError('weights\'s length is zero')
        self.__prob = [0.0] * len(weights)
        self.__alias = [0] * len(weights)
        large = []
        small = []
        avg = 1.0 / len(weights)
        for i in range(len(weights)):
            if weights[i] > avg:
                large.append(i)
            else:
                small.append(i)
        weights_ = weights[:]
        while len(large) > 0 and len(small) > 0:
            less = small.pop()
            more = large.pop()
            self.__prob[less] = weights_[less] * len(weights_)
            self.__alias[less] = more
            weights_[more] = weights_[more] + weights_[less] - avg
            if weights_[more] > avg:
                large.append(more)
            else:
                small.append(more)
        while len(small) > 0:
            less = small.pop()
            self.__prob[less] = 1.0
        while len(large) > 0:
            more = large.pop()
            self.__prob[more] = 1.0
 
    def next(self):
        column = self.next_long(len(self.__prob))
        coin_toss = random.random() < self.__prob[column]
        return column if coin_toss else self.__alias[column]
 
    def next_long(self, n):
        return int(math.floor(n * random.random()))
