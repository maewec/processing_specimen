# -*- coding: utf-8 -*-

"""
Обработка набора фронтов трещин
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

class Specimen:
    def __init__(self, list_fronts, list_names=None):
        """Определение фронтов КИН
        Parameters:
            list_fronts - список путей к контурам
            list_names - соответствующие им имена"""

        self.list_fronts = list_fronts
        if list_names is None:
            self.list_names = [x for x in range(len(self.list_fronts))]
        else:
            self.list_names = list_names

        self.table = dict()
        for i in range(len(list_fronts)):
            self.table[self.list_names[i]] = self.__read_file(self.list_fronts[i])

    NAMES_COUNTUR = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
    NAME_INDEX = ['ang']
    NAMES_COORD = ['ang', 'rad', 'z']
    NAME_DROP = ['node']
    NAMES_COLUMNS = NAME_DROP + NAMES_COORD + NAMES_COUNTUR

    def __read_file(self, path):
        names = Specimen.NAMES_COLUMNS
        index_col = Specimen.NAME_INDEX
        drop = Specimen.NAME_DROP
        return pd.read_table(path, delim_whitespace=True, names=names,
                index_col=index_col).drop(columns=drop)

    def plot_fronts(self, cont='all'):
        for name in self.table:
            self.plot_front(name, cont)

    def plot_front(self, name, cont='all'):
        drop = ['rad', 'z']
        cur_table = self.table[name].drop(columns=drop)
        if name in self.list_names:
            if cont != 'all':
                cur_table = cur_table[cont]
            cur_table.plot(title=name)
        else:
            Exception('Нет фронта '+name)

    def plot_all_fronts(self, cont='c2', return_obj=False):
        if not return_obj:
            plt.figure(figsize=(15, 10))
        for name in self.table:
            ax = self.table[name][cont].plot(label=name)
        if not return_obj:
            plt.legend()
        else:
            return ax

    @staticmethod
    def __moving_average(array, num=5):
        """Сглаживание фронтов с помощью скользящего среднего"""
        res = np.zeros_like(array)
        delta_1 = int(num/2)
        delta_2 = num - delta_1
        length = len(array)
        for i in range(length):
            minn = i-delta_1
            maxx = i+delta_2
            inc = 1
            if maxx <= length - 1:
                inc = 1
                ind = np.arange(minn, maxx)
            else:
                maxx -= length
                inc = -1
                ind = np.concatenate((np.arange(minn, length), np.arange(0, maxx)))
            summ = array[ind].sum()
            res[i] = summ / num
        return res

    def copy_smooth(self, num):
        """Создает копию объекта со сглаженными фронтами"""
        obj_copy = copy.deepcopy(self)
        for name in obj_copy.table:
            for contour in Specimen.NAMES_COUNTUR:
                arr = np.array(obj_copy.table[name][contour])
                obj_copy.table[name][contour] = self.__moving_average(arr, num=num)
        return obj_copy

