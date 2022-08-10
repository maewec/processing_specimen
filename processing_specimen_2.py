# -*- coding: utf-8 -*-

"""
Обработка набора фронтов трещин
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


    def __read_file(self, path):
        names = ('node', 'ang', 'rad', 'z', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6')
        index_col = 'ang'
        drop = 'node'
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

    def plot_all_fronts(self, cont='c2'):
        plt.figure(figsize=(15, 10))
        for name in self.table:
            self.table[name][cont].plot(label=name)
        plt.legend()

