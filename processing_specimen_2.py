# -*- coding: utf-8 -*-

"""
Обработка набора фронтов трещин
"""
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

class Specimen:
    def __init__(self, list_fronts, list_names=None, rad_obr=None, rad_def=None, force=None):
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

        self.rad_obr = rad_obr
        if self.rad_obr:
            self.s_obr = np.pi * self.rad_obr ** 2
        else:
            self.s_obr = None
        self.rad_def = rad_def
        self.force = force

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

    def plot_geom_front(self, cont='c3'):
        """Печать геометрии фронтов трещины с нанесенными значениями КИН"""
        
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='polar')

        rad_max = 0
        
        for name in self.table:
            tab = self.table[name]
            deg = np.array(np.deg2rad(tab.index))
            rad = np.array(tab['rad'])
            kin = (np.array(tab[cont])[:-1] + np.array(tab[cont])[1:])/2

            points = np.array([deg, rad]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = plt.Normalize(kin.min(), kin.max())
            lc = LineCollection(segments, cmap='jet', norm=norm)
            lc.set_array(kin)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            ax.grid(False)
            if rad_max < rad.max():
                rad_max = rad.max()

        ax.set_theta_zero_location('S')
        ax.set_theta_direction(1)
        ax.set_ylim(0, rad_max+0.5)

        div = 100
        deg360 = np.linspace(0, 2*np.pi, div)
        if self.rad_obr:
            plt.plot(deg360, np.full(div, self.rad_obr), 'k')
        if self.rad_def:
            plt.plot(deg360, np.full(div, self.rad_def), 'k')
        
    @staticmethod
    def __integral_front(rads, angs, flag_radian=True):
        """Принимает массивы радиусов и углов, возвращает общую площадь"""
        if not flag_radian:
            angs = np.radians(angs)
        dangs = np.roll(angs, -1) - angs
        si = rads * np.roll(rads, -1) * np.sin(dangs) / 2
        return si.sum() 

    def nominal_stress(self, rad_obr=None, force=None):
        if rad_obr:
            self.rad_obr = rad_obr
            self.s_obr = np.pi * self.rad_obr ** 2
        if force:
            self.force = force

        if self.rad_obr and self.force:
            names = list()
            s_sechs = list()
            s_noms = list()
            names.append('nocrack')
            s_sechs.append(self.s_obr)
            s_noms.append(self.force / self.s_obr)
            for name in self.table:
                deg = np.array(np.deg2rad(self.table[name].index))
                rad = np.array(self.table[name]['rad'])
                s_cr = self.__integral_front(rad, deg, flag_radian=True)
                s_sech = self.s_obr - s_cr
                s_nom = self.force / s_sech
                names.append(name)
                s_sechs.append(s_sech)
                s_noms.append(s_nom)
            self.nominal_table = pd.DataFrame({
                'name': names,
                's_sech': s_sechs,
                's_nom': s_noms})
            self.nominal_table = self.nominal_table.set_index('name')
            display(self.nominal_table)

