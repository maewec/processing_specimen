# -*- coding: utf-8 -*-

"""
Обработка набора фронтов трещин
"""
import copy
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import linregress
import itertools

from padmne.pdnforcrack.forcrack import OneCycle

class Specimen:
    def __init__(self, list_fronts, list_names=None, rad_obr=None, rad_def=None,
                 force=None, r_asymmetry=None):
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
        self.r_asymmetry = r_asymmetry
        self.dir_sif = list()
        self.nominal_table = None
        self.cge_ct = list()

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

        # линии снятия значений КИН
        i = 0
        for sif in self.dir_sif:
            ang = sif.angle
            plt.plot([ang, ang], [0, self.rad_obr], '--', color='r')
            plt.text(ang, self.rad_obr, 'Путь '+str(i))
        
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

    def create_sif(self, angle, contour):
        direct_sif = pd.DataFrame()
        for name in self.table:
            df = self.table[name]
            k = df.iloc[(np.abs(df.index-angle)).argsort()[:1]]
            direct_sif = direct_sif.append(k)
            direct_sif = direct_sif[['rad', contour]]
        direct_sif = direct_sif.rename(columns={contour: 'c'})
        number = len(self.dir_sif)
        self.dir_sif.append(SIF(direct_sif, angle, self, number))
        display(direct_sif)

    def sif(self, number):
        return self.dir_sif[number]

    def set_cge_ct(self, c, m, name=''):
        self.cge_ct.append([c, m, name])

    def get_cge_ct(self, num=None):
        if num:
            return self.cge_ct[num]
        else:
            return self.cge_ct


class SIF:
    def __init__(self, table, angle, specimen, path_n=None):
        self.sif_table = table
        self.angle = angle
        self.specimen = specimen
        self.r_asymmetry = specimen.r_asymmetry
        self.path_n = path_n
        self.fract_table = None
        self.m = None
        self.c = None
        self.rvalue = None
        self._xline = None
        self._yline = None
        self._length = None

    # цветовые комбинации для графиков
    COLORS = (x for x in itertools.product([0, 1], repeat=3))

    def add_fract(self, text, display_on=False):
        """Добавить данные фрактографии
        Три колонки:
        rad - радиус,
        n - число циклов,
        d - прирост за цикл"""

        data = pd.read_table(StringIO(text), sep='\s+', index_col='rad')
        data = data.sort_index()
        self.fract_table = data
        if display_on:
            display(self.fract_table)

    def concat(self, mpa2kgs=True, display_on=False):
        """Аппроксимация и совмещение КИН и данных фрактографии"""
        self.res_table = self.fract_table.copy()
        self.res_table['sif'] = np.interp(self.fract_table.index,
                self.sif_table['rad'],
                self.sif_table['c'])
        if mpa2kgs:
            self.res_table['sif'] = self.res_table['sif'] / 9.8
        if self.r_asymmetry:
            self.res_table['sif'] = self.res_table['sif'] * (1 - self.r_asymmetry)
        if display_on:
            display(self.res_table)

    def solve_cge(self, drop_left=0, drop_right=0):
        """Определение коэффициентов СРТУ"""
        self.drop_left = drop_left
        self.drop_right = drop_right
        self._length = len(self.res_table)
        x = self.res_table['sif'].iloc[drop_left: self._length - drop_right]
        y = self.res_table['d'].iloc[drop_left: self._length - drop_right]
        slope, intercept, rvalue, pvalue, stderr = linregress(np.log10(x), np.log10(y))
        self.m = slope
        self.c = 10**intercept
        self.rvalue = rvalue
        print('m = {:.3f}\nC = {:.6e}\nR = {:.3f}'.format(self.m, self.c, self.rvalue))
        
        self._xline = np.linspace(self.res_table['sif'].min(), self.res_table['sif'].max(), 100)
        self._yline = self.c * self._xline ** self.m

    def plot_cge(self, plot_coef=True, plot_drop_points=True, figure=None, position=None):
        if figure:
            color_coef = SIF.COLORS[self.path_n]
            color_drop = SIF.COLORS[self.path_n]
            color = SIF.COLORS[self.path_n]
        else:
            color_coef = '#000000'
            color_drop = '#888888'
            color = '#ff0000'
            figure = plt.figure(figsize=(15, 10), dpi=200)
        if not position:
            ax = figure.add_subplot(1, 1, 1)
        else:
            ax = figure.add_subplot(*position)

        x = self.res_table['sif'].iloc[self.drop_left: self._length - self.drop_right]
        y = self.res_table['d'].iloc[self.drop_left: self._length - self.drop_right]
        ax.plot(x, y, 'o', color=color, label='Эксперимент #{}'.format(self.path_n))

        if plot_coef:
            ax.plot(self._xline, self._yline, color=color_coef, label='Аппроксимация #{}'.format(self.path_n))

        if plot_drop_points:
            if self.drop_right or self.drop_left:
                drop = [x for x in range(self.drop_left)] +\
                       [x for x in range(self._length - self.drop_right, self._length)]
                x_drop = self.res_table['sif'].iloc[drop]
                y_drop = self.res_table['d'].iloc[drop]
                ax.plot(x_drop, y_drop, 'x', color=color_drop, label='Эксперимент (сброс) #{}'.format(self.path_n))

        ax.legend(fontsize=20)
        ax.grid(which='both', alpha=0.4)
        ax.set_xlabel('$КИН, кгс/мм^{3/2}$', fontsize=20)
        ax.set_ylabel('dl/dN, мм/цикл', fontsize=20)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='both', labelsize=20)
        return ax

    def plot_comparison(self):
        crack_spec = OneCycle(self.res_table.index, self.res_table['sif'], self.c, self.m)
        cycle_spec = crack_spec.get_number_cycle(self.res_table.index[0])
        crack_list = list()
        for cge in self.specimen.get_cge_ct():
            c, m, name = cge
            cr = OneCycle(self.res_table.index, self.res_table['sif'], c, m)
            cycle = cr.get_number_cycle(self.res_table.index[0])
            crack_list.append([cr, cycle, name])

        plt.figure(figsize=(15,10), dpi=200)
        
        sdvig = self.res_table['n'].min()
        plt.scatter(self.res_table['n'] - sdvig, self.res_table.index,
                    label='Фрактография', color='cyan')

        plt.plot(np.arange(cycle_spec), crack_spec.arr_length_crack,
                 label='Обработанные данные')
        for crack in crack_list:
            cr, cycle, name = crack
            plt.plot(np.arange(cycle), cr.arr_length_crack,
                 label='CT '+name)

        plt.legend()
        plt.grid()
        plt.xlabel('Число циклов')
        plt.ylabel('Длина от очага, мм')
        plt.show()

