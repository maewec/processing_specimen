# -*- coding: utf-8 -*-

"""
Обработка набора фронтов трещин
"""
import copy
import os
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import linregress
import itertools

from padmne.pdnforcrack.forcrack import OneCycle

class Specimen:
    def __init__(self, list_fronts, list_names=None, rad_obr=None, rad_def=None,
                 force=None, r_asymmetry=None, temp=None, name='',
                 sdvig_x=0, sdvig_y=0):
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
        self.temp = temp
        self.name = name
        self.sdvig_x = sdvig_x
        self.sdvig_y = sdvig_y
        self.dir_sif = list()
        self.nominal_table = None
        self.cge_ct = list()
        self.sif_cloud = None

    NAMES_COUNTUR = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
    NAME_INDEX = ['ang']
    NAMES_COORD = ['ang', 'rad', 'z']
    NAME_DROP = ['node']
    NAMES_COLUMNS = NAME_DROP + NAMES_COORD + NAMES_COUNTUR

    def __read_file(self, path):
        names = Specimen.NAMES_COLUMNS
        index_col = Specimen.NAME_INDEX
        drop = Specimen.NAME_DROP
        df = pd.read_table(path, delim_whitespace=True, names=names,
                           index_col=index_col).drop(columns=drop)
        # отрицательные углы конвертируются в положительные
        df.index = np.where(df.index < 0, 360 + df.index, df.index)
        df = df.sort_index()
        return df

    def plot_fronts(self, cont='all', **kwargs):
        for name in self.table:
            self.plot_front(name, cont, **kwargs)

    def plot_front(self, name, cont='all', **kwargs):
        drop = ['rad', 'z']
        cur_table = self.table[name].drop(columns=drop)
        if name in self.list_names:
            if cont != 'all':
                cur_table = cur_table[cont]
            cur_table.plot(title=name, **kwargs)
        else:
            Exception('Нет фронта '+name)

    def plot_all_fronts(self, cont, return_obj=False):
        """
        Parameters:
            cont - str, dict - название контура либо словарь с названиями трещин и контурами
            """
        if not return_obj:
            plt.figure(figsize=(15, 10))

        cont_dict = self.__cont_dict(cont)

        for name in self.table:
            cont = cont_dict[name]
            ax = self.table[name][cont].plot(label=name)

        if not return_obj:
            plt.legend()
        else:
            return ax

    def __cont_dict(self, cont):
        if not isinstance(cont, dict):
            cont_dict = {}
            for name in self.table:
                cont_dict[name] = cont
        else:
            cont_dict = cont
        return cont_dict

    @staticmethod
    def __moving_average(array, num=5):
        """Сглаживание фронтов с помощью скользящего среднего"""
        res = np.zeros_like(array)
        delta_1 = int(num/2)
        delta_2 = num - delta_1
        for i in range(-delta_1, delta_2):
            res += np.roll(array, i)
        res = res / num
        return res

    def copy_smooth(self, num):
        """Создает копию объекта со сглаженными фронтами"""
        obj_copy = copy.deepcopy(self)
        for name in obj_copy.table:
            for contour in Specimen.NAMES_COUNTUR:
                arr = np.array(obj_copy.table[name][contour])
                obj_copy.table[name][contour] = self.__moving_average(arr, num=num)
        return obj_copy

    def plot_geom_front(self, cont='c3', plot_rad_obr=True, plot_rad_def=True, dir_theta_null='S'):
        """Печать геометрии фронтов трещины с нанесенными значениями КИН"""

        cont_dict = self.__cont_dict(cont)
        
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='polar')

        rad_max = 0
        
        for name in self.table:
            tab = self.table[name]
            deg = np.array(np.deg2rad(tab.index))
            rad = np.array(tab['rad'])

            cont = cont_dict[name]

            kin = (np.array(tab[cont])[:-1] + np.array(tab[cont])[1:])/2

            points = np.array([deg, rad]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = plt.Normalize(kin.min(), kin.max())
            lc = LineCollection(segments, cmap='jet', norm=norm)
            lc.set_array(kin)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            if rad_max < rad.max():
                rad_max = rad.max()

        ax.set_theta_zero_location(dir_theta_null)
        ax.set_theta_direction(1)
        ax.set_ylim(0, rad_max+0.5)
        ax.grid(False)

        if self.rad_obr and plot_rad_obr:
            rad, deg360 = self.__sdvig(self.rad_obr)
            ax.plot(deg360, rad, 'k')
            ax.set_ylim(0, self.rad_obr+0.5)
        if self.rad_def and plot_rad_def:
            rad, deg360 = self.__sdvig(self.rad_def)
            ax.plot(deg360, rad, 'k')

        ax.scatter([0], [0], color='r', marker='+', linewidth=10)

        # линии снятия значений КИН
        for sif in self.dir_sif:
            ang = np.deg2rad(sif.angle)
            ax.plot([ang, ang], [0, self.rad_obr], '--', color='r')
            ax.text(ang, self.rad_obr, 'Путь '+str(sif.path_n))

        # облако точек с sif_cloud
        if self.sif_cloud:
            df = self.sif_cloud.table
            for index, row in df.iterrows():
                ax.scatter(np.deg2rad(df['ang']), df['rad'],
                           color='k', marker='x', zorder=10)

    def __sdvig(self, rad):
        div = 100
        deg360 = np.linspace(0, 2*np.pi, div)
        rad = np.full(div, rad)
        x = rad * np.sin(deg360) + self.sdvig_x
        y = rad * np.cos(deg360) + self.sdvig_y
        rad = np.sqrt(np.square(x) + np.square(y))
        with np.errstate(divide='ignore'):
            deg360 = np.where(y != 0, np.arctan(x / y), 1)
        deg360 = np.where(y > 0, deg360, deg360 + np.pi)
        return rad, deg360

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

    def create_sif(self, angle, contour, for_dir_sif=True, print_decart_coord=False):

        cont_dict = self.__cont_dict(contour)

        direct_sif = pd.DataFrame()
        for name in self.table:
            contour = cont_dict[name]
            df = self.table[name]
            k = df.iloc[(np.abs(df.index-angle)).argsort()[:1]]
            k = k[['rad', contour]].rename(columns={contour: 'c'})
            direct_sif = direct_sif.append(k)

        if print_decart_coord:
            x = np.sin(np.deg2rad(angle)) * self.rad_obr
            y = np.cos(np.deg2rad(angle)) * self.rad_obr
            print('x = {:.5f}\ny = {:.5f}'.format(x, y))


        if for_dir_sif:
            number = len(self.dir_sif)
            self.dir_sif.append(SIF(direct_sif, angle, self, number))
            display(direct_sif)
        else:
            return direct_sif
        
    def clean_sif(self):
        self.dir_sif = []

    def get_sif(self, number=None):
        if type(number) == int:
            return self.dir_sif[number]
        else:
            return self.dir_sif

    def set_cge_ct(self, c, m, name=''):
        self.cge_ct.append([c, m, name])

    def get_cge_ct(self, num=None):
        if type(num) == int:
            return self.cge_ct[num]
        else:
            return self.cge_ct

    def plot_cge(self, plot_coef=True, plot_drop_points=True, plot_cge_ct=False):
        figure = plt.figure(figsize=(15, 10), dpi=200)
        ax = figure.add_subplot(1, 1, 1)
        length = len(self.get_sif())
        for sif in self.get_sif():
            sif.plot_cge(plot_coef=plot_coef, plot_drop_points=plot_drop_points, plot_cge_ct=False, ax=ax)
        if plot_cge_ct:
            sif = self.get_sif(0)
            min_ = sif._xline.min()
            max_ = sif._xline.max()
            for sif in self.get_sif():
                if sif._xline.min() < min_:
                    min_ = sif._xline.min()
                if sif._xline.max() > max_:
                    max_ = sif._xline.max()
            for cge in self.get_cge_ct():
                c, m, name = cge
                xline = np.linspace(min_, max_, 100)
                y = c * xline ** m
                ax.plot(xline, y, '--', label='CT '+name)
            ax.legend()

    def plot_cgr(self, sdvig=False):
        figure = plt.figure(figsize=(15, 10), dpi=200)
        ax = figure.add_subplot(1, 1, 1)
        length = len(self.get_sif())
        for sif in self.get_sif():
            sif.plot_cgr(sdvig=sdvig, ax=ax)

    def set_sif_cloud(self, table, reverse_ang=False):
        self.sif_cloud = SIF2(table, self, reverse_ang=reverse_ang)

    def get_sif_cloud(self):
        return self.sif_cloud



class SIF:
    """
    КИН вдоль одной линии по заданному углу
    """
    def __init__(self, table, angle, specimen, path_n=None):
        self.sif_table = table
        self.angle = angle
        self.specimen = specimen
        self.r_asymmetry = specimen.r_asymmetry
        self.path_n = path_n
        self.name = 'Путь #{}'.format(self.path_n)
        self.fract_table = None
        self.res_table = None
        self.m = None
        self.c = None
        self.k_max = None
        self.rvalue = None
        self._xline = None
        self._yline = None
        self._length = None

    # цветовые комбинации для графиков
    COLORS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0),
              (1, 0, 1), (1, 1, 0), (0, 0, 0.5), (0, 0.5, 0), (0, 0.5, 0.5),
              (0, 0.5, 1), (0, 1, 0.5), (0.5, 0, 0), (0.5, 0, 0.5), (0.5, 0, 1),
              (0.5, 0.5, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 1), (0.5, 1, 0), (0.5, 1, 0.5),
              (0.5, 1, 1), (1, 0, 0.5), (1, 0.5, 0), (1, 0.5, 0.5), (1, 0.5, 1),
              (1, 1, 0.5)]

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
        self.k_max = self.res_table['sif'].max()

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

    def plot_cge(self, plot_coef=True, plot_drop_points=True, plot_cge_ct=False, ax=None):
        if ax:
            color_coef = SIF.COLORS[self.path_n]
            color_drop = SIF.COLORS[self.path_n]
            color = SIF.COLORS[self.path_n]
        else:
            figure = plt.figure(figsize=(15, 10), dpi=200)
            ax = figure.add_subplot(1, 1, 1)
            color_coef = '#000000'
            color_drop = '#888888'
            color = '#ff0000'

        x = self.res_table['sif'].iloc[self.drop_left: self._length - self.drop_right]
        y = self.res_table['d'].iloc[self.drop_left: self._length - self.drop_right]
        ax.plot(x, y, 'o', color=color, label=self.name)

        if plot_coef:
            ax.plot(self._xline, self._yline, color=color_coef, label='Аппроксимация '+self.name)
        
        if plot_drop_points:
            if self.drop_right or self.drop_left:
                drop = [x for x in range(self.drop_left)] +\
                       [x for x in range(self._length - self.drop_right, self._length)]
                x_drop = self.res_table['sif'].iloc[drop]
                y_drop = self.res_table['d'].iloc[drop]
                ax.plot(x_drop, y_drop, 'x', color=color_drop, label=self.name+' (сброс)')

        if plot_cge_ct:
            for cge in self.specimen.get_cge_ct():
                c, m, name = cge
                y = c * self._xline ** m
                ax.plot(self._xline, y, label='CT '+name)

        ax.legend(fontsize=20)
        ax.grid(which='both', alpha=0.4)
        ax.set_xlabel('$КИН, кгс/мм^{3/2}$', fontsize=20)
        ax.set_ylabel('$dl/dN, мм/цикл$', fontsize=20)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='both', labelsize=20)
        return ax

    def plot_cgr(self, sdvig=False, ax=None):
        if ax:
            color = SIF.COLORS[self.path_n]
        else:
            figure = plt.figure(figsize=(15, 10), dpi=200)
            ax = figure.add_subplot(1, 1, 1)
            color = '#ff0000'
        if sdvig:
            x = self.res_table.index - self.res_table.index[0]
        else:
            x = self.res_table.index
        y = self.res_table['d']
        ax.plot(x, y, 'o-', color=color, label=self.name)

        ax.legend(fontsize=20)
        ax.grid(which='both', alpha=0.4)
        ax.set_xlabel('$Длина, мм$', fontsize=20)
        ax.set_ylabel('$dl/dN, мм/цикл$', fontsize=20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        return ax

    def plot_comparison(self, interpol=1):
        crack_spec = OneCycle(self.res_table.index, np.array(self.res_table['sif']), self.c, self.m,
                              interpol=interpol)
        cycle_spec = crack_spec.get_number_cycle(self.res_table.index[0])
        crack_list = list()
        for cge in self.specimen.get_cge_ct():
            c, m, name = cge
            cr = OneCycle(self.res_table.index, np.array(self.res_table['sif']), c, m, interpol=interpol)
            cycle = cr.get_number_cycle(self.res_table.index[0])
            crack_list.append([cr, cycle, name])

        # первый график
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

        # второй график
        plt.figure(figsize=(15,10), dpi=200)
        plt.scatter(self.res_table.index, self.res_table['d'],
                    label='Фрактография', color='cyan')
        minimum, maximum = self.res_table['d'].min(), self.res_table['d'].max()
        plt.plot(crack_spec.arr_length_crack[:-1], np.diff(crack_spec.arr_length_crack),
                 label='Обработанные данные')
        for crack in crack_list:
            cr, cycle, name = crack
            d = np.diff(cr.arr_length_crack)
            plt.plot(cr.arr_length_crack[:-1], d, label='CT '+name)
            if minimum > d.min():
                minimum = d.min()
            if maximum < d.max():
                maximum = d.max()
        plt.legend()
        plt.grid()
        plt.xlabel('Длина от очага, мм')
        plt.ylabel('Скорость роста, мм/цикл')
        plt.ylim(minimum, maximum)
        plt.show()

    def save_table(self, path='.', name=None):
        if not name:
            name_spec = self.specimen.name
            num = self.path_n
            name = '{}_path{}.csv'.format(name_spec, num)
        full_path = os.path.join(path, name)
        self.res_table.to_csv(full_path, index=True, sep=';')



class SIF2:
    """
    КИН по облаку точек с уникальными радиусами и углами
    """
    def __init__(self, table, specimen, reverse_ang=False):
        """
        Parameters:
        table - таблица с колонками name, rad, ang, d
        """
        self.table = pd.read_table(StringIO(table), sep='\s+')
        self.table['ang'] = np.where(self.table['ang'] < 0,
                                     self.table['ang'] + 360,
                                     self.table['ang'])
        if reverse_ang:
            self.table['ang'] = 360 - self.table['ang']
        self.specimen = specimen
        self.r_asymmetry = specimen.r_asymmetry
        self.ang0 = 0
        self.ang1 = 360
        self.rad0 = 0
        self.rad1 = self.table['rad'].max()
        self.parent = None

    CPOOL = ['#0000c8', '#1579ff', '#00c7dd',
             '#28ffb9', '#39ff00', '#aaff00',
             '#ffe300', '#ff7100', '#ff0000']
    CMAP = mpl.colors.ListedColormap(CPOOL, 'indexed')

    def create_sif(self, contour, mpa2kgs=True):
        self.table['sif'] = None
        for index, row in self.table.iterrows():
            rad = row['rad']
            ang = row['ang']
            direct_sif = self.specimen.create_sif(ang, contour, for_dir_sif=False)
            kin = np.interp(rad, direct_sif['rad'], direct_sif['c'])
            self.table['sif'].iloc[index] = kin
        if mpa2kgs:
            self.table['sif'] = self.table['sif'] / 9.8
        if self.r_asymmetry:
            self.table['sif'] = self.table['sif'] * (1 - self.r_asymmetry)

    def select_group(self, ang0=0, ang1=360, rad0=0, rad1=100):
        obj_copy = copy.deepcopy(self)
        obj_copy.table = obj_copy.table[(obj_copy.table['rad']>rad0) &\
                                        (obj_copy.table['rad']<rad1) &\
                                        (obj_copy.table['ang']>ang0) &\
                                        (obj_copy.table['ang']<ang1)]
        obj_copy.ang0 = ang0
        obj_copy.ang1 = ang1        
        obj_copy.rad0 = rad0        
        obj_copy.rad1 = rad1
        obj_copy.parent = self
        return obj_copy

    def select_by_rate(self, drop_rate_min=0, drop_rate_max='max'):
        obj_copy = copy.deepcopy(self)
        if drop_rate_max == 'max':
            drop_rate_max = obj_copy.table['d'].max()
        obj_copy.table = obj_copy.table[(obj_copy.table['d']>=drop_rate_min) &\
                                        (obj_copy.table['d']<=drop_rate_max)]
        obj_copy.parent = self
        return obj_copy

    def plot_geom(self, ang0=0, ang1=360, rad0=0, rad1='max',
                  color_rate=True, plot_specimen=True, plot_parent=False,
                  dir_theta_null='S'):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='polar')
        ax.set_theta_zero_location(dir_theta_null)
        ax.set_theta_direction(1)
        ax.grid(False)

        spec = self.specimen
        if plot_specimen:
            if spec.rad_obr:
                rad, deg360 = spec._Specimen__sdvig(spec.rad_obr)
                ax.plot(deg360, rad, 'k')
                rad_obr = spec.rad_obr
            if spec.rad_def:
                rad, deg360 = spec._Specimen__sdvig(spec.rad_def)
                ax.plot(deg360, rad, 'k')

        
        df = self.table
        if color_rate:
            ticks = np.linspace(df['d'].min(), df['d'].max(), SIF2.CMAP.N)
            norm = mpl.colors.BoundaryNorm(ticks, SIF2.CMAP.N)
            sc = ax.scatter(np.deg2rad(df['ang']), df['rad'], c=df['d'],
                            cmap=SIF2.CMAP, norm=norm)
            cbar = fig.colorbar(sc, ax=ax, ticks=ticks, norm=norm, shrink=0.5,
                                orientation='horizontal', pad=0.03)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(list(map('{:.3e}'.format, ticks)))
            cbar.ax.tick_params(rotation=45)
        else:
            ax.scatter(np.deg2rad(df['ang']), df['rad'],
                       color='k', marker='o', zorder=10)

        ax.set_xlim(np.deg2rad(ang0), np.deg2rad(ang1))
        if rad1 == 'max':
            rad1 = df['rad'].max()
            if plot_specimen:
                if rad_obr > rad1:
                    rad1 = rad_obr
        ax.set_ylim(rad0, rad1*1.1)



class UniteSpecimen(Specimen):
    def __init__(self, r_asymmetry=None):
        self.r_asymmetry = r_asymmetry
        self.cge_ct = list()
        self.dir_sif = list()

    def create_sif(self, filepath):
        num = len(self.dir_sif)
        sif = SIF_from_file(filepath, specimen=self, path_n=num)
        self.dir_sif.append(sif)
        return sif



class SIF_from_file(SIF):
    def __init__(self, filepath, specimen, path_n=None, name=None):
        if name:
            self.name = name
        else:
            self.name = os.path.splitext(os.path.basename(filepath))[0]

        self.specimen = specimen
        self.r_asymmetry = specimen.r_asymmetry
        self.path_n = path_n

        self.res_table = pd.read_csv(filepath, sep=';', index_col='rad')

        self.m = None
        self.c = None
        self.k_max = None
        self.rvalue = None
        self._xline = None
        self._yline = None
        self._length = None



