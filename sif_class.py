from ._import import *


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

