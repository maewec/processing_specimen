from ._import import *
from .groupsif_class import GroupSIF

figsize = (15, 10)
figsize2 = (6, 4)
dpi = 300

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
        self.curve_front_data = specimen.curve_front_data
        self.ang0 = 0
        self.ang1 = 360
        self.rad0 = 0
        self.rad1 = self.table['rad'].max()
        self.parent = None
        self.m = None
        self.c = None
        self.rvalue = None
        self._xline = None
        self._yline = None
        self._yline_sko = None
        self.table_cycle = None

        self.curve_min = None
        self.curve_max = None
        self.define_minmax_curve()

    CPOOL = ['#0000c8', '#1579ff', '#00c7dd',
             '#28ffb9', '#39ff00', '#aaff00',
             '#ffe300', '#ff7100', '#ff0000']
    CMAP = mpl.colors.ListedColormap(CPOOL, 'indexed')

    def define_minmax_curve(self):
        if self.curve_front_data:
            self.sort('rad')
            index = self.table['rad'].idxmin()
            self.curve_min = self.curve_front_data.search_curve(self.table.loc[index, 'rad'],
                                               self.table.loc[index, 'ang'])
            index = self.table['rad'].idxmax()
            self.curve_max = self.curve_front_data.search_curve(self.table.loc[index, 'rad'],
                                               self.table.loc[index, 'ang'])

    def get_curve(self, index):
        return self.curve_front_data.search_curve(self.table.loc[index, 'rad'],
                                                  self.table.loc[index, 'ang'])

    def sort(self, column='sif'):
        self.table = self.table.sort_values(by=[column])

    def create_sif(self, contour, mpa2kgs=True):
        self.table['sif'] = None
        for index, row in self.table.iterrows():
            rad = row['rad']
            ang = row['ang']
            direct_sif = self.specimen.create_sif(ang, contour, for_dir_sif=False)
            kin = np.interp(rad, direct_sif['rad'], direct_sif['c'])
            self.table.loc[index, 'sif'] = kin
        if mpa2kgs:
            self.table['sif'] = self.table['sif'] / 9.8
        if self.r_asymmetry:
            self.table['sif'] = self.table['sif'] * (1 - self.r_asymmetry)

        self.sort()

    def select_group(self, ang0=0, ang1=360, rad0=0, rad1=100,
                     marker='o'):
        obj_copy = copy.deepcopy(self)
        # выбор по радиусу
        obj_copy.table = obj_copy.table[(obj_copy.table['rad']>=rad0) &\
                                        (obj_copy.table['rad']<=rad1)]
        # выбор по углу
        if ang0 > ang1:
            obj_copy.table = pd.concat([obj_copy.table[(obj_copy.table['ang']>=ang0)],
                                        obj_copy.table[(obj_copy.table['ang']<=ang1)]])
        else:
            obj_copy.table = obj_copy.table[(obj_copy.table['ang']>=ang0) &\
                                            (obj_copy.table['ang']<=ang1)]
        obj_copy.ang0 = ang0
        obj_copy.ang1 = ang1        
        obj_copy.rad0 = rad0        
        obj_copy.rad1 = rad1
        obj_copy.parent = self
        obj_copy.define_minmax_curve()
        return obj_copy

    def select_by_rate(self, drop_rate_min=0, drop_rate_max='max',
                       marker='o'):
        obj_copy = copy.deepcopy(self)
        if drop_rate_max == 'max':
            drop_rate_max = obj_copy.table['d'].max()
        obj_copy.table = obj_copy.table[(obj_copy.table['d']>=drop_rate_min) &\
                                        (obj_copy.table['d']<=drop_rate_max)]
        obj_copy.parent = self
        obj_copy.define_minmax_curve()
        return obj_copy

    def drop_points(self, index, marker='o'):
        """
        Parameter:
        index - list или int индексов таблицы
        """
        obj_copy = copy.deepcopy(self)
        obj_copy.table = obj_copy.table.drop(index=index)
        obj_copy.parent = self
        obj_copy.define_minmax_curve()
        return obj_copy


    def solve_cge(self):
        """Определение коэффициентов СРТУ"""
        self.sort()
        df = self.table
        x = df['sif'].to_numpy(dtype=float)
        y = df['d'].to_numpy(dtype=float)
        slope, intercept, rvalue, pvalue, stderr = linregress(np.log10(x), np.log10(y))
        self.m = slope
        self.c = 10**intercept
        self.rvalue = rvalue
        # C c учетом СКО
        self.c_sko = {i: self.c*10**(i*stderr) for i in [-3, -1, 1, 3]}
        print('m = {:.3f}\nC = {:.6e}\nR = {:.3f}'.format(self.m, self.c, self.rvalue))
        print(stderr)
        print(self.c_sko)
        
        self._xline = np.linspace(df['sif'].min(), df['sif'].max(), 100)
        self._yline = self.c * self._xline ** self.m
        self._yline_sko = {i: self.c_sko[i] * self._xline ** self.m for i in self.c_sko}

    def plot_cge(self, plot_coef=True, plot_cge_ct=True, ax=None, marker='o',
                 comment_num_points=False, group=None, sko=False, print_text_coef=True,
                 print_coef_in_legend=True):
        """
        Parameter:
        sko - включение отображения СКО, True или список степеней [-3, -1, 1, 4]
        """
        if ax:
            if isinstance(group, GroupSIF):
                name, id_ = group.find(self)
                label = '№{} {}'.format(id_, name)
                label2 = label + '; '
                color_coef = COLORS[id_]
                color = COLORS[id_]
            else:
                color_coef = '#000000'
                color = '#ff0000'
                label = 'Данные'
                label2 = ''
        else:
            figure = plt.figure(figsize=figsize2, dpi=dpi)
            ax = figure.add_subplot(1, 1, 1)
            color_coef = '#000000'
            color = '#ff0000'
            label = 'Данные'
            label2 = ''

        df = self.table
        x = df['sif'].to_numpy(dtype=float)
        y = df['d'].to_numpy(dtype=float)
        ax.plot(x, y, marker=marker, linestyle='', color=color)

        if comment_num_points:
            for index, row in df.iterrows():
                ax.text(row['sif'], row['d'], str(index))

        if print_coef_in_legend:
            label2 = '{}C={:.4e}; m={:.4f}'.format(label2, self.c, self.m)
        if plot_coef:
            ax.plot(self._xline, self._yline, color=color_coef,
                    label=label2)

        if plot_cge_ct:
            for cge in self.specimen.get_cge_ct():
                c, m, name = cge
                y = c * self._xline ** m
                ax.plot(self._xline, y, label='ВР '+name)

        if sko:
            if isinstance(sko, bool):
                sko = list(self.c_sko.keys())
            for i in sko:
                ax.plot(self._xline, self._yline_sko[i], color=color_coef,
                        linestyle='--')

        if print_text_coef:
            if sko:
                c_copy = self.c_sko.copy()
                c_copy[0] = self.c
                c_copy = dict(sorted(c_copy.items()))
            else:
                c_copy = dict()
                c_copy[0] = self.c
            text = ''
            for i in c_copy:
                c_text = '{:.4e}'.format(c_copy[i])
                c_text = '{}\cdot 10^{{{}}}'.format(*c_text.split('e'))
                t = '$dl/dN = {} \cdot \Delta K ^{{{:.4f}}}$\n'.format(c_text, self.m)
                text += t
            bbox = dict(boxstyle="round", fc='white', alpha=0.4)
            ax.text(0.95, 0.05, text[:-1], ha='right', va='bottom', transform=ax.transAxes, bbox=bbox)


        ax.legend()
        ax.grid(which='both', alpha=0.4)
        ax.set_xlabel('$размах\ КИН,\ кгс/мм^{3/2}$')
        ax.set_ylabel('$dl/dN,\ мм/цикл$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='both')

        if not isinstance(group, GroupSIF):
            xmin, xmax = ax.set_xlim()
            xmin = int(xmin // 10 * 10)
            xmax = int(xmax // 10 * 10 +10)
            xticks = np.arange(xmin, xmax+1, 10)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, rotation=45)
        return ax

    def plot_length_of_cycle(self, cm_parent=0, plot_ct=True, ax=None, interpol=1,
                             group=None):
        """Зависимость длины трещины от количества циклов по фракт данным и по
        вычисленным C и m
        Parameter:
            cm_parent - 0 - c и m данного экземпляра, 1 - экземпляра родителя и тд
        """
        if ax:
            if isinstance(group, GroupSIF):
                name, id_ = group.find(self)
                label = '{} {}'.format(id_, name)
                color_coef = '#000000'
                color = COLORS[id_]
            else:
                label = 'Эксперимент'
                color_coef = '#000000'
                color = '#ff0000'
        else:
            figure = plt.figure(figsize=(6, 4), dpi=dpi)
            ax = figure.add_subplot(1, 1, 1)
            label = 'Эксперимент'
            color_coef = '#000000'
            color = '#ff0000'

        self.sort('rad')

        obj = self
        for i in range(cm_parent):
            obj = obj.parent
        c = obj.c
        m = obj.m
        rad = self.table['rad'].to_numpy(dtype=float)
        sif = self.table['sif'].to_numpy(dtype=float)
        initial_length = self.table['rad'].min()
        crack_spec = OneCycle(rad, sif, c, m, interpol=interpol)
        cycle_spec = crack_spec.get_number_cycle(initial_length)

        # новые скорости
        cm_text = 'C = {:.4e}; m = {:.4f}'.format(c, m)
        ax.plot(np.arange(cycle_spec), crack_spec.arr_length_crack,
                label=cm_text, color=color_coef)
        
        # стандартные скорости
        if plot_ct:
            for c_ct, m_ct, name in self.specimen.get_cge_ct():
                cr = OneCycle(rad, sif, c_ct, m_ct, interpol=interpol)
                cycle = cr.get_number_cycle(initial_length)
                ax.plot(np.arange(cycle), cr.arr_length_crack,
                        label='ВР '+name)

        # расстояние между соседними точками
        inc = self.table['rad'].diff().to_numpy()[1:]
        # средние скорости для этих отрезков
        d = self.table['d'].to_numpy()
        d = np.average(np.vstack((d[:-1], d[1:])), axis=0)
        # число циклов для этих отрезков и кумулятивный массив
        n_inc = inc / d
        n = n_inc.cumsum()
        # расстояние и циклы от нуля
        r = self.table['rad'].to_numpy()
        n = np.concatenate(([.0], n))
        ax.scatter(n, r, label=label, color=color)

        ax.grid(which='both', alpha=0.4)
        ax.legend()
        ax.set_xlabel('Число циклов')
        ax.set_ylabel('Длина трещины')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=initial_length)
        return ax

    def drop_dev_max(self, part_top=0.025, part_bottom=0.025):
        """Удаление точек сверху и снизу и создание нового экземпляра
        Parameters:
            part_top - доля точек, удаляемая сверху
            part_bottom - доля точек, удаляемая снизу
        """
        obj_copy = copy.deepcopy(self)
        obj_copy.solve_cge()
        obj_copy.table['dev_log'] = None
        m = obj_copy.m
        clog = np.log10(obj_copy.c)
        for index, row in obj_copy.table.iterrows():
            k0 = np.log10(row['sif'])
            d0 = np.log10(row['d'])
            dev = d0 - m*k0 - clog
            obj_copy.table.loc[index, 'dev_log'] = dev
        le = len(obj_copy.table)
        drop_num_top = int(round(part_top*le, 0))
        drop_num_bottom = int(round(part_bottom*le, 0))
        obj_copy.sort('dev_log')

        obj_copy.table = obj_copy.table.iloc[drop_num_bottom: le - drop_num_top]

        obj_copy.solve_cge()
        obj_copy.parent = self
        obj_copy.define_minmax_curve()
        obj_copy.sort()
        return obj_copy

    def solve_cycle(self, display_table='True'):
        self.sort('rad')
        table = self.table
        # расстояние между соседними точками
        inc = table['rad'].diff().to_numpy()[1:]
        # средние скорости для этих отрезков
        d = table['d'].to_numpy()
        d = np.average(np.vstack((d[:-1], d[1:])), axis=0)
        # число циклов для этих отрезков и кумулятивный массив
        n_inc = inc / d
        n = n_inc.cumsum()
        # расстояние и циклы от нуля
        r = table['rad'].to_numpy()
        n = np.concatenate(([.0], n))
        # разворот циклов
        n_rev = n - n[-1]
        # запись в таблицу
        table['n'] = n
        table['n_rev'] = n_rev
        if display_table:
            display(table)

    def plot_cycle(self, reverse_rate=False, sdvig_r=False, ax=None, marker='o',
                       plot_total_cycle=True,
                       comment_num_points=False, group=None):
        if ax:
            if isinstance(group, GroupSIF):
                name, id_ = group.find(self)
                label = '{} {}'.format(id_, name)
                color = COLORS[id_]
            else:
                color = '#ff0000'
                label = 'Эксперимент'
        else:
            figure = plt.figure(figsize=figsize, dpi=dpi)
            ax = figure.add_subplot(1, 1, 1)
            color = '#ff0000'
            label = 'Эксперимент'

        self.sort('rad')
        table = self.table
        r = table['rad'].to_numpy()

        if sdvig_r:
            r = r - r[0]
        if reverse_rate:
            n_name = 'n_rev'
        else:
            n_name = 'n'
        n = table[n_name].to_numpy()
        ax.plot(n, r, marker=marker, color=color, label=label)

        if comment_num_points:
            for index, row in table.iterrows():
                ax.text(table.loc[index, n_name],
                        table.loc[index, 'rad'],
                        ' {:.0f}'.format(index))
        if reverse_rate:
            index = 0
        else:
            index = -1
        if plot_total_cycle:
            ax.text(n[index], r[index], ' {:.0f}'.format(n[index]))
        if isinstance(group, GroupSIF):
            print('{} - {:.0f}'.format(label, n[index]))

        ax.legend(fontsize=20)
        ax.grid(which='both', alpha=0.4)
        ax.set_xlabel('Циклы', fontsize=20)
        ax.set_ylabel('Радиус, мм', fontsize=20)
        ax.tick_params(axis='both', which='both', labelsize=20)
        return ax

    def rad_for_cycle(self, delta_n=1000, reverse_rate=True,
                      cycle_min='min', cycle_max='max', display_table=True):
        """Выдается таблица с радиусами и углами точек для заданных циклов"""
        self.sort('rad')
        table = self.table
        if reverse_rate:
            n_name = 'n_rev'
        else:
            n_name = 'n'

        if cycle_min == 'min':
            cycle_min = table[n_name].min()
        else:
            cycle_min = cycle_min

        if cycle_max  == 'max':
            cycle_max = table[n_name].max()
        else:
            cycle_max = cycle_max

        if reverse_rate:
            n_cyc = np.concatenate((np.arange(cycle_max, cycle_min, -delta_n),
                                    [cycle_min]))
        else:
            n_cyc = np.concatenate((np.arange(cycle_min, cycle_max, delta_n),
                                    [cycle_max]))
        r_cyc = np.interp(n_cyc, table[n_name].to_numpy(dtype=float),
                          table['rad'].to_numpy(dtype=float))
        # убираем зависимость от смены угла 360 и 0
        # потом возвращаем обратно
        ang = table['ang'].to_numpy(dtype=float)
        if (np.abs(np.diff(ang)) > 180).any():
            ang = np.where(ang > 180, ang - 360, ang)
        ang_cyc = np.interp(n_cyc, table[n_name].to_numpy(dtype=float),
                            ang)
        # возвращаем обратно
        ang_cyc = np.where(ang_cyc < 0, ang_cyc + 360, ang_cyc)

        tab_cycle = pd.DataFrame({'rad': r_cyc, 'ang': ang_cyc, 'n': n_cyc})
        self.table_cycle = tab_cycle
        if display_table:
            display(self.table_cycle)

    def plot_geom(self, ang0=0, ang1=360, rad0=0, rad1='max',
                  color_rate=True, plot_specimen=True,
                  dir_theta_null='S', comment_num_points=False,
                  plot_curve_front_data=True):
        figure = plt.figure(figsize=(10, 10), dpi=dpi)
        ax = figure.add_subplot(projection='polar')
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
            cbar = figure.colorbar(sc, ax=ax, ticks=ticks, norm=norm, shrink=0.5,
                                orientation='horizontal', pad=0.03)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(list(map('{:.3e}'.format, ticks)))
            cbar.ax.tick_params(rotation=45)
        else:
            ax.scatter(np.deg2rad(df['ang']), df['rad'],
                       color='k', marker='o', zorder=10)

        ax.scatter([0], [0], color='r', marker='+', linewidth=10)

        if comment_num_points:
            for index, row in df.iterrows():
                ax.text(np.deg2rad(row['ang']), row['rad'], str(index), zorder=20)

        ax.set_xlim(np.deg2rad(ang0), np.deg2rad(ang1))
        if rad1 == 'max':
            rad1 = df['rad'].max()
            if plot_specimen:
                if rad_obr > rad1:
                    rad1 = rad_obr
        ax.set_ylim(rad0, rad1*1.1)

        # печать кривых фронта для крайних точек
        if plot_curve_front_data:
            if self.curve_min:
                self.curve_min.plot_ax(ax)
            if self.curve_max:
                self.curve_max.plot_ax(ax)
        return ax

    def concatenate(self, objs_list, group_obj=None, marker='o'):
        obj_copy = copy.deepcopy(self)
        if isinstance(objs_list, list):
            lst = [obj_copy.table]
            for sif in objs_list:
                lst.append(sif.table)
        elif isinstance(objs_list, SIF2):
            lst = [obj_copy.table, objs_list.table]
        obj_copy.table = pd.concat(lst)

        obj_copy.parent = self
        obj_copy.define_minmax_curve()
        return obj_copy

    def copy(self):
        return copy.deepcopy(self)

