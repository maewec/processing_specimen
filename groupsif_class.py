from ._import import *

linestyles = ('dotted', 'dashed', 'dashdot',
             (0, (5, 10)), (5, (10, 3)))

class GroupSIF:
    def __init__(self):
        self.group_obj = []
        self.group_name = []
        self.group_id = []
        self.marker = []
        self.__i = 0

    def add(self, obj, name='', marker='o'):
        self.group_obj.append(obj)
        self.group_name.append(name)
        self.marker.append(marker)
        self.group_id.append(len(self.group_obj))
        self.__i = 0

    def delete(self, obj=None, name=None, id_=None):
        if obj:
            i = self.group_obj.index(obj)
        elif name:
            i = self.group_name.index(name)
        elif id_:
            i = self.group_id.index(id_)
        del self.group_obj[i]
        del self.group_name[i]
        del self.group_id[i]
    
    def find(self, obj=None, name=None, id_=None):
        if obj:
            i = self.group_obj.index(obj)
            return self.group_name[i], self.group_id[i]
        elif name:
            i = self.group_name.index(name)
            return self.group_obj[i], self.group_id[i]
        elif id_:
            i = self.group_id.index(id_)
            return self.group_obj[i], self.group_name[i]

    def set_table(self, id_, table):
        obj, name = self.find(id_=id_)
        obj.table = table

    def solve_cge(self):
        for pack in self:
            print('{} {}'.format(pack['id'], pack['name']))
            pack['sif'].solve_cge()

    def plot_cge(self, plot_coef=True, plot_cge_ct=True, comment_num_points=False, markersize=5):
        figure = plt.figure(figsize=(6, 4), dpi=300)
        ax = figure.add_subplot(1, 1, 1)
        for pack in self:
            ax = pack['sif'].plot_cge(plot_coef=plot_coef, plot_cge_ct=False, ax=ax,
                                 marker=pack['marker'], comment_num_points=comment_num_points,
                                 group=self, print_text_coef=False, markersize=markersize)

        xline = np.linspace(*ax.set_xlim(), 100)

        if plot_cge_ct:
            k = 0
            for cge in pack['sif'].specimen.get_cge_ct():
                c, m, name = cge
                y = c * xline ** m
                ax.plot(xline, y, color='k', linestyle=linestyles[k], label='ВР '+name, zorder=9)
                k += 1
        ax.legend()

        xmin, xmax = ax.set_xlim()
        xmin = int(xmin // 10 * 10)
        xmax = int(xmax // 10 * 10 +10)
        xticks = np.arange(xmin, xmax+1, 10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45)
        return ax

    def plot_length_of_cycle(self, cm_parent=0, plot_ct=True, interpol=1):
        """Зависимость длины трещины от количества циклов по фракт данным и по
        вычисленным C и m
        Parameter:
            cm_parent - 0 - c и m данного экземпляра, 1 - экземпляра родителя и тд
                или объект SIF2 от которого берутся c и m
        """
        for pack in self:
            figure = plt.figure(figsize=(6, 4), dpi=300)
            ax = figure.add_subplot(1, 1, 1)
            sif = pack['sif']
            ax = sif.plot_length_of_cycle(cm_parent=cm_parent, plot_ct=plot_ct,
                    ax=ax, interpol=interpol, group=self)

    def solve_cycle(self, display_table=True):
        for pack in self:
            pack['sif'].solve_cycle(display_table=display_table)

    def plot_cycle(self, reverse_rate=False, sdvig_r=False, comment_num_points=False,
                       plot_total_cycle=True, ax=None):
        if ax==None:
            figure = plt.figure(figsize=(15, 10), dpi=200)
            ax = figure.add_subplot(1, 1, 1)
        for pack in self:
            pack['sif'].plot_cycle(reverse_rate=reverse_rate, sdvig_r=sdvig_r, ax=ax,
                                   marker=pack['marker'], plot_total_cycle=plot_total_cycle,
                                   comment_num_points=comment_num_points,
                                   group=self)
        return ax

    def find_form_from_cycle(self, delta_n=1000, reverse_rate=True, display_table=True):
        if reverse_rate:
            n_name = 'n_rev'
            cycle = self.group_obj[0].table[n_name].min()
        else:
            n_name = 'n'
            cycle = self.group_obj[0].table[n_name].max()

        for pack in self:
            if reverse_rate:
                cyc_ = pack['sif'].table[n_name].min()
                if cycle < cyc_:
                    cycle = cyc_
            else:
                cyc_ = pack['sif'].table[n_name].max()
                if cycle > cyc_:
                    cycle = cyc_
        if reverse_rate:
            cycle_min = cycle
            cycle_max = 'max'
        else:
            cycle_min = 'min'
            cycle_max = cycle

        for pack in self:
            pack['sif'].rad_for_cycle(delta_n=delta_n, reverse_rate=reverse_rate,
                                      cycle_min=cycle_min, cycle_max=cycle_max,
                                      display_table=display_table)

    def cut_to_equivalent(self, outer=True, inner=True):
        """Обрезка лишних точек для создания эквивалентных расстояний"""
        a0_min = 0
        a0_max = 1000000
        for pack in self:
            a0 = pack['sif'].curve_min.rads_initial[0]
            if a0 >= a0_min:
                a0_min = a0
                min_sif_id = pack['id']
                min_sif = pack['sif']
            a0 = pack['sif'].curve_max.rads_initial[0]
            if a0 <= a0_max:
                a0_max = a0
                max_sif_id = pack['id']
                max_sif = pack['sif']

        for pack in self:
            sif = pack['sif']
            table = sif.table
            if inner:
                if pack['id'] != min_sif_id:
                    min_index = table['rad'].idxmin()
                    ang1 = table.loc[min_index, 'ang']
                    rad1 = table.loc[min_index, 'rad']
                    for index, row in table.drop(index=min_index).iterrows():
                        ang2 = row['ang']
                        rad2 = row['rad']
                        rad_curve1_min = min_sif.curve_min.rad_from_ang(ang1)
                        rad_curve2_min = min_sif.curve_min.rad_from_ang(ang2)
                        if (rad_curve1_min >= rad1) and (rad_curve2_min <= rad2):
                            break
                        else:
                            ang1 = ang2
                            rad1 = rad2
                    name_min_sr = 'sredn_min'
                    rad_min_sr = rad_curve2_min
                    ang_min_sr = ang2
                    d_min_sr = np.interp(rad_min_sr, table['rad'].to_numpy(dtype=float),
                                         table['d'].to_numpy(dtype=float))
                    if 'sif' in table:
                        sif_min_sr = np.interp(rad_min_sr, table['rad'].to_numpy(dtype=float),
                                               table['sif'].to_numpy(dtype=float))

                    table = table[table['rad']>=rad_curve2_min]
                    new_ind = -1000
                    table.loc[new_ind] = table.iloc[0]
                    table.loc[new_ind, 'name'] = name_min_sr
                    table.loc[new_ind, 'rad'] = rad_min_sr
                    table.loc[new_ind, 'ang'] = ang_min_sr
                    table.loc[new_ind, 'd'] = d_min_sr
                    if 'sif' in table:
                        table.loc[new_ind, 'sif'] = sif_min_sr
                    table = table.sort_values(by='rad')

            if outer:
                if pack['id'] != max_sif_id:
                    max_index = table['rad'].idxmax()
                    ang1 = table.loc[max_index, 'ang']
                    rad1 = table.loc[max_index, 'rad']
                    for index, row in table.iloc[::-1].drop(index=max_index).iterrows():
                        ang2 = row['ang']
                        rad2 = row['rad']
                        rad_curve1_max = max_sif.curve_max.rad_from_ang(ang1)
                        rad_curve2_max = max_sif.curve_max.rad_from_ang(ang2)
                        if (rad_curve1_max <= rad1) and (rad_curve2_max >= rad2):
                            break
                        else:
                            ang1 = ang2
                            rad1 = rad2
                    name_max_sr = 'sredn_max'
                    rad_max_sr = rad_curve2_max
                    ang_max_sr = ang2
                    d_max_sr = np.interp(rad_max_sr, table['rad'].to_numpy(dtype=float),
                                         table['d'].to_numpy(dtype=float))
                    if 'sif' in table:
                        sif_max_sr = np.interp(rad_max_sr, table['rad'].to_numpy(dtype=float),
                                               table['sif'].to_numpy(dtype=float))

                    table = table[table['rad']<=rad_curve2_max]
                    new_ind = 1000
                    table.loc[new_ind] = table.iloc[-1]
                    table.loc[new_ind, 'name'] = name_max_sr
                    table.loc[new_ind, 'rad'] = rad_max_sr
                    table.loc[new_ind, 'ang'] = ang_max_sr
                    table.loc[new_ind, 'd'] = d_max_sr
                    if 'sif' in table:
                        table.loc[new_ind, 'sif'] = sif_max_sr
                    table = table.sort_values(by='rad')

            self.set_table(pack['id'], table)
            sif.define_minmax_curve()

    def copy(self):
        new_obj = copy.copy(self)
        new_obj.group_obj  = copy.copy(new_obj.group_obj)
        new_obj.group_name = copy.copy(new_obj.group_name)
        new_obj.group_id   = copy.copy(new_obj.group_id)
        new_obj.marker     = copy.copy(new_obj.marker)
        new_obj.__i = 0
        return new_obj

    def deepcopy(self):
        return copy.deepcopy(self)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            pack = {'sif': self.group_obj[self.__i],
                    'name': self.group_name[self.__i],
                    'id': self.group_id[self.__i],
                    'marker': self.marker[self.__i]}
            self.__i += 1
            return pack
        except IndexError:
            self.__i = 0
            raise StopIteration

    def plot_geom(self, ang0=0, ang1=360, rad0=0, rad1='max',
                  plot_rad_obr=True, plot_rad_def=False, plot_parent=False,
                  dir_theta_null='S', comment_num_points=False,
                  plot_curve_front_data=True, plot_curve_front_data_spec=True, settings=dict(),
                  ax=None):
        """
        Parameter:
            settings - словарь с настройками:
                color_curve - цвет кривых
                alpha_curve - прозрачность кривых
                only_min_max_curve - только минимальная и максимальная кривые
                axis_switch - 'off', 'on' - выключение осей и шкал
        """

        if ax==None:
            fig = plt.figure(figsize=(10,10), dpi=300)
            ax = fig.add_subplot(projection='polar')
            ax.set_theta_zero_location(dir_theta_null)
            ax.set_theta_direction(1)
            ax.grid(False)

        for pack in self:
            sif = pack['sif']
            df = sif.table
            sc = ax.plot(np.deg2rad(df['ang']), df['rad'], linestyle='',
                         color=COLORS[pack['id']], marker=pack['marker'], zorder=10,
                         label='{} {}'.format(pack['id'], pack['name']))

            if comment_num_points:
                for index, row in df.iterrows():
                    ax.text(np.deg2rad(row['ang']), row['rad'], str(index), zorder=20)

        # печать кривых фронта для крайних точек
            if plot_curve_front_data:
                if sif.curve_min:
                    sif.curve_min.plot_ax(ax, color=COLORS[pack['id']], alpha=1)
                if sif.curve_max:
                    sif.curve_max.plot_ax(ax, color=COLORS[pack['id']], alpha=1)

        ax.scatter([0], [0], color='r', marker='+', linewidth=10)

        spec = pack['sif'].specimen

        # печать образца и дефекта
        if spec.rad_obr and plot_rad_obr:
            rad, deg360 = spec._Specimen__sdvig(spec.rad_obr)
            ax.plot(deg360, rad, 'k')
        if spec.rad_obr:
            ax.set_ylim(0, spec.rad_obr+0.5)
        if spec.rad_def and plot_rad_def:
            rad, deg360 = spec._Specimen__sdvig(spec.rad_def)
            ax.plot(deg360, rad, 'k')

        # печать кривых для образца
        if plot_curve_front_data_spec:
            if spec.curve_front_data:
                try:
                    color = settings['color_curve']
                except KeyError:
                    color='k'
                try:
                    alpha = settings['alpha_curve']
                except KeyError:
                    alpha = 0.3
                try:
                    only_min_max = settings['only_min_max_curve']
                except KeyError:
                    only_min_max = True
                ax = spec.curve_front_data.plot_ax_initial(ax, color=color,
                        alpha=alpha, only_min_max=only_min_max)

        try:
            axis_switch =  settings['axis_switch']
            ax.axis(axis_switch)
        except:
            pass

        ax.set_xlim(np.deg2rad(ang0), np.deg2rad(ang1))
        if rad1 == 'max':
            rad1 = df['rad'].max()
            if plot_rad_obr:
                if spec.rad_obr > rad1:
                    rad1 = spec.rad_obr
        ax.set_ylim(rad0, rad1*1.1)

        ax.legend()
        return ax

