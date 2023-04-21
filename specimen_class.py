from ._import import *
from .sif_class import SIF, SIF_from_file
from .sif2_class import SIF2
from .imagespecimen_class import ImageSpecimen


class Specimen:
    def __init__(self, list_fronts=None, list_names=None, reverse_ang=False, rad_obr=None, rad_def=None,
                 force=None, r_asymmetry=None, temp=None, name='',
                 sdvig_x=0, sdvig_y=0, curve_front_data=None):
        """Определение фронтов КИН
        Parameters:
            list_fronts - список путей к контурам
            list_names - соответствующие им имена
            reverse_ang - разворачивать ли фронты по оси симметрии 0 - 180,
              можно задать список для каждого фронта"""

        self.list_fronts = list_fronts
        if list_fronts:
            if list_names is None:
                self.list_names = [x for x in range(len(self.list_fronts))]
            else:
                self.list_names = list_names
            if isinstance(reverse_ang, bool):
                self.reverse_ang = []
                for i in range(len(list_fronts)):
                    self.reverse_ang.append(reverse_ang)
            else:
                self.reverse_ang = reverse_ang
        else:
            self.list_names = list_names

        self.table = dict()
        if list_fronts:
            for i in range(len(list_fronts)):
                self.table[self.list_names[i]] = self.__read_file(i)

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
        self.curve_front_data = curve_front_data
        self.dir_sif = list()
        self.nominal_table = None
        self.cge_ct = list()
        self.sif_cloud = None

    NAMES_COUNTUR = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
    NAME_INDEX = ['ang']
    NAMES_COORD = ['ang', 'rad', 'z']
    NAME_DROP = ['node']
    NAMES_COLUMNS = NAME_DROP + NAMES_COORD + NAMES_COUNTUR

    def __read_file(self, i):
        path = self.list_fronts[i]
        names = Specimen.NAMES_COLUMNS
        index_col = Specimen.NAME_INDEX
        drop = Specimen.NAME_DROP
        df = pd.read_table(path, delim_whitespace=True, names=names,
                           index_col=index_col).drop(columns=drop)
        # отрицательные углы конвертируются в положительные
        df.index = np.where(df.index < 0, 360 + df.index, df.index)
        if self.reverse_ang[i]:
            df.index = 360 - df.index
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

    def plot_geom_front(self, cont=None, plot_rad_obr=True, plot_rad_def=True,
                        dir_theta_null='S', plot_curve_front_data=True,
                        save_plot_with_image=None, settings=dict()):
        """Печать геометрии фронтов трещины с нанесенными значениями КИН
        Parameter:
            settings - словарь с настройками:
                color_sif_cloud - цвет точек
                marker_sif_cloud - маркер точек
                color_curve - цвет кривых
                alpha_curve - прозрачность кривых
                only_min_max_curve - только минимальная и максимальная кривые
                axis_switch - 'off', 'on' - выключение осей и шкал
                sif_plot_general - True, False - раскраска КИН общая по всем фронтам или раздельная
                """

        cont_dict = self.__cont_dict(cont)
        
        fig = plt.figure(figsize=(10, 10), dpi=250)
        ax = fig.add_subplot(projection='polar')

        rad_max = 0
        kin_min = 10000
        kin_max = 0
        if cont:
            try:
                sif_plot_general = settings['sif_plot_general']
            except KeyError:
                sif_plot_general = False
            if sif_plot_general:
                for name in self.table:
                    tab = self.table[name]
                    kin = (np.array(tab[cont])[:-1] + np.array(tab[cont])[1:])/2
                    kin_min0 = kin.min()
                    kin_max0 = kin.max()
                    if kin_min0 < kin_min:
                        kin_min = kin_min0
                    if kin_max0 > kin_max:
                        kin_max = kin_max0

            for name in self.table:
                tab = self.table[name]
                deg = np.array(np.deg2rad(tab.index))
                rad = np.array(tab['rad'])

                cont = cont_dict[name]

                kin = (np.array(tab[cont])[:-1] + np.array(tab[cont])[1:])/2

                points = np.array([deg, rad]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                if sif_plot_general:
                    norm = plt.Normalize(kin_min, kin_max)
                else:
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

        ax.scatter([0], [0], color='r', marker='+', linewidth=2)

        # линии снятия значений КИН
        for sif in self.dir_sif:
            ang = np.deg2rad(sif.angle)
            ax.plot([ang, ang], [0, self.rad_obr], '--', color='r')
            ax.text(ang, self.rad_obr, 'Путь '+str(sif.path_n))

        # облако точек с sif_cloud
        if self.sif_cloud:
            df = self.sif_cloud.table
            try:
                color = settings['color_sif_cloud']
            except KeyError:
                color='k'
            try:
                marker = settings['marker_sif_cloud']
            except KeyError:
                marker = 'x'
            for index, row in df.iterrows():
                ax.scatter(np.deg2rad(df['ang']), df['rad'],
                           color=color, marker=marker, zorder=10)

        # исходные кривые фронтов
        if plot_curve_front_data:
            if self.curve_front_data:
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
                ax = self.curve_front_data.plot_ax_initial(ax, color=color,
                        alpha=alpha, only_min_max=only_min_max)

        try:
            axis_switch =  settings['axis_switch']
            ax.axis(axis_switch)
        except:
            pass

        # если есть объект с изображением поверхности излома, то сохранить его
        if isinstance(save_plot_with_image, ImageSpecimen):
            save_plot_with_image.save_fig(fig)

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
            direct_sif = direct_sif.sort_values(by='rad')

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

