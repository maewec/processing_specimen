from ._import import *



class CurveFrontsInterp:
    """Построение фронта и его аппроксимация по точкам"""
    def __init__(self, data):
        """
        Parameters:
        data - список двумерных массив со строками радиусов и углов для разных положений фронта
        """
        self.data = data
        self.curves = [CurveFront8point(dat[0], dat[1]) for dat in data]

    def plot_ax_initial(self, ax, color='k', alpha=0.3, only_min_max=False):
        if only_min_max:
            for curv in [self.curves[0], self.curves[-1]]:
                ax = curv.plot_ax(ax, color=color, alpha=alpha)
        else:
            for curv in self.curves:
                ax = curv.plot_ax(ax, color=color, alpha=alpha)
        return ax

    def search_curve(self, rad, ang):
        "Получение новых опорных точек кривой, проходящей через точку rad, ang"
        num_angs = 1000
        l_c = len(self.curves)
        rads4ang = np.zeros(l_c)
        curv_neutral_rad_arr = np.zeros((l_c, num_angs))
        i = 0
        for curv in self.curves:
            curv_neutral = curv.convert_neutral(num_angs=num_angs)
            curv_neutral_rad_arr[i,:] = curv_neutral.rads_initial
            rads4ang[i] = curv_neutral.rad_from_ang(ang)
            i += 1
        angs = np.degrees(curv_neutral.angs_initial)
        rad_new = np.zeros(num_angs)
        for i in range(num_angs):
            rad_new[i] = np.interp(rad, rads4ang, curv_neutral_rad_arr[:, i])
        return CurveFront8point(rads=rad_new, angs=angs)

    def insert_dop_form(self, list_df, angs_arr):
        """Добавить на существующие кривые фронтов врезку с новыми радиусами и углами,
        list_df - полученные таблицы, в которых определен рост циклов
        Расположение таблиц в листе по часовой стрелке
        angs_arr - вектор углов, которые остаются от исходной кривой"""
        # у всех таблиц индексы одинаковые, начинаем проходить по одному
        for index in list_df[0].index:
            # таблица в которой собрана одна строка других таблиц
            df_row = pd.DataFrame(columns=['rad', 'ang'])
            for df in list_df:
                df_row = df_row.append(df.loc[index, ['rad', 'ang']])
            rad_0 = df_row.iloc[0]['rad']
            ang_0 = df_row.iloc[-1]['ang']
            curve_0 = self.search_curve(rad_0, ang_0)
            rad_1 = df_row.iloc[0]['rad']
            ang_1 = df_row.iloc[-1]['ang']
            curve_1 = self.search_curve(rad_1, ang_1)




class CurveFront8point:
    """Описание фронта по 8 радиусам, расположенным с шагом 45 градусов в цилиндрической СК"""
    def __init__(self, rads, angs=None):
        """
        Parameters:
        dat - двухмерный массив со строками радиусов и углов для разных положений фронта
        """
        self.rads = rads
        if angs is None:
            l = len(self.rads)
            self.angs = np.arange(0, 2*np.pi, 2*np.pi/l)
        else:
            self.angs = np.radians(angs)
        self.rads_initial = self.rads
        self.angs_initial = self.angs
        # расширение пределов массива для плавного соединения сплайна
        self.rads = np.concatenate((self.rads, self.rads, self.rads))
        self.angs = np.concatenate((self.angs-2*np.pi, self.angs,
                                    self.angs+2*np.pi))

        self.__tck = interpolate.splrep(self.angs, self.rads, k=3)

    def plot_ax(self, ax, color='k', alpha=0.3):
        angs = np.linspace(0, 2*np.pi, 1000)
        rads = interpolate.splev(angs, self.__tck, der=0)
        ax.plot(angs, rads, color=color, alpha=alpha)   
        return ax

    def rad_from_ang(self, ang):
        ang = np.radians(ang)
        rad = interpolate.splev(ang, self.__tck, der=0)
        return rad

    def convert_neutral(self, num_angs=1000):
        angs = np.linspace(0, 360, num_angs, endpoint=False)
        rads = self.rad_from_ang(angs)
        return CurveFront8point(rads, angs)

    def convert_angs(self, angs_arr):
        """Получение кривой с опорными радиусами в массиве заданных углов angs_arr"""
        angs = angs_arr
        rads = self.rad_from_ang(angs)
        return CurveFront8point(rads, angs)

