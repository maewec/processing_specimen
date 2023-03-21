from ._import import *



class ImageSpecimen:
    """Описание изображения поверхности образца для подложки графика"""

    count = 0

    def __init__(self, pathfile, point_center_pix, point_center_mm,
                 point_rot_pix, point_rot_mm, angle=0):
        """Parameter:
        pathfile - путь до файла
        point_center_pix - координаты центра в пикселях
        point_center_mm - координаты центра в мм
        point_rot_pix - координаты второй точки в пикселях
        point_rot_mm - координаты второй точки в мм
        angle - дополнительный поворот в градусах"""

        self.pathfile = pathfile
        self.pcp = np.array(point_center_pix)
        self.pcm = np.array(point_center_mm)
        self.prp = np.array(point_rot_pix)
        self.prm = np.array(point_rot_mm)
        self.angle = angle

        self.img = Image.open(pathfile)
        # в одном мм пикселей
        self.pix_in_mm = np.sqrt(
                                 np.sum(np.square(self.pcp - self.prp)) /\
                                 np.sum(np.square(self.pcm - self.prm))
                                 )
        self.name = os.path.splitext(os.path.basename(self.pathfile))[0]

    def save_fig(self, fig):
        ax = fig.axes[0]
        # прозначный фон для графика
        ax.patch.set_alpha(0)

        # пикселей в мм графика
        l = 1
        pix_graph = (ax.transData.transform([0, 0]) - ax.transData.transform([0, l]))[1]
        pix_graph = np.abs(pix_graph)
        k_resize = pix_graph / self.pix_in_mm

        # новые размеры для изображения
        new_x = int(round(self.img.size[0] * k_resize, 0))
        new_y = int(round(self.img.size[1] * k_resize, 0))
        img = self.img.resize((new_x, new_y), resample=Image.BILINEAR)

        # выравниваем горизонт по второй точке
        dif = self.pcp - self.prp
        ang = np.degrees(np.arctan(dif[1]/dif[0]))
        # и добавляем ручной поворот
        # т.к. при внедрении картинки в график задан аргумент origin='lower',
        # угол идет с минусом
        ang = -ang + self.angle

        # находим координаты в пикселях нулевой точки графика
        # и пересчитываем новые координаты центральной точки изображения
        # в пикселях с учетом ресайза
        point_center_graph = ax.transData.transform([0, 0])
        point_center_graph = np.array(np.around(point_center_graph, 0), dtype=int)
        pcp_resize = self.pcp * k_resize
        pcp_resize = np.array(np.around(pcp_resize, 0), dtype=int)
        # т.к. при внедрении картинки в график задан аргумент origin='lower',
        # то он переворачивает картинку по вертикали, надо перечитать координаты
        pcp_resize[1] = new_y - pcp_resize[1]
        sdvig = point_center_graph - pcp_resize

        # т.к. при внедрении картинки в график задан аргумент origin='lower',
        # надо обратно перевернуть изображение (см. выше)
        img = ImageOps.flip(img)

        # поворот относительно центральной точки
        img = img.rotate(angle=ang, expand=False, center=tuple(pcp_resize),
                         resample=Image.BILINEAR, fillcolor='white')

        # внедрение изображения в график
        fig.figimage(img, sdvig[0], sdvig[1], zorder=-1, resize=False, origin='lower')

        name = '{}_{:2.0f}.png'.format(self.name, ImageSpecimen.count)
        ImageSpecimen.count += 1
        fig.savefig(name)

