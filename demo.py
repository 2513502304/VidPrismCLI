'''
修改 processing.py 的 get_color_codes 函数，为其添加 color_func 与 func_kwargs 参数以测试不同颜色函数对比之间的差异
'''

from colorama import just_fix_windows_console, Fore, Back, Style, Cursor
from skimage.color import deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc
from typing import Sequence, MutableSequence, Final
import numpy as np
import cv2 as cv
import string
import random

# 颜色吸管：https://photokit.com/colors/eyedropper/?lang=zh
# normal color
NORMAL_BLACK: Final = (12, 12, 12)  # #0c0c0c
NORMAL_RED: Final = (197, 15, 31)  # #c50f1f
NORMAL_GREEN: Final = (19, 161, 14)  # #13a10e
NORMAL_YELLOW: Final = (193, 156, 0)  # #c19c00
NORMAL_BLUE: Final = (0, 55, 218)  # #0037da
NORMAL_MAGENTA: Final = (136, 23, 152)  # #881798
NORMAL_CYAN: Final = (58, 150, 221)  # #3a96dd
NORMAL_WHILE: Final = (204, 204, 204)  # #cccccc
# dim color
DIM_BLACK: Final = (6, 6, 6)  # #060606
DIM_RED: Final = (98, 7, 15)  # #62070f
DIM_GREEN: Final = (9, 80, 7)  # #095007
DIM_YELLOW: Final = (96, 78, 0)  # #604e00
DIM_BLUE: Final = (0, 27, 109)  # #001b6d
DIM_MAGENTA: Final = (68, 11, 76)  # #440b4c
DIM_CYAN: Final = (29, 75, 110)  # #1d4b6e
DIM_WHILE: Final = (102, 102, 102)  # #666666
# bright color
BRIGHT_BLACK: Final = (118, 118, 118)  # #767676
BRIGHT_RED: Final = (231, 72, 86)  # #e74856
BRIGHT_GREEN: Final = (22, 198, 12)  # #16c60c
BRIGHT_YELLOW: Final = (249, 241, 165)  # #f9f1a5
BRIGHT_BLUE: Final = (59, 120, 255)  # #3b78ff
BRIGHT_MAGENTA: Final = (180, 0, 158)  # #b4009e
BRIGHT_CYAN: Final = (97, 214, 214)  # #61d6d6
BRIGHT_WHILE: Final = (242, 242, 242)  # #f2f2f2

# 颜色选择器
switcher = {
    # normal color
    NORMAL_BLACK: Style.NORMAL + Fore.BLACK,
    NORMAL_RED: Style.NORMAL + Fore.RED,
    NORMAL_GREEN: Style.NORMAL + Fore.GREEN,
    NORMAL_YELLOW: Style.NORMAL + Fore.YELLOW,
    NORMAL_BLUE: Style.NORMAL + Fore.BLUE,
    NORMAL_MAGENTA: Style.NORMAL + Fore.MAGENTA,
    NORMAL_CYAN: Style.NORMAL + Fore.CYAN,
    NORMAL_WHILE: Style.NORMAL + Fore.WHITE,
    # dim color
    DIM_BLACK: Style.DIM + Fore.BLACK,
    DIM_RED: Style.DIM + Fore.RED,
    DIM_GREEN: Style.DIM + Fore.GREEN,
    DIM_YELLOW: Style.DIM + Fore.YELLOW,
    DIM_BLUE: Style.DIM + Fore.BLUE,
    DIM_MAGENTA: Style.DIM + Fore.MAGENTA,
    DIM_CYAN: Style.DIM + Fore.CYAN,
    DIM_WHILE: Style.DIM + Fore.WHITE,
    # bright color
    BRIGHT_BLACK: Style.BRIGHT + Fore.BLACK,  # Fore.LIGHTBLACK_EX
    BRIGHT_RED: Style.BRIGHT + Fore.RED,  # Fore.LIGHTRED_EX
    BRIGHT_GREEN: Style.BRIGHT + Fore.GREEN,  # Fore.LIGHTGREEN_EX
    BRIGHT_YELLOW: Style.BRIGHT + Fore.YELLOW,  # Fore.LIGHTWHITE_EX
    BRIGHT_BLUE: Style.BRIGHT + Fore.BLUE,  #  Fore.LIGHTBLUE_EX
    BRIGHT_MAGENTA: Style.BRIGHT + Fore.MAGENTA,  # Fore.LIGHTMAGENTA_EX
    BRIGHT_CYAN: Style.BRIGHT + Fore.CYAN,  # Fore.LIGHTCYAN_EX
    BRIGHT_WHILE: Style.BRIGHT + Fore.WHITE,  # Fore.LIGHTYELLOW_EX
}


def deltaE_rgb(rgb1: Sequence[Sequence[int]], rgb2: Sequence[Sequence[int]], channel_axis: int = -1, weight: Sequence[int] = (1, 1, 1)):
    '''
    附带权重的 RGB 色彩空间中两点之间的欧几里得距离
    ---
    Reference
    ---
    https://wikimedia.org/api/rest_v1/media/math/render/svg/766971fac976a11f71166fb485df533072c886fb
    '''
    r1, g1, b1 = np.moveaxis(rgb1.astype(np.float32), source=channel_axis, destination=0)[:3]
    r2, g2, b2 = np.moveaxis(rgb2.astype(np.float32), source=channel_axis, destination=0)[:3]
    r_w, g_w, b_w = weight
    return np.sqrt(r_w * (r2 - r1)**2 + g_w * (g2 - g1)**2 + b_w * (b2 - b1)**2)


def deltaE_approximation_rgb(rgb1: Sequence[Sequence[int]], rgb2: Sequence[Sequence[int]], channel_axis: int = -1):
    '''
    一种低成本近似方法
    这个公式的结果非常接近 L*u*v*（具有修改后的亮度曲线），更重要的是，它是一种更稳定的算法：它不存在一个颜色范围，在这个范围内会突然给出远离最优结果的结果
    ---
    Reference
    ---
    https://wikimedia.org/api/rest_v1/media/math/render/svg/766971fac976a11f71166fb485df533072c886fb
    https://web.archive.org/web/20210327221240/https://www.compuphase.com/cmetric.htm
    '''
    r1, g1, b1 = np.moveaxis(rgb1.astype(np.float32), source=channel_axis, destination=0)[:3]
    r2, g2, b2 = np.moveaxis(rgb2.astype(np.float32), source=channel_axis, destination=0)[:3]
    r_mean = (r1 + r2) / 2
    delat_r = r2 - r1
    delat_g = g2 - g1
    delta_b = b2 - b1
    return np.sqrt((2 + r_mean / 256) * delat_r**2 + 4 * delat_g**2 + (2 + (255 - r_mean) / 256) * delta_b**2)


def get_color_codes(data: Sequence[Sequence[int]], color_func: callable = deltaE_ciede2000, func_kwargs: dict = {}, enhance_color: bool = True) -> list[str]:
    '''
    获取颜色映射代码列表
    ---
    :param data: rgb list-like or ndarray, shape=(N, 3)
    :param color_func: 衡量颜色差异的可调用对象，该函数接受的第一个参数为参考颜色（Lab 色彩空间），第二个参数为比较颜色（Lab 色彩空间），函数签名请阅览 skimage.color 中提供的 deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc 函数。可选。默认为 skimage.color.deltaE_ciede2000
    :param func_kwargs: 要传递给 color_func 的关键字参数，可选。默认为空字典
    :param enhance_color: 是否增强颜色细节，由 8 色添加到 24 色，可选。默认为 True
    :return: 颜色映射代码列表
    '''
    #! 使用矢量化操作提高性能，并配合 numpy 默认行存储的特性对其内存布局进行计算优化，以保证所有的计算都发生在行操作 (axis=1) 而非列操作 (axis=0) 上
    if enhance_color:
        colormap = list(switcher.keys())
    else:
        colormap = list(switcher.keys())[:8]
    # 输入的 rgb 颜色，shape=(N, 1, 3)
    rgbs = np.asarray(data, dtype=np.uint8).reshape(-1, 1, 3)
    # rgb 颜色列表，shape=(M, 1, 3)
    colors = np.asarray(colormap, np.uint8).reshape(-1, 1, 3)
    # 判断 rgb 颜色与 colors 颜色列表大小，以适用不同策略，进行计算优化
    N, M = rgbs.shape[0], colors.shape[0]
    # 将 rgb 颜色转换为 lab 颜色，shape=(N, 3)
    labs = cv.cvtColor(rgbs, cv.COLOR_RGB2LAB).reshape(-1, 3)
    # 将 rgb 颜色列表转换为 lab 颜色列表，shape=(M, 3)
    lab_colors = cv.cvtColor(colors, cv.COLOR_RGB2LAB).reshape(-1, 3)
    #! 大多数情况下，N >> M
    if M < N:
        # 计算所有颜色之间的距离，shape=(M, N)，并将其转置以匹配形状 (N, M)，采用行操作 (axis=1) 而非列操作 (axis=0)
        lab_colors = lab_colors[:, np.newaxis, :]
        distances = np.asarray([color_func(lab_color, labs, **func_kwargs) for lab_color in lab_colors]).T
    else:
        # 计算所有颜色之间的距离，shape=(N, M)
        labs = labs[:, np.newaxis, :]
        distances = np.asarray([color_func(lab_colors, lab, **func_kwargs) for lab in labs])
    # 找到最小距离的索引，shape=(N, )
    min_index = np.argmin(distances, axis=1)
    # 最佳匹配颜色，shape=(N, 3)
    best_color = colors[min_index].reshape(-1, 3)
    # 返回颜色映射代码
    return [switcher.get(tuple(bc), Style.NORMAL + Fore.BLACK) for bc in best_color]


# 字符表
charset = string.digits + string.ascii_letters + string.punctuation

image = cv.imdecode(np.fromfile('./Data/茜特菈莉.png', dtype=np.uint8), cv.IMREAD_REDUCED_COLOR_8)
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)


def show_color_metric_difference(data: Sequence[Sequence[int]], color_func: callable, func_kwargs: dict = {}) -> None:
    '''
    评估不同颜色差异计算方法的显示效果
    :param data: rgb list-like or ndarray, shape=(N, 3)
    :param color_func: 衡量颜色差异的可调用对象，该函数接受的第一个参数为参考颜色（Lab 色彩空间），第二个参数为比较颜色（Lab 色彩空间），函数签名请阅览 skimage.color 中提供的 deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc 函数。可选。默认为 skimage.color.deltaE_ciede2000
    :param func_kwargs: 要传递给 color_func 的关键字参数，可选。默认为空字典
    '''
    h, w = data.shape[:2]

    # 卡车封路
    for _ in range(7):
        print(Style.BRIGHT + Fore.RED + '**' * w + Fore.RESET)

    # color 8
    rgb_color_codes = get_color_codes(rgb.reshape(-1, 3), enhance_color=False, color_func=color_func, func_kwargs=func_kwargs)
    symbols = [c + ''.join(random.choices(charset, k=2)) if (i + 1) % w != 0 else '\n' + c + ''.join(random.choices(charset, k=2)) for i, c in enumerate(rgb_color_codes)]
    print(''.join(symbols))

    # color 24
    rgb_color_codes = get_color_codes(rgb.reshape(-1, 3), color_func=color_func, func_kwargs=func_kwargs)
    symbols = [c + ''.join(random.choices(charset, k=2)) if (i + 1) % w != 0 else '\n' + c + ''.join(random.choices(charset, k=2)) for i, c in enumerate(rgb_color_codes)]
    print(''.join(symbols))

    # 卡车封路
    for _ in range(7):
        print(Style.BRIGHT + Fore.RED + '**' * w + Fore.RESET)


if __name__ == '__main__':
    # deltaE_rgb
    show_color_metric_difference(rgb, deltaE_rgb)
    # deltaE_rgb with weight=(2, 4, 3)
    show_color_metric_difference(rgb, deltaE_rgb, {'weight': (2, 4, 3)})

    # deltaE_approximation_rgb
    show_color_metric_difference(rgb, deltaE_approximation_rgb)

    # deltaE_cie76
    show_color_metric_difference(rgb, deltaE_cie76)

    # deltaE_cie2000 with kL=1
    show_color_metric_difference(rgb, deltaE_ciede2000, {'kL': 1})
    # deltaE_cie2000 with kL=2
    show_color_metric_difference(rgb, deltaE_ciede2000, {'kL': 2})

    # deltaE_ciede94 with kL=1
    show_color_metric_difference(rgb, deltaE_ciede94, {'kL': 1})
    # deltaE_ciede94 with kL=2
    show_color_metric_difference(rgb, deltaE_ciede94, {'kL': 2})

    # deltaE_cmc with kL=1
    show_color_metric_difference(rgb, deltaE_cmc, {'kL': 1})
    # deltaE_cmc with kL=2
    show_color_metric_difference(rgb, deltaE_cmc, {'kL': 2})
