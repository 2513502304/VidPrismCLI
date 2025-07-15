from colorama import just_fix_windows_console, Fore, Back, Style, Cursor
from skimage.color import deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc
from typing import Sequence, MutableSequence, Final, Callable
from enum import Enum, auto
import numpy as np
import cv2 as cv

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

# 前景颜色选择器
fore_switcher = {
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

# 背景颜色选择器
back_switcher = {
    # normal color
    NORMAL_BLACK: Style.NORMAL + Back.BLACK,
    NORMAL_RED: Style.NORMAL + Back.RED,
    NORMAL_GREEN: Style.NORMAL + Back.GREEN,
    NORMAL_YELLOW: Style.NORMAL + Back.YELLOW,
    NORMAL_BLUE: Style.NORMAL + Back.BLUE,
    NORMAL_MAGENTA: Style.NORMAL + Back.MAGENTA,
    NORMAL_CYAN: Style.NORMAL + Back.CYAN,
    NORMAL_WHILE: Style.NORMAL + Back.WHITE,
    # dim color
    DIM_BLACK: Style.DIM + Back.BLACK,
    DIM_RED: Style.DIM + Back.RED,
    DIM_GREEN: Style.DIM + Back.GREEN,
    DIM_YELLOW: Style.DIM + Back.YELLOW,
    DIM_BLUE: Style.DIM + Back.BLUE,
    DIM_MAGENTA: Style.DIM + Back.MAGENTA,
    DIM_CYAN: Style.DIM + Back.CYAN,
    DIM_WHILE: Style.DIM + Back.WHITE,
    # bright color
    BRIGHT_BLACK: Style.BRIGHT + Back.BLACK,  # Back.LIGHTBLACK_EX
    BRIGHT_RED: Style.BRIGHT + Back.RED,  # Back.LIGHTRED_EX
    BRIGHT_GREEN: Style.BRIGHT + Back.GREEN,  # Back.LIGHTGREEN_EX
    BRIGHT_YELLOW: Style.BRIGHT + Back.YELLOW,  # Back.LIGHTWHITE_EX
    BRIGHT_BLUE: Style.BRIGHT + Back.BLUE,  #  Back.LIGHTBLUE_EX
    BRIGHT_MAGENTA: Style.BRIGHT + Back.MAGENTA,  # Back.LIGHTMAGENTA_EX
    BRIGHT_CYAN: Style.BRIGHT + Back.CYAN,  # Back.LIGHTCYAN_EX
    BRIGHT_WHILE: Style.BRIGHT + Back.WHITE,  # Back.LIGHTYELLOW_EX
}


def deltaE_rgb(
    rgb1: Sequence[Sequence[int]],
    rgb2: Sequence[Sequence[int]], 
    channel_axis: int = -1, 
    weight: Sequence[int] = (1, 1, 1),
) -> np.ndarray:
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


def deltaE_approximation_rgb(
    rgb1: Sequence[Sequence[int]],
    rgb2: Sequence[Sequence[int]],
    channel_axis: int = -1, 
) -> np.ndarray:
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


def pixel2cluster_color(
    data: Sequence[Sequence[int]], 
    func: Callable = deltaE_ciede2000, 
    mode: str = 'lab', 
    func_kwargs: dict = {}, 
    enhance_color: bool = True,
) -> np.ndarray:
    '''
    获取聚类颜色映射列表
    ---
    :param data: rgb list-like or ndarray, shape=(N, 3)
    :param func: 衡量颜色差异的可调用对象，该函数接受的第一个参数为参考颜色，第二个参数为比较颜色，函数签名请阅览 skimage.color 中提供的 deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc 函数。可选。默认为 skimage.color.deltaE_ciede2000
    :param mode: 衡量 data 颜色差异的颜色空间，必须与 func 计算所使用的颜色空间对应，可选。默认为 'lab'
    :param func_kwargs: 要传递给 func 的关键字参数，可选。默认为空字典
    :param enhance_color: 是否增强颜色细节，由 8 色添加到 24 色，可选。默认为 True
    :return: 聚类颜色映射列表，shape=(N, 3)
    '''
    #! 使用矢量化操作提高性能，并配合 numpy 默认行存储的特性对其内存布局进行计算优化，以保证所有的计算都发生在行操作 (axis=1) 而非列操作 (axis=0) 上
    # 是否增强颜色细节，由 8 色添加到 24 色
    if enhance_color:
        colormap = list(fore_switcher.keys())   # list(back_switcher.keys())
    else:
        colormap = list(fore_switcher.keys())[:8]   # list(back_switcher.keys())[:8]
    # 输入的 rgb 颜色，shape=(N, 1, 3)
    rgbs = np.asarray(data, dtype=np.uint8).reshape(-1, 1, 3)
    # rgb 颜色列表，shape=(M, 1, 3)
    colors = np.asarray(colormap, np.uint8).reshape(-1, 1, 3)
    # 判断 rgb 颜色与 colors 颜色列表大小，以适用不同策略，进行计算优化
    N, M = rgbs.shape[0], colors.shape[0]
    # 匹配模式
    match mode.lower():
        case 'rgb':
            # rgb 比较颜色，shape=(N, 3)
            comparison_colors = rgbs.reshape(-1, 3)
            # rgb 参考颜色，shape=(M, 3)
            reference_colormap = colors.reshape(-1, 3)
        case 'lab':
            # lab 比较颜色，shape=(N, 3)
            comparison_colors = cv.cvtColor(rgbs, cv.COLOR_RGB2LAB).reshape(-1, 3)
            # lab 参考颜色，shape=(M, 3)
            reference_colormap = cv.cvtColor(colors, cv.COLOR_RGB2LAB).reshape(-1, 3)
        case _:
            raise NotImplementedError()
    #! 大多数情况下，N >> M
    if M < N:
        # 计算所有颜色之间的距离，shape=(M, N)，并将其转置以匹配形状 (N, M)，采用行操作 (axis=1) 而非列操作 (axis=0)
        reference_colormap = reference_colormap[:, np.newaxis, :]
        distances = np.asarray([func(reference_color, comparison_colors, **func_kwargs) for reference_color in reference_colormap]).T
    else:
        # 计算所有颜色之间的距离，shape=(N, M)
        comparison_colors = comparison_colors[:, np.newaxis, :]
        distances = np.asarray([func(reference_colormap, comparison_color, **func_kwargs) for comparison_color in comparison_colors])
    # 找到最小距离的索引，shape=(N, )
    min_index = np.argmin(distances, axis=1)
    # 聚类颜色映射列表，shape=(N, 3)
    cluster_color = colors[min_index].reshape(-1, 3)
    # 返回聚类颜色映射列表
    return cluster_color


def pixel2cluster_color_code(
    data: Sequence[Sequence[int]],
    func: Callable = deltaE_ciede2000, 
    mode: str = 'lab', 
    func_kwargs: dict = {}, 
    enable_background: bool = False,
    enhance_color: bool = True, 
) -> list[str]:
    '''
    获取聚类颜色映射代码列表
    ---
    :param data: rgb list-like or ndarray, shape=(N, 3)
    :param func: 衡量颜色差异的可调用对象，该函数接受的第一个参数为参考颜色，第二个参数为比较颜色，函数签名请阅览 skimage.color 中提供的 deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc 函数。可选。默认为 skimage.color.deltaE_ciede2000
    :param mode: 衡量 data 颜色差异的颜色空间，必须与 func 计算所使用的颜色空间对应，可选。默认为 'lab'
    :param func_kwargs: 要传递给 func 的关键字参数，可选。默认为空字典
    :param enable_background: 是否绘制背景，仅在 color_mode 为 CLUSTERCOLOR 时使用，可选。默认为 False
    :param enhance_color: 是否增强颜色细节，由 8 色添加到 24 色，可选。默认为 True
    :return: 聚类颜色映射代码列表，shape=(N, )
    '''
    # 聚类颜色映射列表，shape=(N, 3)
    cluster_color = pixel2cluster_color(data=data, func=func, mode=mode, func_kwargs=func_kwargs, enhance_color=enhance_color, )
    # 聚类颜色映射代码列表，shape=(N, )
    if enable_background:
        # 若启用背景，则使用背景颜色
        cluster_color_code = [back_switcher.get(tuple(bc), Style.NORMAL + Back.BLACK) for bc in cluster_color]
    else:
        # 否则使用前景颜色
        cluster_color_code =  [fore_switcher.get(tuple(bc), Style.NORMAL + Fore.BLACK) for bc in cluster_color]
    # 返回聚类颜色映射代码列表
    return cluster_color_code


# 转换为 ANSI 转义序列，使用 np.frompyfunc 创建 ufunc
pixel2true_color_code: Callable[[Sequence[int], Sequence[int], Sequence[int]], Sequence[str]] = np.frompyfunc(lambda r, g, b: f'\033[38;2;{r};{g};{b}m', 3, 1)


class COLORMODE(Enum):
    '''颜色模式'''
    CLUSTERCOLOR = auto() # 聚类色
    TRUECOLOR = auto() # 真彩色
