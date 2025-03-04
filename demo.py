from skimage.color import deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc
from typing import Sequence, MutableSequence, Final
import numpy as np
import cv2 as cv
import string
import random
from processing import deltaE_rgb, deltaE_approximation_rgb, get_color_codes, ufunc_pixel2color_codes


def show_color_metric_difference(data: Sequence[Sequence[int]], func: callable, mode: str = 'lab', func_kwargs: dict = {}) -> None:
    '''
    评估不同颜色差异计算方法的显示效果
    ---
    :param data: rgb list-like or ndarray, shape=(N, 3)
    :param func: 衡量颜色差异的可调用对象，该函数接受的第一个参数为参考颜色，第二个参数为比较颜色，函数签名请阅览 skimage.color 中提供的 deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc 函数。可选。默认为 skimage.color.deltaE_ciede2000
    :param mode: 衡量 data 颜色差异的颜色空间，必须与 func 计算所使用的颜色空间对应，可选。默认为 'lab'
    :param func_kwargs: 要传递给 func 的关键字参数，可选。默认为空字典
    '''
    h, w = data.shape[:2]

    # color 8
    rgb_color_codes = get_color_codes(data.reshape(-1, 3), enhance_color=False, mode=mode, func=func, func_kwargs=func_kwargs)
    symbols = [c + ''.join(random.choices(charset, k=2)) if (i + 1) % w != 0 else '\n' + c + ''.join(random.choices(charset, k=2)) for i, c in enumerate(rgb_color_codes)]
    print(''.join(symbols))

    # color 24
    rgb_color_codes = get_color_codes(data.reshape(-1, 3), mode=mode, func=func, func_kwargs=func_kwargs)
    symbols = [c + ''.join(random.choices(charset, k=2)) if (i + 1) % w != 0 else '\n' + c + ''.join(random.choices(charset, k=2)) for i, c in enumerate(rgb_color_codes)]
    print(''.join(symbols))


if __name__ == '__main__':
    # 字符表
    charset = string.digits + string.ascii_letters + string.punctuation

    image = cv.imdecode(np.fromfile('./Data/茜特菈莉.png', dtype=np.uint8), cv.IMREAD_REDUCED_COLOR_8)
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # deltaE_rgb
    show_color_metric_difference(rgb, deltaE_rgb, 'rgb')
    # deltaE_rgb with weight=(2, 4, 3)
    show_color_metric_difference(rgb, deltaE_rgb, 'rgb', func_kwargs={'weight': (2, 4, 3)})

    # deltaE_approximation_rgb
    show_color_metric_difference(rgb, deltaE_approximation_rgb, 'rgb')

    # deltaE_cie76
    show_color_metric_difference(rgb, deltaE_cie76)

    # deltaE_cie2000 with kL=1
    show_color_metric_difference(rgb, deltaE_ciede2000, func_kwargs={'kL': 1})
    # deltaE_cie2000 with kL=2
    show_color_metric_difference(rgb, deltaE_ciede2000, func_kwargs={'kL': 2})

    # deltaE_ciede94 with kL=1
    show_color_metric_difference(rgb, deltaE_ciede94, func_kwargs={'kL': 1})
    # deltaE_ciede94 with kL=2
    show_color_metric_difference(rgb, deltaE_ciede94, func_kwargs={'kL': 2})

    # deltaE_cmc with kL=1
    show_color_metric_difference(rgb, deltaE_cmc, func_kwargs={'kL': 1})
    # deltaE_cmc with kL=2
    show_color_metric_difference(rgb, deltaE_cmc, func_kwargs={'kL': 2})

    # # true color
    # h, w = rgb.shape[:2]
    # rgb_color_codes = ufunc_pixel2color_codes(rgb[..., 0], rgb[..., 1], rgb[..., 2]).reshape(-1)
    # symbols = [c + ''.join(random.choices(charset, k=2)) if (i + 1) % w != 0 else '\n' + c + ''.join(random.choices(charset, k=2)) for i, c in enumerate(rgb_color_codes)]
    # print(''.join(symbols))
