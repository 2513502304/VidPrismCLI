from colorama import just_fix_windows_console, Fore, Back, Style, Cursor
from colorama.ansi import code_to_chars, set_title, clear_screen, clear_line
from skimage.color import deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc
from playsound import playsound
from typing import Sequence, MutableSequence, Final, Callable
from enum import Enum, auto
from rich.progress import track
import subprocess
import numpy as np
import os
import shutil
import cv2 as cv
import time
import string
import random
from utils import logger, console

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


def split_va(input_file_path: str, output_video_path: str, output_audio_path: str) -> None:
    '''
    拆分音视频
    ---
    :param input_file_path: 携带音视频信息的输入文件路径
    :param output_video_path: 仅携带视频信息的输出文件路径
    :param output_audio_path: 仅携带音频信息的输出文件路径
    :return: None
    '''
    # 拆分视频
    cmds = [
        f'ffmpeg -i "{input_file_path}" -an -c:v copy "{output_video_path}"', f'ffmpeg -i "{input_file_path}" -vn -c:a copy "{output_audio_path}"',
    ]
    for cmd in cmds:
        command = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            check=True,
        )
        args = command.args
        returncode = command.returncode
        stdout = command.stdout.decode('utf-8')
        stderr = command.stderr.decode('utf-8')
        logger.info(f'{args = }')
        logger.info(f'{returncode = }')
        logger.info(f'{stdout}')
        logger.info(f'{stderr}')


def merge_va(input_video_path: str, input_audio_path: str, output_file_path: str) -> None:
    '''
    合成音视频
    ---
    :param input_video_path: 仅携带视频信息的输入文件路径
    :param input_audio_path: 仅携带音频信息的输入文件路径
    :param output_file_path: 携带音视频信息的输出文件路径
    :return: None
    '''
    # 合成视频
    cmds = [
        f'ffmpeg -i "{input_video_path}" -i "{input_audio_path}" -c:v copy -c:a copy "{output_file_path}"',
    ]
    for cmd in cmds:
        command = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            check=True,
        )
        args = command.args
        returncode = command.returncode
        stdout = command.stdout.decode('utf-8')
        stderr = command.stderr.decode('utf-8')
        logger.info(f'{args = }')
        logger.info(f'{returncode = }')
        logger.info(f'{stdout}')
        logger.info(f'{stderr}')


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


def get_color_codes(data: Sequence[Sequence[int]], func: Callable = deltaE_ciede2000, mode: str = 'lab', func_kwargs: dict = {}, enhance_color: bool = True) -> list[str]:
    '''
    获取颜色映射代码列表
    ---
    :param data: rgb list-like or ndarray, shape=(N, 3)
    :param func: 衡量颜色差异的可调用对象，该函数接受的第一个参数为参考颜色，第二个参数为比较颜色，函数签名请阅览 skimage.color 中提供的 deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc 函数。可选。默认为 skimage.color.deltaE_ciede2000
    :param mode: 衡量 data 颜色差异的颜色空间，必须与 func 计算所使用的颜色空间对应，可选。默认为 'lab'
    :param func_kwargs: 要传递给 func 的关键字参数，可选。默认为空字典
    :param enhance_color: 是否增强颜色细节，由 8 色添加到 24 色，可选。默认为 True
    :return: 颜色映射代码列表
    '''
    #! 使用矢量化操作提高性能，并配合 numpy 默认行存储的特性对其内存布局进行计算优化，以保证所有的计算都发生在行操作 (axis=1) 而非列操作 (axis=0) 上
    # 是否增强颜色细节，由 8 色添加到 24 色
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
    # 最佳匹配颜色，shape=(N, 3)
    best_color = colors[min_index].reshape(-1, 3)
    # 返回颜色映射代码
    return [switcher.get(tuple(bc), Style.NORMAL + Fore.BLACK) for bc in best_color]


class COLORMODE(Enum):
    '''颜色模式'''
    CLUSTERCOLOR = auto() # 聚类色
    TRUECOLOR = auto() # 真彩色


# 转换为 ANSI 转义序列，使用 np.frompyfunc 创建 ufunc
ufunc_pixel2color_codes = np.frompyfunc(lambda r, g, b: f'\033[38;2;{r};{g};{b}m', 3, 1)


class RLC():
    '''游程编码类，支持灰度图像和彩色图像的游程编码'''

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.height, self.width = image.shape[:2]
        try:
            self.channel = image.shape[2]
        except:
            self.channel = 1
        self.rindex_map = {index: [] for index in range(self.height)}

    def encode(self) -> dict:
        '''游程编码'''
        # 遍历图像像素
        for i in range(self.height):
            # 当前行
            row = self.image[i]
            # 灰度图像
            if self.channel == 1:
                # 逐行像素值依次相减找到变化点
                diff = np.diff(row, n=1, axis=-1, prepend=row[0] - 1)
                # 获取当前行变化点的索引
                row_idx = np.where(diff != 0)[0]
            # 彩色图像
            else:
                # 逐行像素值依次相减找到变化点
                diff = np.diff(row, n=1, axis=0, prepend=[row[0] - 1])
                # 获取当前行变化点的索引
                row_idx = np.where(np.any(diff != 0, axis=-1))[0]
            # 获取当前行游程值
            values = row[row_idx]
            # 计算当前行游程值长度
            counts = np.diff(np.append(row_idx, self.width), n=1, axis=-1)
            # 存储编码
            self.rindex_map[i] = list(zip(values, counts))
        return self.rindex_map

    def decode(self, rindex_map: dict = None) -> np.ndarray:
        '''游程解码'''
        if rindex_map is None:
            rindex_map = self.rindex_map
        # 灰度图像
        if self.channel == 1:
            image = np.zeros((self.height, self.width), dtype=self.image.dtype)
        # 彩色图像
        else:
            image = np.zeros((self.height, self.width, self.channel), dtype=self.image.dtype)
        # 遍历行索引表
        for index, value in rindex_map.items():
            # 解码当前行
            if self.channel == 1:
                # 灰度图像
                pixels = np.concatenate([np.repeat(record[0], record[1]) for record in value])
            else:
                # 彩色图像
                pixels = np.concatenate([np.tile(record[0], (record[1], 1)) for record in value])
            # 填充到图像中
            image[index] = pixels
        return image
    

def video2txt(input_video_path: str, output_txt_dir: str, aspect: int, enhance_detail: bool = True, enhance_color: bool = True, color_mode: COLORMODE | str = COLORMODE.CLUSTERCOLOR, enhance_memory: bool = True) -> bool:
    '''
    视频转文本
    ---
    :param input_video_path: 携带视频信息的输入文件路径
    :param output_txt_dir: 输出视频帧文本的文件夹路径
    :param aspect: 预处理为文本前，将图像宽高放缩至不小于该参数
    :param enhance_detail: 是否增强图像细节，为图像边缘添加白色描边。建议在 color_mode 为 CLUSTERCOLOR 时设置为 True，color_mode 为 TRUECOLOR 时设置为 False，可选。默认为 True
    :param enhance_color: 是否增强颜色细节，由 8 色添加到 24 色。仅在 color_mode 为 CLUSTERCOLOR 时使用，可选。默认为 True
    :param color_mode: 颜色模式，为 CLUSTERCOLOR（聚类为 8/24 色的临近色）、TRUECOLOR（24 位真彩色）之一，可选。默认为 CLUSTERCOLOR
    :param enhance_memory: 是否减小内存占用，使用游程编码，开启此选项可显著减少每个视频帧的文本文件大小，但会略微减慢视频转换的处理速度，可选。默认为 True
    :return: 若转换成功，返回 True，否则返回 False
    '''
    # 颜色模式
    if isinstance(color_mode, COLORMODE):
        pass
    elif isinstance(color_mode, str):
        color_mode = color_mode.upper()
    else:
        raise ValueError()
    # 打开视频
    capture = cv.VideoCapture(input_video_path)
    # 若视频打开失败
    if not capture.isOpened():
        logger.error('不受支持的视频文件或错误的视频路径')
        return False
    # 视频名称
    video_name = os.path.basename(input_video_path).split('.')[0]
    # 新建视频帧文本的文件夹
    try:
        os.makedirs(output_txt_dir)
    except FileExistsError as e:
        # 若文件存在，可选择性地省去耗时的重新生成结果文件
        logger.warning('当前结果文件已存在，是否替换该结果文件：\n- 0：不替换\n- 1：替换')
        # 直到用户输入正确
        while True:
            try:
                sign = input()
                assert sign in ['0', '1']
                if sign == '1':
                    shutil.rmtree(output_txt_dir, True)
                    os.makedirs(output_txt_dir)
                else:
                    return True
                break
            except AssertionError as e:
                logger.warning('输入错误，请重新输入')
    # 字符集
    charset = string.digits + string.ascii_letters + string.punctuation
    # 当前视频帧
    current_frame = 1
    logger.info('正在对视频帧进行文本转换，请稍后')
    start = time.time()
    # 处理视频的每一帧图像
    with console.status('文本转换中...', spinner='earth') as status:
        while True:
            s = time.time()
            # 获取视频的每一帧图像
            ret, image = capture.read()
            if not ret:
                logger.warning('相机已断开连接或视频文件中没有更多帧')
                break
            # 从第一帧中获取缩放高宽
            if current_frame == 1:
                # 由于将图像像素替换为符号来表示会导致图像过大，此处重置图像大小
                h, w = image.shape[:2]
                while h > aspect or w > aspect:
                    h //= 2
                    w //= 2
            image = cv.resize(image, (w, h), interpolation=cv.INTER_CUBIC)
            # 增强图像细节
            if enhance_detail:
                # 灰度图
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                # 提取边缘
                edges = cv.Canny(gray, threshold1=100, threshold2=200, L2gradient=True)
                # 添加白色描边
                image[edges == 255] = 255
            # 转换为 rgb
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # 使用游程编码
            if enhance_memory:
                # 游程编码
                rindex_map = RLC(rgb).encode()
                # 符号集
                symbols = ''
                # 遍历行索引表
                for index, value in rindex_map.items():
                    # 解码当前行，仅获取游程值而忽略游程长度（即获取当前行内的唯一连续子像素值）
                    # inhomogeneous shape
                    pixels = np.asarray([record[0] for record in value])    # (-1, 3)
                    counts = np.asarray([record[1] for record in value])    # (-1, )
                    # 真彩色模式
                    if color_mode == COLORMODE.TRUECOLOR or color_mode == COLORMODE.TRUECOLOR.name:
                        # 获取颜色映射代码列表
                        rgb_color_codes = ufunc_pixel2color_codes(pixels[..., 0], pixels[..., 1], pixels[..., 2])
                    # 8/24 色模式
                    elif color_mode == COLORMODE.CLUSTERCOLOR or color_mode == COLORMODE.CLUSTERCOLOR.name:
                        # 获取颜色映射代码列表
                        rgb_color_codes = get_color_codes(pixels, enhance_color=enhance_color)
                    # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                    symbols += ''.join([c + ''.join(random.choices(charset, k=2*count)) for c, count in zip(rgb_color_codes, counts)]) + '\n'
                # 将二值化图像替换为符号
                with open(f'{output_txt_dir}/{video_name}_{current_frame:0>7d}.txt', 'w') as f:
                    f.write(symbols)
            else:
                # 真彩色模式
                if color_mode == COLORMODE.TRUECOLOR or color_mode == COLORMODE.TRUECOLOR.name:
                    # 获取颜色映射代码列表
                    rgb_color_codes = ufunc_pixel2color_codes(rgb[..., 0], rgb[..., 1], rgb[..., 2]).reshape(-1)
                # 8/24 色模式
                elif color_mode == COLORMODE.CLUSTERCOLOR or color_mode == COLORMODE.CLUSTERCOLOR.name:
                    # 获取颜色映射代码列表
                    rgb_color_codes = get_color_codes(rgb.reshape(-1, 3), enhance_color=enhance_color)
                # 将二值化图像替换为符号
                with open(f'{output_txt_dir}/{video_name}_{current_frame:0>7d}.txt', 'w') as f:
                    # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                    symbols = [
                        c + ''.join(random.choices(charset, k=2)) if (i + 1) % w != 0 else '\n' + c + ''.join(random.choices(charset, k=2)) for i, c in enumerate(rgb_color_codes)
                    ]
                    f.write(''.join(symbols))
            e = time.time()
            logger.info(f'第 {current_frame} 帧文本转换完成，处理时间：{e - s}s')
            current_frame += 1
    capture.release()
    end = time.time()
    logger.info(f'文本转换完成，处理时间：{end - start}s，平均每帧 {(end - start) / (current_frame - 1)}s')
    return True


def show(input_txt_dir: str, input_audio_path: str, title: str = '', interval: float = 0) -> None:
    '''
    在命令行中打印彩色视频
    ---
    :param input_txt_dir: 输出视频帧文本的文件夹路径
    :param input_audio_path: 仅携带音频信息的输入文件路径
    :param title: 命令行窗口标题，可选。默认为 ''
    :param interval: 控制每次打印的时间间隔，可选。默认为 0
    :return: None
    '''
    # 修复 Windows 控制台的颜色问题
    just_fix_windows_console()
    # 文本文件
    files = os.listdir(input_txt_dir)
    # 文本列表
    symbols = []
    # 依次读取每一个文本文件内容到列表中
    logger.info('正在读取文件中，请稍后')
    start = time.time()
    for file in track(files, description='读取文件中...'):
        with open(os.path.join(input_txt_dir, file), 'r', encoding='utf-8') as f:
            symbols.append(f.read())
    end = time.time()
    logger.info(f'读取文件完成，处理时间：{end - start}s，平均每帧 {(end - start) / len(files)}s')
    # 播放音乐
    playsound(input_audio_path, block=False)
    # 设置命令行窗口标题
    print(set_title(title))
    # 将光标移动到起始位置
    pos = Cursor.POS()
    # 依次打印每个文本文件内容
    start_time = time.time()
    for i, symbol in enumerate(symbols):
        # 当前时间
        current_time = time.time()
        # 实际经过时间
        elapsed_time = current_time - start_time
        # 每帧预期时间
        expected_time = i * interval
        # 时间差
        delta_time = elapsed_time - expected_time
        # 若生成文本文件内容过小，则 print 操作相对快速
        if delta_time < 0:  # 减慢打印速度
            # 将光标移动到起始位置，原地更新内容，减少屏幕闪烁
            print(pos + symbol, flush=True)
            time.sleep(interval - delta_time)
        # 若生成文本文件内容过大，则 print 操作相对耗时
        else:  # 跳过当前帧，以实现抽帧效果，与音频同步
            continue
    # 清屏
    os.system('cls')
    # reset all settings
    print(Fore.RESET + Back.RESET + Style.RESET_ALL)
