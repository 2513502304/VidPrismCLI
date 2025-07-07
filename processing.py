from colorama import just_fix_windows_console, Fore, Back, Style, Cursor
from colorama.ansi import code_to_chars, set_title, clear_screen, clear_line
from playsound import playsound
from rich.progress import track
import shutil
import os
import time
import string
import random
import numpy as np
import cv2 as cv

from utils import logger, console, safe_imread, safe_imwrite
from color import pixel2cluster_color, pixel2cluster_color_code, pixel2true_color_code, COLORMODE
from media import split_va, merge_va
from rlc import RLC


def ink_ratio(
    char: str,
    fontFace: int = cv.FONT_HERSHEY_SIMPLEX,
    fontScale: float = 1.0,
    thickness: int = 1,
) -> float:
    """
    计算字符在其外接矩形（bounding box）内实际被“墨水”覆盖的面积与整个矩形面积的比值

    Args:
        char (str): 输入文本字符
        fontFace (int, optional): 要使用的字体，请参阅 cv.HersheyFonts. Defaults to cv.FONT_HERSHEY_SIMPLEX.
        fontScale (float, optional): 字体比例因子乘以特定于字体的基本大小. Defaults to 1.0.
        thickness (int, optional): 用于呈现文本的线条粗细. Defaults to 1.

    Returns:
        float: 字符在其外接矩形（bounding box）内实际被“墨水”覆盖的面积与整个矩形面积的比值
    """
    # 计算文本字符串的宽度和高度，返回包含指定文本的矩形框大小以及相对于最底部文本点的基线的 y 坐标
    size, baseline = cv.getTextSize(char, fontFace=fontFace, fontScale=fontScale, thickness=thickness)
    # 创建一个黑色图像，大小为包含整个字符的最小外接矩形
    image = np.zeros((size[1] + baseline + thickness // 2, size[0]), np.uint8)
    # 绘制文本
    cv.putText(image, char, (0, size[1]), fontFace=fontFace, fontScale=fontScale, color=255, thickness=thickness, lineType=cv.LINE_AA, bottomLeftOrigin=False)
    # 计算墨水覆盖面积
    ink_area = cv.countNonZero(image)
    # 计算外接矩形面积
    bounding_box_area = image.shape[0] * image.shape[1]
    return ink_area / bounding_box_area if bounding_box_area > 0 else 0.0


def draw_text(
    text: str,
    color: np.ndarray = None,
    fontFace: int = cv.FONT_HERSHEY_SIMPLEX,
    fontScale: float = 1.0,
    thickness: int = 1,
) -> np.ndarray:
    """
    创建一个包含全部文本的最小外接矩形（bounding box）并绘制文本，自动处理换行符 '\\n'

    Args:
        text (str): 输入文本字符串
        color (np.ndarray): 文本颜色，形状为 (M, N, 3) 的 BGR 颜色数组. Defaults to None.
        fontFace (int, optional): 要使用的字体，请参阅 cv.HersheyFonts. Defaults to cv.FONT_HERSHEY_SIMPLEX.
        fontScale (float, optional): 字体比例因子乘以特定于字体的基本大小. Defaults to 1.0.
        thickness (int, optional): 用于呈现文本的线条粗细. Defaults to 1.

    Returns:
        np.ndarray: 绘制后的图像
    """
    # 计算文本字符串的宽度和高度，返回包含指定文本的矩形框大小以及相对于最底部文本点的基线的 y 坐标
    size, baseline = cv.getTextSize('x', fontFace=fontFace, fontScale=fontScale, thickness=thickness)
    # 将文本按行分割
    lines = text.split('\n')
    # 删除最后一个空行（此行为 symbols 使用列表推导式构造的一个特例）
    lines.pop(-1)
    # 每行文本的宽度，包含线条粗细
    letter_width, line_width = size[0], size[0] * len(lines[0])
    # 每行文本的高度，包含基线和线条粗细
    letter_height = line_height = size[1] + baseline + thickness // 2  #!由于 OpenCV 的 putText 函数不支持多行文本，因此无论文本中有多少行，都会将其视为一行进行绘制，因此不需要考虑文本宽度，而文本高度则始终为一行
    # 创建一个黑色图像，大小为包含整个字符的最小外接矩形
    image = np.zeros((line_height * len(lines), line_width, 3), np.uint8)
    # 若未指定颜色，则使用默认颜色
    if color is None:
        # 默认颜色为白色
        color = np.full((len(lines), len(lines[0]), 3), 255, dtype=np.uint8)
    # 绘制图像起始点
    x, y = 0, line_height  # 从左下角开始绘制文本
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            # 绘制文本
            cv.putText(
                image,
                char,
                (letter_width * j, y),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color[i, j].tolist(),
                thickness=thickness,
                lineType=cv.LINE_AA,
                bottomLeftOrigin=False,
            )
        # 递增 y 坐标
        y += line_height
    return image


def video2symbol(
    input_video_path: str,
    output_symbol_dir: str,
    aspect: int,
    color_mode: COLORMODE | str = COLORMODE.TRUECOLOR,
    enhance_detail: bool = False,
    enhance_lightness: bool = True,
    enhance_color: bool = False,
    enhance_memory: bool = True,
) -> bool:
    '''
    视频转字符画文本
    ---
    :param input_video_path: 携带视频信息的输入文件路径
    :param output_symbol_dir: 输出视频帧文本的文件夹路径
    :param aspect: 预处理为文本前，将图像宽高放缩至不小于该参数
    :param color_mode: 颜色模式，为 CLUSTERCOLOR（聚类为 8/24 色的临近色）、TRUECOLOR（24 位真彩色）之一，可选。默认为 CLUSTERCOLOR
    :param enhance_detail: 是否增强图像细节，为图像边缘添加白色描边。建议在 color_mode 为 CLUSTERCOLOR 时设置为 True，color_mode 为 TRUECOLOR 时设置为 False，可选。默认为 True
    :param enhance_lightness: 是否增强亮度细节，按照图像像素亮度排序字符集，优先使用亮度较高的字符，可选。默认为 True
    :param enhance_color: 是否增强颜色细节，由 8 色添加到 24 色。仅在 color_mode 为 CLUSTERCOLOR 时使用，可选。默认为 True
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
        os.makedirs(output_symbol_dir)
    except FileExistsError as e:
        # 若文件存在，可选择性地省去耗时的重新生成结果文件
        logger.warning('当前结果文件已存在，是否替换该结果文件：\n- 0：不替换\n- 1：替换')
        # 直到用户输入正确
        while True:
            try:
                sign = input()
                assert sign in ['0', '1']
                if sign == '1':
                    shutil.rmtree(output_symbol_dir, True)
                    os.makedirs(output_symbol_dir)
                else:
                    return True
                break
            except AssertionError as e:
                logger.warning('输入错误，请重新输入')
    # 字符集
    charset = string.digits + string.ascii_letters + string.punctuation
    # 增强亮度细节
    if enhance_lightness:
        # 计算每个字符的墨水覆盖率
        char_ratios = {char: ink_ratio(char) for char in charset}
        # 按照墨水覆盖率从低到高排序
        sorted_char_ratios = sorted(char_ratios.items(), key=lambda item: item[1], reverse=False)
        # 更新字符集
        new_charset = np.asarray([char for char, _ in sorted_char_ratios])
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
            # 图像像素个数
            pixel_count = h * w
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
            # 增强亮度细节
            if enhance_lightness:
                # 取后处理的灰度图像得到亮度值
                lightness = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
                # 将亮度值归一化到 [0, 1] 范围，并映射到字符集的索引
                char_index = ((lightness - lightness.min()) / (lightness.max() - lightness.min()) * (len(new_charset) - 1)).astype(int)
                # 将图像转换为字符，shape=(-1, )
                char_image = new_charset[char_index].reshape(-1)
                # 当前字符位置
                char_position = 0
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
                    pixels = np.asarray([record[0] for record in value])  # (-1, 3)
                    counts = np.asarray([record[1] for record in value])  # (-1, )
                    # 真彩色模式
                    if color_mode == COLORMODE.TRUECOLOR or color_mode == COLORMODE.TRUECOLOR.name:
                        # 获取颜色映射代码列表
                        rgb_color_codes = pixel2true_color_code(pixels[..., 0], pixels[..., 1], pixels[..., 2])
                    # 8/24 色模式
                    elif color_mode == COLORMODE.CLUSTERCOLOR or color_mode == COLORMODE.CLUSTERCOLOR.name:
                        # 获取颜色映射代码列表
                        rgb_color_codes = pixel2cluster_color_code(pixels, enhance_color=enhance_color)
                    # 增强亮度细节
                    if enhance_lightness:
                        # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                        symbols += ''.join(
                            [c + ''.join(np.repeat(
                                char_image[char_position:(char_position := char_position + count)],
                                repeats=2,
                            )) for c, count in zip(rgb_color_codes, counts)]) + '\n'
                    else:
                        # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                        symbols += ''.join([c + ''.join(random.choices(charset, k=2 * count)) for c, count in zip(rgb_color_codes, counts)]) + '\n'
            # 不使用游程编码优化内存
            else:
                # 真彩色模式
                if color_mode == COLORMODE.TRUECOLOR or color_mode == COLORMODE.TRUECOLOR.name:
                    # 获取颜色映射代码列表
                    rgb_color_codes = pixel2true_color_code(rgb[..., 0], rgb[..., 1], rgb[..., 2]).reshape(-1)
                # 8/24 色模式
                elif color_mode == COLORMODE.CLUSTERCOLOR or color_mode == COLORMODE.CLUSTERCOLOR.name:
                    # 获取颜色映射代码列表
                    rgb_color_codes = pixel2cluster_color_code(rgb.reshape(-1, 3), enhance_color=enhance_color)
                # 增强亮度细节
                if enhance_lightness:
                    # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                    symbols = ''.join([
                        c + ''.join(np.repeat(
                            char_image[char_position:(char_position := char_position + 1)],
                            repeats=2,
                        )) if (i + 1) % w != 0 else '\n' + c + ''.join(np.repeat(
                            char_image[char_position:(char_position := char_position + 1)],
                            repeats=2,
                        )) for i, c in enumerate(rgb_color_codes)
                    ])
                else:
                    # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                    symbols = ''.join([
                        c + ''.join(random.choices(charset, k=2)) if (i + 1) % w != 0 else '\n' + c + ''.join(random.choices(charset, k=2)) for i, c in enumerate(rgb_color_codes)
                    ])
            # 保存文本文件
            with open(f'{output_symbol_dir}/{video_name}_{current_frame:0>7d}.txt', 'w') as f:
                f.write(symbols)
            e = time.time()
            logger.info(f'第 {current_frame} 帧文本转换完成，处理时间：{e - s}s')
            current_frame += 1
    capture.release()
    end = time.time()
    logger.info(f'文本转换完成，处理时间：{end - start}s，平均每帧 {(end - start) / (current_frame - 1)}s')
    return True


def symbol_show(
    input_symbol_dir: str,
    input_audio_path: str,
    title: str = '',
    interval: float = 0,
) -> None:
    '''
    在命令行中打印字符画视频
    ---
    :param input_symbol_dir: 输出视频帧文本的文件夹路径
    :param input_audio_path: 仅携带音频信息的输入文件路径
    :param title: 命令行窗口标题，可选。默认为 ''
    :param interval: 控制每次打印的时间间隔，可选。默认为 0
    :return: None
    '''
    # 修复 Windows 控制台的颜色问题
    just_fix_windows_console()
    # 设置命令行窗口标题
    print(set_title(title))
    # 文本文件
    files = os.listdir(input_symbol_dir)
    # 文本列表
    symbols = []
    # 依次读取每一个文本文件内容到列表中
    logger.info('正在读取文件中，请稍后')
    start = time.time()
    for file in track(files, description='读取文件中...'):
        with open(os.path.join(input_symbol_dir, file), 'r', encoding='utf-8') as f:
            symbols.append(f.read())
    end = time.time()
    logger.info(f'读取文件完成，处理时间：{end - start}s，平均每帧 {(end - start) / len(files)}s')
    # 播放音乐
    playsound(input_audio_path, block=False)
    # 将光标移动到起始位置字符
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


def video2media(
    input_video_path: str,
    output_media_dir: str,
    aspect: int,
    color_mode: COLORMODE | str = COLORMODE.TRUECOLOR,
    enhance_detail: bool = False,
    enhance_lightness: bool = True,
    enhance_color: bool = False,
) -> bool:
    '''
    视频转字符画图像
    ---
    :param input_video_path: 携带视频信息的输入文件路径
    :param output_media_dir: 输出视频帧文本的文件夹路径
    :param aspect: 预处理为文本前，将图像宽高放缩至不小于该参数
    :param color_mode: 颜色模式，为 CLUSTERCOLOR（聚类为 8/24 色的临近色）、TRUECOLOR（24 位真彩色）之一，可选。默认为 CLUSTERCOLOR
    :param enhance_detail: 是否增强图像细节，为图像边缘添加白色描边。建议在 color_mode 为 CLUSTERCOLOR 时设置为 True，color_mode 为 TRUECOLOR 时设置为 False，可选。默认为 True
    :param enhance_lightness: 是否增强亮度细节，按照图像像素亮度排序字符集，优先使用亮度较高的字符，可选。默认为 True
    :param enhance_color: 是否增强颜色细节，由 8 色添加到 24 色。仅在 color_mode 为 CLUSTERCOLOR 时使用，可选。默认为 True
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
        os.makedirs(output_media_dir)
    except FileExistsError as e:
        # 若文件存在，可选择性地省去耗时的重新生成结果文件
        logger.warning('当前结果文件已存在，是否替换该结果文件：\n- 0：不替换\n- 1：替换')
        # 直到用户输入正确
        while True:
            try:
                sign = input()
                assert sign in ['0', '1']
                if sign == '1':
                    shutil.rmtree(output_media_dir, True)
                    os.makedirs(output_media_dir)
                else:
                    return True
                break
            except AssertionError as e:
                logger.warning('输入错误，请重新输入')
    # 字符集
    charset = string.digits + string.ascii_letters + string.punctuation
    # 增强亮度细节
    if enhance_lightness:
        # 计算每个字符的墨水覆盖率
        char_ratios = {char: ink_ratio(char) for char in charset}
        # 按照墨水覆盖率从低到高排序
        sorted_char_ratios = sorted(char_ratios.items(), key=lambda item: item[1], reverse=False)
        # 更新字符集
        new_charset = np.asarray([char for char, _ in sorted_char_ratios])
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
            # 图像像素个数
            pixel_count = h * w
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
            # 真彩色模式
            if color_mode == COLORMODE.TRUECOLOR or color_mode == COLORMODE.TRUECOLOR.name:
                # 获取颜色映射列表，shape=(h, w, 3)
                rgb_color = rgb
                # 转换为待填充的 bgr 颜色，shape=(h, w, 3)
                bgr_color = image
            # 8/24 色模式
            elif color_mode == COLORMODE.CLUSTERCOLOR or color_mode == COLORMODE.CLUSTERCOLOR.name:
                # 获取颜色映射列表，shape=(h, w, 3)
                rgb_color = np.asarray(pixel2cluster_color(rgb.reshape(-1, 3), enhance_color=enhance_color)).reshape(*rgb.shape[:2], 3)
                # 转换为待填充的 bgr 颜色，shape=(h, w, 3)
                bgr_color = cv.cvtColor(rgb_color, cv.COLOR_RGB2BGR)
            # 增强亮度细节
            if enhance_lightness:
                # 取后处理的灰度图像得到亮度值
                lightness = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
                # 将亮度值归一化到 [0, 1] 范围，并映射到字符集的索引
                char_index = ((lightness - lightness.min()) / (lightness.max() - lightness.min()) * (len(new_charset) - 1)).astype(int)
                # 将图像转换为字符，shape=(-1, )
                char_image = new_charset[char_index].reshape(-1)
                # 当前字符位置
                char_position = 0
                # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                symbols = ''.join([
                    ''.join(np.repeat(
                        char_image[char_position:(char_position := char_position + 1)],
                        repeats=2,
                    )) if (i + 1) % w != 0 else '\n' + ''.join(np.repeat(
                        char_image[char_position:(char_position := char_position + 1)],
                        repeats=2,
                    )) for i in range(pixel_count)
                ])
                bgr_color = np.repeat(bgr_color, 2, axis=1)  # shape=(h, 2 * w, 3)
                # 绘制文本图像
                text_image = draw_text(symbols, bgr_color)
            else:
                # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                symbols = ''.join([''.join(random.choices(charset, k=2)) if (i + 1) % w != 0 else '\n' + ''.join(random.choices(charset, k=2)) for i in range(pixel_count)])
                bgr_color = np.repeat(bgr_color, 2, axis=1)  # shape=(h, 2 * w, 3)
                # 绘制文本图像
                text_image = draw_text(symbols, bgr_color)
            # 保存图像文件
            safe_imwrite(f'{output_media_dir}/{video_name}_{current_frame:0>7d}.jpg', text_image)
            e = time.time()
            logger.info(f'第 {current_frame} 帧图像转换完成，处理时间：{e - s}s')
            current_frame += 1
    capture.release()
    end = time.time()
    logger.info(f'图像转换完成，处理时间：{end - start}s，平均每帧 {(end - start) / (current_frame - 1)}s')
    return True
