from colorama import just_fix_windows_console, Fore, Back, Style, Cursor
from playsound import playsound
from typing import Sequence, MutableSequence, Final
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
BLACK: Final = (12, 12, 12)  # #0c0c0c
RED: Final = (197, 15, 31)  # #c50f1f
GREEN: Final = (19, 161, 14)  # #13a10e
YELLOW: Final = (193, 156, 0)  # #c19c00
BLUE: Final = (0, 55, 218)  # #0037da
MAGENTA: Final = (136, 23, 152)  # #881798
CYAN: Final = (58, 150, 221)  # #3a96dd
WHILE: Final = (193, 193, 193)  # #c1c1c1

# 颜色选择器
switcher: Final = {
    BLACK: Fore.BLACK,
    RED: Fore.RED,
    GREEN: Fore.GREEN,
    YELLOW: Fore.YELLOW,
    BLUE: Fore.BLUE,
    MAGENTA: Fore.MAGENTA,
    CYAN: Fore.CYAN,
    WHILE: Fore.WHITE,
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
    cmds = [f'ffmpeg -i {input_file_path} -an -c:v {output_video_path}', f'ffmpeg -i {input_file_path} -vn -c:a {output_audio_path}']
    for cmd in cmds:
        command = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            check=True,
        )
        args = ' '.join(command.args)
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
    cmds = [f'ffmpeg -i {input_video_path} -i {input_audio_path} -c:v copy -c:a copy {output_file_path}']
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


def get_color_code(rgb: Sequence[int]) -> str:
    '''
    获取颜色映射代码
    ---
    :param rgb: rgb 列表
    :return: 颜色映射代码
    '''
    # 输入的 rgb 颜色
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3)
    # 将 rgb 颜色转换为 lab 颜色
    lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB).reshape(3)
    # rgb 颜色列表
    colors = np.asarray(list(switcher.keys()), np.uint8).reshape(-1, 1, 3)
    # 将 rgb 颜色列表转换为 lab 颜色列表
    lab_colors = cv.cvtColor(colors, cv.COLOR_RGB2LAB).reshape(-1, 3)
    # 计算所有颜色之间的欧氏距离
    distances = np.linalg.norm(lab_colors - lab, axis=1)
    # 找到最小距离的索引
    min_index = np.argmin(distances)
    # 最佳匹配颜色
    best_color = colors[min_index].reshape(3)
    # 返回颜色映射代码
    return switcher.get(tuple(best_color), Fore.BLACK)


def video2txt(input_video_path: str, output_txt_dir: str, aspect: int, enhance_detail: bool) -> bool:
    '''
    视频转文本
    ---
    :param input_video_path: 携带视频信息的输入文件路径
    :param output_txt_dir: 输出视频帧文本的文件夹路径
    :param aspect: 预处理为文本前，将图像宽高放缩至不小于该参数
    :param enhance_detail: 是否增强图像细节，为图像边缘添加白色描边
    :return: 若转换成功，返回 True，否则返回 False
    '''
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
            # 将二值化图像替换为符号
            with open(f'{output_txt_dir}/{video_name}_{current_frame:0>7d}.txt', 'w') as f:
                for height in range(h):
                    line = ''
                    for width in range(w):
                        # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                        line += get_color_code(rgb[height][width]) + ''.join(random.choices(charset, k=2))
                    f.write(line + '\n')
            e = time.time()
            logger.info(f'第 {current_frame} 帧文本转换完成，处理时间：{e - s}s')
            current_frame += 1
    capture.release()
    end = time.time()
    logger.info(f'文本转换完成，处理时间：{end - start}s')
    return True


def show(input_txt_dir: str, input_audio_path: str, interval: float = 0) -> None:
    '''
    在命令行中打印彩色视频
    ---
    :param output_txt_dir: 输出视频帧文本的文件夹路径
    :param input_audio_path: 仅携带音频信息的输入文件路径
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
    logger.info(f'读取文件完成，处理时间：{end - start}s')
    # 播放音乐
    playsound(input_audio_path, block=False)
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
            os.system('cls')
            print(symbol, flush=True)
            time.sleep(interval - delta_time)
        # 若生成文本文件内容过大，则 print 操作相对耗时
        else:  # 跳过当前帧，以实现抽帧效果，与音频同步
            continue
    # reset all settings
    print(Fore.RESET, Back.RESET, Style.RESET_ALL)
