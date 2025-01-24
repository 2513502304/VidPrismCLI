from colorama import just_fix_windows_console, Fore, Back, Style, Cursor
from playsound import playsound
from typing import Sequence, MutableSequence, Final
import subprocess
import numpy as np
import os
import shutil
import cv2 as cv
import time
import string
import random
from utils import logger

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
    rgb = np.array(rgb)
    # 颜色列表
    colors = np.array(list(switcher.keys()))
    # 最佳匹配颜色
    best_color = np.array([np.inf, np.inf, np.inf])
    for c in colors:
        if np.sum(np.abs(c - rgb)) < np.sum(np.abs(best_color - rgb)):
            best_color = c
    # 返回颜色映射代码
    return switcher.get(tuple(best_color), Fore.BLACK)


def video2txt(input_video_path: str, output_txt_dir: str, aspect: int = 64) -> bool:
    '''
    视频转文本
    ---
    :param input_video_path: 携带视频信息的输入文件路径
    :param output_txt_dir: 输出视频帧文本的文件夹路径
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
    while True:
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
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # 将二值化图像替换为符号
        with open(f'{output_txt_dir}/{video_name}_{current_frame:0>7d}.txt', 'w') as f:
            for height in range(h):
                line = ''
                for width in range(w):
                    # 由于在 cmd 中字符的高度是宽度的两倍，这里使用两个字符进行填充
                    line += get_color_code(rgb_image[height][width]) + ''.join(random.choices(charset, k=2))
                f.write(line + '\n')
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
    for file in files:
        with open(os.path.join(input_txt_dir, file), 'r', encoding='utf-8') as f:
            symbols.append(f.read())
    end = time.time()
    logger.info(f'读取文件完成，处理时间：{end - start}')
    # 播放音乐
    playsound(input_audio_path, block=False)
    # 依次打印每个文本文件内容
    for symbol in symbols:
        start = time.time()
        print(symbol, flush=True)
        os.system('cls')
        end = time.time()
        ptime = end - start
        if interval > ptime:
            # 减慢打印速度
            time.sleep(interval - ptime)
