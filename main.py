'''
Author: 未来可欺 2513502304@qq.com
Date: 2025-01-24 15:27:02
LastEditors: 未来可欺 2513502304@qq.com
LastEditTime: 2025-07-15 13:53:57
Description: 命令行中打印的彩色视频
'''

import cv2 as cv
import os
from processing import split_va, merge_va, symbol_show
# try:
#     from cy_video2txt import video2txt  # Cython 代码相比于 Python 代码具有更稳定的性能，每次运行的时长标准差相较于更小
# except ModuleNotFoundError as e:
#     from processing import video2txt
from processing import video2symbol, video2media
from utils import logger
import settings


def main():
    # 若 file_path 不为 None，拆分音视频并赋值 video_path 与 audio_path 参数
    if settings.file_path is not None:
        file_name = os.path.basename(settings.file_path)
        video_dir = os.path.join(settings.output_dir, 'video')
        audio_dir = os.path.join(settings.output_dir, 'audio')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        video_path = os.path.join(video_dir, file_name.split('.')[0] + '.mp4')
        audio_path = os.path.join(audio_dir, file_name.split('.')[0] + '.m4a')
        if not os.path.exists(video_path) and not os.path.exists(video_path):
            # 拆分音视频
            split_va(
                input_file_path=settings.file_path,
                output_video_path=video_path,
                output_audio_path=audio_path,
            )
    else:
        video_path = settings.video_path
        audio_path = settings.audio_path
    # 打开视频
    capture = cv.VideoCapture(video_path)
    # 若视频打开失败
    if not capture.isOpened():
        logger.error('不受支持的视频文件或错误的视频路径')
        return
    # 获取视频帧数
    fps = capture.get(cv.CAP_PROP_FPS)
    # 每帧间隔
    interval = 1 / fps
    capture.release()
    # 字符画文本输出
    if settings.output_mode.lower() == 'symbol':
        symbol_dir = os.path.join(settings.output_dir, 'symbol')
        # 视频转字符画文本
        if video2symbol(
                input_video_path=video_path,
                output_symbol_dir=symbol_dir,
                aspect=settings.aspect,
                color_mode=settings.color_mode,
                enable_background=settings.enable_background,
                enhance_detail=settings.enhance_detail,
                enhance_lightness=settings.enhance_lightness,
                enhance_color=settings.enhance_color,
                enhance_memory=settings.enhance_memory,
        ):
            # 在命令行中打印字符画视频
            symbol_show(
                input_symbol_dir=symbol_dir,
                input_audio_path=audio_path,
                title=settings.title,
                interval=interval,
            )
    # 字符画图像以及视频输出
    elif settings.output_mode.lower() == 'media':
        media_dir = os.path.join(settings.output_dir, 'media')
        # 视频转字符画图像
        if video2media(
                input_video_path=video_path,
                output_media_dir=media_dir,
                aspect=settings.aspect,
                color_mode=settings.color_mode,
                enable_background=settings.enable_background,
                enhance_detail=settings.enhance_detail,
                enhance_lightness=settings.enhance_lightness,
                enhance_color=settings.enhance_color,
        ):
            # TODO：图像转视频
            pass


if __name__ == "__main__":
    main()
