'''
Author: 未来可欺 2513502304@qq.com
Date: 2025-01-24 15:27:02
LastEditors: 未来可欺 2513502304@qq.com
LastEditTime: 2025-03-07 02:07:31
Description: 命令行中打印的彩色视频
'''

import cv2 as cv
import os
from processing import split_va, merge_va, show
# try:
#     from cy_video2txt import video2txt  # Cython 代码相比于 Python 代码具有更稳定的性能，每次运行的时长标准差相较于更小
# except ModuleNotFoundError as e:
#     from processing import video2txt
from processing import video2txt
from utils import logger
import settings


def main():
    # 打开视频
    capture = cv.VideoCapture(settings.video_path)
    # 若视频打开失败
    if not capture.isOpened():
        logger.error('不受支持的视频文件或错误的视频路径')
        return
    # 获取视频帧数
    fps = capture.get(cv.CAP_PROP_FPS)
    # 每帧间隔
    interval = 1 / fps
    capture.release()
    # file_path 不为 None
    if settings.file_path is not None:
        file_name = os.path.basename(settings.file_path)
        video_dir = os.path.join(settings.output_dir, 'video')
        audio_dir = os.path.join(settings.output_dir, 'audio')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        video_path = os.path.join(video_dir, file_name)
        audio_path = os.path.join(audio_dir, file_name)
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
    txt_dir = os.path.join(settings.output_dir, 'symbol')
    # 视频转文本
    if video2txt(
            input_video_path=video_path,
            output_txt_dir=txt_dir,
            aspect=settings.aspect,
            enhance_detail=settings.enhance_detail,
            enhance_color=settings.enhance_color,
            color_mode=settings.color_mode,
            enhance_memory=settings.enhance_memory,
    ):
        # 在命令行中打印彩色视频
        show(
            input_txt_dir=txt_dir,
            input_audio_path=audio_path,
            title=settings.title,
            interval=interval,
        )


if __name__ == "__main__":
    main()
