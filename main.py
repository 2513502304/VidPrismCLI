'''
Author: 未来可欺 2513502304@qq.com
Date: 2025-01-24 15:27:02
LastEditors: 未来可欺 2513502304@qq.com
LastEditTime: 2025-01-24 19:56:29
Description: 命令行中打印的彩色视频
'''

import cv2 as cv
import os
from processing import split_va, merge_va, video2txt, show
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
    ):
        # 在命令行中打印彩色视频
        show(
            input_txt_dir=txt_dir,
            input_audio_path=audio_path,
            interval=interval,
        )


if __name__ == "__main__":
    main()
