from rich.console import Console
from rich.logging import RichHandler
import logging

import numpy as np
import cv2 as cv

import os

# 控制台对象
console = Console()

# 日志记录
logging.basicConfig(
    format='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[RichHandler(console=console, )],
)
logger = logging.getLogger('VidPrismCLI')


def safe_imread(file: str) -> np.ndarray:
    """
    支持非 ASCII 编码路径的图像读取

    Args:
        file (str): 要读取的图像路径

    Returns:
        np.ndarray: 读取的 BGR 图像
    """
    # 以 uint8 数据类型，读取并解析二进制图像文件中的数据
    buffer = np.fromfile(file, dtype=np.uint8)
    # 解码数据为图像
    image = cv.imdecode(buffer, cv.IMREAD_COLOR)
    assert image is not None, f"Failed to load image: {file}"
    return image


def safe_imwrite(file: str, image: np.ndarray) -> True:
    """
    支持非 ASCII 编码路径的图像写入

    Args:
        file (str): 要写入的图像路径
        image (np.ndarray): 要写入的图像

    Returns:
        True: 是否写入成功
    """
    # 获取图像扩展名
    root, ext = os.path.splitext(file)
    # 将图像编码为指定文件格式的二进制数据
    ret, buffer = cv.imencode(ext, image)
    # 编码成功
    if ret:
        # 写入数据
        buffer.tofile(file)
        return True
    # 编码失败
    else:
        return False
