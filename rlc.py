import numpy as np


class RLC():
    '''游程编码类，支持灰度图像和彩色图像的游程编码'''

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.height, self.width = image.shape[:2]
        try:
            self.channel = image.shape[2]
        except IndexError as e:
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
