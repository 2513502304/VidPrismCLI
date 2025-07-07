from typing import Literal

# 仅携带视频信息的输入文件路径
video_path: str | None = './Data/《原神》角色预告-「茜特菈莉：星之眼的注视」.mp4'

# 仅携带音频信息的输入文件路径
audio_path: str | None = './Data/《原神》角色预告-「茜特菈莉：星之眼的注视」.m4a'

# 命令行窗口标题
title: str | None = '《原神》角色预告-「茜特菈莉：星之眼的注视」'

# 携带音视频信息的输入文件路径，若给出该参数，则忽略 video_path 与 audio_path 参数
file_path: str | None = None

# 输出模式，'symbol' 为字符画文本输出，用于在命令行中显示；'media' 为字符画图像以及视频输出，用于在播放器中显示
output_mode: Literal['symbol', 'media'] = 'symbol'

# 输出结果文件目录
# - symbol：若 output_mode 为 'symbol'，则存储文本格式的字符画结果文件
# - media：若 output_mode 为 'media'，则存储图像以及视频格式的字符画结果文件
# - video：若 file_path 不为 None 且该路径有效，则存储拆分原视频后，仅携带视频信息的视频文件
# - audio：若 file_path 不为 None 且该路径有效，则存储拆分原视频后，仅携带音频信息的音频文件
output_dir: str = './Data'

# 预处理为文本前，将图像宽高放缩至不小于该参数
# 建议不要超过 256，否则将在命令行中无法完全显示所有输出
# 同时，若该值过小，则会导致输出质量较差
aspect: int = 256

# 是否增强图像细节，为图像边缘添加白色描边。建议在 color_mode 为 CLUSTERCOLOR 时设置为 True，color_mode 为 TRUECOLOR 时设置为 False
enhance_detail: bool = False

# 是否增强亮度细节，按照图像像素亮度排序字符集，优先使用亮度较高的字符
enhance_lightness: bool = True

# 是否增强颜色细节，由 8 色添加到 24 色。仅在 color_mode 为 CLUSTERCOLOR 时使用
enhance_color: bool = True

# 颜色模式，为 CLUSTERCOLOR（聚类为 8/24 色的临近色）、TRUECOLOR（24 位真彩色）之一
color_mode: Literal['CLUSTERCOLOR', 'TRUECOLOR'] = 'TRUECOLOR'

# 是否减小内存占用，使用游程编码，开启此选项可显著减少每个视频帧的文本文件大小，但会略微减慢视频转换的处理速度
enhance_memory: bool = True
