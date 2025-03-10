# 仅携带视频信息的输入文件路径
video_path = './Data/《原神》角色预告-「茜特菈莉：星之眼的注视」.mp4'

# 仅携带音频信息的输入文件路径
audio_path = './Data/《原神》角色预告-「茜特菈莉：星之眼的注视」.m4a'

# 命令行窗口标题
title = '《原神》角色预告-「茜特菈莉：星之眼的注视」'

# 携带音视频信息的输入文件路径，若给出该参数，则忽略 video_path 与 audio_path 参数
file_path = None

# 输出结果文件目录
# - symbol：存储以文本形式的结果文件
# - video：若 file_path 不为 None 且有效，则存储拆分原视频后，仅携带视频信息的视频文件
# - audio：若 file_path 不为 None 且有效，则存储拆分原视频后，仅携带音频信息的音频文件
output_dir = './Data'

# 预处理为文本前，将图像宽高放缩至不小于该参数
# 建议不要超过 256，否则将在命令行中无法完全显示所有输出
# 同时，若该值过小，则会导致输出质量较差
aspect = 256

# 是否增强图像细节，为图像边缘添加白色描边。建议在 color_mode 为 CLUSTERCOLOR 时设置为 True，color_mode 为 TRUECOLOR 时设置为 False
enhance_detail = False

# 是否增强颜色细节，由 8 色添加到 24 色。仅在 color_mode 为 CLUSTERCOLOR 时使用
enhance_color = True

# 颜色模式，为 CLUSTERCOLOR（聚类为 8/24 色的临近色）、TRUECOLOR（24 位真彩色）之一
color_mode = 'TRUECOLOR'

# 是否减小内存占用，使用游程编码，开启此选项可显著减少每个视频帧的文本文件大小，但会略微减慢视频转换的处理速度
enhance_memory = True
