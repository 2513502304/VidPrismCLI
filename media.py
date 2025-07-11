import subprocess

from utils import logger, console


def split_va(
    input_file_path: str,
    output_video_path: str,
    output_audio_path: str,
) -> None:
    '''
    拆分音视频
    ---
    :param input_file_path: 携带音视频信息的输入文件路径
    :param output_video_path: 仅携带视频信息的输出文件路径
    :param output_audio_path: 仅携带音频信息的输出文件路径
    :return: None
    '''
    # 拆分视频
    cmds = [
        f'ffmpeg -i "{input_file_path}" -an -c:v copy "{output_video_path}"',
        f'ffmpeg -i "{input_file_path}" -vn -c:a copy "{output_audio_path}"',
    ]
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


def merge_va(
    input_video_path: str,
    input_audio_path: str,
    output_file_path: str,
) -> None:
    '''
    合成音视频
    ---
    :param input_video_path: 仅携带视频信息的输入文件路径
    :param input_audio_path: 仅携带音频信息的输入文件路径
    :param output_file_path: 携带音视频信息的输出文件路径
    :return: None
    '''
    # 合成视频
    cmds = [
        f'ffmpeg -i "{input_video_path}" -i "{input_audio_path}" -c:v copy -c:a copy "{output_file_path}"',
    ]
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
