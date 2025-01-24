import logging

# 日志记录
logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s (%(filename)s %(funcName)s %(lineno)d): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger('VidPrismCLI')
