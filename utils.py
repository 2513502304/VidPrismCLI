from rich.console import Console
from rich.logging import RichHandler
import logging

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
