import logging

def setup_logger():
    # 创建一个日志记录器
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)  # 设置最低捕获级别

    # 创建一个文件处理器，并设置级别和格式
    file_handler = logging.FileHandler('../../app.log')
    file_handler.setLevel(logging.INFO)
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)

    # 创建一个控制台处理器，可以用于调试
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)  # 控制台只显示错误及以上级别的日志

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger