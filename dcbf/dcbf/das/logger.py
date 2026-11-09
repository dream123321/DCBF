import logging

def _temperature_text(value):
    if value is None or value == []:
        return "off"
    values = value if isinstance(value, (list, tuple)) else [value]
    rendered = []
    for item in values:
        try:
            number = float(item)
            rendered.append(f"{number:g} K")
        except (TypeError, ValueError):
            rendered.append(f"{item} K")
    return "[" + ", ".join(rendered) + "]"


def format_log_context(main_name, gen_name=None, npt=None, nvt=None):
    parts = [str(main_name)]
    if gen_name is not None:
        parts.append(str(gen_name))
    parts.extend((f"NPT={_temperature_text(npt)}", f"NVT={_temperature_text(nvt)}"))
    return "[" + " | ".join(parts) + "]"


def context_from_mlp_md(workspace, mlp_md):
    npt = []
    nvt = []
    for stage in mlp_md or []:
        if not isinstance(stage, dict):
            continue
        if stage.get("npt"):
            npt.extend(stage["npt"] if isinstance(stage["npt"], list) else [stage["npt"]])
        if stage.get("nvt"):
            nvt.extend(stage["nvt"] if isinstance(stage["nvt"], list) else [stage["nvt"]])
    from pathlib import Path
    path = Path(workspace)
    return format_log_context(path.parent.name, path.name, npt, nvt)


class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"{self.extra['context']} {msg}", kwargs


def setup_logger(context=None):
    # 创建一个日志记录器
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)  # 设置最低捕获级别

    # 创建一个文件处理器，并设置级别和格式
    file_handler = logging.FileHandler('../../app.log')
    file_handler.setLevel(logging.INFO)
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M',
    )
    file_handler.setFormatter(formatter)

    # 创建一个控制台处理器，可以用于调试
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)  # 控制台只显示错误及以上级别的日志

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return _ContextAdapter(logger, {"context": context}) if context else logger
