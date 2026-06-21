import os
import sys
import time
import argparse
import multiprocessing as mp
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from ase.io import read, iread, write
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from tqdm import tqdm
from loguru import logger

try:
    from dcbf.das.file_conversion import write_normalized_extxyz
except ImportError:
    write_normalized_extxyz = None

RESULT_INFO_KEYS = {"energy", "forces", "stress", "stress_GPa", "virial", "pbc"}

# 检查pymlip是否可用
try:
    from pymlip.core import MTPCalactor, PyConfiguration

    PYMLIP_AVAILABLE = True
except ImportError:
    PYMLIP_AVAILABLE = False
    logger.warning("pymlip未安装，SUS2计算器将不可用")

# 定义可用的计算器类型
CALCULATORS = {
    'nep': 'NEP',
    'mace': 'MACECalculator',
    'chgnet': 'CHGNetCalculator',
    'dp': 'DP',
    'm3gnet': 'PESCalculator',
    'mattersim': 'MatterSimCalculator',
    'sus2': 'SUS2Calculator',  # 新增SUS2计算器
}

WORKER_CALCULATOR = None


def configure_logger(log_level: str):
    """配置主进程或子进程日志。"""
    logger.remove()
    logger.add(sys.stderr, level=log_level)


def limit_blas_threads():
    """多进程 CPU 计算时，限制每个 worker 的底层线程数，避免过度抢占。"""
    for env_name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(env_name, "1")


class SUS2Calculator(Calculator):
    """
    SUS2 calculator based on ase Calculator
    基于MTP的SUS2势函数计算器
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self,
                 potential: str = "p.sus2",
                 ele_list: Optional[List[str]] = None,
                 compute_stress: bool = True,
                 stress_weight: float = 1.0,
                 print_EK: bool = True,
                 **kwargs):
        """
        Args:
            potential (str): xxx.sus2 或 xxx.mtp 势函数文件
            ele_list (List[str]): 元素符号列表，例如 ["Al", "O"]
            compute_stress (bool): 是否计算应力
            stress_weight (float): 应力权重因子
            print_EK (bool): 是否打印能量信息
            **kwargs:
        """
        if not PYMLIP_AVAILABLE:
            raise ImportError("pymlip 未安装。请先安装 pymlip 以支持SUS2计算器")

        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = compute_stress
        self.print_EK = print_EK
        self.stress_weight = stress_weight
        self.mtpcalc = MTPCalactor(self.potential)

        if ele_list is None:
            raise ValueError("SUS2计算器需要指定元素列表 (ele_list)")
        self.unique_numbers = [atomic_numbers[ele] for ele in ele_list]

        logger.info(f"SUS2计算器初始化完成")
        logger.info(f"势函数文件: {self.potential}")
        logger.info(f"元素列表: {ele_list} (原子序数: {self.unique_numbers})")

    def calculate(
            self,
            atoms: Optional[Atoms] = None,
            properties: Optional[list] = None,
            system_changes: Optional[list] = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms对象
            properties (list): 需要计算的属性列表
            system_changes (list): 监测原子系统的变化
        """
        properties = properties or ["energy"]
        system_changes = system_changes or self.all_changes
        super().calculate(atoms=atoms, properties=properties,
                          system_changes=system_changes)

        # 转换为pymlip配置
        cfg = PyConfiguration.from_ase_atoms(atoms, unique_numbers=self.unique_numbers)
        V = atoms.cell.volume if atoms.cell.volume > 0 else 1.0

        # 执行SUS2计算
        self.mtpcalc.calc(cfg)

        # 获取能量和力
        energy = np.array(cfg.energy)
        forces = cfg.force

        self.results['energy'] = energy
        self.results['forces'] = forces

        # 计算应力（如果需要）
        if self.compute_stress and hasattr(cfg, 'stresses') and cfg.stresses is not None:
            try:
                # SUS2/MTP输出的应力是完整张量，转换为Voigt记法
                stresses = cfg.stresses
                # Voigt记法: [xx, yy, zz, yz, xz, xy]
                self.results['stress'] = -np.array([
                    stresses[0, 0],  # xx
                    stresses[1, 1],  # yy
                    stresses[2, 2],  # zz
                    stresses[1, 2],  # yz
                    stresses[0, 2],  # xz
                    stresses[0, 1]  # xy
                ]) * self.stress_weight / V
            except (IndexError, AttributeError) as e:
                logger.debug(f"应力计算失败: {e}")
                pass


def scf(stru, calculator):
    """单点能量计算函数

    Args:
        stru: ASE原子结构
        calculator: ASE计算器实例

    Returns:
        atoms: 包含计算结果的Atoms对象
    """
    stru.calc = calculator
    e = stru.get_potential_energy()
    f = stru.get_forces()

    # 创建新的Atoms对象
    atoms = Atoms(stru.get_chemical_symbols(),
                  positions=stru.get_positions(),
                  cell=stru.get_cell(),
                  pbc=stru.get_pbc())
    for key, value in stru.info.items():
        if key not in RESULT_INFO_KEYS:
            atoms.info[key] = value

    # 存储能量和力
    atoms.info['energy'] = e
    atoms.arrays['forces'] = f

    # 尝试获取应力
    try:
        s = stru.get_stress(voigt=False)  # 先尝试获取完整张量
        if s is not None and len(s) == 9:
            atoms.info['stress'] = s
            atoms.info['stress_GPa'] = s * 160.21766208  # 转换为GPa
            virial = -1 * s * stru.get_volume()
            atoms.info['virial'] = virial
        else:
            # 如果get_stress返回的是Voigt记法，转换为完整张量
            s_voigt = stru.get_stress(voigt=True)
            six2nine = np.array([s_voigt[0], s_voigt[5], s_voigt[4],
                                 s_voigt[5], s_voigt[1], s_voigt[3],
                                 s_voigt[4], s_voigt[3], s_voigt[2]])
            atoms.info['stress'] = six2nine
            atoms.info['stress_GPa'] = six2nine * 160.21766208
            virial = -1 * six2nine * stru.get_volume()
            atoms.info['virial'] = virial
    except (NotImplementedError, AttributeError) as e:
        logger.debug(f"应力计算不被支持或失败: {e}")
        pass

    atoms.info['pbc'] = "T T T"
    atoms.pbc = [True, True, True]

    return atoms


def setup_calculator(calc_type, model_path=None, device='cpu', ele_list=None):
    """设置计算器

    Args:
        calc_type: 计算器类型 ('nep', 'mace', 'chgnet', 'dp', 'm3gnet', 'mattersim', 'sus2')
        model_path: 模型文件路径
        device: 计算设备 ('cpu'或'cuda')
        ele_list: 元素列表 (仅对SUS2计算器必需)

    Returns:
        calculator: ASE计算器实例
    """

    if calc_type == 'nep':
        from pynep.calculate import NEP
        if not model_path:
            raise ValueError("NEP计算器需要指定模型文件路径")
        return NEP(model_path)

    elif calc_type == 'mace':
        try:
            from mace.calculators import MACECalculator
        except ImportError:
            raise ImportError("请安装MACE包: pip install mace-torch")

        if not model_path:
            raise ValueError("MACE计算器需要指定模型文件路径")
        return MACECalculator(model_paths=model_path, device=device)

    elif calc_type == 'dp':
        try:
            from deepmd.calculator import DP
        except ImportError:
            raise ImportError("请安装DeePMD包: pip install deepmd-kit")

        if not model_path:
            raise ValueError("DP计算器需要指定模型文件路径")
        return DP(model=model_path, device=device)

    elif calc_type == 'chgnet':
        try:
            from chgnet.model import CHGNet
            from chgnet.model.dynamics import CHGNetCalculator

            if model_path:
                model = CHGNet.from_file(model_path)
            else:
                # 使用默认模型
                model = CHGNet.load()

            return CHGNetCalculator(model, use_device=device)
        except ImportError:
            raise ImportError("请安装CHGNet包: pip install chgnet")

    elif calc_type == 'm3gnet':
        try:
            import matgl
            matgl.set_backend("DGL")
            from matgl.ext.ase import PESCalculator

            if model_path:
                model = matgl.load_model(model_path)
            else:
                # 使用预训练模型
                model = matgl.load_model("M3GNet-MP-2021.2.8-PES")

            return PESCalculator(model)
        except ImportError:
            raise ImportError("请安装matgl包: pip install matgl")

    elif calc_type == 'mattersim':
        try:
            from mattersim.forcefield import MatterSimCalculator

            logger.info(f"初始化MatterSim计算器，使用设备: {device}")

            if model_path:
                calculator = MatterSimCalculator(
                    load_path=model_path,
                    device=device
                )
            else:
                calculator = MatterSimCalculator(device=device)

            return calculator

        except ImportError:
            raise ImportError("请安装MatterSim包: pip install mattersim")
        except Exception as e:
            raise ImportError(f"MatterSim初始化失败: {e}")

    elif calc_type == 'sus2':
        if not PYMLIP_AVAILABLE:
            raise ImportError("pymlip未安装，无法使用SUS2计算器")

        if not model_path:
            raise ValueError("SUS2计算器需要指定势函数文件路径")

        if ele_list is None:
            raise ValueError("SUS2计算器需要指定元素列表，请使用 --ele_list 参数")

        logger.info(f"初始化SUS2计算器")
        logger.info(f"势函数文件: {model_path}")
        logger.info(f"元素列表: {ele_list}")

        return SUS2Calculator(
            potential=model_path,
            ele_list=ele_list,
            compute_stress=True
        )

    else:
        raise ValueError(f"不支持的计算器类型: {calc_type}. 可选: {list(CALCULATORS.keys())}")


def load_structures(input_path: Path) -> List[Atoms]:
    """读取输入结构；对 xyz/extxyz 显式预读取到内存，便于并行计算。"""
    suffix = input_path.suffix.lower()

    if suffix in {'.xyz', '.extxyz'}:
        logger.info("检测到 xyz/extxyz 输入，预先读取全部结构到内存")
        structures = read(input_path, index=':')
        if isinstance(structures, Atoms):
            return [structures]
        return list(structures)

    logger.info("读取输入结构到内存")
    return list(iread(input_path))


def resolve_num_workers(requested_workers: int) -> int:
    """解析用户指定的并行进程数。"""
    available_cpus = os.cpu_count() or 1

    if requested_workers <= 0:
        return available_cpus

    if requested_workers > available_cpus:
        logger.warning(f"请求的 CPU 核数 {requested_workers} 超过可用核数 {available_cpus}，将使用 {available_cpus}")
        return available_cpus

    return requested_workers


def init_worker(calc_type: str,
                model_path: Optional[str],
                device: str,
                ele_list: Optional[List[str]],
                log_level: str):
    """为每个进程初始化独立计算器，避免跨进程序列化 calculator。"""
    global WORKER_CALCULATOR
    limit_blas_threads()
    configure_logger(log_level)
    WORKER_CALCULATOR = setup_calculator(
        calc_type=calc_type,
        model_path=model_path,
        device=device,
        ele_list=ele_list
    )


def process_structure(task: Tuple[int, Atoms]):
    """子进程执行单个结构的静态计算。"""
    global WORKER_CALCULATOR
    index, structure = task

    if WORKER_CALCULATOR is None:
        raise RuntimeError("worker 计算器未初始化")

    try:
        atoms = scf(structure, WORKER_CALCULATOR)
        return index, atoms, None
    except Exception as exc:
        return index, None, str(exc)


def write_result(output_file: Path, atoms: Atoms, output_format: str, append_mode: bool):
    """统一写出计算结果。"""
    if output_format.lower() in {"xyz", "extxyz"} and write_normalized_extxyz is not None:
        write_normalized_extxyz(output_file, atoms, append=append_mode)
        return
    write(output_file, atoms, format=output_format, append=append_mode)


def build_output_file(input_path: Path, output_dir: Path, output_format: str, calc_type: str, suffix_text: str) -> Path:
    """生成输出文件名。"""
    stem = input_path.stem
    suffix = f"_{suffix_text}" if suffix_text else ""
    extension = input_path.suffix

    if output_format == input_path.suffix.lstrip('.'):
        output_filename = f"{calc_type}_{stem}{suffix}{extension}"
    elif output_format == 'extxyz':
        output_filename = f"{calc_type}_{stem}{suffix}.xyz"
    else:
        output_filename = f"{calc_type}_{stem}{suffix}.{output_format}"
    return output_dir / output_filename


def count_xyz_frames(input_path: Path) -> int:
    """按 xyz/extxyz 帧边界统计结构数。"""
    frames = 0
    with input_path.open('r', encoding='utf-8', errors='replace', newline='') as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            if not first.strip():
                continue
            try:
                natoms = int(first.strip().split()[0])
            except Exception as exc:
                raise ValueError(f"无法解析第 {frames + 1} 帧的原子数行: {first!r}") from exc
            header = handle.readline()
            if not header:
                raise ValueError(f"第 {frames + 1} 帧缺少 header 行")
            for _ in range(natoms):
                if not handle.readline():
                    raise ValueError(f"第 {frames + 1} 帧原子坐标块不完整")
            frames += 1
    return frames


def split_xyz_raw(input_path: Path, part_dir: Path, split_workers: int) -> Tuple[List[Path], List[int]]:
    """把 xyz/extxyz 输入按连续帧切成 N 份，不重写数值格式。"""
    total_frames = count_xyz_frames(input_path)
    if total_frames == 0:
        return [], []

    part_dir.mkdir(parents=True, exist_ok=True)
    part_count = min(split_workers, total_frames)
    frames_per_part = (total_frames + part_count - 1) // part_count
    part_paths = [part_dir / f"{input_path.stem}_part{i}{input_path.suffix}" for i in range(part_count)]
    part_counts = [0] * part_count

    handles = [path.open('w', encoding='utf-8', newline='') for path in part_paths]
    frame_index = 0
    try:
        with input_path.open('r', encoding='utf-8', errors='replace', newline='') as src:
            while True:
                first = src.readline()
                if not first:
                    break
                if not first.strip():
                    continue
                natoms = int(first.strip().split()[0])
                header = src.readline()
                if not header:
                    raise ValueError(f"第 {frame_index + 1} 帧缺少 header 行")
                part_index = min(frame_index // frames_per_part, part_count - 1)
                out = handles[part_index]
                out.write(first)
                out.write(header)
                for _ in range(natoms):
                    line = src.readline()
                    if not line:
                        raise ValueError(f"第 {frame_index + 1} 帧原子坐标块不完整")
                    out.write(line)
                part_counts[part_index] += 1
                frame_index += 1
    finally:
        for handle in handles:
            handle.close()

    return part_paths, part_counts


def _split_worker_thread_count(split_workers: int, num_workers: int, device: str) -> int:
    """估算每个底层进程可用线程数，避免 split 和内部 worker 同时超订阅。"""
    visible_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    inner_workers = max(1, num_workers if device == 'cpu' else 1)
    total_processes = max(1, split_workers * inner_workers)
    return max(1, visible_cpus // total_processes)


def run_split_workers(args, input_path: Path, output_file: Path, output_dir: Path) -> Tuple[int, int]:
    """把输入切分为多个临时 xyz/extxyz 文件，并行预测后按原顺序拼接。"""
    input_suffix = input_path.suffix.lower()
    output_format = args.format.lower()
    if input_suffix not in {'.xyz', '.extxyz'}:
        raise ValueError("--split-workers 仅支持 xyz/extxyz 输入；其它格式请使用 --num-workers")
    if output_format not in {'xyz', 'extxyz'}:
        raise ValueError("--split-workers 仅支持 xyz/extxyz 输出格式，因为结果需要按帧拼接")
    if args.device != 'cpu':
        raise ValueError("--split-workers 当前仅支持 CPU 预测；GPU 请使用 --num-workers 1")

    split_root = output_dir / f".{output_file.stem}_split_workers"
    if split_root.exists():
        shutil.rmtree(split_root)
    part_input_dir = split_root / "inputs"
    part_output_dir = split_root / "outputs"
    part_output_dir.mkdir(parents=True, exist_ok=True)

    part_paths, part_counts = split_xyz_raw(input_path, part_input_dir, args.split_workers)
    if not part_paths:
        logger.warning("未读取到任何结构，程序结束")
        return 0, 0

    threads_per_process = _split_worker_thread_count(len(part_paths), args.num_workers, args.device)
    logger.info(
        f"启用数据集切分并行: split_workers={len(part_paths)}, "
        f"每份 num_workers={args.num_workers}, 每底层进程线程数={threads_per_process}"
    )
    logger.info(f"分片帧数: {part_counts}")

    processes = []
    for index, part_path in enumerate(part_paths):
        child_output_dir = part_output_dir / f"part{index}"
        child_output_dir.mkdir(parents=True, exist_ok=True)
        child_suffix = f"split_part{index}"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(part_path),
            "--calc_type",
            args.calc_type,
            "--device",
            args.device,
            "--output",
            str(child_output_dir),
            "--format",
            args.format,
            "--suffix",
            child_suffix,
            "--num-workers",
            str(args.num_workers),
            "--log-level",
            args.log_level,
        ]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.ele_list:
            cmd.extend(["--ele_list", *args.ele_list])

        env = os.environ.copy()
        for env_name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
        ):
            env[env_name] = str(threads_per_process)

        stdout_path = child_output_dir / "stdout.log"
        stderr_path = child_output_dir / "stderr.log"
        stdout = stdout_path.open("wb")
        stderr = stderr_path.open("wb")
        process = subprocess.Popen(cmd, cwd=str(Path.cwd()), env=env, stdout=stdout, stderr=stderr)
        processes.append((index, process, stdout, stderr, child_output_dir))

    statuses = []
    for index, process, stdout, stderr, child_output_dir in processes:
        status = process.wait()
        stdout.close()
        stderr.close()
        statuses.append(status)
        if status != 0:
            logger.error(f"分片 {index} 预测失败，returncode={status}，日志目录: {child_output_dir}")

    if any(status != 0 for status in statuses):
        raise RuntimeError(f"切分并行预测失败，return codes: {statuses}；临时目录保留在 {split_root}")

    child_xyz_files = []
    for index in range(len(part_paths)):
        child_output_dir = part_output_dir / f"part{index}"
        files = sorted(
            path for path in child_output_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.xyz', '.extxyz'}
        )
        if len(files) != 1:
            raise RuntimeError(f"分片 {index} 输出文件数量异常: {files}；临时目录保留在 {split_root}")
        child_xyz_files.append(files[0])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "ab" if args.append else "wb"
    with output_file.open(open_mode) as merged:
        for child_file in child_xyz_files:
            with child_file.open("rb") as handle:
                shutil.copyfileobj(handle, merged, length=1024 * 1024)

    successful = sum(part_counts)
    logger.info(f"切分并行预测完成，输出文件: {output_file}")
    shutil.rmtree(split_root)
    return successful, 0


def main():
    parser = argparse.ArgumentParser(
        description='使用不同机器学习势函数进行结构能量计算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python calc.py input.xyz --calc_type nep --model model.txt
  python calc.py input.extxyz --calc_type mace --model model.pth --device cuda
  python calc.py input.cif --calc_type chgnet --output results
  python calc.py input.xyz --calc_type mattersim --device cuda
  python calc.py input.xyz --calc_type sus2 --model model.sus2 --ele_list Al O  # 使用SUS2计算器
  python calc.py input.xyz --calc_type sus2 --model model.mtp --ele_list Al     # 使用MTP格式的势函数
  python calc.py input.xyz --calc_type mace --suffix test                       # 添加自定义后缀
        """
    )

    parser.add_argument('input', help='输入结构文件')
    parser.add_argument('--calc_type', '-c',
                        choices=list(CALCULATORS.keys()),
                        default='sus2',
                        help='计算器类型 (默认: sus2)')
    parser.add_argument('--model', '-m',
                        help='模型/势函数文件路径')
    parser.add_argument('--ele_list', '-e',
                        nargs='+',
                        help='元素列表，例如: --ele_list Al O (仅对SUS2计算器必需)')
    parser.add_argument('--device', '-d',
                        default='cpu',
                        choices=['cpu', 'cuda'],
                        help='计算设备 (cpu或cuda，默认: cpu)')
    parser.add_argument('--output', '-o',
                        default='out_files',
                        help='输出目录 (默认: out_files)')
    parser.add_argument('--format', '-f',
                        default='extxyz',
                        help='输出格式 (默认: extxyz)')
    parser.add_argument('--append', '-a',
                        action='store_true',
                        help='追加到输出文件而不是覆盖')
    parser.add_argument('--log-level', '-l',
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别 (默认: INFO)')
    parser.add_argument('--suffix', '-s',
                        default='',
                        help='在输出文件名中添加后缀 (例如: --suffix test 生成 input_test.xyz)')
    parser.add_argument('--num-workers', '-n',
                        type=int,
                        default=1,
                        help='CPU 并行进程数；1 为串行，0 或负数表示使用全部可用 CPU 核')
    parser.add_argument('--split-workers',
                        type=int,
                        default=1,
                        help='将 xyz/extxyz 数据集切分为 N 份并行预测后拼接；1 表示不切分 (默认: 1)')

    args = parser.parse_args()

    # 配置logger
    configure_logger(args.log_level)

    if args.split_workers < 1:
        logger.error("--split-workers 必须大于等于 1")
        sys.exit(1)

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = build_output_file(input_path, output_dir, args.format, args.calc_type, args.suffix)
    logger.info(f"输出文件: {output_file}")

    try:
        # 自动检测CUDA可用性
        if args.device == 'cuda':
            if args.calc_type == 'sus2':
                logger.warning("SUS2 仅支持 CPU，将使用 CPU 进行计算")
                args.device = 'cpu'
            else:
                try:
                    import torch
                    if torch.cuda.is_available():
                        logger.info(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        logger.warning("未检测到可用 CUDA，将使用 CPU 进行计算")
                        args.device = 'cpu'
                except ImportError:
                    logger.warning("未能导入 torch 检查 CUDA，继续按用户指定设备初始化")

        args.num_workers = resolve_num_workers(args.num_workers)
        parallel_enabled = args.device == 'cpu' and args.num_workers > 1

        if args.device != 'cpu' and args.num_workers > 1:
            logger.warning("GPU 模式不启用多进程并行，已自动切换为单进程")
            args.num_workers = 1
            parallel_enabled = False

        if args.split_workers > 1:
            logger.info("开始切分数据集并行预测...")
            start_time = time.time()
            successful, failed = run_split_workers(args, input_path, output_file, output_dir)
            elapsed = (time.time() - start_time) / 60
            logger.info("=" * 50)
            logger.info("计算完成!")
            logger.info(f"成功: {successful} 个结构")
            logger.info(f"失败: {failed} 个结构")
            logger.info(f"总时间: {elapsed:.2f} 分钟")
            if successful > 0:
                logger.info(f"平均每个结构: {elapsed * 60 / successful:.2f} 秒")
            logger.info("=" * 50)
            return

        if parallel_enabled:
            limit_blas_threads()
            logger.info(f"启用 CPU 并行计算，worker 数: {args.num_workers}")
        else:
            logger.info("使用单进程计算")

        calculator = None
        if not parallel_enabled:
            calculator = setup_calculator(
                args.calc_type,
                args.model,
                args.device,
                args.ele_list
            )

        logger.info(f"使用 {CALCULATORS[args.calc_type]} 计算器")
        if args.model:
            logger.info(f"模型文件: {args.model}")
        logger.info(f"计算设备: {args.device}")

    except Exception as e:
        logger.error(f"初始化预测失败: {e}")
        sys.exit(1)

    try:
        structures = load_structures(input_path)
        logger.info(f"读取到 {len(structures)} 个结构")
    except Exception as e:
        logger.error(f"读取输入文件失败: {e}")
        sys.exit(1)

    if len(structures) == 0:
        logger.warning("未读取到任何结构，程序结束")
        sys.exit(0)

    # 执行计算
    logger.info("开始计算...")
    start_time = time.time()

    successful = 0
    failed = 0
    append_mode = args.append

    if parallel_enabled:
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                    max_workers=args.num_workers,
                    mp_context=ctx,
                    initializer=init_worker,
                    initargs=(args.calc_type, args.model, args.device, args.ele_list, args.log_level)
            ) as executor:
                results = executor.map(
                    process_structure,
                    enumerate(structures)
                )

                for index, atoms, error in tqdm(results, total=len(structures), desc="计算进度"):
                    if error is not None:
                        logger.error(f"结构 {index + 1} 计算失败: {error}")
                        failed += 1
                        continue

                    try:
                        write_result(output_file, atoms, args.format, append_mode)
                        successful += 1
                        append_mode = True
                    except Exception as exc:
                        logger.error(f"结构 {index + 1} 写出失败: {exc}")
                        failed += 1

        except Exception as e:
            logger.error(f"并行计算失败: {e}")
            sys.exit(1)
    else:
        for i, stru in enumerate(tqdm(structures, desc="计算进度")):
            try:
                atoms = scf(stru, calculator)
                write_result(output_file, atoms, args.format, append_mode)
                successful += 1
                append_mode = True
            except Exception as e:
                logger.error(f"结构 {i + 1} 计算失败: {e}")
                failed += 1
                continue

    end_time = time.time()
    elapsed = (end_time - start_time) / 60

    # 输出统计信息
    logger.info("=" * 50)
    logger.info(f"计算完成!")
    logger.info(f"成功: {successful} 个结构")
    logger.info(f"失败: {failed} 个结构")
    logger.info(f"总时间: {elapsed:.2f} 分钟")
    if successful > 0:
        logger.info(f"平均每个结构: {elapsed * 60 / successful:.2f} 秒")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
