#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare DFT and SUS² results: energy, force and stress.
Scatter plots are density-coloured; unified combined plot.
Command line interface version with CFG file support.

作者信息:
    ============================================================
    作者: 黄晶
    单位: 南方科技大学
    邮箱: 2760344463@qq.com
    开发时间: 2026.1.19
    修改: 添加MLIP名称自定义功能
    修改: 容许XYZ文件没有应力信息
    修改(v3): 力/应力误差显示改为按分量统计，与logout中的MAE/RMSE对齐
    修改: 应力单位可通过 --stress-unit 在 eV/GPa 间切换，默认eV
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import argparse
from multiprocessing import Pool
from ase.io import iread
import time, sys, os
import tempfile
from sklearn.metrics import r2_score

EV_PER_A3_TO_GPA = 160.21766208


class CFGConverter:
    """CFG文件转换器，将CFG格式转换为XYZ格式"""

    def __init__(self, elements=None):
        """
        初始化CFG转换器

        Parameters:
        -----------
        elements : list, optional
            元素符号列表，默认为['C']
        """
        if elements is None:
            elements = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca',
                        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb',
                        'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa',
                        'U', 'Np', 'Pu']
        self.elements = elements
        self.map_dict = {i: ele for i, ele in enumerate(elements)}

    def convert_cfg_to_xyz(self, cfg_file, xyz_file=None, keep_temp=False):
        """
        将CFG文件转换为XYZ格式

        Parameters:
        -----------
        cfg_file : str
            输入的CFG文件路径
        xyz_file : str, optional
            输出的XYZ文件路径，如果为None则创建临时文件
        keep_temp : bool
            是否保留临时文件

        Returns:
        --------
        str : 输出的XYZ文件路径
        """
        if xyz_file is None:
            # 创建临时文件
            temp_fd, temp_path = tempfile.mkstemp(suffix='.xyz', prefix='temp_')
            os.close(temp_fd)
            xyz_file = temp_path
        else:
            temp_path = None

        print(f"Converting CFG file: {cfg_file} -> {xyz_file}")

        with open(cfg_file) as f:
            lines = f.readlines()

        cfgcnt = 0
        for line in lines:
            if line == ' Size\n':
                cfgcnt += 1

        # print(f"Found {cfgcnt} configurations in {cfg_file}")

        cntr = 1
        with open(xyz_file, 'w') as ff:
            for i in range(len(lines)):
                if lines[i] != 'BEGIN_CFG\n':
                    continue

                # 读取原子数量
                size = int(lines[i + 2].split()[0])

                # 读取能量
                energy = float(lines[i + 9 + size].split()[0])

                # 读取晶格参数
                lat = lines[i + 4].split() + lines[i + 5].split() + lines[i + 6].split()
                lat = [float(x) for x in lat]

                # 读取应力（处理对称性）
                stress = lines[i + 11 + size].split()
                stress = [float(x) for x in stress]
                _stress = [stress[0], stress[5], stress[4],
                           stress[5], stress[1], stress[3],
                           stress[4], stress[3], stress[2]]

                # 读取原子信息
                atoms = []
                for j in range(size):
                    words = lines[i + 8 + j].split()
                    atom_info = [
                        self.map_dict[int(words[1])],  # 元素符号
                        float(words[2]),  # x坐标
                        float(words[3]),  # y坐标
                        float(words[4]),  # z坐标
                        float(words[-3]),  # Fx
                        float(words[-2]),  # Fy
                        float(words[-1])  # Fz
                    ]
                    atoms.append(atom_info)

                # 写入XYZ格式
                ff.write(f"{size}\n")
                ff.write(
                    f'Lattice="{lat[0]:12.8f} {lat[1]:12.8f} {lat[2]:12.8f} '
                    f'{lat[3]:12.8f} {lat[4]:12.8f} {lat[5]:12.8f} '
                    f'{lat[6]:12.8f} {lat[7]:12.8f} {lat[8]:12.8f}" '
                    f'Properties=species:S:1:pos:R:3:forces:R:3 '
                    f'energy={energy:12.8f} '
                    f'virial="{_stress[0]:12.8f} {_stress[1]:12.8f} {_stress[2]:12.8f} '
                    f'{_stress[3]:12.8f} {_stress[4]:12.8f} {_stress[5]:12.8f} '
                    f'{_stress[6]:12.8f} {_stress[7]:12.8f} {_stress[8]:12.8f}" '
                    f'pbc="T T T"\n'
                )

                for atom in atoms:
                    ff.write(
                        f'{atom[0]} {atom[1]:12.8f} {atom[2]:12.8f} {atom[3]:12.8f} '
                        f'{atom[4]:12.8f} {atom[5]:12.8f} {atom[6]:12.8f}\n'
                    )

                if cntr % 100 == 0:
                    print(f"  Processed {cntr} configurations...")
                cntr += 1

        print(f"Conversion complete. Created {xyz_file} with {cntr - 1} configurations.")

        self.temp_file = temp_path if not keep_temp else None
        return xyz_file

    def cleanup(self):
        """清理临时文件"""
        if self.temp_file and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
            print(f"Removed temporary file: {self.temp_file}")


class DFTSUS2Comparator:
    """DFT和SUS2结果比较器"""

    def __init__(self, args):
        """
        初始化比较器

        Parameters:
        -----------
        args : argparse.Namespace
            命令行参数
        """
        self.args = args

        # 从args获取参数
        self.num_processes = args.num_processes
        self.cmap_name = args.cmap
        self.scatter_size = args.scatter_size
        self.output_file = args.output
        self.show_R2 = args.show_r2
        self.save_data = args.save_data
        self.bins = args.bins
        self.force_mode = args.force_mode  # 力显示模式
        self.elements = args.elements  # 元素列表
        self.keep_temp = args.keep_temp  # 是否保留临时文件
        self.mlip_name = args.mlip_name  # MLIP名称
        self.skip_structure_indices = sorted({idx for idx in args.skip_structure_indices if idx > 0})
        self.stress_unit = args.stress_unit
        self.stress_unit_label = 'GPa' if self.stress_unit == 'GPa' else 'eV'
        self.stress_csv_suffix = 'GPa' if self.stress_unit == 'GPa' else 'eV'
        self.signed_axis_symmetric = args.signed_axis_symmetric

        # 新增：应力可用标志
        self.stress_available = True

        # 设置全局字体和样式
        self._setup_plot_style(
            fontsize=args.fontsize,
            tick_labelsize=args.tick_labelsize,
            legend_fontsize=args.legend_fontsize,
            linewidth=args.linewidth
        )

        # 初始化CFG转换器
        self.cfg_converter = CFGConverter(elements=self.elements)

    def _setup_plot_style(self, fontsize=30, tick_labelsize=30,
                          legend_fontsize=30, linewidth=4):
        """设置绘图样式"""
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams.update({
            'font.size': fontsize,
            'axes.titlesize': fontsize,
            'axes.titleweight': 'bold',
            'axes.labelsize': fontsize,
            'axes.labelweight': 'bold',
            'axes.linewidth': 5,
            'xtick.labelsize': tick_labelsize,
            'ytick.labelsize': tick_labelsize,
            'legend.fontsize': legend_fontsize,
            'legend.title_fontsize': legend_fontsize,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': linewidth,
            'ytick.major.width': linewidth,
            'xtick.major.size': 14,
            'ytick.major.size': 14,
            'xtick.minor.width': linewidth,
            'ytick.minor.width': linewidth,
            'xtick.minor.size': 7,
            'ytick.minor.size': 7,
        })

    def _process_input_file(self, input_file):
        """
        处理输入文件，如果是CFG格式则转换为XYZ

        Parameters:
        -----------
        input_file : str
            输入文件路径

        Returns:
        --------
        str : 处理后的XYZ文件路径
        """
        if input_file.lower().endswith('.cfg'):
            # CFG文件，需要转换
            xyz_file = self.cfg_converter.convert_cfg_to_xyz(
                input_file,
                keep_temp=self.keep_temp
            )
            return xyz_file
        elif input_file.lower().endswith('.xyz'):
            # 已经是XYZ文件，直接使用
            return input_file
        else:
            raise ValueError(f"Unsupported file format: {input_file}. "
                             f"Expected .cfg or .xyz files.")

    def hist2d_density(self, x, y, bins=None):
        """用 2D 直方图快速估算每个点的密度"""
        if bins is None:
            bins = self.bins
        h, xe, ye = np.histogram2d(x, y, bins=bins)
        ix = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
        iy = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
        return h[ix, iy]

    def density_scatter(self, ax, x, y, bins=None):
        """在指定 ax 上绘制密度散点并返回 scatter 句柄"""
        if bins is None:
            bins = self.bins
        z = self.hist2d_density(x, y, bins)
        order = z.argsort()  # 低密度点先画
        sc = ax.scatter(x[order], y[order],
                        c=z[order], s=self.scatter_size, cmap=self.cmap_name,
                        norm=mcolors.LogNorm(), edgecolors='none')
        return sc


    def _set_equal_parity_axes(self, ax, x_data=None, y_data=None, symmetric_zero=False, padding=None, nbins=5):
        """Use identical x/y limits and ticks for DFT-vs-MLIP parity plots.

        Limits are based on the actual plotted data plus a small padding.  The
        tick locator is only used to choose tick positions; it no longer expands
        the plotting range.  signed quantities such as force/stress components
        can be centered around zero with symmetric_zero=True.
        """
        if padding is None:
            padding = getattr(self.args, 'axis_padding', 0.04)

        if x_data is None or y_data is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lim_low = min(xlim[0], ylim[0])
            lim_high = max(xlim[1], ylim[1])
        else:
            x_arr = np.asarray(x_data, dtype=float).reshape(-1)
            y_arr = np.asarray(y_data, dtype=float).reshape(-1)
            both = np.concatenate([x_arr, y_arr])
            both = both[np.isfinite(both)]
            if both.size == 0:
                lim_low, lim_high = -1.0, 1.0
            else:
                lim_low = float(np.min(both))
                lim_high = float(np.max(both))

        if symmetric_zero:
            max_abs = max(abs(lim_low), abs(lim_high))
            if max_abs == 0:
                max_abs = 1.0
            max_abs *= (1.0 + padding)
            lim_low, lim_high = -max_abs, max_abs
        else:
            span = lim_high - lim_low
            if span <= 0:
                span = max(abs(lim_low), 1.0)
            lim_low -= span * padding
            lim_high += span * padding

        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)

        locator = MaxNLocator(nbins=nbins, symmetric=symmetric_zero)
        ticks = np.asarray(locator.tick_values(lim_low, lim_high), dtype=float)
        ticks = ticks[np.isfinite(ticks)]
        eps = max(abs(lim_low), abs(lim_high), 1.0) * 1e-9
        ticks = ticks[(ticks >= lim_low - eps) & (ticks <= lim_high + eps)]
        if symmetric_zero and not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.append(ticks, 0.0))
        if ticks.size >= 2:
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

        ax.set_aspect('equal', adjustable='box')
        return lim_low, lim_high

    def compute_energy(self, a_dft, a_sus):
        """计算能量"""
        e_dft = a_dft.get_potential_energy() / a_dft.get_global_number_of_atoms()
        e_sus = a_sus.get_potential_energy() / a_sus.get_global_number_of_atoms()
        return e_dft, e_sus

    def compute_forces(self, a_dft, a_sus):
        """计算力"""
        return a_dft.get_forces(), a_sus.get_forces()

    def _matrix_from_voigt_or_flat(self, values):
        """Convert stress/virial stored as 6 Voigt or 9 flat values to 3x3."""
        arr = np.array(values, dtype=float)
        if arr.shape == (3, 3):
            return arr
        arr = arr.reshape(-1)
        if arr.size == 9:
            return arr.reshape(3, 3)
        if arr.size == 6:
            return np.array([
                [arr[0], arr[5], arr[4]],
                [arr[5], arr[1], arr[3]],
                [arr[4], arr[3], arr[2]],
            ], dtype=float)
        raise ValueError(f"Unsupported stress/virial shape: {arr.shape}")

    def _get_virial_matrix(self, atoms):
        """Return virial matrix in eV; accept either virial or ASE stress labels."""
        if 'virial' in atoms.info:
            return self._matrix_from_voigt_or_flat(atoms.info['virial'])

        stress = None
        if 'stress' in atoms.info:
            stress = atoms.info['stress']
        elif atoms.calc is not None and 'stress' in atoms.calc.results:
            stress = atoms.calc.results['stress']
        else:
            try:
                stress = atoms.get_stress(voigt=False)
            except Exception:
                return None

        volume = atoms.get_volume()
        if volume <= 0:
            return None
        stress_matrix = self._matrix_from_voigt_or_flat(stress)
        return -stress_matrix * volume

    def compute_stress_components_ev(self, atoms_dft, atoms_sus):
        """
        计算应力/virial的6个独立分量；默认eV，--stress-unit GPa时转换为GPa
        口径与训练程序logout一致: xx, yy, zz, yz, xz, xy
        如果缺少virial/stress信息或GPa转换时体积无效，返回None
        """
        try:
            dft_s = self._get_virial_matrix(atoms_dft)
            sus_s = self._get_virial_matrix(atoms_sus)
            if dft_s is None or sus_s is None:
                return None, None

            dft_components = np.array([
                dft_s[0, 0], dft_s[1, 1], dft_s[2, 2],
                dft_s[1, 2], dft_s[0, 2], dft_s[0, 1]
            ])
            sus_components = np.array([
                sus_s[0, 0], sus_s[1, 1], sus_s[2, 2],
                sus_s[1, 2], sus_s[0, 2], sus_s[0, 1]
            ])
            if self.stress_unit == 'GPa':
                dft_volume = atoms_dft.get_volume()
                sus_volume = atoms_sus.get_volume()
                if dft_volume <= 0 or sus_volume <= 0:
                    return None, None
                dft_components = dft_components / dft_volume * EV_PER_A3_TO_GPA
                sus_components = sus_components / sus_volume * EV_PER_A3_TO_GPA
            return dft_components, sus_components
        except (KeyError, TypeError, ValueError):
            return None, None

    def compute_stress_components_both_units(self, atoms_dft, atoms_sus):
        """Return stress/virial components in both eV and GPa.

        The eV values are virial components.  The GPa values use the same
        convention as the existing --stress-unit GPa path:
        component_GPa = component_eV / volume * 160.21766208.
        """
        try:
            dft_s = self._get_virial_matrix(atoms_dft)
            sus_s = self._get_virial_matrix(atoms_sus)
            if dft_s is None or sus_s is None:
                return None

            dft_ev = np.array([
                dft_s[0, 0], dft_s[1, 1], dft_s[2, 2],
                dft_s[1, 2], dft_s[0, 2], dft_s[0, 1]
            ])
            sus_ev = np.array([
                sus_s[0, 0], sus_s[1, 1], sus_s[2, 2],
                sus_s[1, 2], sus_s[0, 2], sus_s[0, 1]
            ])

            dft_volume = atoms_dft.get_volume()
            sus_volume = atoms_sus.get_volume()
            if dft_volume <= 0 or sus_volume <= 0:
                dft_gpa = None
                sus_gpa = None
            else:
                dft_gpa = dft_ev / dft_volume * EV_PER_A3_TO_GPA
                sus_gpa = sus_ev / sus_volume * EV_PER_A3_TO_GPA

            return {
                'eV': (dft_ev, sus_ev),
                'GPa': None if dft_gpa is None or sus_gpa is None else (dft_gpa, sus_gpa)
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _stress_metric_bundle(self, dft_components, sus_components, unit):
        """Build the full stress metric dict for one unit."""
        dft_all = dft_components.flatten()
        sus_all = sus_components.flatten()
        diff = dft_all - sus_all

        return {
            'dft_all': dft_all,
            'sus_all': sus_all,
            'dft_components': dft_components,
            'sus_components': sus_components,
            'mae': np.mean(np.abs(diff)),
            'rmse': np.sqrt(np.mean(diff ** 2)),
            'r2': r2_score(dft_all, sus_all),
            'mae_xx': np.mean(np.abs(dft_components[:, 0] - sus_components[:, 0])),
            'mae_yy': np.mean(np.abs(dft_components[:, 1] - sus_components[:, 1])),
            'mae_zz': np.mean(np.abs(dft_components[:, 2] - sus_components[:, 2])),
            'mae_yz': np.mean(np.abs(dft_components[:, 3] - sus_components[:, 3])),
            'mae_xz': np.mean(np.abs(dft_components[:, 4] - sus_components[:, 4])),
            'mae_xy': np.mean(np.abs(dft_components[:, 5] - sus_components[:, 5])),
            'rmse_xx': np.sqrt(np.mean((dft_components[:, 0] - sus_components[:, 0]) ** 2)),
            'rmse_yy': np.sqrt(np.mean((dft_components[:, 1] - sus_components[:, 1]) ** 2)),
            'rmse_zz': np.sqrt(np.mean((dft_components[:, 2] - sus_components[:, 2]) ** 2)),
            'rmse_yz': np.sqrt(np.mean((dft_components[:, 3] - sus_components[:, 3]) ** 2)),
            'rmse_xz': np.sqrt(np.mean((dft_components[:, 4] - sus_components[:, 4]) ** 2)),
            'rmse_xy': np.sqrt(np.mean((dft_components[:, 5] - sus_components[:, 5]) ** 2)),
            'r2_xx': r2_score(dft_components[:, 0], sus_components[:, 0]),
            'r2_yy': r2_score(dft_components[:, 1], sus_components[:, 1]),
            'r2_zz': r2_score(dft_components[:, 2], sus_components[:, 2]),
            'r2_yz': r2_score(dft_components[:, 3], sus_components[:, 3]),
            'r2_xz': r2_score(dft_components[:, 4], sus_components[:, 4]),
            'r2_xy': r2_score(dft_components[:, 5], sus_components[:, 5]),
            'available': True,
            'unit': unit,
        }

    def load_data(self, dft_file, sus_file):
        """加载DFT和SUS2数据，自动处理CFG格式"""
        t0 = time.time()

        # 处理输入文件
        dft_xyz = self._process_input_file(dft_file)
        sus_xyz = self._process_input_file(sus_file)

        # 加载XYZ数据
        dft = list(iread(dft_xyz))
        sus = list(iread(sus_xyz))

        if self.skip_structure_indices:
            invalid_indices = [idx for idx in self.skip_structure_indices if idx > len(dft)]
            if invalid_indices:
                raise ValueError(
                    f"Skip indices exceed DFT frame count ({len(dft)}): {invalid_indices[:10]}"
                )

            skip_zero_based = {idx - 1 for idx in self.skip_structure_indices}
            dft = [atoms for frame_idx, atoms in enumerate(dft) if frame_idx not in skip_zero_based]
            print(
                f"Skipped {len(skip_zero_based)} DFT structures using 1-based indices: "
                f"{self.skip_structure_indices}"
            )

        print(
            f'Data loaded in {time.time() - t0:.1f}s, '
            f'frames: DFT={len(dft)}, {self.mlip_name}={len(sus)}'
        )

        # 如果使用了临时文件，记录它们以便后续清理
        self.dft_temp = dft_xyz if dft_xyz != dft_file else None
        self.sus_temp = sus_xyz if sus_xyz != sus_file else None

        if len(dft) != len(sus):
            raise ValueError(
                f"Frame count mismatch after filtering: DFT={len(dft)}, {self.mlip_name}={len(sus)}. "
                "Please provide matching --skip-structure-indices for the missing MLIP frames."
            )

        return dft, sus

    def compute_all_metrics(self, dft, sus):
        """计算能量、力和应力的所有指标"""

        # 计算能量数据
        with Pool(processes=self.num_processes) as pool:
            energy_res = pool.starmap(self.compute_energy, zip(dft, sus))
        e_dft, e_sus = map(np.array, zip(*energy_res))

        # 计算能量指标
        mae_e = np.mean(np.abs(e_dft - e_sus)) * 1000
        rmse_e = np.sqrt(np.mean((e_dft - e_sus) ** 2)) * 1000
        r2_e = r2_score(e_dft, e_sus)

        # 计算力数据
        with Pool(processes=self.num_processes) as pool:
            force_res = pool.starmap(self.compute_forces, zip(dft, sus))
        f_dft_vec = np.vstack([r[0] for r in force_res])
        f_sus_vec = np.vstack([r[1] for r in force_res])

        # 力的不同表示
        f_dft_mag = np.linalg.norm(f_dft_vec, axis=1)
        f_sus_mag = np.linalg.norm(f_sus_vec, axis=1)

        # 力的分量
        f_dft_x = f_dft_vec[:, 0]
        f_dft_y = f_dft_vec[:, 1]
        f_dft_z = f_dft_vec[:, 2]
        f_sus_x = f_sus_vec[:, 0]
        f_sus_y = f_sus_vec[:, 1]
        f_sus_z = f_sus_vec[:, 2]

        # 力误差分成两套口径：
        # 1. 分量口径：与logout中的force_MAE(meV/A)一致
        # 2. 合力口径：用于magnitude模式下的图内标注
        f_dft_all = f_dft_vec.flatten()
        f_sus_all = f_sus_vec.flatten()
        diff_components = f_dft_all - f_sus_all
        diff_mag = f_dft_mag - f_sus_mag

        mae_f_comp = np.mean(np.abs(diff_components)) * 1000
        rmse_f_comp = np.sqrt(np.mean(diff_components ** 2)) * 1000
        r2_f_comp = r2_score(f_dft_all, f_sus_all)

        mae_f_mag = np.mean(np.abs(diff_mag)) * 1000
        rmse_f_mag = np.sqrt(np.mean(diff_mag ** 2)) * 1000
        r2_f_mag = r2_score(f_dft_mag, f_sus_mag)

        # 计算力分量指标 - 用于打印
        mae_f_x = np.mean(np.abs(f_dft_x - f_sus_x)) * 1000
        mae_f_y = np.mean(np.abs(f_dft_y - f_sus_y)) * 1000
        mae_f_z = np.mean(np.abs(f_dft_z - f_sus_z)) * 1000

        rmse_f_x = np.sqrt(np.mean((f_dft_x - f_sus_x) ** 2)) * 1000
        rmse_f_y = np.sqrt(np.mean((f_dft_y - f_sus_y) ** 2)) * 1000
        rmse_f_z = np.sqrt(np.mean((f_dft_z - f_sus_z) ** 2)) * 1000

        r2_f_x = r2_score(f_dft_x, f_sus_x)
        r2_f_y = r2_score(f_dft_y, f_sus_y)
        r2_f_z = r2_score(f_dft_z, f_sus_z)

        # ===== 尝试计算应力数据（如果可用）=====
        print("Checking for stress information...")
        stress_available = True
        stress_metrics_by_unit = {}
        try:
            with Pool(processes=self.num_processes) as pool:
                stress_res = pool.starmap(self.compute_stress_components_both_units, zip(dft, sus))

            valid_stress = [item for item in stress_res if item is not None and item.get('eV') is not None]

            if len(valid_stress) == 0:
                print("No stress data found in files. Stress comparison will be skipped.")
                stress_available = False
            else:
                print(f"Found stress data for {len(valid_stress)} configurations.")

                ev_pairs = [item['eV'] for item in valid_stress]
                dft_ev_components = np.array([d for d, _ in ev_pairs])
                sus_ev_components = np.array([s for _, s in ev_pairs])
                stress_metrics_by_unit['eV'] = self._stress_metric_bundle(
                    dft_ev_components, sus_ev_components, 'eV'
                )

                gpa_pairs = [item['GPa'] for item in valid_stress if item.get('GPa') is not None]
                if gpa_pairs:
                    dft_gpa_components = np.array([d for d, _ in gpa_pairs])
                    sus_gpa_components = np.array([s for _, s in gpa_pairs])
                    stress_metrics_by_unit['GPa'] = self._stress_metric_bundle(
                        dft_gpa_components, sus_gpa_components, 'GPa'
                    )
                else:
                    print("Stress GPa conversion skipped because valid volumes were not available.")

                active_unit = self.stress_unit_label
                if active_unit not in stress_metrics_by_unit:
                    active_unit = 'eV'
                active_stress = stress_metrics_by_unit[active_unit]

                dft_all = active_stress['dft_all']
                sus_all = active_stress['sus_all']
                dft_components = active_stress['dft_components']
                sus_components = active_stress['sus_components']
                mae_s = active_stress['mae']
                rmse_s = active_stress['rmse']
                r2_s = active_stress['r2']
                mae_s_xx = active_stress['mae_xx']
                mae_s_yy = active_stress['mae_yy']
                mae_s_zz = active_stress['mae_zz']
                mae_s_yz = active_stress['mae_yz']
                mae_s_xz = active_stress['mae_xz']
                mae_s_xy = active_stress['mae_xy']
                rmse_s_xx = active_stress['rmse_xx']
                rmse_s_yy = active_stress['rmse_yy']
                rmse_s_zz = active_stress['rmse_zz']
                rmse_s_yz = active_stress['rmse_yz']
                rmse_s_xz = active_stress['rmse_xz']
                rmse_s_xy = active_stress['rmse_xy']
                r2_s_xx = active_stress['r2_xx']
                r2_s_yy = active_stress['r2_yy']
                r2_s_zz = active_stress['r2_zz']
                r2_s_yz = active_stress['r2_yz']
                r2_s_xz = active_stress['r2_xz']
                r2_s_xy = active_stress['r2_xy']

        except Exception as e:
            print(f"Error processing stress data: {e}")
            print("Stress comparison will be skipped.")
            stress_available = False

        if not stress_available:
            dft_all = np.array([])
            sus_all = np.array([])
            dft_components = np.array([])
            sus_components = np.array([])
            mae_s = float('nan')
            rmse_s = float('nan')
            r2_s = float('nan')
            mae_s_xx = mae_s_yy = mae_s_zz = mae_s_yz = mae_s_xz = mae_s_xy = float('nan')
            rmse_s_xx = rmse_s_yy = rmse_s_zz = rmse_s_yz = rmse_s_xz = rmse_s_xy = float('nan')
            r2_s_xx = r2_s_yy = r2_s_zz = r2_s_yz = r2_s_xz = r2_s_xy = float('nan')
            self.stress_available = False
        else:
            self.stress_available = True

        # 保存数据（如果需要）
        if self.save_data:
            self._save_data_to_csv(
                e_dft, e_sus,
                f_dft_vec, f_sus_vec, f_dft_mag, f_sus_mag,
                dft_components, sus_components if self.stress_available else None
            )

        return {
            'energy': {
                'dft': e_dft, 'sus': e_sus,
                'mae': mae_e, 'rmse': rmse_e, 'r2': r2_e
            },
            'force': {
                'dft_mag': f_dft_mag, 'sus_mag': f_sus_mag,
                'dft_all': f_dft_all, 'sus_all': f_sus_all,
                'dft_vec': f_dft_vec, 'sus_vec': f_sus_vec,
                'dft_x': f_dft_x, 'dft_y': f_dft_y, 'dft_z': f_dft_z,
                'sus_x': f_sus_x, 'sus_y': f_sus_y, 'sus_z': f_sus_z,
                'mae_components': mae_f_comp, 'rmse_components': rmse_f_comp, 'r2_components': r2_f_comp,
                'mae_magnitude': mae_f_mag, 'rmse_magnitude': rmse_f_mag, 'r2_magnitude': r2_f_mag,
                'mae_x': mae_f_x, 'mae_y': mae_f_y, 'mae_z': mae_f_z,
                'rmse_x': rmse_f_x, 'rmse_y': rmse_f_y, 'rmse_z': rmse_f_z,
                'r2_x': r2_f_x, 'r2_y': r2_f_y, 'r2_z': r2_f_z
            },
            'stress': {
                'dft_all': dft_all, 'sus_all': sus_all,
                'dft_components': dft_components, 'sus_components': sus_components,
                'mae': mae_s, 'rmse': rmse_s, 'r2': r2_s,
                'mae_xx': mae_s_xx, 'mae_yy': mae_s_yy, 'mae_zz': mae_s_zz,
                'mae_yz': mae_s_yz, 'mae_xz': mae_s_xz, 'mae_xy': mae_s_xy,
                'rmse_xx': rmse_s_xx, 'rmse_yy': rmse_s_yy, 'rmse_zz': rmse_s_zz,
                'rmse_yz': rmse_s_yz, 'rmse_xz': rmse_s_xz, 'rmse_xy': rmse_s_xy,
                'r2_xx': r2_s_xx, 'r2_yy': r2_s_yy, 'r2_zz': r2_s_zz,
                'r2_yz': r2_s_yz, 'r2_xz': r2_s_xz, 'r2_xy': r2_s_xy,
                'available': self.stress_available,
                'unit': self.stress_unit_label,
                'all_units': stress_metrics_by_unit
            }
        }

    def _save_data_to_csv(self, e_dft, e_sus, f_dft_vec, f_sus_vec,
                          f_dft_mag, f_sus_mag, dft_components, sus_components):
        """保存数据到CSV文件"""
        import pandas as pd

        # 创建输出目录
        os.makedirs('output_data', exist_ok=True)

        # 保存能量数据
        energy_df = pd.DataFrame({
            'DFT_Energy_eV_per_atom': e_dft,
            'SUS2_Energy_eV_per_atom': e_sus
        })
        energy_df.to_csv('output_data/energy_data.csv', index=False)

        # 保存力数据（分量）
        force_components_df = pd.DataFrame({
            'DFT_Fx_eV_per_A': f_dft_vec[:, 0],
            'DFT_Fy_eV_per_A': f_dft_vec[:, 1],
            'DFT_Fz_eV_per_A': f_dft_vec[:, 2],
            'SUS2_Fx_eV_per_A': f_sus_vec[:, 0],
            'SUS2_Fy_eV_per_A': f_sus_vec[:, 1],
            'SUS2_Fz_eV_per_A': f_sus_vec[:, 2]
        })
        force_components_df.to_csv('output_data/force_components_data.csv', index=False)

        # 保存力数据（合力）
        force_magnitude_df = pd.DataFrame({
            '|F_DFT|_eV_per_A': f_dft_mag,
            '|F_SUS2|_eV_per_A': f_sus_mag
        })
        force_magnitude_df.to_csv('output_data/force_magnitude_data.csv', index=False)

        # 保存应力数据（如果可用）
        if sus_components is not None and len(sus_components) > 0:
            stress_unit = self.stress_csv_suffix
            stress_df = pd.DataFrame({
                f'DFT_stress_xx_{stress_unit}': dft_components[:, 0],
                f'DFT_stress_yy_{stress_unit}': dft_components[:, 1],
                f'DFT_stress_zz_{stress_unit}': dft_components[:, 2],
                f'DFT_stress_yz_{stress_unit}': dft_components[:, 3],
                f'DFT_stress_xz_{stress_unit}': dft_components[:, 4],
                f'DFT_stress_xy_{stress_unit}': dft_components[:, 5],
                f'SUS2_stress_xx_{stress_unit}': sus_components[:, 0],
                f'SUS2_stress_yy_{stress_unit}': sus_components[:, 1],
                f'SUS2_stress_zz_{stress_unit}': sus_components[:, 2],
                f'SUS2_stress_yz_{stress_unit}': sus_components[:, 3],
                f'SUS2_stress_xz_{stress_unit}': sus_components[:, 4],
                f'SUS2_stress_xy_{stress_unit}': sus_components[:, 5]
            })
            stress_df.to_csv('output_data/stress_data.csv', index=False)
            print("Stress data saved to 'output_data/stress_data.csv'")
        else:
            print("No stress data to save.")

        print("Data saved to 'output_data/' directory")

    def create_combined_plot(self, dft_file, sus_file):
        """
        创建一行两列或三列的综合图：能量、力、应力（如果可用）
        """
        print(f"Loading data: DFT={dft_file}, {self.mlip_name}={sus_file}")

        # 加载数据（自动处理CFG格式）
        dft, sus = self.load_data(dft_file, sus_file)

        # 计算所有指标
        print("Computing metrics...")
        t0 = time.time()
        metrics = self.compute_all_metrics(dft, sus)
        print(f"Metrics computed in {(time.time() - t0) / 60:.2f} min")

        # ===== 根据应力可用性确定子图数量 =====
        if self.stress_available:
            n_subplots = 3
            fig, axs = plt.subplots(1, 3, figsize=self.args.figsize)
            subplot_labels = ['(a)', '(b)', '(c)']
        else:
            n_subplots = 2
            # 调整图形宽度（应力图宽度的一半）
            fig_width = self.args.figsize[0] * 0.7
            fig, axs = plt.subplots(1, 2, figsize=(fig_width, self.args.figsize[1]))
            subplot_labels = ['(a)', '(b)']
            print("Stress data not available. Creating plot with energy and force only.")

        # 子图 (a): 能量
        sc_energy = self._plot_energy_subfigure(
            axs[0],
            metrics['energy']['dft'],
            metrics['energy']['sus'],
            metrics['energy']['mae'],
            metrics['energy']['rmse'],
            metrics['energy']['r2'],
            subplot_labels[0]
        )

        # 子图 (b): 力（根据模式选择）
        sc_force = self._plot_force_subfigure(
            axs[1],
            metrics['force'],
            self.force_mode,
            subplot_labels[1]
        )

        # 子图 (c): 应力（如果可用）
        if self.stress_available:
            self._plot_stress_subfigure(
                axs[2],
                metrics['stress']['dft_all'],
                metrics['stress']['sus_all'],
                metrics['stress']['mae'],
                metrics['stress']['rmse'],
                metrics['stress']['r2'],
                subplot_labels[2]
            )

        # 添加统一的colorbar（使用力的散点对象）
        plt.tight_layout()
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sc_force, cax=cbar_ax, label='density')
        cbar.outline.set_linewidth(self.args.linewidth)
        cbar.set_label('Density', fontsize=self.args.cbar_fontsize, fontweight='bold')
        cbar.ax.tick_params(which='major', labelsize=self.args.cbar_tick_size,
                            width=2, length=10, direction='in')
        cbar.ax.tick_params(which='minor', labelsize=self.args.cbar_tick_size - 8,
                            width=2, length=6, direction='in')

        # 保存图片
        plt.savefig(self.output_file, dpi=self.args.dpi, bbox_inches='tight')
        plt.close()

        print(f"Combined plot saved as '{self.output_file}'")

        # 清理临时文件
        self._cleanup_temp_files()

        # 打印总结信息
        self._print_summary(metrics)

        return metrics

    def _cleanup_temp_files(self):
        """清理临时文件"""
        if hasattr(self, 'dft_temp') and self.dft_temp and not self.keep_temp:
            if os.path.exists(self.dft_temp):
                os.remove(self.dft_temp)
                print(f"Removed temporary DFT file: {self.dft_temp}")

        if hasattr(self, 'sus_temp') and self.sus_temp and not self.keep_temp:
            if os.path.exists(self.sus_temp):
                os.remove(self.sus_temp)
                print(f"Removed temporary SUS2 file: {self.sus_temp}")

    def _plot_energy_subfigure(self, ax, dft_data, sus_data, mae, rmse, r2, label='(a)'):
        """绘制能量子图"""
        sc = self.density_scatter(ax, dft_data, sus_data)

        # 设置统一范围、统一x/y刻度，并添加理想线
        lim_low, lim_high = self._set_equal_parity_axes(ax, dft_data, sus_data, symmetric_zero=False)
        ax.plot([lim_low, lim_high], [lim_low, lim_high], '--',
                color='gray', linewidth=self.args.linewidth)

        # 添加指标文本
        if self.show_R2:
            text = f'MAE: {mae:.3f} (meV/atom)\nRMSE: {rmse:.3f} (meV/atom)\nR\u00B2: {r2:.3f}'
        else:
            text = f'MAE: {mae:.3f} (meV/atom)\nRMSE: {rmse:.3f} (meV/atom)'

        ax.text(0.05, 0.9, text, transform=ax.transAxes, va='top',
                bbox=dict(facecolor='white', alpha=.8),
                fontsize=self.args.annotation_fontsize)

        # 设置标签和标题 - 使用MLIP名称
        ax.set_xlabel('DFT Energy (eV/atom)', fontweight='bold')
        ax.set_ylabel(f'{self.mlip_name} Energy (eV/atom)', fontweight='bold')
        ax.text(0.03, 0.98, label, transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='top')
        return sc

    def _plot_force_subfigure(self, ax, force_metrics, force_mode, label='(b)'):
        """绘制力子图"""
        if force_mode == 'magnitude':
            # 模式1：显示合力大小
            return self._plot_force_magnitude(ax, force_metrics, label)
        elif force_mode == 'components':
            # 模式2：显示力分量（在同一张图上，使用密度颜色）
            return self._plot_force_components(ax, force_metrics, label)
        else:
            raise ValueError(f"Unknown force mode: {force_mode}")

    def _plot_force_magnitude(self, ax, force_metrics, label='(b)'):
        """绘制合力大小子图"""
        dft_data = force_metrics['dft_mag']
        sus_data = force_metrics['sus_mag']
        mae = force_metrics['mae_magnitude']
        rmse = force_metrics['rmse_magnitude']
        r2 = force_metrics['r2_magnitude']

        sc = self.density_scatter(ax, dft_data, sus_data)

        # 设置统一范围、统一x/y刻度，并添加理想线
        lim_low, lim_high = self._set_equal_parity_axes(ax, dft_data, sus_data, symmetric_zero=False)
        ax.plot([lim_low, lim_high], [lim_low, lim_high], '--',
                color='gray', linewidth=self.args.linewidth)

        # 添加指标文本
        if self.show_R2:
            text = f'MAE: {mae:.3f} (meV/Å)\nRMSE: {rmse:.3f} (meV/Å)\nR\u00B2: {r2:.3f}'
        else:
            text = f'MAE: {mae:.3f} (meV/Å)\nRMSE: {rmse:.3f} (meV/Å)'

        ax.text(0.05, 0.9, text, transform=ax.transAxes, va='top',
                bbox=dict(facecolor='white', alpha=.8),
                fontsize=self.args.annotation_fontsize)

        # 设置标签和标题 - 使用MLIP名称
        ax.set_xlabel('DFT |Force| (eV/Å)', fontweight='bold')
        ax.set_ylabel(f'{self.mlip_name} |Force| (eV/Å)', fontweight='bold')
        ax.text(0.03, 0.98, label, transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='top')
        return sc

    def _plot_force_components(self, ax, force_metrics, label='(b)'):
        """绘制力分量子图（三个分量合并显示，使用密度颜色）"""
        # 分量模式下直接使用所有Fx/Fy/Fz分量，统计口径与logout保持一致
        dft_all = force_metrics['dft_all']
        sus_all = force_metrics['sus_all']
        mae = force_metrics['mae_components']
        rmse = force_metrics['rmse_components']
        r2 = force_metrics['r2_components']

        # 绘制密度散点图（与合力模式相同）
        sc = self.density_scatter(ax, dft_all, sus_all)

        # 设置统一范围、统一x/y刻度；是否关于0对称由 --signed-axis-symmetric 控制
        lim_low, lim_high = self._set_equal_parity_axes(ax, dft_all, sus_all, symmetric_zero=self.signed_axis_symmetric)
        ax.plot([lim_low, lim_high], [lim_low, lim_high], '--',
                color='gray', linewidth=self.args.linewidth)

        # 添加指标文本（与合力模式相同）
        if self.show_R2:
            text = f'MAE: {mae:.3f} (meV/Å)\nRMSE: {rmse:.3f} (meV/Å)\nR\u00B2: {r2:.3f}'
        else:
            text = f'MAE: {mae:.3f} (meV/Å)\nRMSE: {rmse:.3f} (meV/Å)'

        ax.text(0.05, 0.9, text, transform=ax.transAxes, va='top',
                bbox=dict(facecolor='white', alpha=.8),
                fontsize=self.args.annotation_fontsize)

        # 设置标签和标题 - 使用MLIP名称
        ax.set_xlabel('DFT Force Component (eV/Å)', fontweight='bold')
        ax.set_ylabel(f'{self.mlip_name} Force Component (eV/Å)', fontweight='bold')
        ax.text(0.03, 0.98, label, transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='top')
        return sc

    def _plot_stress_subfigure(self, ax, dft_data, sus_data, mae, rmse, r2, label='(c)'):
        """绘制应力子图"""
        # 检查是否有数据
        if len(dft_data) == 0:
            ax.text(0.5, 0.5, 'No stress data available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=self.args.annotation_fontsize, style='italic')
            ax.set_xlabel(f'DFT Stress Component ({self.stress_unit_label})', fontweight='bold')
            ax.set_ylabel(f'{self.mlip_name} Stress Component ({self.stress_unit_label})', fontweight='bold')
            ax.text(0.03, 0.98, label, transform=ax.transAxes,
                    fontsize=40, fontweight='bold', va='top')
            return None

        sc = self.density_scatter(ax, dft_data, sus_data)

        # 设置统一范围、统一x/y刻度；是否关于0对称由 --signed-axis-symmetric 控制
        lim_low, lim_high = self._set_equal_parity_axes(ax, dft_data, sus_data, symmetric_zero=self.signed_axis_symmetric)
        ax.plot([lim_low, lim_high], [lim_low, lim_high], '--',
                color='gray', linewidth=self.args.linewidth)

        # 添加指标文本（检查是否为有效数字）
        if not np.isnan(mae) and not np.isnan(rmse):
            if self.show_R2 and not np.isnan(r2):
                text = f'MAE: {mae:.3f} ({self.stress_unit_label})\nRMSE: {rmse:.3f} ({self.stress_unit_label})\nR²: {r2:.3f}'
            else:
                text = f'MAE: {mae:.3f} ({self.stress_unit_label})\nRMSE: {rmse:.3f} ({self.stress_unit_label})'
        else:
            text = 'Stress metrics\nnot available'

        ax.text(0.05, 0.9, text, transform=ax.transAxes, va='top',
                bbox=dict(facecolor='white', alpha=.8),
                fontsize=self.args.annotation_fontsize)

        # 设置标签和标题 - 使用MLIP名称
        ax.set_xlabel(f'DFT Stress Component ({self.stress_unit_label})', fontweight='bold')
        ax.set_ylabel(f'{self.mlip_name} Stress Component ({self.stress_unit_label})', fontweight='bold')
        ax.text(0.03, 0.98, label, transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='top')
        return sc

    def _print_summary(self, metrics):
        """打印总结信息"""
        print("\n" + "=" * 60)
        print(f"SUMMARY OF METRICS (DFT vs {self.mlip_name}):")
        print("=" * 60)

        # 能量
        print(f"Energy:    MAE = {metrics['energy']['mae']:.3f} meV/atom, "
              f"RMSE = {metrics['energy']['rmse']:.3f} meV/atom, "
              f"R² = {metrics['energy']['r2']:.3f}")

        # 力（分量指标，与logout一致）
        print(f"Force (components, aligned with logout):")
        print(f"  MAE = {metrics['force']['mae_components']:.3f} meV/Å, "
              f"RMSE = {metrics['force']['rmse_components']:.3f} meV/Å, "
              f"R² = {metrics['force']['r2_components']:.3f}")
        print(f"  Fx: MAE={metrics['force']['mae_x']:.3f} meV/Å, "
              f"RMSE={metrics['force']['rmse_x']:.3f} meV/Å, "
              f"R²={metrics['force']['r2_x']:.3f}")
        print(f"  Fy: MAE={metrics['force']['mae_y']:.3f} meV/Å, "
              f"RMSE={metrics['force']['rmse_y']:.3f} meV/Å, "
              f"R²={metrics['force']['r2_y']:.3f}")
        print(f"  Fz: MAE={metrics['force']['mae_z']:.3f} meV/Å, "
              f"RMSE={metrics['force']['rmse_z']:.3f} meV/Å, "
              f"R²={metrics['force']['r2_z']:.3f}")

        # 力（合力指标）
        print(f"Force (magnitude |F|):")
        print(f"  MAE = {metrics['force']['mae_magnitude']:.3f} meV/Å, "
              f"RMSE = {metrics['force']['rmse_magnitude']:.3f} meV/Å, "
              f"R² = {metrics['force']['r2_magnitude']:.3f}")

        # 应力（如果可用）
        def print_stress_metrics(stress, title=None):
            unit = stress['unit']
            if title is None:
                title = f"Stress (components, {unit}):"
            print(title)
            print(f"  MAE = {stress['mae']:.3f} {unit}, "
                  f"RMSE = {stress['rmse']:.3f} {unit}, "
                  f"R² = {stress['r2']:.3f}")
            print(f"  Sxx: MAE={stress['mae_xx']:.3f} {unit}, "
                  f"RMSE={stress['rmse_xx']:.3f} {unit}, "
                  f"R²={stress['r2_xx']:.3f}")
            print(f"  Syy: MAE={stress['mae_yy']:.3f} {unit}, "
                  f"RMSE={stress['rmse_yy']:.3f} {unit}, "
                  f"R²={stress['r2_yy']:.3f}")
            print(f"  Szz: MAE={stress['mae_zz']:.3f} {unit}, "
                  f"RMSE={stress['rmse_zz']:.3f} {unit}, "
                  f"R²={stress['r2_zz']:.3f}")
            print(f"  Syz: MAE={stress['mae_yz']:.3f} {unit}, "
                  f"RMSE={stress['rmse_yz']:.3f} {unit}, "
                  f"R²={stress['r2_yz']:.3f}")
            print(f"  Sxz: MAE={stress['mae_xz']:.3f} {unit}, "
                  f"RMSE={stress['rmse_xz']:.3f} {unit}, "
                  f"R²={stress['r2_xz']:.3f}")
            print(f"  Sxy: MAE={stress['mae_xy']:.3f} {unit}, "
                  f"RMSE={stress['rmse_xy']:.3f} {unit}, "
                  f"R²={stress['r2_xy']:.3f}")

        if metrics['stress']['available']:
            print_stress_metrics(metrics['stress'])
            all_stress_units = metrics['stress'].get('all_units', {})
            active_unit = metrics['stress']['unit']
            for other_unit in ('eV', 'GPa'):
                if other_unit != active_unit and other_unit in all_stress_units:
                    print_stress_metrics(
                        all_stress_units[other_unit],
                        title=f"Stress (components, {other_unit}; converted for reporting):"
                    )
        else:
            print("Stress:    Not available in input files")
        print("=" * 60)


def sus2_plot_errors_main():
    """主函数：解析命令行参数并运行DFT和SUS2比较"""
    parser = argparse.ArgumentParser(
        description='DFT和SUS2结果比较工具 - 支持CFG和XYZ格式，生成能量、力、应力的综合比较图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  使用命令 mlp_exe calc-efs xxx.mtp xxx.cfg pre_xxx.cfg(pre_xxx.cfg是SUS2-MLIP预测的cfg)

  # 基本使用（默认显示力分量，支持CFG或XYZ格式）
  python sus2_plot_errors.py dft_data.cfg sus2_data.cfg
  python sus2_plot_errors.py dft_data.xyz sus2_data.xyz
  python sus2_plot_errors.py dft_data.cfg sus2_data.xyz 

  # 指定MLIP名称（如MTP, GAP, NEP等）
  python sus2_plot_errors.py dft.cfg sus2.cfg --mlip-name MTP
  python sus2_plot_errors.py dft.cfg sus2.cfg --mlip-name GAP

  # 多元素系统（如Na-Ca-Ti-O）
  python sus2_plot_errors.py dft.cfg sus2.cfg --elements Na Ca Ti O

  # 显示力分量（Fx, Fy, Fz）
  python sus2_plot_errors.py dft.cfg sus2.xyz --force-mode components

  # 保留临时文件
  python sus2_plot_errors.py dft.cfg sus2.cfg --keep-temp True

  # 自定义参数
  python sus2_plot_errors.py dft.cfg sus2.cfg --num-processes 32 --output comparison.jpg
  python sus2_plot_errors.py dft.cfg sus2.cfg --force-mode components --save-data True
        """
    )

    # 必需参数
    parser.add_argument('dft_file', type=str, help='DFT数据文件路径 (.cfg 或 .xyz格式)')
    parser.add_argument('sus_file', type=str, help='SUS2数据文件路径 (.cfg 或 .xyz格式)')

    # 新增参数：MLIP名称
    parser.add_argument('--mlip-name', type=str, default='SUS²',
                        help='MLIP名称，用于图例和标签（如MTP, GAP, NEP, SUS²等），默认: SUS²')

    # 新增参数：CFG文件相关
    parser.add_argument('--elements', type=str, nargs='+',
                        default=['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
                                 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                                 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                                 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                                 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                                 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'],
                        help='元素符号列表（按原子类型编号顺序），例如：--elements Na Ca Ti O')
    parser.add_argument('--keep-temp', type=str, default='False',
                        help='是否保留临时转换的XYZ文件 (True/False)')
    parser.add_argument('--skip-structure-indices', type=int, nargs='*', default=[],
                        help='比较前跳过DFT文件中的1-based结构序号，例如：--skip-structure-indices 16374 25884')

    # 所有可选参数的定义（使用元组列表）
    optional_args = [
        # 力显示模式
        ('--force-mode', str, 'components', '力显示模式: components-分量Fx/Fy/Fz(默认, 与logout对齐), magnitude-合力大小',
         {'choices': ['magnitude', 'components']}),

        # 应力显示单位
        ('--stress-unit', str, 'ev', '应力显示/输出单位: ev 或 GPa；默认ev显示',
         {'choices': ['eV', 'ev', 'GPa', 'gpa']}),

        # 并行处理参数
        ('--num-processes', int, 24, '并行进程数', {}),

        # 绘图参数
        ('--output', str, 'efs.jpg', '输出图片文件名', {}),
        ('--figsize', float, [30, 10], '图形尺寸 (宽度 高度)', {'nargs': 2, 'metavar': ('WIDTH', 'HEIGHT')}),
        ('--dpi', int, 300, '输出图片DPI', {}),
        ('--cmap', str, 'Spectral_r', '颜色映射名称', {}),
        ('--scatter-size', int, 10, '散点大小', {}),
        ('--bins', int, 120, '直方图bin数量,与密度相关', {}),
        ('--axis-padding', float, 0.04, 'parity plot坐标轴留白比例；默认0.04表示真实数据范围外多留4%', {}),
        ('--signed-axis-symmetric', str, 'False', 'force/stress有符号分量坐标轴是否强制关于0对称 (True/False)；默认False更贴合实际数据范围',
         {'type': lambda x: (str(x).lower() in ['true', '1', 'yes', 'y'])}),

        # 字体和样式参数
        ('--fontsize', int, 30, '字体大小', {}),
        ('--tick-labelsize', int, 30, '刻度标签大小', {}),
        ('--legend-fontsize', int, 20, '图例字体大小', {}),
        ('--title-fontsize', int, 32, '标题字体大小', {}),
        ('--annotation-fontsize', int, 20, '注解字体大小', {}),
        ('--cbar-fontsize', int, 28, '颜色条字体大小', {}),
        ('--cbar-tick-size', int, 22, '颜色条刻度标签大小', {}),
        ('--linewidth', float, 4, '线宽', {}),

        # 功能开关
        ('--show-r2', str, 'True', '是否显示R²指标 (True/False)',
         {'type': lambda x: (str(x).lower() in ['true', '1', 'yes', 'y'])}),
        ('--save-data', str, 'False', '是否保存原始数据到CSV文件 (True/False)',
         {'type': lambda x: (str(x).lower() in ['true', '1', 'yes', 'y'])})
    ]

    # 批量添加可选参数
    for arg_name, arg_type, arg_default, arg_help, arg_extras in optional_args:
        # 对于已经明确指定type的参数（如lambda函数），使用arg_extras中的type
        if 'type' in arg_extras:
            parser.add_argument(arg_name,
                                default=arg_default,
                                help=arg_help,
                                **arg_extras)
        else:
            parser.add_argument(arg_name,
                                type=arg_type,
                                default=arg_default,
                                help=arg_help,
                                **arg_extras)

    parser.add_argument('--version', action='version',
                        version='DFT-SUS2 Comparator v1.6 (stress unit selectable; default eV)')

    # 解析参数
    args = parser.parse_args()

    # 转换keep-temp字符串为布尔值
    if isinstance(args.keep_temp, str):
        args.keep_temp = args.keep_temp.lower() in ['true', '1', 'yes', 'y']

    args.stress_unit = 'GPa' if str(args.stress_unit).lower() == 'gpa' else 'eV'

    print("=" * 60)
    print("DFT-MLIP COMPARATOR (with stress handling)")
    print("=" * 60)
    print(f"DFT data: {args.dft_file}")
    print(f"MLIP data: {args.sus_file}")
    print(f"MLIP name: {args.mlip_name}")
    print(f"Elements: {args.elements}")
    print(f"Output file: {args.output}")
    print(f"Number of processes: {args.num_processes}")
    print(f"Force display mode: {args.force_mode}")
    print(f"Color map: {args.cmap}")
    print(f"Show R²: {args.show_r2}")
    print(f"Save data: {args.save_data}")
    print(f"Keep temp files: {args.keep_temp}")
    print("-" * 60)

    # 创建比较器实例
    comparator = DFTSUS2Comparator(args)

    # 创建综合图
    total_t0 = time.time()
    try:
        metrics = comparator.create_combined_plot(
            dft_file=args.dft_file,
            sus_file=args.sus_file
        )

        print(f"\nTotal execution time: {(time.time() - total_t0) / 60:.2f} min")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sus2_plot_errors_main()
