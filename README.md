# DCBF SUS2-SH CPU 离线版

支持 static-radial 和 coordination-effect 的采样、描述符筛选、Reduce 与 checkpoint 接续训练。训练和 MD 使用 CPU；LAMMPS 包含 SUS2 embedded、PLUMED 2.8.2 和 MC atom/swap。

## 安装

解压到新的目录后执行：

```bash
bash install.sh
source activate.sh
bash verify.sh
dcbf -h
```

安装和日常使用不需要下载依赖，也不需要集群的 Intel 模块。目标平台为 Linux x86_64、AVX2 CPU，glibc 至少 2.17。更换安装位置时应重新解压发行压缩包，不能直接搬动已展开的 Conda 环境。

安装脚本会恢复 Windows 压缩过程可能丢失的 Linux 运行库符号链接，并由包内 wrapper 优先使用随包提供的 GCC/Intel MPI 运行库。不要直接绕过 `runtime/bin/run_cpu.sh` 调用 `lmp-cpu`；如果目标目录挂载为 `noexec`，请换到允许执行二进制的目录。

VASP、ABACUS、CP2K、QE 及赝势由用户提供。请根据自己的集群修改队列、核数、DFT 环境和命令。包内 MPI 仅由 SUS2/LAMMPS 启动脚本设置，DFT 使用其自身配置的 MPI。

`xyz2cfg` 现在同时支持周期结构和非周期分子。输入 XYZ/EXTXYZ 没有晶格且关闭 PBC 时，输出 CFG 会省略 `Supercell` 段；带晶格的周期帧仍按原格式写出。

## 配置

```json
{
  "run_dir": "./workspace",
  "init_dataset": {
    "xyz_input": null,
    "structure_source_dir": "./stru",
    "init_source_dir": "./init",
    "builder": {"enabled": true, "md": {"enabled": true}}
  },
  "sampling": {
    "scheduler": {
      "submission_backend": "bsub",
      "train": {
        "sus2_mlp_exe": "${DCBF_HOME}/runtime/bin/mlp-sus2",
        "train_env": "source ${DCBF_HOME}/activate.sh",
        "original_command": "{sus2_mlp_exe} train {initial_yaml}",
        "subsequent_command": "{sus2_mlp_exe} train {continuation_yaml}",
        "queue": "33", "cores": 40, "ptile": 40,
        "model_yaml": "./init/sus2-static.yaml",
        "sampling": {"initial_iterations": 600, "subsequent_iterations": 300},
        "final": {"enabled": true, "model_yaml": "./init/sus2-effect.yaml", "iterations": 3000, "model_name": "final.sus2"}
      },
      "lmp": {
        "lmp_exe": "${DCBF_HOME}/runtime/bin/lmp_mpi",
        "lmp_env": "source ${DCBF_HOME}/activate.sh",
        "queue": "33", "cores": 40, "ptile": 40
      },
      "scf": {
        "queue": "33", "cores": 40, "ptile": 40,
        "cal_engine": "vasp", "template_dir": "./init",
        "calc_dir_num": 10, "force_threshold": 20,
        "clean_dcbf_environment": true,
        "dft_env": "module load vasp/5.4.4",
        "dft_command": "mpirun -np $NP vasp_std"
      }
    }
  }
}
```

`model_yaml` 位于 `sampling.scheduler.train`，描述采样阶段的模型架构、static/effect 模式和损失权重。默认采样使用较快的 `init/sus2-static.yaml`。最终训练可以在 `train.final.model_yaml` 中单独指定：`static-radial` 自动从最新完整 checkpoint 接续，`coordination-effect` 自动重新初始化并使用全部已标注数据；不再填写 `mode`。初始 MD 固定使用包内 87 元素 `sus2_87.sus2`，不再填写 `model_path` 或元素列表。普通后续代和候选更新仍从最新完整 checkpoint 接续；接续 YAML 不含 `model` 架构组，使用 `data.model` 和 `initialization.parameters: same`；BFGS 历史会重新开始。

采样筛选默认使用模型 metadata 中的二体和三体 outer scalar，四体不进入默认采样筛选。Reduce 仍可在自己的 JSON 中显式填写 `body_list`，例如只选 `two`、`three`、`four`，或使用 `all`。

训练、LAMMPS 和 SCF 配置分别位于 `scheduler.train`、`scheduler.lmp` 和 `scheduler.scf`。程序路径和 `source` 环境命令会在配置中显式保留；`${DCBF_HOME}` 在运行配置中解析为实际安装目录。错误路径会报错。CLI 默认配置使用 `~/.dcbf/sus2sh_cli_defaults.json`，不会读取或覆盖旧版 `cli_defaults.json`。

安装时会检查 `bsub` 和 `sbatch`，并把包内所有示例 JSON 的 `submission_backend` 自动改成检测到的调度器；检测结果保存在 `runtime/scheduler.detected.json`。如果安装节点暂时没有调度器命令，示例 JSON 保持原值，之后可设置 `DCBF_SUBMISSION_BACKEND=bsub` 或 `DCBF_SUBMISSION_BACKEND=sbatch` 后重新运行 `install.sh`。运行时不再临时猜测调度器，而是使用 JSON 中已保存的值。

```bash
mkdir my_case
cd my_case
dcbf create-init
# 修改 JSON 中的 DFT 设置、模型 YAML 和输入数据后运行：
bash bsub.sh
# 启明节点使用：
DCBF_CONFIG=dcbf.init_dataset.vasp.qiming.json bash bsub.sh
```

`create-init` 会生成两个服务器配置：`dcbf.init_dataset.vasp.test.json` 和
`dcbf.init_dataset.vasp.qiming.json`。两份配置的采样阶段使用 `init/sus2-static.yaml`，最终训练默认使用
`init/sus2-effect.yaml` 从头重训；静态和配位 YAML 都统一放在
`init/`，并附带 `bsub.sh` 和 `bsub.lsf` 两个提交脚本。示例不再复制 `initial.xyz`。
Si 的演示数据与模型用于软件验收，不是通用势，
也不代表对其他材料的精度保证。多元素训练时程序会从训练集推断元素顺序，并在生成的训练 YAML 中自动写入 `model.species_count`；续训会检查 checkpoint 与训练集的元素数一致。

提交脚本使用已安装包的 `DCBF_HOME`。本地运行：

```bash
export DCBF_HOME=/path/to/DCBF-SUS2SH-CPU-20261102-sus2_87
bash bsub.sh
```

启明 LSF 运行：

```bash
export DCBF_HOME=/path/to/DCBF-SUS2SH-CPU-20261102-sus2_87
bsub < bsub.lsf
```

通过 `DCBF_CONFIG=dcbf.init_dataset.vasp.test.json` 可以切换脚本使用的服务器配置。`bsub.sh` 不使用前台模式，直接按配置运行；包内任务的 bsub/sbatch 行为由安装时写入 JSON 的 `submission_backend` 决定。

## Reduce 与描述符

```bash
dcbf reduce source/DCBF/example/reduce/user_sus2/dcbf.reduce.candidate_only.example.json
```

PCA 覆盖分析作为独立命令保留，并使用与 Reduce 相同的 SUS2 metadata 描述符解析：

```bash
dcbf coverage-pca --input sampled_training_dataset.xyz --query query.xyz \
  --model current.sus2 --body-list two three --show-input
```

它会写出 `coverage_summary.csv`、`pca_txt/` 和 `pca_coverage.png`。`--run-dir`
可以从已准备的工作目录读取 `sampled_training_dataset.xyz`、已有 query 文件和最新
`.sus2` 模型。没有现成 query 时，程序会从 `run_dir/stru` 读取结构，调用
当前运行配置中的 SUS2 LAMMPS `pair_style sus2/embedded` 生成
`coverage_query_lammps/query.xyz`；调度模式沿用 `scheduler.lmp_*`，本地模式
只有显式指定 `--lammps-run-mode local` 才启用。PCA 只用于分析和画图，不改变采样筛选；采样仍只使用
`mlp_encode_model`。旧版 `--mtp`、`--mtp-type` 固定列号和 DAS 入口不会由此命令恢复。

采样示例默认在采样和可选最终训练都完成后自动执行这一步，并从工作目录选择最新的完整 `.sus2` checkpoint。它按 `main` 字段拆分
LAMMPS 轨迹，默认选择 init、loop 中间段和最后一段，并以最后一段 loop 作为 coverage grid，结果写入 `xyz_pca_coverage_results/` 并随 summary 一起保存。
如果只想关闭自动分析，在 JSON 的 `sampling.coverage_plot` 中设置 `enabled: false`；它只做后处理，不参与候选选择。

公开配置只需要填写 `enabled`、`body_list`、`query_structures`、`loop_select`、`coverage_grid`、`pca_fit_source`、`descriptor_workers` 和 `coverage_workers`；输出目录、二维模式、LAMMPS 调度和查询缓存等使用内部默认值。手动 `dcbf coverage-pca` 的既有参数和语义保持不变，`--coverage-workers` 是新增的可选参数。

Reduce 的 `reduce.starting_model_path` 填用于描述符计算的 `.sus2` 模型，也可以填 `"builtin:sus2_87"`。后者自动核对模型 SHA-256 并使用随包保存的完整 87 元素映射，省略 `ele` 和 `sort_ele` 即可；例如 Si 的模型类型号是 12，不能按输入文件仅含 Si 而改成 0。自定义模型仍通过 `parameter.ele` 给出完整元素顺序，自定义非原子序映射时设置 `sort_ele: false`。

`models/sus2_87.sus2` 是用户提供的 SUS2-SH v1.8.0 模型的原样副本，原文件名为 `current.best.mtp`。其 outer 描述符二/三/四/五/六体维度为 4/20/56/48/16，共 144；内部 gate 元数据另外保存。该模型的计算接口为 SUS2-SH，文件名只表示其 87 元素通用势用途。初始数据集 MD 与 Reduce 可共用该模型；采样迭代训练保持 YAML 新建和 checkpoint 接续流程。支持某个元素不等于已经验证所有材料和热力学条件的精度。

通用势 Reduce 示例：`source/DCBF/example/reduce/universal_sus2_87/dcbf.reduce.candidate_only.example.json`。采样 `create-init`
示例使用两份服务器配置，运行前设置种子结构、DFT 队列和赝势。

内置 87 元素势现在也可作为 CPU BFGS 接续训练种子。它继承完整 `.sus2` 的物理参数，
使用原生内核已有的 weight-norm 径向优化坐标；不做径向符号修复、不重新随机初始化。
这不是恢复原 GPU 训练的 Adam、EMA 或学习率历史，BFGS 历史重新开始。
推荐配合 `source/DCBF/example/sus2_87-continue.yaml` 和
`source/DCBF/example/reduce/universal_sus2_87/dcbf.reduce.update_model.json` 使用。
Reduce 在识别原始通用势哈希后自动保留 weight-norm 设置到后续批次、恢复和最终训练；
显式要求对该种子使用 physical 时会提前报错。微调输出不覆盖包内原始通用势，完整87元素编号不变。
自己单独接续微调后的模型时，继续使用这份 weight-norm YAML，并保留完整元素映射。
普通采样仍从 YAML 新建 gen_0，再逐代接续，不自动把初始 MD 势当作训练种子。

独立接续训练（先生成输入与提交脚本，检查后加 `--submit`）：

```bash
dcbf train data.xyz --model builtin:sus2_87 \
  --model-yaml "$DCBF_ROOT/example/sus2_87-continue.yaml" \
  --work-dir ./sus2_87_finetune --queue 33 --cores 40 --ptile 40 --max-iter 1000
```

后续使用 `sus2_87_finetune/current.sus2` 作为 `--model`，继续使用同一 YAML；同时保留生成的
`current.sus2.elements.json`。该文件用模型哈希绑定完整元素编号，避免只含Si的数据被错误重编号为0。
优化目标和权重以 CPU YAML 为准，不会把 GPU 的 DPA4-MAE/20:20:5 自动解释为相同的原生损失。
CPU dense BFGS 的 Hessian 内存随参数数平方增长；本87元素势约有39013个优化坐标，
一份完整矩阵约11.34 GiB，不应按小型单元素模型估算资源。示例默认显示40核，实际运行仍需根据可用内存调整。
此固定版本在训练帧数少于MPI进程数时会退回非分布式矩阵路径，多活跃进程可能各分配完整矩阵；训练数据过少时不要盲目扩大进程数。
检查训练日志中的 `distributed-dense-bfgs` 和 `train_comm_is_world=true`。
8进程、6帧的验收尝试曾被信号9终止，LSF峰值内存50.8 GB；这不是任意MPI核数可用的承诺。
更大数据集可按资源调整进程数，但需重新核对通信域和内存。示例步数不是收敛或材料精度保证。

输出第 q 个 outer descriptor 与 `sh_scalar_info[q]` 对齐；`alpha_moment_mapping` 描述输出顺序中的底层 moment 索引。`all` 选择全部 outer scalar，`two/three/four/five/six` 分别选择 metadata 中 body_order 为 2/3/4/5/6 的列，维度随模型动态变化；采样默认只使用 `two` 和 `three`。gate、joint/readout 列不自动混入这些分组。effect 的 outer body 分类不等于整个配位依赖势能具有严格的有限体阶截断。

模型文件底层仍可能以 `MTP` 开头；本版本通过 `potential_tag=SUS2-SH`、有效 metadata 与 `.sus2` 格式共同识别新版模型。仅把旧 `.mtp` 改扩展名不能变成新版模型。

保留 candidate-only、reference-guided、块式/并行解析、缓存、断点续跑、状态布居、最小覆盖及全局去冗余。PCA 覆盖分析仍可通过独立的 `coverage-pca` 命令运行；它不参与采样选择。DAS、固定分歧阈值和 mean-descriptor 编码不再提供。

### Reduce 中执行 SCF 和模型更新

新增 `reduce.update_model_after_selection`、`reduce.run_scf_labeling`，默认均关闭。启用后，仍按原分块顺序处理候选池，
每轮执行“覆盖率评估 → 按阶段预算筛选 → SCF（或读取已有 EF/EFS）→ 合并 → checkpoint 接续”，
下一轮用更新模型和扩大的训练集重新筛选剩余池，不是从头重扫候选池。训练和 DFT 沿用采样的字段，不要求填写 MD 参数。

每轮开始时计算当前训练集对剩余候选池的逐元素覆盖率（按 body 取 min），用
`reduce.selection_budget_by_coverage` 和 `reduce.coverage_target_by_stage` 取第一个尚未达标阶段的预算；
达到最终阶段阈值即以 `coverage_target_reached` 停止。阶段由实际覆盖率推进，`max_update_rounds`
只是防止无限循环的安全上限，达到时 `stop_reason=max_update_rounds_reached`，可在同一 checkpoint 上中途调大。
一次性 `candidate_only` / `reference_guided` 可以同时省略这两个字段，不做任何预算/阈值截断，
输出全部去冗余/min-cover 结果；显式保留旧预算时只按第一个阶段截断，且先按 min-cover 贡献降序排序。
`reduce.max_total_new_structures` 已废弃：出现时打印一次警告并忽略；`max_new_labels` 仍会被拒绝。
每轮原始筛选、预算内接纳结构、标签、模型和日志均保留。
设置 `reduce.remain_xyz: null` 可关闭剩余 XYZ 输出，但来源索引仍保留；省略该字段保持原来的默认输出。

详见 [Reduce 更新与恢复说明](source/DCBF/example/reduce/REDUCE_UPDATES.zh.md)。
同目录提供已有标签续训、仅 SCF，以及 SCF 加续训三套 JSON 示例。

## 增强 MD

普通 MD 使用 `init/lmp_in.py`。PLUMED 或 MCMD 可分别选用 `lmp_in_plumed.py`、`lmp_in_mcmd.py` 作为 `lmp_in.py`。PLUMED 的 collective variables 和偏置参数必须按实际体系设置；MC 需要至少两种模型原子类型。候选结构的能量力重算应去除偏置和交换 fix。

初始数据集的 ASE MD/relax 适配器通过包内 `pysuperlinear.SuperlinearCalculator` 常驻 C++ 模型求能量、力和应力，不生成临时 CFG，也不为每个时间步启动外部 SUS2 进程。长时间生产采样仍使用 LAMMPS 路径。

## 验收与构建

`verification/verify_portable.py` 检查两模型的 descriptor、Reduce 多条 I/O 路径、缓存与续跑、native/LAMMPS E/F/virial、PLUMED、MC 和初始 MD。报告写入运行目录下新建的 `.verification/` 子目录，不随正式运行包保存历史验收报告。

`runtime/COMPONENTS.json` 和 SHA-256 文件记录实际组件。`source/upstream/`、SUS2 源码快照、embedded 接口和 `build/` 提供源代码与构建说明。重编译需要相应编译器、开发库与 CMake；运行已编译的发行包不需要它们。
## 采样筛选的低内存路径

采样阶段会按 MD 目录和 descriptor_chunk_size 生成有序分片，直接解析到磁盘 memmap descriptor store。分片解析使用 encoding_cores 的受控并行度；处理完一个 body 后释放对应内存。筛选结果、帧顺序和原有 coverage/minimum-cover 规则保持一致。

高级参数可以写在 mlp_encode_model 的参数中：

```yaml
descriptor_chunk_size: auto
keep_intermediate: false
```

省略时自动使用 1000000 帧分片并清理文本中间文件。需要复核 cfg/out 时设置 keep_intermediate=true。普通采样配置无需增加字段。
