# Reduce 内接 SCF 和势函数续训

沿用 `dcbf reduce config.json`。`reference_guided` 仍按原来的 chunk 范围依次处理一遍。
`chunk_size` 默认 1000000，是近似分块目标，不是严格每块帧数。更新模型只影响后续轮次；已经处理的块不会自动重扫。

| 新参数 | 默认值 | 用途 |
|---|---|---|
| `reduce.update_model_after_selection` | false | 每轮成功接纳数据后，从上一完整 checkpoint 接续训练；仅 reference_guided |
| `reduce.run_scf_labeling` | false | 对本轮接纳结构统一做固定结构 SCF |
| `reduce.max_update_rounds` | 1 | 仅安全上限；达到时 `stop_reason=max_update_rounds_reached`，可在 checkpoint 中调大后继续 |
| `reduce.selection_budget_by_coverage` | 无 | 各阶段结构预算，例如 `[12,8,5]` 或 `[5000,3000,2000]`；update/SCF 必填 |
| `reduce.coverage_target_by_stage` | 无 | 各阶段逐元素覆盖率目标；update/SCF 必填，达到最终值即 `coverage_target_reached` |

一次性 `candidate_only` / `reference_guided` 可以同时省略预算和目标：不做任何预算/阈值截断，
输出全部去冗余/min-cover 结果。旧字段 `max_total_new_structures` 已废弃：出现时打印一次警告并忽略，
不再限制累计新增结构数；旧字段 `max_new_labels` 仍会被拒绝。

两开关都关闭、未指定标签上限时仍执行原 Reduce。无 SCF 的更新要求种子和候选都有 E/F；应力可选。
`training.labels` 可省略或设 auto；全部有应力/virial 时使用 EFS，否则使用 EF 并把 virial_weight 设为0。
显式 efs 要求每个训练结构都有有效应力；不补造零应力。

训练参数沿用 `training.sampling.subsequent_iterations`；模板示例为1000步，省略时代码默认300。
`potential.model_yaml` 提供损失和训练配置，必须与输入 checkpoint 的 static/effect 类型一致。 仅在 `update_model_after_selection: true` 时需要；一次性选样和仅 SCF 可省略。
架构、完整元素顺序和物理参数来自 `reduce.model_path`；续训固定 parameters=same，不做随机初始化。
续训一律提交到集群调度器，复用采样的 LSF/Slurm 提交方式；`training.submit` 不再支持本地训练。
若旧配置仍写 `training.submit: false`，该值会被忽略并打印一次提示，训练仍提交集群。
`training.final` 为可选最终训练，默认关闭。SUS2、MPI可执行程序默认取包内程序，无需填写路径。

内置 `builtin:sus2_87` 87元素势可开启 update_model_after_selection 和 training.final，
接续时继承完整模型参数，使用 CPU 原生 BFGS 与 weight-norm 径向优化坐标。
原始种子的内容哈希可识别其设置，即使复制改名也有效；后续批次保持同样的设置。
建议使用 `../sus2_87-continue.yaml`，自己的后续微调模型也继续使用此 YAML。
不恢复 GPU Adam/EMA/学习率历史，不做符号修复，不重新随机初始化，不改变完整87元素映射。

87元素接续示例使用2个MPI进程，训练数据不少于2帧。不要直接套用40核训练3帧种子：
此固定原生版本在训练帧数小于进程数时会退回非分布式矩阵路径，多活跃进程可能各分配
约11.34 GiB的完整BFGS矩阵。应检查日志的 `distributed-dense-bfgs` 和 `train_comm_is_world=true`。
8进程、6帧的验收尝试曾被信号9终止、峰值50.8 GB；目前不能承诺任意MPI核数都可运行。

SCF 参数使用顶层 `scheduler` 和 `dft`。字段与采样相同：`scf_cal_engine`、队列、核数、
`dft_env`、`dft_command`、`calc_dir_num`、`force_threshold` 和 `pending_warning_hours`。
新增 `dft.template_dir` 指向用户的 INCAR/INPUT/qe.in/cp2k.inp 等模板。
VASP 使用 NSW=0；模板需有 POTCAR_dir 或可解析的 POTCAR，或已配置 vaspkit。
SCF 示例里的队列40核仅是采样参数示例，运行前请按服务器设置 dft_env 和模板。
VASP、赝势和用户SCF工作目录不随安装包分发。

每块全部选中帧在 `work_dir/updates/chunks/chunk_*/selected_raw.xyz`。
预算内接纳帧和对应标签在 `work_dir/updates/main_0/gen_*/selected_new.xyz`、`labeled_new.xyz`。
同目录保留 train_after.xyz、model_before.sus2、train/run.yaml、checkpoint、日志和SCF原始输出。
`batch.json` 和 `label_sources.json` 记录原始候选帧编号与实际标签来源。
顶层 sample 输出按 append_current 决定是否带种子，remain 包含所有未入库候选，包括预算未接纳和未处理的块。
可显式设置 `reduce.remain_xyz=null`，不输出剩余XYZ，也不复制各块的剩余XYZ。
省略该字段仍沿用旧版默认输出。即使设为null，来源帧编号和原始池仍足以精确重建未入库部分。

每轮开始时先计算“当前训练集对剩余候选池”的逐元素覆盖率（按 body 取 min），再用
`selection_budget_by_coverage` / `coverage_target_by_stage` 决定本轮预算：取第一个尚未达标的阶段，
所以阶段完全由实际覆盖率推进，而不是按轮数硬编码。本轮结束合并标签后再次计算覆盖率并写入
`selection_report.json` 和 checkpoint；达到最终阶段阈值即以 `coverage_target_reached` 停止。
`max_update_rounds` 只是防止无限循环的安全上限，达到时 `stop_reason=max_update_rounds_reached`。
选样先按 min-cover 贡献（覆盖的未覆盖环境类数）降序排序，再按阶段预算截断。

同一个配置重跑会从 `reduce_update_checkpoint.json` 继续，不重算已成功SCF、不重复入库。
中途可以调大 `max_update_rounds`：它不参与 checkpoint 的严格设置签名，只校验新值不小于已完成轮数；
调小到已完成轮数以下会明确报错。其他预算/阈值/训练字段变化仍会拒绝续跑，并保留现场；请使用新工作目录。
模型更新后，后续轮次重新编码累计参考集和分箱，旧模型描述符不会混用。
输入文件、DFT模板或其他关键配置变化会拒绝续跑，并保留现场；请使用新 work_dir。
SCF或训练本身报错时保留作业和结果。提交回执丢失时按唯一作业名找回；无法核实时停止，避免重复作业。

普通 SUS2 示例位于 `reduce/user_sus2/`，内置 87 元素模型示例位于 `reduce/universal_sus2_87/`。
两组都提供 `candidate_only`、`reference_guided`、`scf`、`update_model_after_selection` 和
`update_model_after_selection_scf` 五种配置。
data/seed.xyz 和 data/pool.xyz 来自包内 Si75 示例，不构成新材料势函数精度证明。

`encoding_cores: -1` 表示使用当前进程可见的全部 CPU，`dimension_min_cover_workers: -1` 表示自动选择可见 CPU。
`xyz_io_mode=fast_extxyz` 原生支持没有 `Lattice` 的合法 EXTXYZ（按非周期结构处理，
生成的 CFG 不写 `Supercell`），不会因此回退 ASE。只有其它快路径无法解析的 schema
才回退 ASE，并在报告的 `xyz_io_fallbacks` 中记录。
