# DCBF Update - 2026-09-24

## 中文

- 修正 NPT 晶胞体积截断：首次出现 `V/V0 > npt_max_cell_volume_filter_factor` 后，当前帧及其后续帧永久丢弃；即使后续体积恢复，也不会重新进入采样、覆盖率或候选筛选。
- DAS 现在先对完整 MD 轨迹逐帧执行 NPT 体积检查，再对稳定前缀执行 ambiguity 筛选。即使 ambiguity 阈值没有变化，启用 NPT 体积过滤时仍会执行体积检查；NVT 和 `null` 体积过滤保持原行为。
- `dump2cfg()` 改为单遍读取轨迹，同时检查原子数和晶胞体积，记录实际处理帧数、保留帧数、首次失败 step 和失败原因。正常轨迹的 CFG 输出与原版本保持一致，不改变 `.out` 字段不足时的既有处理行为。
- Reduce 的低内存处理：连续 `float64/int64` 描述符存储、分块编码、快速 XYZ/EXTXYZ I/O、紧凑索引、逐维度最小覆盖和全局反向去冗余；不改变联合最小覆盖、预算、状态布居或筛选数学逻辑。
- `reference_guided` Reduce 新增按 chunk 自动断点续跑：每个完成 chunk 原子保存输入/模型/筛选参数指纹和已选结构；中断后重新运行同一命令会从下一个未完成 chunk 继续。`keep_intermediate=false` 仍为默认值，成功后清理中间文件；输入、模型或筛选参数改变时自动拒绝旧 checkpoint。

## English

- Fixed NPT cell-volume truncation: once `V/V0 > npt_max_cell_volume_filter_factor` is observed, the current frame and the entire remaining suffix are permanently discarded. Later volume recovery cannot re-enter sampling, coverage, or candidate selection.
- DAS now checks every MD frame for the NPT volume guard before ambiguity filtering is applied to the stable prefix. The volume check still runs when ambiguity thresholds are unchanged; NVT and `null` volume filtering retain their previous behavior.
- `dump2cfg()` now reads each trajectory in one pass while checking atom counts and cell volumes, and records processed/kept frames, the first failed step, and the failure reason. Normal CFG output remains byte-identical to the previous path, and the existing behavior for incomplete `.out` fields is unchanged.
- Reduce low-memory path: contiguous `float64/int64` descriptor storage, chunked encoding, fast XYZ/EXTXYZ I/O, compact indices, per-dimension minimum cover, and global reverse pruning. Joint minimum cover, budgets, state-population constraints, and selection mathematics are unchanged.
- `reference_guided` Reduce now supports automatic chunk-level resume: after each completed chunk it atomically stores input/model/selection fingerprints and selected structures, and rerunning the same command resumes from the next unfinished chunk. `keep_intermediate=false` remains the default and cleans intermediates after success; changing inputs, the model, or selection parameters invalidates the old checkpoint.
