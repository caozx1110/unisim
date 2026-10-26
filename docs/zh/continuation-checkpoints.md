# 连续训练检查点

[English](../en/continuation-checkpoints.md)

`SimBackend.export_continuation_state()` 在完整步边界返回独立的 CPU 数组。 `import_continuation_state(state)` 将兼容状态恢复到已初始化的后端，同时保留 原生数组地址和张量别名。其他适配器抛出 `NotImplementedError`，回放快照不能替代。 当前 MJWarp 实现也拒绝固定模型变体及自定义 host/native 物理回调，因为这些额外 状态尚无检查点合同。

MJWarp 的源码实现将 schema 绑定到编译模型、原生包版本、环境数与容量配置、 图执行模式和子步采样绑定。保存全部原生 Data 数组（包括求解器 warmstart、 休眠、接触和约束状态）、随机化 Model 字段及派生常量、适配器缓存、待施加外力， 以及子步加速度与接触历史。导入先完整验证，再原位复制，不执行 reset 或 forward。 原生复制失败后标记后端不可继续使用。只读构造数组必须完全一致。 CUDA 图、事件及原生句柄重新构造，不直接序列化。

调用方必须暂停推进，另行保存任务/manager 状态、replay、learner 和所有 RNG， 并原子发布完整检查点。此适配器没有引入引擎 RNG。CPU 内存为一份独立快照加 至多一个原生字段的临时副本；`nbytes` 和 `nbytes_by_field` 报告载荷字节数。 模型、原生版本或图模式不一致时拒绝恢复。

此可选源码 API 已有 CPU 验证覆盖。每个训练 owner 在宣称完整恢复前仍须完成 真实 CUDA 新进程连续性对照。重新构造图本身不能证明数值轨迹一致；测试必须 比较保存状态和后续转移，覆盖接触、局部 reset、随机化参数及子步历史。
