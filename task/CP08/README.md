CP08 仅接受 config/selected.json 明确绑定的三档 train2000 实例；source72 为120Hz物理时钟，匹配H5 action76，不是30Hz行号。

撤爪：飞书未给距离定义，按源实现。固定实际抓取臂发生实测张爪、持续双指分离并形成外部支撑，终局窗口内稳定且无 robot_contact。不增加撤离距离或命令静止门。

原 source environment_events.py:25 按最近实测 TCP 选择抓取臂；35–44 检查张爪与释放。TCP 沿用原 X5Kinematics.fk:27–32 的关节6坐标系 [.145,0,0]m，原生 sensor 读取实际 link6 位姿，raw 同时保存 flange 位姿和 TCP，Replay 独立复核该变换。source 文件 SHA 在 config/source.json。

CPU 测试与合并不代表原生首帧、动力学或视频验收。正式 eval profiles 尚未提供。

实际 flange pose 使用 RigidPrim 批量只读视图；先确认已有 rigid-body API 且为 articulation 非根 link，显式关闭 reset_xform_properties、prepare_contact_sensors、track_contact_forces、disable_stablization，并清空 contact filters。这样按 Isaac5.1 SDK 不会补 API、重置变换/速度或修改 robot sleep threshold。
