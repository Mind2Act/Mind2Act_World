# PR6：曲带多米诺与被动响铃

公开编号 PR6；保留源目录/实例 PR8、原生任务 CP11。唯一运行入口为本框架
`tasks.PR6.scene.build`，使用共享 World/X5/runner，无旧环境、专家控制器或跨任务依赖。

规则依据现行飞书 `OUv8dRa1ZoPVS9xqGUbcNi94n6d` 第4–6节，快照 SHA256
`cb254b266e77e8eaf2ca762cc96870ea230ee301f267e25c633d9198e483cfd5`，
规则版本 `pr6-spec-20261030-v1`。Instruction 来自交付版30任务 JSON（SHA256
`9f0ca3f4890dc12b416d1ed738c0fa0adc30a90130ff6c0753dec2fb2167d4ae`），
Goal/RECIPE 来自上述现行快照，任务资产通过资产库 resolver。

计分为最高满足阶段0/10/20/100，不累加。无有效摆放、违规接触或第9次推动为0；
历史有效摆放为10；同回合从首块起合法激活、相邻接触、真实倒伏的前缀达到半数向上取整为20；
全部合法倒伏、末块碰铃体、铃体碰被动铃舌并达到实例响铃角度为100。
未响铃属于有效数值失败，可得0/10/20；不因原生状态尚在运行而判 INVALID。
初始推动、360 tick无进展后的允许重启共用8次预算，整个接触批次按进入时状态分类。
重新接触正在倒下的多米诺（含未激活修复目标）为违规；允许从前一原生采样直立静止状态开始的合法初推/重启，已获准持续的接触直到分离不重复计数。

每个120 Hz tick记录原生物体位姿、线/角速度、接触点/法线/分离距离；从真实包围盒、
台面支撑、释放、稳定和曲带几何重建历史摆放。私有 JSONL 追加写一次，证据是固定大小收据，
终局核验SHA并重放一次；不使用专家阶段、动作序号或声称的里程碑。

来源 `pr8-four-case-native-v4`，源码 revision
`094bd912a18b6a83ea1ac83d4b22683971a079ac`；逐文件SHA见 `config/source.json`。
只读来源为 GPU server `evaluations/replay-new15-20261031/code/snapshot/cases/N15/native/PR8`，
训练源 `runtime/PR8/abc-ebe28d3/source`。原生 `native.py` SHA
`6db31a2f77f9ffc8bac43d462039a02f9c33887f9ce96e5f6c5be30241f5b2eb` 已核对；
仅在显式自有 objects 子树保留源CCD、props128/8求解器、1mm接触偏移、0 rest offset；
全局120Hz/CCD/TGS交由共享simulator并记录原源期望，不写站台/其他任务/机器人。保留35mm初始夹爪开度、
原始环形铃体与独立被动铃舌碰撞几何，不扩大触发范围、换摩擦或调整成功阈值。
原快照第9节称缺失的 `contact_policy.py` 已在允许源中找到并核SHA
`27299e4e79be8d54e6a0f89277f4938382795aa5092e0bfbd320d211641ed796`。

从指定源实例生成，不选择第一条、不改seed、不覆盖输出：

```sh
python -m mind2act_eval.tasks.PR6.instances \
  --source-spec SELECTED_PR8_SPEC.json --source-spec-sha256 EXPECTED_SHA256 \
  --asset-entries ASSET_LIBRARY_ENTRIES.json --asset-version ASSET_TAG \
  --output NEW_PR6_INSTANCE.json
```

资产 entries 含相对ID `assets.station`/`assets.x5`，如需共享 relocation 则同时提供
`station_relocation_asset_id`。producer 核验完整原始spec字节SHA、内部caseSHA/seed/count，
保留源case ID，生成独立portable configSHA，不把旧config文件字节SHA误当规范JSON SHA。
选定训练来源为 Easy seed13、Medium seed1、Hard seed1（source-seeds-v1），
并非第8节历史H5检查样本；H5字节/回放配对由共享输入桥另行核验。

公共接口待办：原 Session.notice/action_enabled/public 经文本状态提供600 tick观察提示，
三档源display.enabled=false；当前 Observation 无公共status字段。保留canonical instruction，
不追加提示或泄漏私有证据，待T0统一公共接口。此端口不声称模型提示已经接通。

验证仅CPU：物理事实正反例、部分分/边界、源绑定、共享rest accessor、合成USD物理设置、
真实Scene测量接缝、收据篡改和固定大小证据。未启动Kit/GPU。
原生三档回放、三视频解码、首帧同视角及公开提示可见性由root共享队列验收；
未完成前不能宣称GPU、模型或正式评测通过。
