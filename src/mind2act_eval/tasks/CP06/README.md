CP06 selected source port

只接受三档 `cp06-<tier>-train-supplement-v1-000`，seed 分别为610000/620000/630000；正式eval slots未在本PR声明。指令从交付版SHA `9f0ca3f4890dc12b416d1ed738c0fa0adc30a90130ff6c0753dec2fb2167d4ae` 按档/语言唯一加载。

输入使用两层契约：

1. `spec_from_original` 先校验原 `spec.json` 全字节SHA、H5原instance身份和原 `config.task` declared canonical SHA；原声明不改。
2. 独立校验已证明的资源投影、derived Task SHA与完整station+CollisionTable+Task组合SHA。`spec-replay`不能冒充原spec；不能改旧declared hash强行送过旧validator。源Task字节只有明确镜像URI前缀变换；SHA与证明收据在config中。

调用方必须先通过资产库注册original_spec、station、task_source、x5、station_mapping、projection_proof等相对asset IDs，再调用factory。runtime不使用原件中的PRO路径打开资源；内部URI token只存于SHA绑定的外部projection_proof，不进入公开源码；原件路径只用于字节身份和authored-URI投影检查。shared CLI的prepare_station负责重定位；Task/修复版stand的映射目标另做角色校验，不能换成E2旧stand。

station组合取source world.py:39–64的原recipe，保留def6542f修复版stand、隐藏CollisionTable和源static collision设置。原生build通过shared robot prepare/mount/initialize恢复官方camera joints、profile、articulation；Task只保留自身props的128/8 solver和原ZERO、rider、display机构。robot flange使用已有API/non-root前置、显式禁reset/contact preparation/tracking/stabilization的只读实际pose view；robot base pose直接读取共享articulation。没有任务自带robot driver/Kit/world。

计分按当前第5/6节：成功100，否则20*k/N，N=4/6/8，首错之前连续完成的物理步骤单独报告。Easy采用各档一致的实测lift/contact判据，旧121 tick相同命令和exact closed target不作资格条件；不读expert phase/jobs。任一合法库存砝码可入盘，不绑定专家选中的重复砝码ID。公开配置提供绿区区间、砝码库存与Hard游码目标，水果质量/答案不放入公开参数，训练指令不额外改写。

每个physics tick追加完整raw audit；evidence仅含source绑定与小receipt，终局独立验hash/bytes/全部tick并复算。收尾使用360 tick有界窗口与增量物理事实，不逐tick冻结/复算全历史。tick0明确放弃保留完整初始审计并给0，缺失/损坏证据报invalid。

CPU composition/源码和单元检查不代表native PhysX、首帧像素或视频验收；后续由共享队列执行，reference replay不代表模型表现。
