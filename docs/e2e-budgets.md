# E2E 与 coding agent 运行预算

物理预算以 [90 格冻结表](../manifests/budgets/budgets-e2e-p90x1.2.json) 的
`physics_ticks` 为唯一数值来源，30 任务三档统一适用：

| 协议 | budget_version | 物理预算 | 每局墙钟 |
|---|---|---|---:|
| e2e | `e2e-p90x1.2-v1` | 冻结表 `physics_ticks` | 7200 s |
| coding agent | `coding-agent-e2e-x4-v1` | 冻结表 `physics_ticks × 4` | 21600 s |

旧 15 的 published ×4、新 15 的 60000/216000 等占位值不再作为新 coding agent
运行预算。倍率只在协议选择时应用一次；adapter 接收最终 ticks，不再乘 4。
计分规则与 `scoring_version=mind2act-final-scoring-20261105` 不变。

Coding agent 每局上限为 **150 次完整返回的模型决策**。非法动作需要修正的那轮也计 1 次；
动作展开出的控制行、同一次决策内的网络/服务重试不另占决策数。完整返回后持久化计数，
第 151 次决策在请求前被拒绝，`termination=call_budget`，按当时物理状态计分。
E2E 不使用 coding agent 决策额度。

每次决策的可重试基础设施错误，按 coding agent 原生 `max_retries=12` 处理：
首次请求之外最多重试 12 次，即一轮最多 13 次请求。重试耗尽或不可重试服务错误记为
infra，`score=null`，不把失败请求计为完整决策。移除整局 450 次 provider 请求上限。
已存在的累计付费授权 ledger 仍按调用者给定的总授权记录请求，不能自动扩大或重置；
新运行应提供足够的累计授权，不能把旧的 450 当作本局协议上限。

结果及 `call-budget.json` 记录 `calls_used`、`calls_limit=150`、
`decision_retry_limit=12`，删除 `provider_attempts_used/provider_attempt_limit`。
实际请求日志继续逐请求保存审计证据；其中的累计请求量不是另一个整局调用额度。

墙钟是基础设施上限，不是任务得分条件。撞墙钟记录 `termination=wall_timeout`、
`infra_valid=false`、`score=null`，由调度层同 case 重排；不能作为有效 0 分进入榜单。
物理预算耗尽仍按当前任务状态计分。版本化任务必须精确采用上述墙钟，不能自行延长。
单次执行 `retries=0`；基础设施重排与请求内部网络重试是不同层级。

## 冻结与运行选择

表内 P90 来源为每格 50 个训练专家、共 4500 局；时长按 H5 行数和频率计算，
P90 使用线性插值，取整使用有理数：

```text
control_steps = ceil(P90_seconds × 6/5 × 30)
e2e_physics_ticks = control_steps × 4
coding_agent_physics_ticks = e2e_physics_ticks × 4
```

历史 JSON/CSV 中的 `wall_s` 是统计时保存的旧墙钟计算值，仅保留作审计；
**运行时覆盖为统一 7200/21600 s**，不再采用该列。历史 `published_limits` 同样只作审计。
本次不修改统计表、P90 数据、CP01 预算来源证明或历史发布行。

发布者使用同一冻结表绑定新的 source index 和每个 ready catalog：

```sh
python -m mind2act_eval.training_budgets \
  --document original-source-index.json \
  --table manifests/budgets/budgets-e2e-p90x1.2.json \
  --output /fresh/output/source-index.json
```

`bind_case_budgets` 保留原始 `limits`，新增 `limits_e2e`、`budget_version_e2e` 和表的
语义 SHA；验证统计公式、拒绝覆盖既有绑定。随后重建 descriptor SHA 和新的不可变发布代。
缺少冻结 E2E 绑定的版本化任务拒绝运行，coding agent 不回退到旧 published 预算。

显式选择协议，并同时传入对应的最终 `--physics-ticks`、`--wall-s`、`--retries 0`：

```text
--policy-kind e2e --budget-version e2e-p90x1.2-v1
--policy-kind coding_agent --budget-version coding-agent-e2e-x4-v1
```

`selected_limits` 产生最终执行预算，父进程和子进程核验一致，`result.json` 与
`run-budget.json` 记录 `policy_kind`、`budget_version`、`budget_limits`。
未显式选择协议时，历史运行路径继续读取原始 `limits`；既有 PRO G0.5 的代码、release、
队列和 pin 不变。本补丁不切换部署，不重写已跑结果或运行中配置。

## 90 格运行预算表

单位均为 120 Hz physics tick 或墙钟秒。表逐行直接读取冻结 JSON 的 `physics_ticks`。

| 任务 | 难度 | E2E ticks | coding agent ticks（×4） | E2E 墙钟 s | coding agent 墙钟 s |
|---|---|---:|---:|---:|---:|
| CP01 | easy | 7720 | 30880 | 7200 | 21600 |
| CP01 | hard | 22300 | 89200 | 7200 | 21600 |
| CP01 | medium | 15196 | 60784 | 7200 | 21600 |
| CP02 | easy | 16136 | 64544 | 7200 | 21600 |
| CP02 | hard | 16716 | 66864 | 7200 | 21600 |
| CP02 | medium | 12584 | 50336 | 7200 | 21600 |
| CP03 | easy | 13888 | 55552 | 7200 | 21600 |
| CP03 | hard | 17440 | 69760 | 7200 | 21600 |
| CP03 | medium | 20392 | 81568 | 7200 | 21600 |
| CP04 | easy | 24244 | 96976 | 7200 | 21600 |
| CP04 | hard | 36568 | 146272 | 7200 | 21600 |
| CP04 | medium | 26228 | 104912 | 7200 | 21600 |
| CP05 | easy | 7792 | 31168 | 7200 | 21600 |
| CP05 | hard | 13132 | 52528 | 7200 | 21600 |
| CP05 | medium | 9176 | 36704 | 7200 | 21600 |
| CP06 | easy | 4128 | 16512 | 7200 | 21600 |
| CP06 | hard | 6924 | 27696 | 7200 | 21600 |
| CP06 | medium | 6004 | 24016 | 7200 | 21600 |
| CP07 | easy | 13952 | 55808 | 7200 | 21600 |
| CP07 | hard | 24384 | 97536 | 7200 | 21600 |
| CP07 | medium | 22888 | 91552 | 7200 | 21600 |
| CP08 | easy | 4716 | 18864 | 7200 | 21600 |
| CP08 | hard | 11360 | 45440 | 7200 | 21600 |
| CP08 | medium | 6208 | 24832 | 7200 | 21600 |
| CP09 | easy | 11628 | 46512 | 7200 | 21600 |
| CP09 | hard | 37020 | 148080 | 7200 | 21600 |
| CP09 | medium | 22540 | 90160 | 7200 | 21600 |
| CP10 | easy | 7520 | 30080 | 7200 | 21600 |
| CP10 | hard | 10948 | 43792 | 7200 | 21600 |
| CP10 | medium | 9536 | 38144 | 7200 | 21600 |
| DR1 | easy | 3228 | 12912 | 7200 | 21600 |
| DR1 | hard | 5860 | 23440 | 7200 | 21600 |
| DR1 | medium | 3992 | 15968 | 7200 | 21600 |
| DR2 | easy | 2296 | 9184 | 7200 | 21600 |
| DR2 | hard | 2312 | 9248 | 7200 | 21600 |
| DR2 | medium | 2312 | 9248 | 7200 | 21600 |
| DR3 | easy | 4272 | 17088 | 7200 | 21600 |
| DR3 | hard | 7136 | 28544 | 7200 | 21600 |
| DR3 | medium | 4024 | 16096 | 7200 | 21600 |
| DR4 | easy | 1432 | 5728 | 7200 | 21600 |
| DR4 | hard | 2084 | 8336 | 7200 | 21600 |
| DR4 | medium | 2056 | 8224 | 7200 | 21600 |
| E1 | easy | 5736 | 22944 | 7200 | 21600 |
| E1 | hard | 15064 | 60256 | 7200 | 21600 |
| E1 | medium | 10368 | 41472 | 7200 | 21600 |
| E2 | easy | 1952 | 7808 | 7200 | 21600 |
| E2 | hard | 3296 | 13184 | 7200 | 21600 |
| E2 | medium | 2732 | 10928 | 7200 | 21600 |
| F1 | easy | 2916 | 11664 | 7200 | 21600 |
| F1 | hard | 5812 | 23248 | 7200 | 21600 |
| F1 | medium | 4448 | 17792 | 7200 | 21600 |
| F2 | easy | 21316 | 85264 | 7200 | 21600 |
| F2 | hard | 50460 | 201840 | 7200 | 21600 |
| F2 | medium | 38572 | 154288 | 7200 | 21600 |
| M1 | easy | 6848 | 27392 | 7200 | 21600 |
| M1 | hard | 9800 | 39200 | 7200 | 21600 |
| M1 | medium | 7908 | 31632 | 7200 | 21600 |
| M2 | easy | 4572 | 18288 | 7200 | 21600 |
| M2 | hard | 18312 | 73248 | 7200 | 21600 |
| M2 | medium | 8500 | 34000 | 7200 | 21600 |
| P1 | easy | 4628 | 18512 | 7200 | 21600 |
| P1 | hard | 8388 | 33552 | 7200 | 21600 |
| P1 | medium | 5672 | 22688 | 7200 | 21600 |
| P2 | easy | 7456 | 29824 | 7200 | 21600 |
| P2 | hard | 9116 | 36464 | 7200 | 21600 |
| P2 | medium | 8988 | 35952 | 7200 | 21600 |
| PR1 | easy | 752 | 3008 | 7200 | 21600 |
| PR1 | hard | 752 | 3008 | 7200 | 21600 |
| PR1 | medium | 752 | 3008 | 7200 | 21600 |
| PR2 | easy | 904 | 3616 | 7200 | 21600 |
| PR2 | hard | 920 | 3680 | 7200 | 21600 |
| PR2 | medium | 904 | 3616 | 7200 | 21600 |
| PR3 | easy | 1528 | 6112 | 7200 | 21600 |
| PR3 | hard | 1840 | 7360 | 7200 | 21600 |
| PR3 | medium | 1604 | 6416 | 7200 | 21600 |
| PR4 | easy | 3432 | 13728 | 7200 | 21600 |
| PR4 | hard | 4120 | 16480 | 7200 | 21600 |
| PR4 | medium | 4064 | 16256 | 7200 | 21600 |
| PR5 | easy | 2548 | 10192 | 7200 | 21600 |
| PR5 | hard | 2500 | 10000 | 7200 | 21600 |
| PR5 | medium | 2612 | 10448 | 7200 | 21600 |
| PR6 | easy | 8008 | 32032 | 7200 | 21600 |
| PR6 | hard | 18776 | 75104 | 7200 | 21600 |
| PR6 | medium | 15036 | 60144 | 7200 | 21600 |
| S1 | easy | 5524 | 22096 | 7200 | 21600 |
| S1 | hard | 6304 | 25216 | 7200 | 21600 |
| S1 | medium | 6248 | 24992 | 7200 | 21600 |
| S2 | easy | 2888 | 11552 | 7200 | 21600 |
| S2 | hard | 7144 | 28576 | 7200 | 21600 |
| S2 | medium | 4892 | 19568 | 7200 | 21600 |
