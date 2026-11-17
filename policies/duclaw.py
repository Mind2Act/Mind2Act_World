"""duclaw 网关上的 LLM 关节空间 adapter，入口 mind2act_eval.adapters.duclaw:create。

运行需 --policy-kind agent --budget-version coding-agent-e2e-x4-v1（墙钟 21600 s），
预算语义同 roboprobe_feishu，见 llm_e2e 模块说明。
adapter config 必填 log_dir、ledger、ledger_limit，其余见 llm_e2e.DEFAULTS；
密钥从 key_env（默认 MIND2ACT_API_KEY）环境变量读取，不写入 config。
"""
from mind2act_eval.adapters.llm_e2e import DUCLAW, create_adapter


def create(config):
    return create_adapter(config, DUCLAW)
