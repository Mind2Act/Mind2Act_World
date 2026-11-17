"""duclaw 网关上的 LLM 关节空间 e2e adapter，入口 mind2act_eval.adapters.duclaw:create。

adapter config 必填 log_dir、ledger、ledger_limit，其余见 llm_e2e.DEFAULTS；
密钥从 key_env（默认 MIND2ACT_API_KEY）环境变量读取，不写入 config。
"""
from mind2act_eval.adapters.llm_e2e import DUCLAW, create_adapter


def create(config):
    return create_adapter(config, DUCLAW)
