"""LLM joint-space adapter on the duclaw gateway; entry mind2act_eval.adapters.duclaw:create.

Run with --policy-kind agent --budget-version coding-agent-e2e-x4-v1 (wall 21600 s);
budget semantics follow roboprobe_feishu, see the llm_e2e module docstring.
The adapter config requires log_dir, ledger and ledger_limit; see llm_e2e.DEFAULTS
for the rest. The key is read from the key_env environment variable (default
MIND2ACT_API_KEY) and never written to the config.
"""
from mind2act_eval.adapters.llm_e2e import DUCLAW, create_adapter


def create(config):
    return create_adapter(config, DUCLAW)
