"""Joint-action DuClaw factory for the fixed --duclaw-agent mode."""
from mind2act_eval.adapters.llm_e2e import DUCLAW, create_adapter


def create(config):
    return create_adapter(config, DUCLAW)
