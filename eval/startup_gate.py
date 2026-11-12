"""Serialize native initialization on the local startup domain."""
from contextlib import nullcontext
from mind2act_eval.startup_paths import verify_startup_domain


def simulator_gates(paths, physical_gpu, output_dir, *, gpu_selection=None):
    from mind2act_eval.sim import startup_gate
    return (startup_gate(paths.kit, minimum_interval_s=60.,
                         inherited_cooldown=paths.inherited_kit_cooldown,
                         domain_check=lambda: verify_startup_domain(paths)),
            nullcontext(lambda: None))
