"""Exact PR4 case/source admission, separate from shared physics settings."""
from dataclasses import dataclass
from mind2act_eval.contracts import ClockSpec,InstanceSpec

@dataclass(frozen=True)
class SourceBinding:
    difficulty: str
    seed: int
    source_spec_sha256: str
    source_config_sha256: str
    scene_sha256: str


# Identity receipts of the three selected pr4-original-time-v3 source specs.
# Closed case admission; these receipts never select shared physics values.
PR4_SOURCES = (
    SourceBinding("easy", 410000,
                  "91bae0d07468717fc99bb1e8a3a45da502a808b2b648973f0ea559e9c8a070b1",
                  "61832f768989aa1a7e7a0a9fa108b49622ce04e087fd54c03717f1e499e6c918",
                  "26a07213df4820654fc58dc32a77396db3907470b11ea65793013480f106145f"),
    SourceBinding("medium", 420000,
                  "e8697f4bdf7d589f0eaeedb336bec736ec9b40259c88fe5f9b4d930cf7e9ce30",
                  "421ee81960cd04313990a4fb3ae83672f7acea2b54aa8de7eb54656052f840e4",
                  "bb832184c9b81261a465081ef6799f508623daa9234b70e755cd5c04c60542b1"),
    SourceBinding("hard", 430000,
                  "67d453d2d34081d0d8e97f233a1becd751c8ddfd6c18b81e17cfb16c5236adeb",
                  "bbf2dc9a6b601d38ed50094eecea5c0421cca838d37f35fe813a796dc7476e97",
                  "8da390c0d3b70f452b79d8aa55c5c15617e5e4482cef72af563942d4b4c9e7f3"),
)

def validate_case_source(spec):
    if not isinstance(spec,InstanceSpec) or spec.task_id!='PR4':
        raise ValueError('PR4 case source binding requires a PR4 InstanceSpec')
    from mind2act_eval.sim.new15_source_bindings import matches_original_test_binding
    if matches_original_test_binding(spec):
        return
    row = next(value for value in PR4_SOURCES if value.difficulty == spec.difficulty)
    if (spec.source_id != "pr4-original-time-v3"
            or spec.config_version != "height-study-raised20mm-seeded-translation-v1"
            or spec.protocol_version != "v1" or spec.clock != ClockSpec(120, 30, 30)
            or spec.seed != row.seed
            or spec.instance_id != f"PR4-{row.difficulty}-train-seed{row.seed}"
            or any(spec.config.get(key) != getattr(row, key)
                   for key in ("source_spec_sha256", "source_config_sha256", "scene_sha256"))):
        raise ValueError("PR4 selected source case binding mismatch")
    return
