"""F2 section 6: valid all-correct released confirmation = 100, otherwise 0."""
from mind2act_eval.contracts import NativeEvidence, Score, to_json
from mind2act_eval.scoring import SCORING_VERSION
from .provenance import VERSION, effective_config, read_origin, validate_start
from .mechanism import Task
from .geometry import Body, in_region, supported
from .evidence import validate_frame, read_rows


def score(evidence: NativeEvidence) -> Score:
    if evidence.instance_id.startswith('VAL-F2-composition-') or 'gap_composition_spec' in evidence.values:
        from mind2act_eval.gap_composition_inputs import validate_evidence
        validate_evidence(evidence)
    if evidence.task_id != 'F2':
        raise ValueError('F2 scorer received a different task')
    v = to_json(evidence.values)
    c = effective_config(v['source_case'], input_version=v['input_protocol_version'])
    start = v['source_start_tick']
    origin = read_origin(v['source_origin_receipt'], c['difficulty'])
    validate_start(start, origin)
    machine = Task(c, source_start_tick=start)
    initial = v['initial']
    validate_frame(c, initial, native_tick=0, source_start=start)
    machine.sample(initial['f2_raw'])
    last_row = initial
    tick, previous_time, terminal_tick = 0, initial['simulation_time_s'], None
    for row in read_rows(v['measurements']):
        tick += 1
        validate_frame(c, row, native_tick=tick, source_start=start)
        if abs(row['simulation_time_s']-previous_time-1/120) > 1e-6:
            raise ValueError('F2 native measurements do not follow 120 Hz physics')
        previous_time = row['simulation_time_s']
        last_row = row
        if machine.status == 'running':
            machine.sample(row['f2_raw'])
            if machine.status != 'running':
                terminal_tick = tick
    if tick != evidence.tick or terminal_tick != v['terminal_tick']:
        raise ValueError('F2 measurement/terminal receipt mismatch')
    bodies = {name:Body.parse(raw) for name,raw in last_row['f2_raw']['bodies'].items()}
    slots = list(c['regions'])
    terminal_correct = all(in_region(bodies[name], c['regions'][slot]) and supported(name, slot, bodies, c)
                           for name,slot in zip(c['target'], slots))
    success = machine.status == 'success' and not machine.pending_release and terminal_correct
    return Score(100.0 if success else 0.0, dict(task_success=success, scoring_version=SCORING_VERSION, rubric=VERSION,
        input_protocol_version=v['input_protocol_version'], attempts=len(machine.attempts),
        attempt_budget=c['attempt_budget'], native_status=machine.status, reason=machine.reason,
        terminal_tick=terminal_tick, source_start_tick=start, source_last_tick=start+tick,
        native_physics_ticks=tick, machine_physics_ticks=machine.physics_samples))
