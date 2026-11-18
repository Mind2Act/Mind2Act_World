"""Private A2 scoring delta; all non-flight A scores remain byte-for-byte numeric values."""
PEAK_THRESHOLDS=(.5,.75,1.)
RULE_VERSIONS={p:'mind2act-dr3-scoring-a2-candidate-flight-v%03d-v1'%round(p*100) for p in PEAK_THRESHOLDS}
RULE_VERSION=RULE_VERSIONS[.75]
FLIGHT_PARTIAL_SCORES={'easy':7.5,'medium':5.5,'hard':3.75}

def apply_value(A_value,difficulty,flight,peak_threshold=.75):
 if peak_threshold not in RULE_VERSIONS:raise ValueError('unsupported flight peak sensitivity')
 events=[e for e in flight['events'] if e['touchdown_tick']==flight['last_transport_touchdown_tick'] and e['kinematic_criteria_without_peak'] and e['peak_speed_m_s']>=peak_threshold]
 excluded=bool(events)
 return dict(value=FLIGHT_PARTIAL_SCORES[difficulty] if A_value==100. and excluded else A_value,excluded_delivery=excluded,accepted_flight_events=events,rule_version=RULE_VERSIONS[peak_threshold],peak_threshold_m_s=peak_threshold)
