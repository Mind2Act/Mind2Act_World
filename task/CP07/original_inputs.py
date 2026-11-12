"""K0's explicit original eight-piece test protocol; old 13-slot inputs stay separate."""
from math import cos,isclose,radians,sin
from types import SimpleNamespace

from mind2act_eval.contracts import to_json
from mind2act_eval.original_test_inputs import read_audit,source_identity

PROTOCOL='cp07-original-eight-v1'
STOCK={'easy':8,'medium':11,'hard':14}
WORLD_SHA256='ef84be71411f69b0a7dde111ecec713222ebbbca623bb0eea6dbbb727961bd6d'


def sequence(solution,by_id=None):
    supports=solution['supports']
    if by_id is not None:supports=sorted(supports,key=lambda n:(by_id[n]['W'],by_id[n]['T']))
    return [*supports,solution['wedge'],*solution['short'],*solution['long']]


def geometry(piece):
    return [piece[k] for k in ('kind','L','W','T','height0','height1')]


def pose(raw,solution,index):
    # Original world.py/parameters.py recompute each admissible stack, sorted
    # by width/thickness. Unselected raw dest values are teacher remnants.
    by_id={p['name']:p for p in raw['stock']};order=sequence(solution,by_id)
    angle=radians({'easy':7,'medium':8,'hard':9}[raw['difficulty']])
    yaw,pitch=0.,0.
    if index<3:
        p=[-.025,-.1,.770+sum(by_id[n]['T'] for n in order[:index])+by_id[order[index]]['T']/2]
        yaw=radians(90)/2
    elif index==3:p=[-.010,-.1,.804]
    elif index<6:p=[-.025-.012*sin(angle)-.100/2,-.1+(-.019 if index==4 else .019),.814]
    else:
        p=[-.025+.130*cos(angle)/2-.012*sin(angle)/2,-.1+(-.019 if index==6 else .019),
           .804+.008+.130*sin(angle)/2+.012*cos(angle)/2]
        pitch=-angle/2
    return dict(p_m=p,q_wxyz=[cos(yaw)*cos(pitch),-sin(yaw)*sin(pitch),
        cos(yaw)*sin(pitch),sin(yaw)*cos(pitch)])


def authenticate(spec,declaration):
    try:
        raw=read_audit(spec,declaration)
        by_id={p['name']:p for p in raw['stock']}
        if (raw['version']!='train-v2-cpu' or raw['difficulty']!=spec.difficulty or raw['seed']!=spec.seed
                or len(raw['stock'])!=STOCK[spec.difficulty] or len(by_id)!=len(raw['stock'])
                or not raw['solutions'] or len(raw['solutions'])!=raw['solution_count']
                or any(len(sequence(s))!=8 or len(set(sequence(s)))!=8
                       or not set(sequence(s))<=by_id.keys() for s in raw['solutions'])):
            raise ValueError('original eight-piece source condition differs')
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('CP07 authenticated original eight-piece source required') from exc
    return raw


def validate_binding(spec,binding):
    from .binding import goal_pose
    raw=authenticate(spec,binding['original_source']);by_id={p['name']:p for p in raw['stock']}
    if (binding.get('source_protocol')!=PROTOCOL or binding['solutions_scope']!='original-source-solutions'
            or binding['source_manifest_sha256']!=source_identity(spec)['source_configuration_sha256']
            or {p['id'] for p in binding['pieces']}!=by_id.keys()
            or len(binding['solutions'])!=len(raw['solutions'])):
        raise ValueError('CP07 original source stock/solution binding differs')
    if (not isclose(binding['lane_y_m'],-.1,rel_tol=0,abs_tol=1e-8)
            or not isclose(binding['car_start_x_m'],-.190,rel_tol=0,abs_tol=1e-8)
            or any(not isclose(a,b,rel_tol=0,abs_tol=1e-8)
                   for a,b in zip(binding['button_xy_m'],[.255,-.070]))):
        raise ValueError('CP07 original source terrain/button/car frame differs')
    for piece in binding['pieces']:
        original=by_id[piece['id']]
        if (piece['path']!=original['path'] or piece['kind']!=original['kind']
                or piece['geometry_signature']!=geometry(original)):
            raise ValueError('CP07 original stock geometry differs')
    for solution,original in zip(binding['solutions'],raw['solutions']):
        if solution.get('source_solution')!=original or len(solution['slots'])!=8:
            raise ValueError('CP07 every original solution must remain explicit')
        for index,(slot,name) in enumerate(zip(solution['slots'],sequence(original,by_id))):
            names=slot['compatible_piece_ids']
            if (name not in names or any(n not in by_id or geometry(by_id[n])!=geometry(by_id[name]) for n in names)):
                raise ValueError('CP07 original compatible geometry differs')
            for candidate in names:
                target=goal_pose(slot,candidate);expected=pose(raw,original,index)
                if (any(not isclose(a,b,rel_tol=0,abs_tol=1e-9) for a,b in zip(target['p_m'],expected['p_m']))
                        or min(sum((a-sign*b)**2 for a,b in zip(target['q_wxyz'],expected['q_wxyz']))
                               for sign in (-1,1))>1e-14):
                    raise ValueError('CP07 original candidate target pose differs')
    return 8,len(raw['stock'])


def scoring_counts(evidence):
    v=evidence.values
    if v.get('source_protocol')!=PROTOCOL or v.get('solutions_scope')!='original-source-solutions':
        raise ValueError('CP07 explicit original test scoring protocol required')
    try:
        spec=SimpleNamespace(task_id='CP07',instance_id=evidence.instance_id,difficulty=v['difficulty'],
            seed=v['test_source']['source_seed'],config=dict(test_source=v['test_source']))
        raw=authenticate(spec,v)
        if len(v['solutions'])!=len(raw['solutions']):raise ValueError('original solutions missing')
        by_id={p['name']:p for p in raw['stock']}
        for solution,original in zip(v['solutions'],raw['solutions']):
            if to_json(solution.get('source_solution'))!=original:raise ValueError('original solution differs')
            for measured,name in zip(solution['slots'],sequence(original,by_id)):
                piece=measured.get('piece_id')
                if (measured.get('kind')!=by_id[name]['kind']
                        or piece is not None and (piece not in by_id or geometry(by_id[piece])!=geometry(by_id[name]))):
                    raise ValueError('original slot geometry differs')
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('CP07 original test physical source binding differs') from exc
    return 8,len(raw['stock'])
