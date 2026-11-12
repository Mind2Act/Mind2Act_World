"""Public retrieval transplanted from765189d CP05 episodic_memory/Evidence.

The mainline seam stores already-encoded public RGB in the episode instead of
reading legacy environment paths. Metadata paging, exact-reference opening,
content hashes, six-image LRU and two-observation compaction follow the donor.
No scene, phase, scorer, reference or semantic image query is available.
"""
from collections import OrderedDict
from copy import deepcopy
import base64
import hashlib
import json
import re

MAX_RETRIEVED_IMAGES = 6
CAMERAS = ('head','left_wrist','right_wrist')


def _digest(uri):
    if not isinstance(uri,str) or not uri.startswith('data:image/') or ';base64,' not in uri:
        raise ValueError('Only captured public image data is accepted')
    try:
        data = base64.b64decode(uri.partition(',')[2],validate=True)
    except ValueError:
        raise ValueError('Invalid public image encoding') from None
    if not data:
        raise ValueError('Public image data must be nonempty')
    return hashlib.sha256(data).hexdigest()


class EpisodeMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.events = OrderedDict()
        self.retained = OrderedDict()
        self.open_calls = 0

    def add(self, observation_id, tick, images):
        if (not isinstance(observation_id,str) or not re.fullmatch(r'obs-[0-9]+',observation_id)
                or type(tick) is not int or tick<0 or set(images)!=set(CAMERAS)):
            raise ValueError('Invalid captured public observation')
        if observation_id in self.events:
            return
        if self.events and tick < next(reversed(self.events.values()))['sim_step']:
            raise ValueError('Public observations must be archived in time order')
        row = dict(observation_id=observation_id,sim_step=tick,images={})
        for camera,uri in images.items():
            row['images'][camera] = dict(uri=uri,sha256=_digest(uri))
        self.events[observation_id] = row

    def resolve(self, arguments):
        if not isinstance(arguments,dict) or set(arguments)!={'query','event_ref','camera','cursor'}:
            raise ValueError('Invalid memory arguments')
        query,ref,camera,cursor = (arguments[k] for k in ('query','event_ref','camera','cursor'))
        if (not isinstance(query,str) or camera not in CAMERAS or type(cursor) is not int or cursor<0
                or ref is not None and not isinstance(ref,str)):
            raise ValueError('Invalid memory arguments')
        if ref is None:
            words = query.lower().split()
            rows = [{k:row[k] for k in ('observation_id','sim_step')} for row in self.events.values()]
            rows.sort(key=lambda row:-sum(word in json.dumps(row).lower() for word in words))
            rows = [dict(row,sim_seconds=row['sim_step']/120) for row in rows]
            return dict(response_kind='index',events=rows[cursor:cursor+12],
                next_cursor=cursor+12 if cursor+12<len(rows) else None,
                search_scope='public event metadata; not semantic image search')
        if cursor!=0 or ref not in self.events:
            raise ValueError('Unknown or future public observation, or invalid open cursor')
        record = self.events[ref]['images'][camera]
        if _digest(record['uri'])!=record['sha256']:
            raise ValueError('Public image hash mismatch')
        self.open_calls += 1
        key = (ref,camera)
        self.retained.pop(key,None)
        self.retained[key] = dict(ref=ref,camera=camera,sim_step=self.events[ref]['sim_step'],**record)
        while len(self.retained)>MAX_RETRIEVED_IMAGES:
            self.retained.popitem(last=False)
        return dict(response_kind='evidence',status='opened',evidence_refs=[ref],sha256=record['sha256'])

    def request_images(self):
        content,refs = [],[]
        for row in self.retained.values():
            if _digest(row['uri'])!=row['sha256']:
                raise ValueError('Retrieved public image changed after opening')
            content.extend([dict(type='text',text=f"HISTORICAL RETRIEVED observation {row['ref']}; "
                f"environment tick {row['sim_step']}; camera {row['camera']}. This is not the current scene."),
                dict(type='image_url',image_url=dict(url=row['uri']))])
            refs.append({k:row[k] for k in ('ref','camera','sim_step','sha256')})
        return ([dict(role='user',content=content)] if content else []),refs


def _normal(message):
    content = message.get('content')
    return (message.get('role')=='user' and isinstance(content,list)
        and any(part.get('type')=='text' and 'Measured grasp-point state:' in part.get('text','') for part in content))


def _ref(message,index):
    for part in message.get('content',[]):
        if part.get('type')=='text':
            match = re.search(r'\bobservation (obs-[0-9]+)',part.get('text',''))
            if match:return match.group(1)
    return ('unidentified',index)


def prepare_memory_request(memory,messages,current_message):
    """CP05's6ordinary+6retrieved policy, with the current observation last."""
    outgoing = deepcopy(messages)
    if outgoing and _normal(outgoing[-1]):
        outgoing[-1] = deepcopy(current_message)
    else:
        outgoing.append(deepcopy(current_message))
    normal = [index for index,message in enumerate(outgoing) if _normal(message)]
    keep,seen = set(),set()
    for index in reversed(normal):
        ref = _ref(outgoing[index],index)
        if ref not in seen and len(keep)<2:
            keep.add(index);seen.add(ref)
    for index in normal:
        if index not in keep:
            outgoing[index]['content'] = [part if part.get('type')!='image_url' else
                dict(type='text',text='[Earlier public RGB archived; use memory_resolve.]')
                for part in outgoing[index]['content']]
    ordinary = sum(part.get('type')=='image_url' for index in keep for part in outgoing[index]['content'])
    retrieved,refs = memory.request_images()
    if ordinary>6 or len(refs)>6:
        raise ValueError('Public image request exceeds6ordinary+6retrieved')
    # The donor appends retrieval images; place them immediately before Current
    # Observation to satisfy the current parent document's exact message order.
    outgoing[-1:-1] = retrieved
    audit = dict(version='cp-retrieval-unified12-mainline-v1',ordinary_images=ordinary,
        retrieved_images=len(refs),retrieved_refs=refs,total_images=ordinary+len(refs),
        max_ordinary_images=6,max_retrieved_images=6,max_images=12,
        archived_observations=len(memory.events),automatic_demonstration_bundle=False)
    return outgoing,audit
