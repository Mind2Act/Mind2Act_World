from copy import deepcopy

import pytest

from mind2act_eval.adapters.agent_input.memory import EpisodeMemory, prepare_memory_request

IMAGE='data:image/jpeg;base64,aGVsbG8='


def populate(count=14):
    memory=EpisodeMemory()
    for n in range(count):memory.add(f'obs-{n:09d}',n*4,{c:IMAGE for c in ('head','left_wrist','right_wrist')})
    return memory


def args(ref=None,cursor=0):
    return dict(query='',event_ref=ref,camera='head',cursor=cursor)


def test_public_metadata_paging_and_exact_open():
    memory=populate()
    page=memory.resolve(args())
    assert len(page['events'])==12 and page['next_cursor']==12
    assert set(page['events'][0])=={'observation_id','sim_step','sim_seconds'}
    assert len(memory.resolve(args(cursor=12))['events'])==2
    assert memory.resolve(args('obs-000000000'))['status']=='opened'
    assert len(memory.request_images()[1])==1


@pytest.mark.parametrize('arguments',[args('obs-999999999'),args('obs-000000000',1),args(cursor=-1),
    dict(query='',event_ref=None,camera='private_camera',cursor=0)])
def test_unknown_future_camera_and_cursor_are_rejected(arguments):
    with pytest.raises(ValueError):populate(1).resolve(arguments)


def test_opened_images_have_six_entry_lru_and_hash_check():
    memory=populate(7)
    for n in range(7):memory.resolve(args(f'obs-{n:09d}'))
    messages,refs=memory.request_images()
    assert len(refs)==6 and refs[0]['ref']=='obs-000000001'
    assert sum(p['type']=='image_url' for p in messages[0]['content'])==6
    next(iter(memory.retained.values()))['uri']='data:image/jpeg;base64,Zm9v'
    with pytest.raises(ValueError,match='changed'):memory.request_images()
    memory.reset();assert not memory.events and not memory.retained


def normal(n):
    parts=[dict(type='text',text=f'Instruction: Keep the block.\nMeasured grasp-point state:\nCurrent RGB:')]
    for camera in ('head','left_wrist','right_wrist'):
        parts.extend([dict(type='text',text=f'camera {camera} observation obs-{n:09d}'),dict(type='image_url',image_url={'url':IMAGE})])
    return dict(role='user',content=parts)


def test_six_normal_plus_six_retrieved_keep_all_text_and_current_observation_last():
    memory=populate(7)
    for n in range(6):memory.resolve(args(f'obs-{n:09d}'))
    messages=[dict(role='system',content='System'),dict(role='user',content='Goal')]
    for n in range(3):
        messages += [normal(n),dict(role='tool',content=f'public feedback {n}')]
    before=deepcopy(messages)
    outgoing,audit=prepare_memory_request(memory,messages,normal(3))
    assert messages==before
    assert (audit['ordinary_images'],audit['retrieved_images'],audit['total_images'])==(6,6,12)
    assert outgoing[-1]==normal(3)
    assert all(any(row.get('content')==f'public feedback {n}' for row in outgoing) for n in range(3))
    assert outgoing[2]['content'][0]==normal(0)['content'][0]


def test_repeated_readonly_queries_do_not_duplicate_current_rgb():
    memory=populate(1)
    outgoing,audit=prepare_memory_request(memory,[normal(0),dict(role='tool',content='indexed')],normal(0))
    assert audit['ordinary_images']==3 and audit['retrieved_images']==0
    assert outgoing[-1]==normal(0)
