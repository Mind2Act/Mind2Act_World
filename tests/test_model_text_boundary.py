import json
import numpy as np
import pytest

from mind2act_eval.adapters.xpolicylab import public_observation
from mind2act_eval.contracts import Observation
from mind2act_eval.model_text import cjk_count, require_english_model_text


def test_chinese_source_cannot_escape_through_the_actual_payload_boundary():
    text = json.loads('"\\u6309\\u4e0b"')
    obs = Observation(0, 0., (0.,)*6, (0.,)*6, 0., 0., {}, text)
    with pytest.raises(ValueError, match='English text is required') as exc:
        public_observation(obs, {})
    assert cjk_count(str(exc.value)) == 0


def test_approved_english_payload_is_preserved_verbatim():
    text = 'Set resistance to 20 Ω, within ±0.2 Ω.\nRelease; wait 0.5 seconds.'
    obs = Observation(0, 0., (0.,)*6, (0.,)*6, 0., 0.,
                      {'head': np.zeros((1, 1, 3), np.uint8)}, text)
    assert public_observation(obs, {'head': 'cam_high'})['task'] == text
    assert require_english_model_text(text) == text


@pytest.mark.parametrize('text', ['中文', '。', 'カナ', '한글', '\U00020000', *map(chr, [0x1100, 0x3190, 0x32FF, 0xA960, 0xD7B0, 0xFF76, 0x16FE0, 0x1AFF0, 0x1B001, 0x1F200, 0x3FFFF])])
def test_cjk_families_are_rejected(text):
    with pytest.raises(ValueError, match='English text is required'):
        require_english_model_text(text)


