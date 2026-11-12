"""A reviewed metadata-only X5 derivative keeps strict byte identity checks."""
import hashlib

import pytest
import mind2act_eval.robot as robot


def test_only_reviewed_derivative_of_the_requested_official_pin_is_accepted(tmp_path, monkeypatch):
    original, derived = b'original X5 layer', b'X5 with stale export history removed'
    digest = lambda data: hashlib.sha256(data).hexdigest()
    canonical = digest(original)
    monkeypatch.setattr(robot, 'PROFILE', {
        'asset_sha256': canonical,
        'asset_derivatives': [{'source_sha256': canonical, 'sha256': digest(derived)}],
    })
    path = tmp_path / 'ARX.usd'
    for data in (original, derived):
        path.write_bytes(data)
        assert robot.verify_asset(path, canonical) == digest(data)
    # The declared derivative cannot satisfy an unrelated caller's pin.
    with pytest.raises(ValueError, match='pinned asset digest mismatch'):
        robot.verify_asset(path, digest(b'another robot'))
    path.write_bytes(derived + b'new unreviewed edit')
    with pytest.raises(ValueError, match='pinned asset digest mismatch'):
        robot.verify_asset(path, canonical)


def test_a_derivative_of_another_source_does_not_expand_the_official_pin(tmp_path, monkeypatch):
    path = tmp_path / 'ARX.usd'
    path.write_bytes(b'some robot')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(robot, 'PROFILE', {
        'asset_sha256': 'a' * 64,
        'asset_derivatives': [{'source_sha256': 'b' * 64, 'sha256': actual}],
    })
    with pytest.raises(ValueError, match='pinned asset digest mismatch'):
        robot.verify_asset(path, 'a' * 64)
