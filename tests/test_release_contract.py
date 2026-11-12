import json

import pytest

from mind2act_eval.release_contract import (
    ReleaseContractError,
    load_release,
    reject_deprecated_paths,
)


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "old15-release-v1"
    root.mkdir()
    (root / "manifest.json").write_text("{}\n")
    data = {
        "schema": "mind2act.asset_release.v1",
        "asset_name": root.name,
        "manifest": "manifest.json",
        "manifest_sha256": "a" * 64,
        "code_commit": "b" * 40,
        "generated_at": "2026-11-02T06:00:00+00:00",
        "instances": [
            {"task_id": f"T{i}", "instance_id": f"T{i}/{tier}/one", "difficulty": tier}
            for i in range(15) for tier in ("easy", "medium", "hard")
        ],
        "extra_catalog": {"status": "PENDING"},
    }
    (root / "RELEASE.json").write_text(json.dumps(data))
    return root, data


def save(root, data):
    (root / "RELEASE.json").write_text(json.dumps(data))


def test_load_semantics_without_rehashing_manifest_or_assets(release):
    root, data = release
    result = load_release(root)
    assert result.asset_root == root.resolve()
    assert result.manifest_path == root / "manifest.json"
    assert result.manifest_sha256 == data["manifest_sha256"]
    assert len(result.instances) == 45
    assert result.raw["extra_catalog"] == {"status": "PENDING"}
    assert result.code_identity_checked is False
    # The declared digest deliberately differs from this file; runtime does not rehash it.


@pytest.mark.parametrize("change", [
    lambda d: d.update(asset_name="another-root"),
    lambda d: d.update(manifest_sha256="not-a-hash"),
    lambda d: d.update(code_commit="short"),
    lambda d: d.update(generated_at="not-a-time"),
    lambda d: d.update(instances=d["instances"][:-1]),
    lambda d: d["instances"].__setitem__(0, d["instances"][1]),
    lambda d: d["instances"][0].update(difficulty="unknown"),
    lambda d: d["instances"][0].update(instance_id=""),
])
def test_reject_semantically_incomplete_descriptor(release, change):
    root, data = release
    change(data)
    save(root, data)
    with pytest.raises(ReleaseContractError):
        load_release(root)


def test_manifest_cannot_escape_release(release, tmp_path):
    root, data = release
    (tmp_path / "outside.json").write_text("{}")
    data["manifest"] = "../outside.json"
    save(root, data)
    with pytest.raises(ReleaseContractError, match="inside"):
        load_release(root)


def test_manifest_must_exist(release):
    root, data = release
    data["manifest"] = "missing.json"
    save(root, data)
    with pytest.raises(ReleaseContractError, match="manifest"):
        load_release(root)


def test_optional_code_identity_is_not_invented(release):
    root, data = release
    data.pop("code_commit")
    save(root, data)
    value = load_release(root)
    assert value.code_commit is None
    assert value.code_identity_checked is False


def test_verified_code_identity_matches_or_rejects(release):
    root, _ = release
    assert load_release(root, verified_code_commit="b" * 40).code_identity_checked
    with pytest.raises(ReleaseContractError, match="code"):
        load_release(root, verified_code_commit="c" * 40)


def test_actual_source_commit_matches_or_rejects(release, tmp_path):
    root, _ = release
    code = tmp_path / "code"
    code.mkdir()
    (code / "SOURCE_COMMIT").write_text("b" * 40 + "\n")
    assert load_release(root, code_root=code).code_identity_checked
    (code / "SOURCE_COMMIT").write_text("c" * 40 + "\n")
    with pytest.raises(ReleaseContractError, match="SOURCE_COMMIT"):
        load_release(root, code_root=code)


def test_deprecated_parent_is_rejected_even_for_missing_leaf(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "DEPRECATED").write_text("Use the final release")
    with pytest.raises(ReleaseContractError, match="DEPRECATED"):
        reject_deprecated_paths([old / "nested" / "missing.usda"])


def test_symlink_into_deprecated_root_is_rejected(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "DEPRECATED").write_text("old")
    alias = tmp_path / "alias"
    alias.symlink_to(old, target_is_directory=True)
    with pytest.raises(ReleaseContractError, match="DEPRECATED"):
        reject_deprecated_paths([alias / "scene.usda"])


def test_symlink_out_of_deprecated_root_retains_original_ancestor(tmp_path):
    old, healthy = tmp_path / "old", tmp_path / "healthy"
    old.mkdir()
    healthy.mkdir()
    (old / "DEPRECATED").write_text("old")
    (old / "alias").symlink_to(healthy, target_is_directory=True)
    with pytest.raises(ReleaseContractError, match="DEPRECATED"):
        reject_deprecated_paths([old / "alias" / "missing.usda"])


def test_code_alias_out_of_deprecated_root_is_rejected(release, tmp_path):
    root, _ = release
    old, healthy = tmp_path / "old-code", tmp_path / "healthy-code"
    old.mkdir()
    healthy.mkdir()
    (old / "DEPRECATED").write_text("old code")
    (old / "alias").symlink_to(healthy, target_is_directory=True)
    with pytest.raises(ReleaseContractError, match="DEPRECATED"):
        load_release(root, code_root=old / "alias")


def test_canonical_ancestry_not_string_prefix(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "DEPRECATED").write_text("old")
    good = tmp_path / "old-but-current"
    good.mkdir()
    paths = reject_deprecated_paths([old / ".." / good.name / "scene.usda"])
    assert paths == ((good / "scene.usda").resolve(),)


def test_release_root_and_manifest_aliases_use_same_deprecation_guard(release, tmp_path):
    root, data = release
    old = tmp_path / "old"
    old.mkdir()
    (old / "manifest.json").write_text("{}")
    (old / "DEPRECATED").write_text("old")
    (root / "linked.json").symlink_to(old / "manifest.json")
    data["manifest"] = "linked.json"
    save(root, data)
    with pytest.raises(ReleaseContractError, match="DEPRECATED"):
        load_release(root)


def test_unmarked_missing_path_is_not_a_new_existence_gate(tmp_path):
    path = tmp_path / "future" / "asset.usda"
    assert reject_deprecated_paths([path]) == (path.resolve(),)


def test_deprecated_code_root_is_rejected_without_descriptor_commit(release, tmp_path):
    root, data = release
    data.pop("code_commit")
    save(root, data)
    code = tmp_path / "old-code"
    code.mkdir()
    (code / "DEPRECATED").write_text("old code")
    with pytest.raises(ReleaseContractError, match="DEPRECATED"):
        load_release(root, code_root=code)


def test_descriptor_alias_cannot_leave_release(release, tmp_path):
    root, data = release
    outside = tmp_path / "descriptor.json"
    outside.write_text(json.dumps(data))
    (root / "RELEASE.json").unlink()
    (root / "RELEASE.json").symlink_to(outside)
    with pytest.raises(ReleaseContractError, match="inside"):
        load_release(root)
