"""Tier binding for CP04: recipe and thermal profile come from the instance's own spec.

Source: cp04-b-v42-delivery, engine/mind2act_benchmark/integration/compiler.py
native_contract (recipe length 3 for hard, 2 otherwise). The old wrapper compiled
hard/medium requests with the easy recipe; mismatches are rejected here.
"""
from dataclasses import dataclass

from mind2act_eval.contracts import InstanceSpec
from .mechanism.thermal_profile497 import VERSION

RECIPE_LENGTH = {"easy": 2, "medium": 2, "hard": 3}


@dataclass(frozen=True)
class RecipeBinding:
    recipe: tuple[str, ...]
    recipe_indices: tuple[int, ...]
    thermal_difficulty: str
    thermal_profile_version: str


def bind_recipe(spec: InstanceSpec) -> RecipeBinding:
    tier = spec.difficulty
    if tier not in RECIPE_LENGTH:
        raise ValueError("unknown CP04 difficulty: " + tier)
    for other in RECIPE_LENGTH:
        if other != tier and f"_{other}_" in spec.instance_id:
            raise ValueError(f"CP04 instance {spec.instance_id} belongs to {other}, requested {tier}")
    identities = {int(k): v for k, v in spec.config["ingredient_identities"].items()}
    indices = tuple(spec.config["recipe_indices"])
    if len(indices) != RECIPE_LENGTH[tier] or len(set(indices)) != len(indices):
        raise ValueError(f"CP04 {tier} recipe needs {RECIPE_LENGTH[tier]} distinct ingredients: {indices}")
    if any(i not in identities for i in indices):
        raise ValueError(f"CP04 recipe has unknown ingredient index: {indices}")
    if spec.config.get("difficulty") != tier:
        raise ValueError(f"CP04 {tier} config difficulty differs from the instance difficulty")
    profile = spec.config["thermal_profile"]
    thermal = profile["difficulty"]
    if thermal != tier:
        raise ValueError(f"CP04 {tier} spec carries {thermal} thermal profile")
    version = profile.get("version")
    if version != VERSION:
        raise ValueError(f"CP04 unsupported thermal profile version: {version!r}")
    if spec.config.get("thermal_profile_version") != version:
        raise ValueError("CP04 thermal profile version differs from config thermal_profile_version")
    return RecipeBinding(tuple(identities[i] for i in indices), indices, tier, version)
