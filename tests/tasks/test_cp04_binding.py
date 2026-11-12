import pytest
from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.tasks.CP04.binding import bind_recipe
from mind2act_eval.tasks.CP04.mechanism.thermal_profile497 import VERSION

IDS = {"0": "chili", "1": "garlic", "2": "ginger", "3": "onion"}
RECIPES = {"easy": [0, 3], "medium": [0, 3], "hard": [1, 2, 0]}


def spec(difficulty, recipe=None, thermal=None, instance=None):
    return InstanceSpec(
        task_id="CP04", instance_id=instance or f"CP04_{difficulty}_x", difficulty=difficulty,
        source_id="cp04-b-v42-delivery", config_version="v1", asset_version="cp04-final",
        protocol_version="v1",
        config={"difficulty": difficulty,
                "recipe_indices": RECIPES[difficulty] if recipe is None else recipe,
                "ingredient_identities": IDS,
                "thermal_profile": {"difficulty": thermal or difficulty, "version": VERSION},
                "thermal_profile_version": VERSION})


@pytest.mark.parametrize("difficulty,count", [("easy", 2), ("medium", 2), ("hard", 3)])
def test_each_tier_binds_its_own_recipe(difficulty, count):
    bound = bind_recipe(spec(difficulty))
    assert len(bound.recipe) == count
    assert bound.thermal_difficulty == difficulty
    assert bound.thermal_profile_version == VERSION


def test_hard_with_easy_recipe_rejected():
    with pytest.raises(ValueError, match="recipe"):
        bind_recipe(spec("hard", recipe=[0, 3]))


def test_hard_with_easy_thermal_profile_rejected():
    with pytest.raises(ValueError, match="thermal"):
        bind_recipe(spec("hard", thermal="easy"))


def test_instance_of_other_tier_rejected():
    with pytest.raises(ValueError, match="instance"):
        bind_recipe(spec("hard", instance="CP04_easy_train_rebuilt_e93d"))


def test_duplicate_or_unknown_ingredient_rejected():
    with pytest.raises(ValueError):
        bind_recipe(spec("hard", recipe=[0, 0, 3]))
    with pytest.raises(ValueError):
        bind_recipe(spec("hard", recipe=[0, 3, 9]))
