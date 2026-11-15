"""Render a live E2E Goal with the reviewed per-tier RoboProbe Markdown recipe.

The current input is literal per-tier Markdown. Legacy R compilation helpers
remain for archival reproducibility only. Source Python is parsed as data,
never executed or imported.
"""
import ast
import hashlib
import json
import math
import re
import string
from pathlib import Path

from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.adapters.agent_payload import require_english

LEVELS = ('easy', 'medium', 'hard')
FIELDS = ('title', 'capability_en', 'goal_en', 'description_en', 'completion_en', 'scoring_en')
# Explicit per-task allowlist: score tiers such as 100/30/15 are never inferred
# to be difficulty variants. All other authored text remains byte-for-byte.
DIFFICULTY_GROUPS = {'PR1': [('±0.2 / ±0.15 / ±0.12', ('±0.2', '±0.15', '±0.12')),
                           ('30 / 22 / 15', ('30', '22', '15')), ('8 / 4 / 2', ('8', '4', '2'))],
 'PR2': [('±1 / ±0.5 / ±0.5', ('±1', '±0.5', '±0.5')),
         ('30 / 22 / 15', ('30', '22', '15')), ('40 / 20 / 10', ('40', '20', '10'))],
 'PR3': [('30 / 22 / 15', ('30', '22', '15'))],
 'PR5': [('100 / 80 / 74', ('100', '80', '74')),
         ('5° / 2° / 1°', ('5°', '2°', '1°')),
         ('30 / 22 / 15', ('30', '22', '15')),
         ('15 / 11 / 7.5', ('15', '11', '7.5'))],
 'PR6': [('5 / 7 / 9', ('5', '7', '9')),
         ('30 / 22 / 15', ('30', '22', '15')),
         ('3 / 4 / 5', ('3', '4', '5')),
         ('15 / 11 / 7.5', ('15', '11', '7.5'))],
 'DR1': [('3 / 5 / 7', ('3', '5', '7')), ('30 / 22 / 15', ('30', '22', '15'))],
 'DR2': [('4.0 / 3.2 / 2.6', ('4.0', '3.2', '2.6')), ('3 / 2.2 / 1.5', ('3', '2.2', '1.5'))],
 'DR3': [('7.5 / 5.5 / 3.75', ('7.5', '5.5', '3.75'))],
 'DR4': [('30 / 25 / 20', ('30', '25', '20')), ('30 / 22 / 15', ('30', '22', '15'))],
 'E1': [('2 / 4 / 6', ('2', '4', '6')), ('30 / 22 / 15', ('30', '22', '15'))],
 'E2': [('30 / 22 / 15', ('30', '22', '15'))],
 'S1': [('15 / 11 / 7.5', ('15', '11', '7.5')), ('7.5 / 5.5 / 3.75', ('7.5', '5.5', '3.75'))],
 'S2': [('4 × 4 / 5 × 5 / 6 × 6', ('4 × 4', '5 × 5', '6 × 6'))],
 'P1': [('4 / 5 / 7', ('4', '5', '7')),
        ('15 / 11 / 7.5', ('15', '11', '7.5')),
        ('7.5 / 5.5 / 3.75', ('7.5', '5.5', '3.75'))],
 'P2': [('3 / 4 / 4', ('3', '4', '4')),
        ('30 / 22 / 15', ('30', '22', '15')),
        ('20 / 15 / 10', ('20', '15', '10')),
        ('10 / 7.5 / 5', ('10', '7.5', '5'))],
 'F1': [('2 / 4 / 4', ('2', '4', '4')),
        ('15 / 11 / 7.5', ('15', '11', '7.5')),
        ('7.5 / 5.5 / 3.75', ('7.5', '5.5', '3.75'))],
 'F2': [('3 / 4 / 5', ('3', '4', '5')), ('5 / 6 / 7', ('5', '6', '7'))],
 'M1': [('4 / 5 / 6', ('4', '5', '6')),
        ('15 / 11 / 7.5', ('15', '11', '7.5')),
        ('7.5 / 5.5 / 3.75', ('7.5', '5.5', '3.75'))],
 'M2': [('2 / 4 / 8', ('2', '4', '8')), ('30 / 22 / 15', ('30', '22', '15'))],
 'CP01': [('3 / 6 / 9', ('3', '6', '9')), ('15 / 10 / 7', ('15', '10', '7'))],
 'CP02': [('4 / 6 / 6', ('4', '6', '6')), ('15 / 10 / 7', ('15', '10', '7'))],
 'CP03': [('3 / 5 / 7', ('3', '5', '7')), ('15 / 10 / 7', ('15', '10', '7'))],
 'CP04': [('4 / 3 / 2', ('4', '3', '2')), ('2 / 2 / 3', ('2', '2', '3')), ('15 / 10 / 7', ('15', '10', '7'))],
 'CP05': [('8 / 8 / 12', ('8', '8', '12')), ('15 / 10 / 7', ('15', '10', '7'))],
 'CP06': [('1 / 2 / 3', ('1', '2', '3')), ('3 / 4 / 5', ('3', '4', '5')), ('15 / 10 / 7', ('15', '10', '7'))],
 'CP08': [('6 / 4 / 4', ('6', '4', '4')), ('3 / 4 / 5', ('3', '4', '5')), ('15 / 10 / 7', ('15', '10', '7'))],
 'CP09': [('1 / 2 / 3', ('1', '2', '3')), ('15 / 10 / 7', ('15', '10', '7'))],
 'CP10': [('5 / 8 / 12', ('5', '8', '12')),
          ('3 / 4 / 6', ('3', '4', '6')),
          ('15 / 10 / 7', ('15', '10', '7'))]}


def compile_catalog(source_bytes):
    """Reproducibly extract literal r(...) records and string concatenations."""
    bound, tasks = {}, {}
    allowed = {'A_END_EN', 'A_END_ZH', 'B_END_EN', 'B_END_ZH'}

    def data(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in bound:
            return bound[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [data(item) for item in node.elts]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = data(node.left), data(node.right)
            if isinstance(left, str) and isinstance(right, str):
                return left + right
        raise ValueError('Final recipe source must contain literal data only')

    for node in ast.parse(source_bytes).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'r':
            continue  # Never execute the source helper.
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == 'R' and isinstance(node.value, ast.Dict):
                for task_node, row_node in zip(node.value.keys, node.value.values):
                    task, row = ast.literal_eval(task_node), ast.literal_eval(row_node)
                    if task in tasks or len(row) != 11:
                        raise ValueError('Duplicate or malformed final recipe task')
                    english = dict(zip(FIELDS, [row[0], row[1], row[3], row[5], row[7], row[9]]))
                    completion = english['completion_en']
                    kind = 'A' if completion.endswith(bound['A_END_EN']) else 'B' if completion.endswith(bound['B_END_EN']) else None
                    if kind is None:
                        raise ValueError('Missing authoritative episode ending')
                    tasks[task] = dict(english, kind=kind)
                continue
            if name in allowed:
                value = data(node.value)
                if not isinstance(value, str):
                    raise ValueError('Final recipe ending constant must be text')
                bound[name] = value
                continue
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'r'
                and len(node.value.args) == 12 and not node.value.keywords):
            raise ValueError('Unexpected statement in final recipe data authority')
        task, *row = [data(arg) for arg in node.value.args]
        if task in tasks:
            raise ValueError('Duplicate final recipe task')
        english = dict(zip(FIELDS, [row[0], row[1], row[3], row[5], row[7], row[9]]))
        if (not all(isinstance(english[key], str) for key in FIELDS[:-1])
                or not isinstance(english['scoring_en'], list)
                or not all(isinstance(line, str) for line in english['scoring_en'])
                or not english['goal_en'].startswith('Goal: ')):
            raise ValueError('Malformed final English recipe record')
        completion = english['completion_en']
        kind = 'A' if completion.endswith(bound['A_END_EN']) else 'B' if completion.endswith(bound['B_END_EN']) else None
        if kind is None:
            raise ValueError('Missing authoritative episode ending')
        tasks[task] = dict(english, kind=kind)
    if len(tasks) != 30:
        raise ValueError('The final recipe authority must contain 30 tasks')
    authored = {task: {key: row[key] for key in FIELDS} for task, row in tasks.items()}
    english_bytes = json.dumps(authored, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    for task, groups in DIFFICULTY_GROUPS.items():
        joined = '\n'.join([tasks[task][key] for key in FIELDS[:-1]] + tasks[task]['scoring_en'])
        if any(source not in joined for source, values in groups):
            raise ValueError('Difficulty allowlist differs from the authored recipe')
    for task, row in tasks.items():
        row['difficulty_groups'] = [dict(source=source, values=list(values))
                                    for source, values in DIFFICULTY_GROUPS.get(task, ())]
    return dict(scoring_version=SCORING_VERSION,
        goal_source='live_e2e_instruction',
        authority='reviews/final-version-20261105/FINAL-VERSION.md',
        source_path='reviews/final-version-20261105/final_coding_agent_recipes.py',
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_english_sha256=hashlib.sha256(english_bytes).hexdigest(), tasks=tasks)


CATALOG = json.loads(Path(__file__).with_name('data').joinpath('final-goals-scoring.json').read_text())
if CATALOG['scoring_version'] != SCORING_VERSION:
    raise ValueError('Coding recipe/scorer version mismatch')


def compile_roboprobe_catalog(source_bytes):
    """Extract the single reviewed literal RECIPES assignment without execution."""
    tree = ast.parse(source_bytes)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign)
            or len(tree.body[0].targets) != 1 or not isinstance(tree.body[0].targets[0], ast.Name)
            or tree.body[0].targets[0].id != 'RECIPES'):
        raise ValueError('Expected only a literal RECIPES assignment')
    recipes = ast.literal_eval(tree.body[0].value)
    if not isinstance(recipes, dict) or set(recipes) != set(CATALOG['tasks']):
        raise ValueError('RoboProbe recipes must contain all 30 tasks')
    for tiers in recipes.values():
        if not isinstance(tiers, dict) or set(tiers) != set(LEVELS):
            raise ValueError('RoboProbe recipes require exactly three tiers')
        for text in tiers.values():
            if (not isinstance(text, str) or not text.startswith('# ')
                    or re.findall(r'^## (.+)$', text, re.M) != ['Capability dimension', 'Description', 'Scoring']
                    or '| Score | Condition |' not in text):
                raise ValueError('Invalid RoboProbe Markdown recipe structure')
    require_english(recipes)
    return dict(scoring_version=SCORING_VERSION, goal_source='live_e2e_instruction',
        recipe_format='roboprobe-markdown-per-tier',
        source_path='reviews/final-version-20261105/final_coding_agent_recipes_roboprobe.py',
        source_sha256=hashlib.sha256(source_bytes).hexdigest(), recipes=recipes)


ROBOPROBE_CATALOG = json.loads(Path(__file__).with_name('data').joinpath('roboprobe-recipes.json').read_text())
if ROBOPROBE_CATALOG['scoring_version'] != SCORING_VERSION:
    raise ValueError('RoboProbe recipe/scorer version mismatch')
RECIPES = ROBOPROBE_CATALOG['recipes']


def render_public_text(task, difficulty, text, public_parameters):
    """Apply only that task's explicit three-way numeric groups and TARGET."""
    if task not in CATALOG['tasks'] or difficulty not in LEVELS:
        raise ValueError('Unknown coding recipe task or difficulty')
    index = LEVELS.index(difficulty)
    for group in CATALOG['tasks'][task]['difficulty_groups']:
        pattern = r'(?<![\d.])' + re.escape(group['source']) + r'(?!\d|\.\d)'
        text = re.sub(pattern, lambda match: group['values'][index], text)
    if '{TARGET}' in text:
        target = public_parameters.get('target')
        if task not in ('PR1', 'PR2') or type(target) not in (int, float) or not math.isfinite(target):
            raise ValueError('Finite public Goal target is required')
        text = text.replace('{TARGET}', f'{target:g}')
    return text


def validate_runtime_instruction(instruction):
    """Require public English text without coercion, trimming or a catalog fallback."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A nonempty runtime E2E instruction is required')
    require_english(instruction)
    # Keep ordinary English and scientific unit letters, reject other scripts.
    if (not re.search(r'[A-Za-z]', instruction)
            or any(c.isalpha() and c not in string.ascii_letters + 'ΩΩμµ' for c in instruction)):
        raise ValueError('The runtime E2E instruction must be English')
    if 'TASK RECIPE:' in instruction:
        raise ValueError('The runtime E2E instruction cannot contain a recipe section')
    return instruction


def render_coding_input(task, difficulty, public_parameters, *, instruction=None):
    if task not in RECIPES or difficulty not in LEVELS:
        raise ValueError('Unknown coding recipe task or difficulty')
    instruction = validate_runtime_instruction(instruction)
    # Match pinned RoboProbe _goal_content: the recipe is literal Markdown,
    # selected by the actual tier; only trailing whitespace is stripped.
    return f"Goal: {instruction}\n\nTASK RECIPE:\n{RECIPES[task][difficulty].rstrip()}"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compile the final R data authority without executing it')
    parser.add_argument('--format', choices=('roboprobe', 'legacy'), default='roboprobe')
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compiler = compile_roboprobe_catalog if args.format == 'roboprobe' else compile_catalog
    result = compiler(Path(args.source).read_bytes())
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
