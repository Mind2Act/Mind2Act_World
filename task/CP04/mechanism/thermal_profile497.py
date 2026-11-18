"""Predeclared thermal difficulty, identical for train/test; no motion policy."""
import copy
import hashlib
import json
VERSION = 'cooking_thermal497_v3_4_3_2_off_freeze'
WINDOWS = {'easy': 4., 'medium': 3., 'hard': 2.}
LEGACY_WINDOWS = {'easy': 12., 'medium': 8., 'hard': 5.}
EPS_C = 1e-8  # Existing thermal solver's floating-point tolerance, not contact policy.
OLD_TEXT = '在110°C上限前切换保温（有效预热至少80°C）'

def config(level):
    window = WINDOWS[level]
    return dict(version=VERSION, difficulty=level, window_seconds=window,
                green_min_C=80., green_max_C=80.+2.5*window,
                heating_C_per_s=2.5, cooling_C_per_s=0.,
                bounds='inclusive; floating point tolerance 1e-8 C',
                scope='train_and_test', trigger='measured HEAT_to_KEEP_WARM',
                shutdown='physical OFF after observed hold confirmation; freeze actual temperature without cooling; no added deadline')

def window_seconds(level, version=None):
    if version is None: return LEGACY_WINDOWS[level]
    if version != VERSION: raise ValueError('Unknown thermal profile: '+str(version))
    return WINDOWS[level]
