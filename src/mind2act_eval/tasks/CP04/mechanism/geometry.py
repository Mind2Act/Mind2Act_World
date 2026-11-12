"""SI dimensions shared by kitchen-jar rendering, IK and contact geometry."""
FOOD_PIECES_PER_JAR = 3
FOOD_PIECE_SCALE = 1.2
FOOD_PIECE_MASS_KG = .002
JAR_RADIUS = .038
JAR_INNER_RADIUS = .0355
MOUTH_RADIUS = .034
JAR_HEIGHT = .090
JAR_FLOOR = .004
CAP_RADIUS = .040
CAP_Z = .082
CAP_HEIGHT = .014
BODY_GRIP_Z = .034
CAP_GRIP_Z = CAP_Z + CAP_HEIGHT * .85
PITCH = .012
RELEASE_ANGLE = 120.
# Versioned opt-in: old 14mm/120deg profile remains byte-for-byte dimensional default.
import os as _os
GEOMETRY_PROFILE = 'shortturn413_30'
if GEOMETRY_PROFILE not in ['legacy120', 'tallcap407_360', 'tallcap408_120', 'tallcap411_120_32', 'shortturn413_30']:
    raise ValueError('Unknown cooking geometry profile: '+GEOMETRY_PROFILE)
CAP_MASS = .025
CAP_COM_Z = CAP_HEIGHT/2
CAP_INERTIA = (CAP_MASS*(3*CAP_RADIUS**2+CAP_HEIGHT**2)/12,)*2+(CAP_MASS*CAP_RADIUS**2/2,)
if GEOMETRY_PROFILE in ['tallcap407_360', 'tallcap408_120', 'tallcap411_120_32', 'shortturn413_30']:
    CAP_HEIGHT = {'tallcap407_360':.040, 'tallcap408_120':.026, 'tallcap411_120_32':.032, 'shortturn413_30':.026}[GEOMETRY_PROFILE]
    CAP_GRIP_Z = CAP_Z + CAP_HEIGHT - .008
    PITCH = .004 if GEOMETRY_PROFILE == 'tallcap407_360' else .012
    RELEASE_ANGLE = 360. if GEOMETRY_PROFILE == 'tallcap407_360' else 120.
    # Same material density as legacy proxy: annular 2mm wall + solid 2mm roof.
    import math as _math
    _ro, _ri, _t = CAP_RADIUS, .038, .002
    _density = .025 / (_math.pi*(_ro**2-_ri**2)*.012+_math.pi*_ro**2*_t)
    _wall_h = CAP_HEIGHT-_t
    _mw = _density*_math.pi*(_ro**2-_ri**2)*_wall_h
    _mt = _density*_math.pi*_ro**2*_t
    CAP_MASS = _mw+_mt
    CAP_COM_Z = (_mw*_wall_h/2+_mt*(CAP_HEIGHT-_t/2))/CAP_MASS
    _ix = _mw*(3*(_ro**2+_ri**2)+_wall_h**2)/12 + _mw*(_wall_h/2-CAP_COM_Z)**2 + _mt*(3*_ro**2+_t**2)/12 + _mt*(CAP_HEIGHT-_t/2-CAP_COM_Z)**2
    _iz = _mw*(_ro**2+_ri**2)/2+_mt*_ro**2/2
    CAP_INERTIA = (_ix,_ix,_iz)
# Short-turn bayonet-like passive helical guide: 30deg, 1mm nominal rise.
if GEOMETRY_PROFILE == 'shortturn413_30':
    RELEASE_ANGLE = 30.
RELEASE_AXIAL_TOLERANCE = .0002 if GEOMETRY_PROFILE == 'shortturn413_30' else .001
APPLIANCE_X_OFFSET = .110
CONTROL_X = -.159 + APPLIANCE_X_OFFSET
DISPLAY_X = .067 + APPLIANCE_X_OFFSET
POWER_X = CONTROL_X
POWER_Y = -.037
KNOB_Y = -.132
POT_Y_SHIFT = -.095
POT_X_SHIFT = APPLIANCE_X_OFFSET
PAN_CENTER = (-.003861 + POT_X_SHIFT, .119681 + POT_Y_SHIFT)
POT_LID_START = (.365 + APPLIANCE_X_OFFSET, .080, .774)

# Shared roomy 3-column / 2-row drawer layout. Closed local coordinates.
DRAWER_COLUMNS = (-.550, -.425, -.300)
DRAWER_ROWS = (.300, .440)
DRAWER_CENTER_X = -.405
DRAWER_WIDTH = .430
DRAWER_SUPPORT_WIDTH = .386
DRAWER_TRAVEL = .425
DRAWER_Y_OFFSET = .065
def drawer_jar_position(index, opened=True, profile=None):
    if profile is not None:
        if profile.get("profile") != "cooking_simplified_223_easy404" or index not in profile["candidate_indices"]:
            raise ValueError("Unknown profile or unavailable jar")
        x,y=profile["drawer_layout"]["local_xy_by_index"][str(index)]
        return (x, y + DRAWER_Y_OFFSET - (DRAWER_TRAVEL if opened else 0.), .793)
    return (DRAWER_COLUMNS[index % 3], DRAWER_ROWS[index // 3] + DRAWER_Y_OFFSET - (DRAWER_TRAVEL if opened else 0.), .793)
