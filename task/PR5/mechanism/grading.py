"""PR5 成功判定：只依赖每帧实测状态（印章位姿、接触），不依赖仿真器 API，可离线复算。

盖印本身的判定（单印、框内、方向、无拖印）在 shared/imprint/imprint.py 的 judge()；
这里补上 PR5 自己的“印章归位”和“机械臂离开”两项物理证据，并组装最终结果。
"""
import math
import numpy as np
from .imprint import judge

HOME_XY_TOL = .004      # 印章中心与章座中心的水平偏差上限 m
HOME_Z_TOL = .002       # 印面底与章座底板的高度偏差上限 m
HOME_YAW_DEG = 3.
HOME_TILT_DEG = 3.      # 印章倾斜上限
STILL_TOL = .0005       # 最后 0.5 s 内印章位移上限 m（静止）
ARM_CLEAR_Z = .08       # TCP 高出印章原点至少这么多才算离开


def stamp_home(final_pos, final_rot, last_positions, holder_xy, plate_top):
    """印章是否静止地放回章座。last_positions：结束前一段时间的印章位置序列。"""
    p = np.asarray(final_pos)
    tilt = math.degrees(math.acos(float(np.clip(final_rot[2, 2], -1, 1))))
    yaw_error = abs((math.degrees(math.atan2(final_rot[1, 0], final_rot[0, 0])) + 180) % 360 - 180)
    still = max(np.linalg.norm(np.asarray(x) - p) for x in last_positions) < STILL_TOL
    ok = (np.linalg.norm(p[:2] - np.asarray(holder_xy)) < HOME_XY_TOL
          and abs(p[2] - plate_top) < HOME_Z_TOL and tilt < HOME_TILT_DEG and yaw_error < HOME_YAW_DEG and still)
    return bool(ok), {'xy_err_m': float(np.linalg.norm(p[:2] - np.asarray(holder_xy))),
                      'z_err_m': float(p[2] - plate_top), 'yaw_error_deg': yaw_error, 'tilt_deg': tilt, 'still': bool(still)}


def arm_clear(tcp_z, stamp_z, gripper_touching_stamp):
    return bool(tcp_z - stamp_z > ARM_CLEAR_Z and not gripper_touching_stamp)


def score(imprints, level, face_side, home, clear):
    """level：难度参数（box_offset、box_wh、target_yaw、yaw_tol），box 坐标为卡片坐标系。"""
    return judge(imprints, tuple(level['box_offset']), level['box_wh'], face_side,
                 level['target_yaw'], level['yaw_tol'], home, clear)
