"""Contact-only ink mutations; target mask is never used to apply actions."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from .layouts import CONFIG, W, H, PPM, xy_to_pixel


class InkSurface:
    def __init__(self, ink):
        self.ink = np.array(ink, dtype=np.uint8, copy=True)
        self.events = []
        self.changed = 0

    def contact(self, tool, xy, contact, tick, angle=0.0, color=1):
        if not contact:
            return 0
        px, py = xy_to_pixel(xy)
        if not np.isfinite([px, py, angle]).all():
            raise ValueError("Non-finite tool contact")
        # Rasterize a local patch to keep long episodes fast.
        radius = 40 if tool == "eraser" else 10
        x0, y0 = max(0,int(px)-radius), max(0,int(py)-radius)
        x1, y1 = min(W,int(px)+radius+1), min(H,int(py)+radius+1)
        if x1 <= x0 or y1 <= y0:
            return 0
        patch = Image.new("1", (x1-x0,y1-y0))
        d = ImageDraw.Draw(patch)
        if tool == "eraser":
            hx, hy = np.array(CONFIG["eraser_size_m"])*PPM/2
            c, s = np.cos(angle), np.sin(angle)
            corners = [(px+c*x-s*y-x0,py+s*x+c*y-y0) for x,y in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]]
            d.polygon(corners,fill=1)
        elif tool == "pen":
            r = CONFIG["pen_width_m"]*PPM/2 - .5
            d.ellipse((px-r-x0,py-r-y0,px+r-x0,py+r-y0),fill=1)
        else:
            raise ValueError("Unknown tool")
        coverage = np.array(patch, dtype=bool)
        view = self.ink[y0:y1,x0:x1]
        before = view.copy()
        if tool == "eraser":
            view[coverage] = 0
        else:
            view[coverage] = int(color)
        changed = int(np.count_nonzero(before != view))
        if changed:
            self.events.append({"tick":int(tick), "tool":tool, "xy_m":list(map(float,xy)), "changed_pixels":changed})
            self.changed += changed
        return changed


def score_surface(target, actual):
    """Tolerance-aware shape/layout metrics from final actual ink only."""
    tolerance=CONFIG['grading']['tolerance_m']*PPM
    target=np.asarray(target,dtype=np.uint8);actual=np.asarray(actual,dtype=np.uint8)
    nt,na=int(np.count_nonzero(target)),int(np.count_nonzero(actual))
    matched_target=matched_actual=0;per_color=[]
    for color in range(1,len(CONFIG['colors'])+1):
        wanted=target==color;present=actual==color
        dt=distance_transform_edt(~wanted);da=distance_transform_edt(~present)
        mt=int(np.count_nonzero(wanted&(da<=tolerance))) if present.any() else 0
        ma=int(np.count_nonzero(present&(dt<=tolerance))) if wanted.any() else 0
        matched_target+=mt;matched_actual+=ma
        if wanted.any():per_color.append(dict(color=color,recall=mt/int(wanted.sum()),precision=ma/max(1,int(present.sum()))))
    recall=matched_target/max(1,nt);precision=matched_actual/max(1,na)
    extra=(na-matched_actual)/max(1,nt)
    g=CONFIG['grading'];success=(recall>=g['min_recall'] and precision>=g['min_precision'] and extra<=g['max_extra_fraction']
        and all(c['recall']>=g['min_recall'] and c['precision']>=g['min_precision'] for c in per_color))
    union=np.count_nonzero((target>0)|(actual>0))
    return dict(success=bool(success),recall=recall,precision=precision,extra_fraction=extra,missing_fraction=1-recall,
                exact_iou=float(np.count_nonzero((target>0)&(actual==target))/max(1,union)),target_pixels=nt,actual_pixels=na,
                tolerance_m=g['tolerance_m'],source='actual_contact_modified_colored_final_raster',per_color=per_color)
