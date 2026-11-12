"""Deterministic episode geometry. Target geometry stays evaluator-private."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw

CASE_ROOT = Path(__file__).resolve().parents[1]

CONFIG = json.loads((CASE_ROOT / "config/case.json").read_text())
W, H = CONFIG["raster_size"]
BW, BH = CONFIG["board_size_m"]
PPM = W / BW
PALETTE=np.array([(244,245,238)]+[c["rgb"] for c in CONFIG["colors"]],dtype=np.uint8)
INK = tuple(PALETTE[1])
PAPER = (244, 245, 238)


def xy_to_pixel(xy):
    a = np.asarray(xy, dtype=float)
    return np.stack([(a[..., 0] + BW / 2) * PPM,
                     (BH / 2 - a[..., 1]) * PPM], axis=-1)


def pixel_to_xy(rc):
    a = np.asarray(rc, dtype=float)
    return np.stack([a[..., 1] / PPM - BW / 2,
                     BH / 2 - a[..., 0] / PPM], axis=-1)


def raster(paths):
    im = Image.new("L", (W, H))
    d = ImageDraw.Draw(im)
    width = round(CONFIG["pen_width_m"] * PPM)
    for path in paths:
        pts = [tuple(p) for p in xy_to_pixel(path)]
        d.line(pts, fill=255, width=width, joint="curve")
        for x, y in (pts[0], pts[-1]):
            r = width / 2 - 0.5
            d.ellipse((x-r, y-r, x+r, y+r), fill=255)
    return np.array(im) > 0


def board_rgb(mask):
    return PALETTE[np.asarray(mask,dtype=np.uint8)]


def color_layers(paths,colors):
    labels=np.zeros((H,W),dtype=np.uint8)
    for path,color in zip(paths,colors):labels[raster([path])]=color
    return labels


def circle(x, y, r):
    t = np.linspace(0, 2 * math.pi, 101)
    return np.stack([x + r*np.cos(t), y + r*np.sin(t)], axis=1)


def make_case(difficulty="hard", seed=0):
    """Frozen 5 original families x 5 interference patterns x 2 layouts.

    Geometry constraints are checked before physics. Seeds encode strata;
    there is no simulation-success-based selection or resampling.
    """
    if difficulty == 'mid':difficulty='medium'
    if difficulty not in CONFIG['difficulties'] or seed not in range(100):
        raise ValueError('coverage-v1 requires a valid difficulty and condition index 0..49')
    condition = seed % 50
    family, interference, layout = condition//10, (condition//2)%5, condition%2
    triangle=[[-.175,-.055],[-.115,.040],[-.055,-.055],[-.175,-.055]]
    triangle_changes=[(-.115,.040,0.),(-.132,.043,.003),(-.098,.041,-.003),(-.122,.049,-.004),(-.108,.046,.004)]
    apex_x,apex_y,base_slant=triangle_changes[family]
    triangle[1]=[apex_x,apex_y];triangle[0][1]+=base_slant;triangle[2][1]-=base_slant;triangle[3]=triangle[0].copy()
    poly=np.array([[-.165,.087],[-.123,.060],[-.082,.100],[-.034,.072]])
    poly[:,1]+=np.array([[0,0,0,0],[.003,-.006,-.004,.002],[-.003,.004,-.007,-.002],[.005,.002,-.003,.005],[-.004,-.004,.003,-.004]])[family]
    def ellipse(cx,cy,rx,ry):
        t=np.linspace(0,2*math.pi,101);return np.stack([cx+rx*np.cos(t),cy+ry*np.sin(t)],axis=1)
    radius=[(.036,.036),(.034,.038),(.039,.034),(.035,.037),(.038,.035)][family]
    targets=[np.array(triangle),poly]
    if difficulty=='easy':
        targets[1]=ellipse(.102,-.005,*radius)
        y=[-.018,-.008,.002,.012,.022][interference]
        tilt=[0.,.009,-.009,.014,-.014][interference]
        additions=[np.array([[-.190,y-tilt],[.170,y+tilt]])]
    else:
        targets.append(ellipse(.102,.062,*radius))
        y=[.030,.034,.028,.036,.032][interference]
        tilt=[0.,.003,-.003,.004,-.004][interference]
        line=np.array([[-.190,y-tilt],[.177,y+tilt]])
        if difficulty=='medium':
            x=[-.105,-.115,-.120,-.095,-.085][interference]
            additions=[line,np.array([[x,.035],[x,.115]])]
        else:
            quad=np.array([[.055,-.092],[.128,-.092],[.128,-.032],[.055,-.032],[.055,-.092]])
            change=[(0.,0.),(.006,.004),(-.005,-.003),(.004,-.004),(-.004,.005)][family]
            quad[2,0]+=change[0];quad[3,0]+=change[1];targets.append(quad)
            diagonal=[(.095,-.100),(.105,-.095),(.085,-.105),(.105,-.110),(.090,-.090)][interference]
            loop=[(.081,-.063,.028,.028),(.085,-.060,.032,.026),(.090,-.060,.035,.025),(.080,-.055,.030,.030),(.090,-.065,.033,.028)][interference]
            additions=[line,np.array([[-.180,diagonal[0]],[.163,diagonal[1]]]),ellipse(*loop)]
    angle=math.radians(-1.6 if layout==0 else 1.6)
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    shift=np.array([-.003,.002] if layout==0 else [.003,-.002])
    transform=lambda p:.68*(np.asarray(p)@rotation.T)+shift+np.array([0.,-.045])
    if seed >= 50:
        # Evaluation uses genuinely unseen board layouts, never train-pool RNG aliases.
        angle=math.radians(-3.4 if layout==0 else 3.4)
        rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        shift=np.array([-.005,.0035] if layout==0 else [.005,-.0035])
        scale=.665 if layout==0 else .695
        transform=lambda p:scale*(np.asarray(p)@rotation.T)+shift+np.array([0.,-.045])
    targets=[transform(p) for p in targets];additions=[transform(p) for p in additions]
    count=CONFIG['color_count_by_difficulty'][difficulty]
    target_colors=([1,1] if count==1 else [[1,2,1],[2,1,1],[1,1,2],[2,1,2],[1,2,2]][family] if count==2 else [[1,2,3,1],[2,3,1,2],[3,1,2,3],[1,3,2,1],[2,1,3,2]][family])
    addition_colors=([1] if count==1 else ([1,2] if interference%2==0 else [2,1]) if count==2 else [[2,1,3],[1,3,2],[3,2,1],[2,3,1],[3,1,2]][interference])
    target=color_layers(targets,target_colors);work=target.copy()
    for path,color in zip(additions,addition_colors):work[raster([path])]=color
    payload={'schema':'mind2act.cp09-restoration.generated-case.v1','task_id':CONFIG['task_id'],
             'difficulty':difficulty,'seed':int(seed),'target_element_count':len(targets),
             'target_paths_m':[p.tolist() for p in targets],'addition_paths_m':[p.tolist() for p in additions],
             'color_count':count,'target_path_colors':target_colors,'addition_path_colors':addition_colors,
             'cap_open_angle_deg':CONFIG['cap_open_angle_deg'][difficulty],
             'cap_opening_mechanism':'none' if difficulty=='easy' else 'axial-pull-v2',
             'generator_version':'cp09-coverage-v1','coverage':{'original_family':family,'interference_pattern':interference,'layout':layout}}
    if seed >= 50:payload['generator_version']='cp09-heldout-layouts-v1'
    payload['case_id']=f"cp09-coverage-v1-{difficulty}-{seed:02d}-"+hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:12]
    return payload,target,work


def labels_from_rgb(rgb):
    values=np.asarray(rgb,dtype=np.int32)
    return np.argmin(np.sum((values[:,:,None,:]-PALETTE[None,None,:,:].astype(np.int32))**2,axis=-1,dtype=np.int64),axis=-1).astype(np.uint8)



def from_case(case):
    """Rasterize exact SHA-bound acquisition paths, avoiding platform float regeneration."""
    targets=case['target_paths_m'];additions=case['addition_paths_m']
    target=color_layers(targets,case['target_path_colors']);work=target.copy()
    for path,color in zip(additions,case['addition_path_colors']):work[raster([path])]=color
    return case,target,work
