"""Physical assembly evidence and task-dependent support graph."""
import re
import numpy as np


def unwanted_robot_contacts(pairs):
    bad=[]
    for pair,imp in pairs.items():
        if imp<1e-5 or not any('/Robot' in p for p in pair):continue
        if all('/Robot' in p for p in pair):
            a,b=[re.search(r'/Robot(Left|Right)/Asset/([^/]+)',p) for p in pair]
            if a and b and a[1]==b[1]:
                if {a[2],b[2]}<={'camera','camera_base','link6','link7','link8'}:continue
                def number(v):return int(v[4:]) if v.startswith('link') else 0 if v=='base_link' else -50
                if abs(number(a[2])-number(b[2]))<=1:continue
            bad.append(list(pair));continue
        if any('base_link' in p for p in pair) and any('Pedestal' in p for p in pair):continue
        # Only the fingers may contact blocks; a forearm hitting them is invalid.
        if any('/CP11/Blocks/' in p for p in pair) and any('/link7/' in p or '/link8/' in p for p in pair):continue
        bad.append(list(pair))
    return bad
