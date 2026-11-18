"""Material topology for the three requested folding designs (read-only math).

These indices describe immutable pieces of the original cloth. They do not
set a simulated node or assume that the robot has achieved the fold.
"""
import numpy as np


def diagonal(n):
    row,col=np.indices((n,n));edge=(row==0)|(col==0)
    moving=row+col<n-1;stationary=row+col>n-1
    crease=np.flatnonzero(row+col==n-1)
    front=np.flatnonzero(moving)
    partners=((n-1-col)*n+(n-1-row)).ravel()[front]
    return dict(front_edge_ids=np.flatnonzero(edge),moving_ids=front,
        stationary_ids=np.flatnonzero(stationary),partner_ids=partners,crease_ids=crease)


def crease_samples(initial,nodes,crease_ids,count=5):
    """Nearest original crease material to each held point's transverse X."""
    initial=np.asarray(initial);crease_ids=np.asarray(crease_ids)
    return {side:crease_ids[np.argsort(abs(initial[crease_ids,0]-initial[node,0]))[:count]]
        for side,node in nodes.items()}


def quarter_layers(n,direction='rear_hems_backward'):
    if (n-1)%4:raise ValueError('Two equal folds require a grid divisible into four material strips')
    q=(n-1)//4;grid=np.arange(n*n).reshape(n,n)
    if direction=='folded_edge_forward':
        return [grid[3*q:],grid[q::-1],grid[q:2*q+1],grid[3*q:2*q-1:-1]]
    if direction!='rear_hems_backward':raise ValueError('Unknown second-fold direction')
    # Each array runs from the first crease toward the second crease.
    # Ordered bottom to top after forward half-fold then backward half-fold.
    return [grid[2*q:3*q+1],grid[2*q:q-1:-1],grid[:q+1],grid[:3*q-1:-1]]
