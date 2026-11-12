"""Native board material changes: public map cue, hidden map, whole-board alarm."""
from pxr import UsdShade
class BoardDisplay:
    def __init__(self,stage,board):
        self.stage,self.board,self.current=stage,board,None
    def update(self,public):
        visible=(public.get('panel') or {}).get('type')=='map'
        alarm=public.get('board_failed',False)
        state=(visible,alarm)
        if state==self.current:return False
        forbidden={tuple(x) for x in (public.get('panel') or {}).get('forbidden',[])}
        n=self.board['size']
        for x in range(n):
            for y in range(n):
                key=('alarm' if alarm else 'start' if [x,y]==self.board['start'] else 'goal' if [x,y]==self.board['goal']
                     else 'blocked' if visible and (x,y) in forbidden else 'free' if visible else 'neutral')
                prim=self.stage.GetPrimAtPath(f'/World/Task/Board/cell_{x}_{y}')
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(self.stage.GetPrimAtPath('/World/Materials/Navigation_'+key)))
        # Keep the dark backing between the red tiles so alarm never erases the grid.
        self.current=state;return True
