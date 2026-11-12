import sys
from types import ModuleType

def native_modules(monkeypatch):
    def module(name,**values):
        value=ModuleType(name);value.__path__=[]
        for key,item in values.items():setattr(value,key,item)
        monkeypatch.setitem(sys.modules,name,value)
        if '.' in name:
            parent,leaf=name.rsplit('.',1)
            if parent not in sys.modules:module(parent)
            monkeypatch.setattr(sys.modules[parent],leaf,value,raising=False)
        return value
    module('isaacsim.core.prims',SingleRigidPrim=object)
    module('omni.physx',get_physx_simulation_interface=lambda:None)
    module('omni.physx.bindings._physx',ContactEventType=object)
    module('omni.physics.tensors')
    module('pxr.PhysicsSchemaTools');module('pxr.PhysxSchema')
    return module
