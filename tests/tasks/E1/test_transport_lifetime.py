"""CPU ownership checks; these do not exercise the native PhysX backend."""
import gc
from types import SimpleNamespace
import sys
import weakref

import numpy as np

from mind2act_eval.tasks.E1.mechanism.transport import DynamicTransport


def test_transport_keeps_simulation_owner_after_build_scope(monkeypatch):
    class BodyView:
        count = 1

        def get_transforms(self):
            return np.array([[0., 0., 0., 0., 0., 0., 1.]], dtype=np.float32)

        def set_kinematic_targets(self, targets, indices):
            assert targets.shape == (1, 7)
            assert np.isfinite(targets).all()
            assert indices.tolist() == [0]

    class SimulationOwner:
        # Like the SDK Python wrapper, the returned view has no Python reference
        # to this factory. This stub makes no claim about C++ backend ownership.
        def create_rigid_body_view(self, path):
            return BodyView()

    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(UsdGeom=SimpleNamespace(
        Xformable=lambda prim: SimpleNamespace(GetOrderedXformOps=lambda: [object()]))))

    def build_scope():
        transport = DynamicTransport(SimpleNamespace(GetPrimAtPath=lambda path: path), {},
                                     [(0., 1.)], np.array([0.]), [["item"]],
                                     ["/World/Carriers/Batch0/Collision/submit_floor"])
        owner = SimulationOwner()
        owner_ref = weakref.ref(owner)
        transport.initialize(owner)
        transport.before_step(0.)  # same path used while settling inside build
        return transport, owner_ref

    transport, owner_ref = build_scope()
    gc.collect()
    assert owner_ref() is not None, "transport must retain its simulation view after scene build returns"
    transport.before_step(0.)  # first runner tick after build/ready
    del transport
    gc.collect()
    assert owner_ref() is None  # lifetime is owned by the transport, not a global
