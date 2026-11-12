from mind2act_eval.contracts import Action, Observation
from mind2act_eval.adapters.coding_agent import CodingAgentAdapter


def test_coding_agent_transport_receives_public_observation_only():
    class Transport:
        def reset(self, *, seed):
            assert seed == 9
        def call_tool(self, name, **arguments):
            assert name == "act"
            assert tuple(arguments) == ("observation",)
            observation = arguments["observation"]
            assert not hasattr(observation, "evidence")
            assert not hasattr(observation, "assets")
            assert observation.instruction == "public instruction"
            return dict(left_arm=[0.]*6, right_arm=[0.]*6, left_gripper=0., right_gripper=1.)
        def close(self):
            pass
    adapter = CodingAgentAdapter(Transport())
    adapter.reset(9)
    action = adapter.act(Observation(0, 0., (0.,)*6, (0.,)*6, 1., 1., instruction="public instruction"))
    assert isinstance(action, Action)
    assert action.right_gripper == 1.
    adapter.close()
