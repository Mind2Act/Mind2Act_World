"""Zero change policy for native installation checks, not benchmark submissions."""
from mind2act_eval.contracts import Action


class Hold:
    def reset(self, seed):
        pass

    def act(self, observation):
        return Action(observation.left_arm, observation.right_arm,
                      observation.left_gripper, observation.right_gripper)

    def result_metadata(self):
        return dict(policy='hold_current_joints', smoke_only=True, benchmark_submission=False)

    def close(self):
        pass


def create(config):
    return Hold()
