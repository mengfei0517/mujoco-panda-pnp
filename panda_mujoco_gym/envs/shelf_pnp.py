import os
from panda_mujoco_gym.envs.panda_env import FrankaEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "shelf_pnp.xml")


class FrankaShelfPNPEnv(FrankaEnv):
    def __init__(
        self,
        reward_type,
        **kwargs,
    ):
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=False,
            distance_threshold=0.05,
            obj_x_range=0.05,
            obj_y_range=0.2,
            **kwargs,
        )
