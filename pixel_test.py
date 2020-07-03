import numpy as np
import robosuite as suite 
from model_cnn import CNNStemNetwork


env_name  = 'SawyerPickPlace'



env = suite.make(env_name,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_height=84,
        camera_width=84,
        render_collision_mesh=False,
        render_visual_mesh=True,
        camera_name='agentview',
        use_object_obs=False,
        camera_depth=True,
        reward_shaping=True,)


state  = env.reset()
print(state)
print(state["image"])
print(state["image"].shape)




