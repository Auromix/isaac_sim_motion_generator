from isaac_sim_motion_generator.curobo_motion_generator import CuroboMotionGenerator
# Load robot world and user configuration
robot_config_file_name = "ur5_robot_config.yml"
world_config_file_name = "collision_base.yml"
user_config_file_name = "default_user_config.toml"
curobo_motion_generator = CuroboMotionGenerator(
    robot_config_file_name=robot_config_file_name,
    world_config_file_name=world_config_file_name,
    user_config_file_name=user_config_file_name,
)

# Inverse kinematics
example_end_effector_pose = [0.26, -0.04,
                             0.6, 0.09429, -0.62593, 0.68322, -0.36406]
ik_result = curobo_motion_generator.ik(example_end_effector_pose)
print(f"ik result={ik_result}")
