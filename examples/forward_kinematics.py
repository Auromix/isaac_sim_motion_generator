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

# Forward kinematics
example_joint_angles = [1.2, -1.2, 1.2, -1.40, 0.80, 0.6]

fk_result = curobo_motion_generator.fk(example_joint_angles)
print(f"fk result={fk_result}")
