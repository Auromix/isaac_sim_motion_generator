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


# Motion generation
curobo_motion_generator.init_motion_gen()

# Start joint angles and goal joint angles
start_joint_angles = [1.2, -1.2, 1.2, -1.40, 0.80, 0.6]
goal_joint_angles = [-0.42089125514030457, -2.02638840675354,
                     1.973724126815796, -3.3172101974487305, -2.0704729557037354, 0.2292235940694809]

solution = curobo_motion_generator.motion_generate(
    start_joint_angles=start_joint_angles,
    goal_joint_angles=goal_joint_angles,
)
# print(f"motion gen solution={solution}")


# Start joint angles and goal end effector pose
start_joint_angles = [1.2, -1.2, 1.2, -1.40, 0.80, 0.6]
goal_end_effector_pose = [0.2599996328353882, -0.040003832429647446, 0.5999994277954102, -
                          0.09428329765796661, 0.6259337067604065, -0.6832123398780823, 0.3640587031841278]

solution = curobo_motion_generator.motion_generate(
    start_joint_angles=start_joint_angles,
    goal_end_effector_pose=goal_end_effector_pose,
)
# print(f"motion gen solution={solution}")
