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

# Inverse kinematics(Single)
example_end_effector_pose = [0.26, -0.04, 0.6, 0.09429, -0.62593, 0.68322, -0.36406]
ik_result = curobo_motion_generator.ik(example_end_effector_pose)
print(f"ik result={ik_result}")

# Inverse kinematics(Batch)

example_end_effector_poses = [
    [0.26, -0.04, 0.6, 0.09429, -0.62593, 0.68322, -0.36406],
    [
        0.31567856669425964,
        -0.35251304507255554,
        -0.2648656368255615,
        0.2944166958332062,
        0.2446623295545578,
        0.91354900598526,
        0.13743145763874054,
    ],
    [
        0.390546977519989,
        -0.5450860857963562,
        -0.2648656666278839,
        -0.1048457995057106,
        0.36816874146461487,
        -0.0737229585647583,
        0.9208823442459106,
    ],
    [66666, 6666666, 666666, 0.09429, -0.62593, 0.68322, -0.36406],  # Dirty value
]
ik_result = curobo_motion_generator.ik_batch(example_end_effector_poses)
print(f"batch ik result={ik_result}")
