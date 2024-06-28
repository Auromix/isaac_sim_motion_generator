# For Isaac Sim 4.0.0
from isaacsim import SimulationApp

isaac_sim_app_config = {
    "headless": False,
    "width": "1920",
    "height": "1080",
}
simulation_app = SimulationApp(isaac_sim_app_config)

from omni.isaac.core.world import World
from omni.isaac.core.objects import VisualCuboid

from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from isaac_sim_motion_generator.curobo_motion_generator import CuroboMotionGenerator
import numpy as np
import threading
import asyncio


def main():
    # Load robot world and user configuration
    robot_config_file_name = "my_franka_robot_config.yml"
    world_config_file_name = "collision_base.yml"
    user_config_file_name = "isaac_sim_user_config.toml"
    curobo_motion_generator = CuroboMotionGenerator(
        robot_config_file_name=robot_config_file_name,
        world_config_file_name=world_config_file_name,
        user_config_file_name=user_config_file_name,
    )

    # Motion generation
    curobo_motion_generator.init_motion_gen()

    # Instantiate World class
    world = World()
    # Add a ground plane to the stage and register it in the scene
    ground_plane = world.scene.add_default_ground_plane()

    # Add cuboid to the stage
    target_cuboid = VisualCuboid(
        prim_path="/World/Target",  # Path in the stage
        name="target_cuboid",  # Nickname in the scene
        position=np.array([0.6, 0.6, 0.1]),  # Initial position as an array [x, y, z]
        orientation=np.array(
            [1, 0, 0, 0]
        ),  # Initial orientation as an array [qw, qx, qy, qz]
        color=np.array([1, 0, 0]),  # Normalized RGB color (red)
        size=0.05,  # Size of the cuboid as an array [length, width, height]
        scale=np.array([1, 1, 1]),  # Scale factors for the cuboid
    )

    # Add robot to the stage [From isaac sim offcial assets folder]
    # Get isaac sim assets folder root path
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Could not find nucleus server with '/Isaac' folder")

    # Get franka in isaac sim offcial assets folder
    robot_asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

    # Add robot asset reference to stage
    # This will create a new XFormPrim and point it to the usd file as a reference
    add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/MyRobot")

    # Wrap the root of robot prim under a Robot(Articulation) class
    # to use high level api to set/ get attributes as well as initializing physics handles needed
    robot = Robot(prim_path="/World/MyRobot", name="my_robot")
    # Add robot to the scene
    world.scene.add(robot)

    # Init
    target_cuboid_position, target_cuboid_orientation = (
        target_cuboid.get_world_pose()
    )  # wxyz
    target_cuboid_orientation = curobo_motion_generator.wxzy_to_xyzw(
        target_cuboid_orientation
    )  # xyzw
    target_cuboid_pose = np.concatenate(
        [target_cuboid_position, target_cuboid_orientation]
    )

    print("Click PLAY to start.")
    while simulation_app.is_running():
        # simulation_app.update()
        # Step
        world.step(render=True)

        # Wait for play
        if not world.is_playing():
            continue
        # Get current_time_step_index
        current_time_step_index = world.current_time_step_index
        # Init physics handle when simulation starts
        if current_time_step_index < 2:
            # Reset world simulation physics context
            world.reset()
            # Init robot articulation controller physics handle
            robot.initialize()
            # Get controller
            articulation_controller = robot.get_articulation_controller()
            # Get activated joint index
            activated_joint_index = [
                robot.get_dof_index(x) for x in curobo_motion_generator.joint_names
            ]
            # Set default joint positions
            robot.set_joint_positions(
                curobo_motion_generator.default_joint_positions, activated_joint_index
            )
        # Wait for stablization
        if current_time_step_index < 60:
            continue
        # Update obstacle at init
        if current_time_step_index == 60:
            print(
                f"Initializing isaac sim obstacle world with respect to {robot.prim_path}..."
            )
            # Initialize obstacle world
            curobo_motion_generator.update_obstacle_from_isaac_sim(
                world=world,
                robot_prim_path=robot.prim_path,
                ignore_prim_paths=[
                    robot.prim_path,
                    ground_plane.prim_path,
                    target_cuboid.prim_path,
                ],
            )
        # Update obstacle every 500 steps
        if current_time_step_index % 500 == 0:
            curobo_motion_generator.update_obstacle_from_isaac_sim(
                world=world,
                robot_prim_path=robot.prim_path,
                ignore_prim_paths=[
                    robot.prim_path,
                    ground_plane.prim_path,
                    target_cuboid.prim_path,
                ],
            )

        # Get current cube pose
        current_cuboid_position, current_cuboid_orientation = (
            target_cuboid.get_world_pose()
        )  # wxyz
        current_cuboid_orientation = curobo_motion_generator.wxzy_to_xyzw(
            current_cuboid_orientation
        )  # xyzw
        current_cuboid_pose = np.concatenate(
            [current_cuboid_position, current_cuboid_orientation]
        )

        # Check if robot is static
        robot_static = False
        if robot.get_joint_velocities() is not None:
            if np.max(np.abs(robot.get_joint_velocities())) < 0.2:
                robot_static = True

        if (
            # Cuboid moved
            np.linalg.norm(target_cuboid_pose - current_cuboid_pose) > 1e-2
            # Robot not moving now
            and robot_static
        ):
            # Update target cuboid pose
            target_cuboid_pose = current_cuboid_pose
            # Get current robot joint state
            current_curobo_robot_joint_state = (
                curobo_motion_generator.get_joint_state_from_isaac_sim(robot_prim=robot)
            )

            print(target_cuboid_pose)

            solution = curobo_motion_generator.motion_generate(
                start_joint_angles=current_curobo_robot_joint_state.position.cpu().numpy(),
                goal_end_effector_pose=target_cuboid_pose,
            )
            if solution is None:
                print("No motion solution found, skip solution execution")
                continue

            # Convert solution to ArticulationActions
            actions = curobo_motion_generator.convert_solution_to_isaac_sim_action(
                solution=solution, robot_prim=robot
            )

            # Execute motion plan result
            for action_index, action in enumerate(actions):
                articulation_controller.apply_action(action)
                # Step
                world.step(render=True)
                print(f"Action {action_index} executed")

    simulation_app.close()


if __name__ == "__main__":
    main()
