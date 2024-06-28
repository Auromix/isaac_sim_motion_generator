# Curobo
from curobo.util_file import get_robot_configs_path
from curobo.util_file import get_world_configs_path
from curobo.util_file import get_assets_path
from curobo.util_file import get_path_of_dir
from curobo.util_file import load_yaml
from curobo.util_file import join_path
from curobo.util_file import get_filename
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.types.math import Pose
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Cuboid
from curobo.wrap.reacher.ik_solver import IKSolverConfig
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen
from curobo.wrap.reacher.motion_gen import MotionGenConfig
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.wrap.reacher.motion_gen import PoseCostMetric
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel


# Others
import numpy as np
import torch
import toml
import os
import sys
from typing import List, Tuple, Optional


class CuroboMotionGenerator:
    def __init__(
        self,
        robot_config_file_name: str = "ur5.yml",
        world_config_file_name: str = "collision_base.yml",
        user_config_file_name: Optional[str] = None,
    ):
        """Initialize Curobo Motion Generator.

        Args:
            robot_config_file_name:
                curobo robot configuration file name.
            world_config_file_name:
                curobo world configuration file name.
            user_config_file_name:
                Curobo Motion Generator user config file name.
                If None, the default user config file will be loaded.
        """
        # Init logger
        self.logger = self._init_logger()
        # Get path of Curobo Motion Generator
        self.curobo_motion_generator_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.logger.log_debug(
            f"Curobo Motion Generator directory:\n{self.curobo_motion_generator_dir}\n"
        )
        # Init tensor
        self.tensor_args = TensorDeviceType(
            device=torch.device("cuda:0"), dtype=torch.float32
        )
        # Load robot config
        self.robot_config = self.load_robot_config(robot_config_file_name)
        # Load world config
        self.world_config = self.load_world_config(world_config_file_name)
        # Load user config
        self.user_config = self.load_user_config(user_config_file_name)
        # Init kinematics
        self.init_kinematics()

    def _init_logger(self):
        """Initialize logger."""
        from auro_utils.loggers.logger import Logger

        return Logger(log_level="debug", use_file_log=False, log_path=None)

    def load_robot_config(self, robot_config_file_name: str = "ur5.yml"):
        """Load robot configuration.

        Args:
            robot_config_file_name:
                robot configuration file name of the robot.

        Returns:
            robot_config:
                robot configuration instance for Curobo Motion Generator.
        """

        self.logger.log_debug("Loading robot configuration...")
        try:
            # Get robot configuration full path
            robot_config_full_path = join_path(
                get_robot_configs_path(), robot_config_file_name
            )
            self.logger.log_debug(
                f"Robot configuration file full path:\n{ robot_config_full_path}\n"
            )
            # Load robot configuration
            robot_config = load_yaml(robot_config_full_path)["robot_cfg"]
            self.logger.log_debug(f"Robot configuration:\n{robot_config}\n")

            # Get robot urdf file full path and name
            self.urdf_full_path = join_path(
                get_assets_path(), robot_config["kinematics"]["urdf_path"]
            )
            self.urdf_file_name = get_filename(self.urdf_full_path)
            self.logger.log_debug(
                f"Robot URDF file path:\n{ self.urdf_full_path}\n")
            self.logger.log_debug(
                f"Robot URDF file name:\n{ self.urdf_file_name}\n")

            # Get directory path of urdf file
            self.directory_path_of_urdf = get_path_of_dir(self.urdf_full_path)
            self.logger.log_debug(
                f"Directory path of URDF:\n{self.directory_path_of_urdf}\n"
            )

            # This list specifies the names of individual joints in the arm
            self.joint_names = robot_config["kinematics"]["cspace"]["joint_names"]
            self.logger.log_debug(f"Joint names:\n{self.joint_names}\n")
            self.robot_dof = len(self.joint_names)
            self.logger.log_debug(f"Robot DOF:\n{self.robot_dof}\n")

            # Locked joints
            self.locked_joints = robot_config["kinematics"]["lock_joints"]

            self.logger.log_debug(f"Locked joints:\n{self.locked_joints}\n")

            # This list represents the desired positions for each joint when the arm is in a retracted or idle state
            self.initial_joint_angles = robot_config["kinematics"]["cspace"][
                "retract_config"
            ]
            self.logger.log_debug(
                f"Initial joint angles:\n{self.initial_joint_angles}\n"
            )

            # Convert dict to RobotConfig instance
            robot_config_instance = RobotConfig.from_dict(
                robot_config, self.tensor_args
            )
        except Exception as e:
            self.logger.log_error(f"Failed to load robot configuration: {e}")
            sys.exit(1)

        self.robot_config_dict = robot_config

        return robot_config_instance

    def load_world_config(
        self, world_config_file_name: str = "collision_base.yml"
    ) -> WorldConfig:
        """Load world configuration.

        Args:
            world_config_file_name:
                world configuration file name.

        Returns:
            world_config:
                world configuration instance.

        """
        # Load world configuration
        self.logger.log_debug("Loading world configuration...")
        try:
            # Load from standalone world config file
            world_config_full_path = join_path(
                get_world_configs_path(), world_config_file_name
            )
            self.logger.log_debug(
                f"World configuration file path:\n{ world_config_full_path}\n"
            )
            world_config = WorldConfig.from_dict(
                load_yaml(world_config_full_path))

            self.logger.log_debug(f"World configuration:\n{world_config}\n")
        except Exception as e:
            self.logger.log_error(f"Failed to load world configuration: {e}")
            sys.exit(1)
        return world_config

    def load_user_config(self, config_file_name: str = None):
        """Load user configuration.

        Args:
            config_file_name:
                user config filename.
                If None, the default user config file will be loaded.

        """
        self.logger.log_debug("Loading user configuration...")
        try:
            if config_file_name is None:
                user_config_file_path = os.path.join(
                    self.curobo_motion_generator_dir, "config", "default_user_config.toml"
                )

            else:
                user_config_file_path = os.path.join(
                    self.curobo_motion_generator_dir, "config", config_file_name
                )
                # Check if the file exists
                if not os.path.exists(user_config_file_path):
                    self.logger.log_error(
                        f"File {config_file_name} does not exist.")
                    return None

            self.logger.log_debug(
                f"Load Curobo Motion Generator user config from {user_config_file_path}"
            )
            user_config = toml.load(user_config_file_path)
        except Exception as e:
            self.logger.log_error(
                f"Failed to load Curobo Motion Generator user configuration: {e}"
            )
            sys.exit(1)

        return user_config

    def init_kinematics(self):
        # Get robot base link and end effector link
        self.base_link = self.robot_config_dict["kinematics"]["base_link"]
        self.end_effector_link = self.robot_config_dict["kinematics"]["ee_link"]
        self.logger.log_debug(
            f"Robot Base Link:\n{self.base_link}\n")
        self.logger.log_debug(
            f"Robot End Effector Link:\n{self.end_effector_link}\n")

        # Get kinematics model
        self.logger.log_debug(
            "Creating kinematics model instance...")
        self.kinematics_model = CudaRobotModel(
            self.robot_config.kinematics)

        # Instantiate IKSolverConfig
        self.ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg=self.robot_config,
            world_model=self.world_config,
            tensor_args=self.tensor_args,
            rotation_threshold=self.user_config["ik"]["rotation_threshold"],
            position_threshold=self.user_config["ik"]["position_threshold"],
            num_seeds=self.user_config["ik"]["num_seeds"],
            self_collision_check=self.user_config[
                "ik"]["self_collision_check"],
            self_collision_opt=self.user_config["ik"]["self_collision_opt"],
            use_cuda_graph=self.user_config["ik"]["use_cuda_graph"],
            collision_activation_distance=self.user_config[
                "ik"]["collision_activation_distance"],
        )

        # Instantiate IKSolver
        self.ik_solver = IKSolver(self.ik_config)

    def init_motion_gen(self):
        """Initialize motion generation."""
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            # Robot configuration to load
            robot_cfg=self.robot_config,
            # World configuration to load
            world_model=self.world_config,
            # Tensor device type
            tensor_args=self.tensor_args,
            # Interpolation dt to use for output trajectory
            interpolation_dt=self.user_config["motion_generation"]["interpolation_dt"],
            # Distance(in meters) at which collision checking is activated
            collision_activation_distance=self.user_config["motion_generation"][
                "collision_activation_distance"
            ],
            # Maximum number of steps to interpolate
            interpolation_steps=self.user_config["motion_generation"][
                "interpolation_steps"
            ],
            # The number of IK seeds to run per query problem
            num_ik_seeds=self.user_config["motion_generation"]["num_ik_seeds"],
            # The number of trajectory optimization seeds to use per query problem
            num_trajopt_seeds=self.user_config["motion_generation"][
                "num_trajopt_seeds"
            ],
            # The number of iterations of the gradient descent trajectory optimizer
            grad_trajopt_iters=self.user_config["motion_generation"][
                "grad_trajopt_iters"
            ],
            # Whether to evaluate the interpolated trajectory
            evaluate_interpolated_trajectory=self.user_config["motion_generation"][
                "evaluate_interpolated_trajectory"
            ],
            # The number of time steps for the trajectory optimization
            trajopt_tsteps=self.user_config["motion_generation"]["trajopt_tsteps"],
            # Whether to use a CUDA graph, a mechanism for optimizing the execution of CUDA programs
            use_cuda_graph=self.user_config["motion_generation"]["use_cuda_graph"],
            # Number of graph planning seeds to use per query problem
            num_graph_seeds=self.user_config["motion_generation"]["num_graph_seeds"],
            # Collision checker type for curobo
            # Choose from PRIMITIVE, MESH, BLOX
            collision_checker_type=CollisionCheckerType.MESH,
            # Whether to check self-collision
            self_collision_check=self.user_config["motion_generation"][
                "self_collision_check"
            ],
            # Maximum trajectory time in seconds
            maximum_trajectory_time=self.user_config["motion_generation"][
                "maximum_trajectory_time"
            ],
            # Others
            # velocity_scale=self.user_config["motion_generation"]["velocity_scale"],
            # acceleration_scale=self.user_config[
            #     "motion_generation"]["acceleration_scale"],
            jerk_scale=self.user_config["motion_generation"]["jerk_scale"],
            finetune_dt_scale=self.user_config["motion_generation"][
                "finetune_dt_scale"
            ],
        )

        # Instantiate MotionGen with the motion generation configuration
        self.motion_gen = MotionGen(self.motion_gen_config)

        # Init motion generation plan config
        self.motion_gen_plan_config = MotionGenPlanConfig(
            enable_graph=self.user_config["motion_generation"]["plan"]["enable_graph"],
            enable_opt=self.user_config["motion_generation"]["plan"]["enable_opt"],
            use_nn_ik_seed=self.user_config["motion_generation"]["plan"][
                "use_nn_ik_seed"
            ],
            need_graph_success=self.user_config["motion_generation"]["plan"][
                "need_graph_success"
            ],
            max_attempts=self.user_config["motion_generation"]["plan"]["max_attempts"],
            timeout=self.user_config["motion_generation"]["plan"]["timeout"],
            enable_graph_attempt=self.user_config["motion_generation"]["plan"][
                "enable_graph_attempt"
            ],
            partial_ik_opt=self.user_config["motion_generation"]["plan"][
                "partial_ik_opt"
            ],
            success_ratio=self.user_config["motion_generation"]["plan"][
                "success_ratio"
            ],
            fail_on_invalid_query=self.user_config["motion_generation"]["plan"][
                "fail_on_invalid_query"
            ],
            enable_finetune_trajopt=self.user_config["motion_generation"]["plan"][
                "enable_finetune_trajopt"
            ],
            parallel_finetune=self.user_config["motion_generation"]["plan"][
                "parallel_finetune"
            ],
            num_ik_seeds=None,
            num_graph_seeds=None,
            num_trajopt_seeds=None,
        )

        # Warm up the motion gen
        try:
            self.logger.log_info("Warming up motion generation...")
            self.logger.log_warning("It may take a long time, please wait...")
            self.motion_gen.warmup()
        except Exception:
            self.logger.log_error("Warm up motion generation failed.")
        else:
            self.logger.log_success("Warm up motion generation done.")

        self.motion_gen.clear_world_cache()
        self.motion_gen.reset(reset_seed=False)
        self.motion_gen.update_world(self.world_config)

    def xyzw_to_wxyz(self, quaternion: List[float]) -> List[float]:
        """Convert quaternion from [x, y, z, w] to [w, x, y, z].

        Args:
            quaternion:
                quaternion in format of [x, y, z, w]

        Returns:
            quaternion:
                quaternion in format of [w, x, y, z]

        """
        return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

    def wxzy_to_xyzw(self, quaternion: List[float]) -> List[float]:
        """Convert quaternion from [w, x, y, z] to [x, y, z, w].

        Args:
            quaternion:
                quaternion in format of [w, x, y, z]

        Returns:
            quaternion:
                quaternion in format of [x, y, z, w]

        """
        return [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]

    def fk(self, joint_angles: List[float],
           # link_name: str = None
           ) -> List[List[float]]:
        """Calculate forward kinematics(single).

        Args:
            joint_angles:
                joint angles in format of [q1, q2, q3, q4, q5, q6, q7,...]

        Returns:
            link_pose:
                link pose in format of [x, y, z, qx, qy, qz, qw]

        """
        # Get joint angles in tensor format
        joint_angles = [joint_angles]
        joint_angles = self.tensor_args.to_device(
            joint_angles)
        self.logger.log_debug(
            f"Joint Angles:\n{joint_angles}\n")

        # Compute Forward Kinematics
        self.logger.log_info("Computing Forward Kinematics...")
        fk_result = self.kinematics_model.get_state(joint_angles)
        self.logger.log_debug(
            f"Forward Kinematics Result:\n{fk_result}\n")

        end_effector_position = fk_result.ee_position[0].tolist()
        end_effector_orientation_wxyz = fk_result.ee_quaternion[0].tolist()
        # Convert quaternion from [w, x, y, z] to [x, y, z, w]
        end_effector_orientation_xyzw = self.wxzy_to_xyzw(
            end_effector_orientation_wxyz)
        # Concatenate position and orientation [x, y, z, qx, qy, qz, qw]
        end_effector_pose = end_effector_position+end_effector_orientation_xyzw

        self.logger.log_debug(
            f"End Effector Pose[x,y,z,qx,qy,qz,qw]:\n{end_effector_pose}\n")

        link_pose = end_effector_pose

        return link_pose

    def ik(self,
           end_effector_pose: List,
           ) -> Optional[List[float]]:
        """Solve inverse kinematics(single).

        Args:
            end_effector_pose:
                end effector pose in format of [x, y, z, qx, qy, qz, qw]

        Returns:
            joint_angles_single_solution:
                joint angles single solution in format of [q1, q2, q3, q4, q5, q6, q7,...]

        """
        goal_pose = []
        # Get end effector pose
        # x, y, z
        end_effector_position = self.tensor_args.to_device(
            [end_effector_pose[0], end_effector_pose[1], end_effector_pose[2]])
        # qx, qy, qz, qw to qw, qx, qy, qz
        end_effector_orientation = self.tensor_args.to_device(
            [end_effector_pose[6], end_effector_pose[3], end_effector_pose[4], end_effector_pose[5]])

        # Get pose in Pose format
        goal_pose = Pose(
            position=end_effector_position,
            quaternion=end_effector_orientation,
        )
        self.logger.log_debug(
            f"Goal Pose[x,y,z,qx,qy,qz,qw]:\n{goal_pose}\n")

        # Solve Inverse Kinematics (single)
        self.logger.log_info("Solving Inverse Kinematics...")
        ik_result = self.ik_solver.solve_single(goal_pose)
        # Get joint angles single solution
        joint_angles_single_solution = ik_result.solution[ik_result.success]
        # Convert tensor to list
        if joint_angles_single_solution.numel() != 0:
            joint_angles_single_solution = joint_angles_single_solution[0].tolist(
            )
            self.logger.log_debug(
                f"Inverse Kinematics Result:\n{ik_result}\n")
            self.logger.log_debug(
                f"Inverse Kinematics Single Solution:\n{joint_angles_single_solution}\n")
            return joint_angles_single_solution
        else:
            self.logger.log_warning(
                f"Inverse Kinematics No Solution\n")
            return None

    def motion_generate(
        self,
        start_joint_angles: List[float],
        goal_end_effector_pose: Optional[List[float]] = None,
        goal_joint_angles: Optional[List[float]] = None,

    ):
        """Generate robot arm motion.
        Args:
            start_joint_angles:
                start joint angles in format of [q1, q2, q3, q4, q5, q6, q7,...]
            goal_end_effector_pose:
                goal end effector pose in format of [x, y, z, qx, qy, qz, qw]
            goal_joint_angles:
                goal joint angles in format of [q1, q2, q3, q4, q5, q6, q7,...]


        Returns:
            solution_dict_to_return:
                solution dict to return in format of
                {
                "success": Bool,
                "joint_names": [j1, j2, j3, ...],
                "positions": [[joint_angle1, joint_angle2, ...], [joint_angle1, joint_angle2, ...]],
                "velocities": [[...], [...]],
                "accelerations": [[...], [...]],
                "jerks": [[...], [...]]
                }

        """

        if goal_joint_angles is None and goal_end_effector_pose is not None:
            self.logger.log_debug(
                "Start Joint Angles to Goal End Effector Pose Mode")
            # Get start joint angles
            start_joint_angles = start_joint_angles
            start_joint_angles = JointState.from_position(
                position=self.tensor_args.to_device([start_joint_angles]),
                joint_names=self.joint_names[0: len(start_joint_angles)],
            )
            self.logger.log_info(
                f"Start Joint Angles:\n{start_joint_angles.position}"
            )

            # Get goal pose
            goal_pose = goal_end_effector_pose
            goal_pose = Pose(
                # [x,y,z]
                position=self.tensor_args.to_device(
                    [goal_pose[0], goal_pose[1], goal_pose[2]]
                ),
                # [qx,qy,qz,qw] to [qw,qx,qy,qz]
                quaternion=self.tensor_args.to_device(
                    [goal_pose[6], goal_pose[3], goal_pose[4], goal_pose[5]]
                ),
            )
            self.logger.log_debug(f"Pose Instance:\n{goal_pose}\n")
            self.logger.log_info(
                f"Goal Pose[x,y,z,qx,qy,qz,qw]:\n{goal_end_effector_pose}")

            # Plan single(cartesian space)
            try:
                self.logger.log_info("Generating Motion...")
                motion_gen_result = self.motion_gen.plan_single(
                    start_joint_angles, goal_pose, self.motion_gen_plan_config
                )

            except Exception as e:
                self.logger.log_error(
                    f"Error occurred during motion generation. {e}")
                return None

        elif goal_joint_angles is not None and goal_end_effector_pose is None:
            self.logger.log_debug(
                "Start Joint Angles to Goal Joint Angles Mode")

            # Get start joint angles
            start_joint_angles = start_joint_angles
            start_joint_angles = JointState.from_position(
                position=self.tensor_args.to_device(
                    [start_joint_angles]),
                joint_names=self.joint_names[0:len(
                    start_joint_angles)],
            )
            self.logger.log_info(
                f"Start Joint Angles\n{start_joint_angles.position}")

            # Get goal joint angles
            goal_joint_angles = JointState.from_position(
                position=self.tensor_args.to_device(
                    [goal_joint_angles]),
                joint_names=self.joint_names[0:len(
                    goal_joint_angles)],
            )
            self.logger.log_info(
                f"Goal Joint Angles\n{goal_joint_angles.position}")

            # Plan single (joint space)
            try:
                self.logger.log_info("Generating Motion...")
                motion_gen_result = self.motion_gen.plan_single_js(
                    start_joint_angles, goal_joint_angles, self.motion_gen_plan_config)

            except Exception as e:
                self.logger.log_error(
                    f"Error occurred during motion generation. {e}")
                return None

        else:
            raise ValueError("Invalid goal type.")

        # Success check
        if motion_gen_result.success.item():
            # Interpolate trajectory
            interpolated_solution = motion_gen_result.get_interpolated_plan()
            self.logger.log_success("Motion Generation Success.")

            # Generate solution dict to return
            solution_dict_to_return = {"success": motion_gen_result.success.item(),
                                       "joint_names": interpolated_solution.joint_names,
                                       "positions": interpolated_solution.position.cpu().squeeze().numpy().tolist(),
                                       "velocities": interpolated_solution.velocity.cpu().squeeze().numpy().tolist(),
                                       "accelerations": interpolated_solution.acceleration.cpu().squeeze().numpy().tolist(),
                                       "jerks": interpolated_solution.jerk.cpu().squeeze().numpy().tolist(),
                                       "interpolation_dt": motion_gen_result.interpolation_dt}

            return solution_dict_to_return
        else:
            self.logger.log_warning("Motion Generation Failed.")
            self.logger.log_warning(motion_gen_result.status)
            return None
