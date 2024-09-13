# 🦴 Isaac Sim Motion Generator

The Isaac Sim Motion Generator provides a framework for motion generation using cuRobo, a CUDA-accelerated library. This tool allows users to perform forward kinematics, inverse kinematics, and motion generation in Isaac Sim and real-world applications.

> Note: Isaac Sim Motion Generator supports Latest Isaac Sim 4.0.0 and previous versions.

## ⚙️ Installation

Welcome to the Isaac Sim Motion Generator! This guide will help you install and set up the motion generation framework using curobo, and provide a quickstart for running forward kinematics, inverse kinematics, and motion generation examples in isaac sim and real world.

### For Isaac Sim

#### Step 1: isaac sim motion generator

Navigate to the `isaac_sim_motion_generator` directory and install the package:

```bash
cd isaac_sim_motion_generator
# Install in isaac sim python
omni_python -m pip install -e .
```

#### Step 2: Install curobo

cuRobo is a CUDA-accelerated library offering fast robotics algorithms, including kinematics, collision checking, optimization, geometric planning, trajectory optimization, and motion generation, all running significantly faster through parallel computing.

Navigate to the curobo_motion_generation directory and initialize the submodule:

```bash
cd isaac_sim_motion_generator
git submodule init
git submodule update
```

Navigate to the curobo directory and install dependencies and curobo:

```bash
cd curobo
omni_python -m pip install tomli wheel ninja
omni_python -m pip install -e .[isaacsim] --no-build-isolation
```

If you encounter any issues during the installation of curobo, please refer to the [curobo installation guide](https://curobo.org/get_started/1_install_instructions.html) for troubleshooting steps and additional information.

#### Step 3: Configure curobo for isaac sim curobo motion generator

```bash
cd isaac_sim_motion_generator
bash config_isaac_sim_curobo_motion_generator.sh
```

If you want to add your own custom robot model, please refer to the [curobo](https://curobo.org/) for instructions on how to create a custom robot model.

### For System python/Conda python

Isaac sim motion generator can also be installed in system python or conda python.
It could also be used in real world.

Replace the `omni_python` command with the appropriate command for your python environment.
For example, if you are using system python, you would use `python` instead of `omni_python`.

## 🏁 Quickstart

## Forward kinematics

```bash
omni_python examples/forward_kinematics.py
```

## Inverse kinematics

```bash
omni_python examples/inverse_kinematics.py
```

## Motion generation

```bash
omni_python examples/motion_generation.py
```

```bash
omni_python examples/motion_generation_with_collision_avoidance.py
```
