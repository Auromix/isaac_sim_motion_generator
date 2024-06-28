#!/bin/bash
# set -x
set -e

# Get  path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
cd $SCRIPT_DIR

# Get curobo path
if [ ! -d "curobo" ]; then
    echo "curobo folder not exist, please check!"
    exit 1
fi

cd curobo
CUROBO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "CUROBO_DIR: $CUROBO_DIR"

cd $SCRIPT_DIR
echo "Copying robot assets files to curobo..."

copy_robot_assets() {
    local robot=$1
    local robot_config=$2
    local world_config=$3

    echo "Copying $robot"
    cp -r $SCRIPT_DIR/assets/$robot $CUROBO_DIR/src/curobo/content/assets/robot
    cp $SCRIPT_DIR/assets/$robot/$robot_config $CUROBO_DIR/src/curobo/content/configs/robot
    cp $SCRIPT_DIR/assets/$robot/$world_config $CUROBO_DIR/src/curobo/content/configs/world
}

# Copy ur5
copy_robot_assets "ur5" "ur5_robot_config.yml" "ur5_world_config.yml"

# Copy franka
copy_robot_assets "my_franka" "my_franka_robot_config.yml" "my_franka_world_config.yml"

# If you need to copy other robot assets, you can add them here.
# copy_robot_assets "your_robot_name" "your_robot_name_robot_config.yml" "your_robot_name_world_config.yml"

echo "Isaac sim motion generator configuration is done!"
