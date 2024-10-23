#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "scipy",
    "matplotlib",
    "opencv-python",
    "open3d",
    "pybind11",
    # 'jax',
    "toml",
    "auro_utils==0.0.7",
    "usd-core==23.11",
]


test_requirements = [
    "pytest>=3",
]

setup(
    author="Herman Ye",
    author_email="hermanye233@icloud.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="The Isaac Sim Motion Generator provides a framework for motion generation using cuRobo, a CUDA-accelerated library. This tool allows users to perform forward kinematics, inverse kinematics, and motion generation in Isaac Sim and real-world applications.",
    install_requires=requirements,
    include_package_data=True,
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords="isaac_sim_motion_generator",
    name="isaac_sim_motion_generator",
    packages=find_packages(
        include=["isaac_sim_motion_generator", "isaac_sim_motion_generator.*"]
    ),
    test_suite="tests",
    tests_require=test_requirements,
    url="nop",
    version="0.0.1",
    zip_safe=False,
)
