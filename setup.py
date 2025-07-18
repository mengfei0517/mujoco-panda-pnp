from setuptools import setup, find_packages

setup(
    name="panda_mujoco_gym",
    version="0.1.0",
    description="Based on Mujoco, a collection of Panda robotic arm reinforcement learning environments",
    author="Mengfei Fan",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  # Dependencies are listed in requirements.txt
    python_requires=">=3.8",
) 