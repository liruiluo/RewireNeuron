from setuptools import find_packages, setup

# Required dependencies
required = [
    "tensorflow>=2.0",
    "mujoco-py<2.2,>=2.0",
    # "mujoco-py<2.2,>=2.1",
    # "metaworld @ git+https://github.com/rlworkgroup/metaworld.git@0875192baaa91c43523708f55866d98eaf3facaf#egg=metaworld",
    # "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb",
    "pandas",
    "matplotlib",
    "seaborn",
]

setup(
    name="continualworld",
    description="Continual World: A Robotic Benchmark For Continual Reinforcement Learning",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
