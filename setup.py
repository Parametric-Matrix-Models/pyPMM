from setuptools import setup

setup(
    name="ParametricMatrixModels",
    version="0.1.0",
    description="Package for the construction, training, and use of Parametric Matrix Models",
    url="https://github.com/pdcook/PMM",
    author="Patrick Cook",
    author_email="cookpat@frib.msu.edu",
    license="GNU General Public License v3 (GPLv3)",
    packages=["ParametricMatrixModels"],
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
)
