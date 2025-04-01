from setuptools import setup

description = (
    "Colab GraphCast: Learning skillful medium-range global weather forecasting"
)

setup(
    name="colab_graphcast",  # новое имя пакета
    version="0.2.0.dev",
    description=description,
    long_description=description,
    author="DeepMind",
    license="Apache License, Version 2.0",
    keywords="GraphCast Weather Prediction",
    url="https://github.com/deepmind/graphcast",
    packages=["colab_graphcast"],  # имя пакета, под которым он будет импортироваться
    package_dir={"colab_graphcast": "graphcast"},  # указываем, что исходный код в папке graphcast
    install_requires=[
        "cartopy",
        "chex",
        "colabtools",
        "dask",
        "dinosaur-dycore",
        "dm-haiku",
        "dm-tree",
        "jax",
        "jraph",
        "matplotlib",
        "numpy",
        "pandas",
        "rtree",
        "scipy",
        "trimesh",
        "typing_extensions",
        "xarray",
        "xarray_tensorstore"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
