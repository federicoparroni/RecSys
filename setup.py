import setuptools

setuptools.setup(
    name="gabdampar_recommender",
    version="0.0.1",
    author="Gabdampar",
    author_email="damicoedoardo@gmail.com giovanni.gabbolini@gmail.com federico.parroni@live.it",
    description="Another recommender system algorithms collection",
    url="https://github.com/pypa/sampleproject",
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'similaripy'
    ]
)