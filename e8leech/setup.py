from setuptools import setup, find_packages

setup(
    name='e8leech',
    version='0.1.0',
    description='A Python framework for the E8 and Leech lattices',
    author='Jules',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'sympy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
