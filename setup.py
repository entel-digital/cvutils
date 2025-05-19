from setuptools import find_packages, setup

setup(
    name='cvutils',
    packages=find_packages(), 
    version='0.1.0',
    install_requires=[],
    entry_points={
        'console_scripts': [
            'cvutils = cvutils.cli:main',
        ],
    },
    description='Computer vision tools',
    author='Entel Ocean, Odd Industries',
    license='MIT',
)
