from setuptools import find_packages, setup

setup(
    name='cvutils',
    packages=find_packages(), 
    version='0.1.0',
    install_requires=[
        'Click', 'pandas', 'matplotlib', 'scipy', 'posthog', 'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'cvutils = cvutils:cli',
        ],
    },
    description='Computer vision tools',
    author='Entel Ocean, Odd Industries',
    license='',
)
