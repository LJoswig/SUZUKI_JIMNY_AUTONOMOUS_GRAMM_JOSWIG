import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'jimny_localization'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liljim',
    maintainer_email='jimnyspots@gmail.com',
    description='REP-105 localization (odom/map EKFs + navsat_transform) for the Jimny',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wheel_twist = jimny_localization.wheel_twist_node:main',
        ],
    },
)
