from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'to_vehicle'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),  # Include launch directory
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liljim',
    maintainer_email='jimnyspots@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'talker = to_vehicle.to_vehicle:main',
                'data_publisher_node = to_vehicle.test_pub:main',
                'to_vehicle = to_vehicle.to_vehicle:main',  # Ensure to_vehicle executable is listed
        ],
    },
)
