import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'jimny_map'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*.geojson')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liljim',
    maintainer_email='jimnyspots@gmail.com',
    description='OSM road graph import and RViz visualization for the Jimny',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'osm_to_graph = jimny_map.osm_to_graph:main',
            'map_visualizer = jimny_map.map_visualizer_node:main',
        ],
    },
)
