from setuptools import setup

package_name = 'ekf_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Newton Adhikari',
    maintainer_email='newton@havenomail.com',
    description='EKF state estimation ROS2 node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ekf_node = ekf_node.ekf_node:main',
        ],
    },
)
 