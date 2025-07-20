from setuptools import find_packages, setup

package_name = 'etrobo_object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KatsumiOuwa',
    maintainer_email='ouwa@emtechs.co.jp',
    description='etrobo_object_detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ObjectDetectionNode = etrobo_object_detection.ObjectDetectionNode:main'
        ],
    },
)
