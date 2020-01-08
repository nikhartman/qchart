from setuptools import setup, find_packages

setup(
    name='qchart',
    version='0.1',
    description='A tool for plotting data sent through a ZMQ socket.',
    packages=find_packages(),
    install_requires=[
        'zmq',
        'pandas>=0.22',
        'xarray',
        'simplejson',
    ],
)
