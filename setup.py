
from distutils.core import setup

setup(
    name='actionlab',
    version='0.1dev',
    packages=['actionlab'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    author='Dan Gale',
    long_description=open('README.md').read(), # PyPi description
    url='https://github.com/danjgale/action-lab/',
    tests_require=['pytest']
    # install_requires=[],
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest'],
)