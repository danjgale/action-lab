
from distutils.core import setup

setup(
    name='actionlab',
    version='0.0.1',
    packages=['actionlab'],
    license='MIT',
    author='Dan Gale',
    long_description=open('README.md').read(),
    url='https://github.com/danjgale/action-lab/',
    tests_require=[
        'pytest',
        'pytest-cov'
    ]
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'scikit-image',
        'nilearn',
        'nipype'
    ]
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)