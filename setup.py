from setuptools import setup, find_packages

__version__ = '0.0.1'

# try:
#     from setuptools.core import setup
# except ImportError:
#     from distutils.core import setup

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name='fnl_tools',
    version=__version__,
    author='Luke Chang',
    author_email='luke.j.chang@dartmouth.edu',
    url='http://neurolearn.readthedocs.org/en/latest/',
    install_requires=['numpy>=1.9', 'scipy','pandas>=0.16', 'six', 'seaborn',
                    'matplotlib','scikit-learn>=0.18.1','hmmlearn'],
    packages=find_packages(exclude=['tests']),
    package_data={'nltools': ['resources/*']},
    license='LICENSE.txt',
    description='A Python package to analyze fnl data ',
    **extra_setuptools_args
)
