import os
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt')) as f:
  install_reqs = [
    s for s in [
      line.split('#', 1)[0].strip(' \t\n') for line in f
    ] if s != ''
  ]

setup(
    name='sanskrit_tagger',
    version='0.1.5',
    include_package_data=True,
    package_data={
        'sanskrit_tagger': ['data/*.dat'],
    },

    description='Pos tagger tools to use with pas taggers models',

    url='https://github.com/koleslena/sanskrit_pos_tagger',

    author='koleslena',
    author_email='',

    license='GNU GENERAL PUBLIC LICENSE',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License (GPL)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3'
    ],

    keywords='sanskrit samskritam pos tagger pos-tagger nlp',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_reqs,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        # 'dev': ['check-manifest'],
        'test': ['pytest']
    },
)