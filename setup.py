"""pypi compatible setup module.

This setup is based on:
https://packaging.python.org/tutorials/distributing-packages/


"""
from setuptools import setup
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='systemsim',
    version='0.22',
    description='Simulator for dynamical feedback systems and networks.',
    long_description=long_description,
    url='https://github.com/laurensvalk/systemsim',
    author='Laurens Valk',
    author_email='laurensvalk@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='dynamics simulation control',
    python_requires='>=3',
    packages=['systemsim'],
    project_urls={
        'Bug Reports': 'https://github.com/laurensvalk/systemsim/issues',
        'Questions': 'https://github.com/laurensvalk/systemsim/issues',
        'Source': 'https://github.com/laurensvalk/systemsim',
    }
)
