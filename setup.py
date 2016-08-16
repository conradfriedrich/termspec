# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='termspec',
    version='0.0.1',
    description='Word space model to analyze term specificity or semantic specificity.',
    long_description=readme,
    author='Conrad Friedrich',
    author_email='conradfried®@gmail.com',
    url='https://github.com/conradfriedrich/termspec',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
