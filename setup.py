# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:34:24 2025

@author: cami_
"""
import os
from setuptools import setup, find_packages

# Función para leer archivos de forma segura con UTF-8
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

setup(
    name='KPIs',
    version='0.1.0',
    description='Librería para el cálculo de Índices de Resiliencia Térmica',
    
    # Usar la función read_file para asegurar la codificación UTF-8
    long_description=read_file('README.md'), 
    long_description_content_type='text/markdown',
    author='Cami',
    author_email='camilaescuderofiqueni@gmail.com',
    url='https://github.com/camiescuderofiqueni/KPIs',
    
    packages=find_packages(),
    
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
    ],
    
    # Clasificadores (ayudan a la gente a encontrar tu librería en PyPI)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.6',
)