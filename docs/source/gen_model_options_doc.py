# coding=utf-8

"""
Generates rst files for model options
"""
from thetis.configuration import *
from thetis.options import CommonModelOptions, ModelOptions2d, ModelOptions3d, GLSModelOptions, LinearEquationOfStateOptions, SedimentModelOptions


with open('model_options_2d.rst', 'w') as f:
    content = """
.. _model_options_2d:

2D model options
================

This page lists all available options for the 2D model.

"""
    content += rst_all_options(CommonModelOptions) + "\n"
    content += rst_all_options(ModelOptions2d)
    f.write(content)


with open('model_options_3d.rst', 'w') as f:
    content = """
.. _model_options_3d:

3D model options
================

This page lists all available options for the 3D model.

See also :ref:`turbulence_options` and :ref:`eos_options`.

"""
    content += rst_all_options(CommonModelOptions) + "\n"
    content += rst_all_options(ModelOptions3d)
    f.write(content)


with open('turbulence_options.rst', 'w') as f:
    content = """
.. _turbulence_options:

Turbulence model options
========================

This page lists all available options for turbulence closure models.

Generic Length Scale model options
----------------------------------

"""
    content += rst_all_options(GLSModelOptions)
    f.write(content)


with open('eos_options.rst', 'w') as f:
    content = """
.. _eos_options:

Equation of State options
=========================

This page lists all available options for defining the equation of state.

Linear Equation of State
------------------------

"""
    content += rst_all_options(LinearEquationOfStateOptions)
    f.write(content)


with open('sediment_model_options.rst', 'w') as f:
    content = """
.. _sediment_model_options:

2D sediment model options
=========================

This page lists all available options for the 2D sediment model.

"""
    content += rst_all_options(SedimentModelOptions) + "\n"
    f.write(content)
