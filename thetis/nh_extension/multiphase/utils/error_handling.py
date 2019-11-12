# Copyright (C) 2014-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0
"""
Error handling and assertions giving nice user input
and controlled shutdown of Ocellaris
"""


class OcellarisError(Exception):
    def __init__(self, header, description):
        super().__init__('%s: %s' % (header, description))
        self.header = header
        self.description = description


def ocellaris_error(header, description):
    raise OcellarisError(header, description)


def verify_key(name, key, options, loc=None):
    """
    Verify that a key is among a set of options. If not
    give a sensible warning.

    * name should be non-capitalized, ie. 'flower'
    * key should be the user provided input, ie. 'dandelion'
    * options should be allowable inputs, ie. ['rose', 'daisy']
    * loc is optional to provide more context, ie. 'get_flower2()'
    """
    if key not in options:
        loc = ' in %s' % loc if loc is not None else ''
        if len(options) > 1:
            if hasattr(options, 'keys'):
                options = list(options.keys())
            available_options = '\n'.join(' - %r' % m for m in options)
            ocellaris_error(
                'Unsupported %s' % name,
                'The %s %r is not available%s, please use one of:\n%s'
                % (name, key, loc, available_options),
            )
        else:
            available_options = ', '.join('%r' % m for m in options)
            ocellaris_error(
                'Unsupported %s' % name,
                'The %s %r is not available%s, only %s is available'
                % (name, key, loc, available_options),
            )


def verify_field_variable_definition(simulation, vardef, loc, return_var=True):
    """
    Verify that a variable definition like "my field/psi" refers to an existing
    field (here "my field") and contains exactly one forward slash. Optionally
    the field variable with the given name (here "psi") will be returned and in
    that process the existance of that field variable in the field is verified.
    """
    comps = vardef.strip().split('/')
    if len(comps) != 2:
        ocellaris_error(
            'Field variable reference error',
            'Field variable should be on format "field name/varname", found %r in %s'
            % (vardef, loc),
        )
    field_name, var_name = comps

    if field_name not in simulation.fields:
        existing = ','.join(simulation.fields)
        if not existing:
            existing = 'NO FIELDS DEFINED!'
        ocellaris_error(
            'Field missing',
            'No field named %r exists, existing fields: %s' % (field_name, existing),
        )

    field = simulation.fields[field_name]
    if return_var:
        return field.get_variable(var_name)
