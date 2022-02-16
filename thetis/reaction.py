"""
Utilities for loading advection-diffusion-reaction configurations
from dictionaries and YAML files.
"""
from .utility import OrderedDict
import firedrake as fd
import networkx as nx
import sympy
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations
from sympy.utilities.lambdify import lambdify
import yaml


__all__ = ["ADR_Model", "read_tracer_from_yml", "extract_species"]


class ADR_Model(object):
    """
    Data structure to store an ADR model specification.

    Performs validation checks on a user-provided dictionary of strings
    and parses strings representing mathematical expressions into
    SymPy expressions.

    :cvar transformations: Transformations applied to strings parsed as
        SymPy expressions.
    :type transformations: tuple
    :ivar constants:  Dictionary mapping strings used to uniquely identify
        constants to the values of those constants.
    :type constants: dict
    :ivar species: Dictionary mapping strings used to uniquely identify
        biological or chemical species to dictionaries that contain data
        pertaining to those species.
    :type species: dict
    """

    # convert_xor causes "x^2" to be parsed as "x**2"
    transformations = standard_transformations + (convert_xor,)

    reserved_symbols = {
        "pi", "e",
        "x", "y", "z", "t",
        "u", "u_x", "u_y", "u_z"}

    def __init__(self, model_dict, lambdify_modules='numpy'):
        """ADR_Model constructor.

        Uses a user-defined dict to create a data structure containing
        the names and symbols associated with all constants and species
        in the ADR system, the values of all constants, and the
        diffusivities and reaction terms of all species.

        :arg model_dict: Dictionary defining species in an ADR system.
            This must include the keys used to identify each species,
            their full names, their diffusivities and their reaction
            terms.
            Any hard-coded numeric values should be defined under
            model_dict['constants'], and the details of each species
            should be defined under model_dict['species'].
            A dictionary obtained from a correctly-formatted YAML file using
            `yaml.safe_load` should be sufficient.

            **Example**

            .. code-block:: python

                {
                    'constants': {
                        'D1': 8e-05,
                        'D2': 4e-05,
                        'k1': 0.024,
                        'k2': 0.06
                    },
                    'species': {
                        'a': {
                            'diffusion': 'D1',
                            'name': 'Tracer A',
                            'reaction': '-a*b**2 + k1*(1-a)'
                        },
                        'b': {
                            'diffusion': 'D2',
                            'name': 'Tracer B',
                            'reaction': 'a*b^2 - (k1+k2)*b'
                        }
                    }
                }

        :type model_dict: dict
        :kwarg lambdify_modules: Optional argument to pass as the `modules`
            parameter of `sympy.lambdify`.
        :type lambdify_modules: str or dict, optional
        """
        self.constants = {
            "pi": fd.pi,
            "e": fd.e
        }
        self.species = {}
        self.constant_keys = set(model_dict["constants"].keys())
        self.species_keys = set(model_dict["species"].keys())
        self._validate_keys()
        for k, v in model_dict["constants"].items():
            self.constants[k] = float(v)
        for i, (k, v) in enumerate(model_dict["species"].items()):
            # Parse reaction and diffusion terms
            assert isinstance(v["reaction"], str)
            reaction_term = parse_expr(v["reaction"], transformations=self.transformations)
            assert isinstance(v["diffusion"], str)
            diffusion_term = parse_expr(v["diffusion"], transformations=self.transformations)
            # Check that all symbols in these terms are recognised
            self._check_all_symbols_recognised(reaction_term)
            self._check_all_symbols_recognised(diffusion_term)
            # Substitute values of constants into the expressions
            # for the reaction and diffusion terms.
            reaction_term = self._substitute_constants(reaction_term)
            diffusion_term = self._substitute_constants(diffusion_term)
            # Obtain a list of arguments to be passed to the
            # reaction function
            reaction_args = list(reaction_term.free_symbols)
            # Store species-specific data.
            # The reaction expression will now be converted into
            # a callable function.
            self.species[k] = {
                "name": v["name"],
                "index": i,
                "diffusion": float(diffusion_term),
                "reaction":
                {
                    "expression": reaction_term,
                    "args": list(map(str, reaction_args)),
                    "function": lambdify(
                        reaction_args, reaction_term,
                        modules=lambdify_modules)
                }
            }

    def _validate_keys(self):
        # Check that none of the constant or species keys are reserved symbols
        for k in (self.constant_keys | self.species_keys):
            if k in ADR_Model.reserved_symbols:
                print("Symbol \"%s\" is reserved." % k)
                raise RuntimeError
        # Check that self.constant_keys and self.species_keys do not have any
        # elements in common
        shared_keys = self.constant_keys & self.species_keys
        if len(shared_keys) != 0:
            print("The following keys are shared between a species "
                  "and a constant: %s." % ', '.join(shared_keys))
            raise RuntimeError

    def _check_all_symbols_recognised(self, expression):
        for s in expression.free_symbols:
            symbol_str = str(s)
            if symbol_str in ADR_Model.reserved_symbols:
                continue
            if symbol_str in self.constant_keys:
                continue
            if symbol_str in self.species_keys:
                continue
            print("Unrecognised symbol \"%s\" found in the following "
                  "expression:" % symbol_str)
            print(expression)
            raise RuntimeError

    def _substitute_constants(self, expression):
        """
        Substitutes numeric values of `self.constants` into a SymPy expression.

        :arg expression: The expression into which the constant values are
            to be substituted.
        :type expression: SymPy expression
        :return: The updated expression with constants replaced by numeric
            values.
        :rtype: SymPy expression
        """
        for s in expression.free_symbols:
            constant_name = str(s)
            if constant_name in self.constants.keys():
                constant_value = self.constants[constant_name]
                expression = expression.subs(
                    sympy.symbols(constant_name),
                    constant_value)
        return expression

    def list_species_keys(self):
        return sorted(
            list(self.species_keys),
            key=lambda k: self.species[k]['index'])

    @property
    def dependency_graph(self):
        """
        Creates a graph representing dependencies between species.

        Adds a node for each species.
        Adds a directed edge from species s1 to s2 for all (s1, s2) pairs
        in which s2 appears in the reaction term of s1, such that s1 is
        dependent on s2.

        :return: A directed graph representing dependencies between species.
        :rtype: networkx.DiGraph
        """
        dependencies = nx.DiGraph()
        dependencies.add_nodes_from(self.list_species_keys())
        for k in self.species_keys:
            for s in self.species[k]["reaction"]["args"]:
                symbol_str = str(s)
                if symbol_str in self.species_keys:
                    dependencies.add_edge(k, symbol_str)
        return dependencies


def read_tracer_from_yml(filename, lambdify_modules=None):
    r"""
    Constructs and returns an :class:`ADR_Model` object from a YAML file.

    The YAML file is essentially a dictionary, which defines species in
    an ADR system. This must include the keys used to identify each species,
    their full names, their diffusivities and their reaction terms. Any
    hard-coded numeric values should be defined under
    ``model_dict['constants']``, and the details of each species should be
    defined under ``model_dict['species']``.

    **Example**

    .. code-block:: yaml

        constants:
          D1: 8.0e-5
          D2: 4.0e-5
          k1: 0.024
          k2: 0.06

        species:
          a:
            name: Tracer A
            diffusion: D1
            reaction: -a*b**2 + k1*(1-a)
          b:
            name: Tracer B
            diffusion: D2
            reaction: a*b^2 - (k1+k2)*b


    :arg filename: Path to a YAML file containing constant values, tracer
        diffusivities and reaction terms.
    :kwarg lambdify_modules: A string or dictionary to be passed as the
        `modules` parameter of `sympy.lambdify`.
    """
    model_dict = None
    with open(filename, 'r') as f_in:
        model_dict = yaml.safe_load(f_in)
    return ADR_Model(model_dict, lambdify_modules=lambdify_modules)


def extract_species(adr_model, function_space, key_order=None, append_dimension=False):
    r"""
    Extracts tracer species from an :class:`ADR_Model` instance and
    constructs the appropriate :class:`Function` s and coefficients.

    :arg adr_model: the :class:`ADR_Model` instance.
    :arg function_space: The Firedrake :class:`FunctionSpace` in which the
        :class:`Function` for each tracer will reside.
    :kwarg key_order: List of species keys. If not `None`, this
        determines the order in which species are added to the
        :class:`OrderedDict` before it is returned.
    :kwarg append_dimension: If `True`, a suffix will be appended to the
        label of each species from `model_dict`. The form of the suffix is
        `_Nd`, where `N` is the dimensionality of the spatial domain in
        which `function_space` resides.
    """
    key_suffix = ""
    if append_dimension:
        key_suffix = f"_{function_space.mesh().topological_dimension()}d"

    species = key_order
    if species is None:
        # Get tracer labels
        species = adr_model.list_species_keys()

    adr_dict = OrderedDict({s + key_suffix: {} for s in species})
    # NOTE: the order they are added are the order we assume for now

    # Create Functions and extract diffusion coefficients
    for s in species:
        adr_dict[s + key_suffix]["function"] = fd.Function(
            function_space, name=adr_model.species[s]["name"])
        adr_dict[s + key_suffix]["diffusivity"] = fd.Constant(
            adr_model.species[s]["diffusion"])

    # Reaction terms must be added in a separate loop, after all
    # functions have been created.
    for s in species:
        f = adr_model.species[s]["reaction"]["function"]
        args = [
            adr_dict[s2 + key_suffix]["function"]
            for s2 in adr_model.species[s]["reaction"]["args"]
        ]
        adr_dict[s + key_suffix]["reaction_terms"] = f(*args)

    return adr_dict
