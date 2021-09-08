import firedrake as fd
import networkx as nx
import sympy
from sympy.parsing.sympy_parser import \
    convert_xor, \
    parse_expr, \
    standard_transformations
from sympy.utilities.lambdify import lambdify


class ADR_Model:
    """Data structure to store an ADR model specification.

    Performs validation checks on a user-provided dictionary of strings
    and parses strings representing mathematical expressions into
    SymPy expressions.

    Class attributes
    ----------------
    transformations : tuple[function]
        Transformations applied to strings parsed as sympy expressions.

    Instance attributes
    -------------------
    constants : dict[str, float]
        Dictionary mapping strings used to uniquely identify constants
        to the values of those constants.
    species : dict[str, dict]
        Dictionary mapping strings used to uniquely identify biological
        or chemical species to dictionaries that contain data pertaining
        to those species.
    """

    # convert_xor causes "x^2" to be parsed as "x**2"
    transformations = (
        standard_transformations + (convert_xor,))

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

        Parameters
        ----------
        model_dict : dict[str, dict]
            A dict defining all species in the ADR system, including
            the symbols by which they are represented in equations,
            their full names, diffusivity constants and reaction
            terms.
            Any hard-coded numeric values should be defined under
            model_dict['constants'], and the details of each species
            should be defined under model_dict['species'].
            Example:
            { 'constants': {
                'D1': 8e-05,
                'D2': 4e-05,
                'k1': 0.024,
                'k2': 0.06 },
              'species': {
                'a': {
                  'diffusion': 'D1',
                  'name': 'Tracer A',
                  'reaction': '-a*b**2 + k1*(1-a)'},
                'b': {
                  'diffusion': 'D2',
                  'name': 'Tracer B',
                  'reaction': 'a*b^2 - (k1+k2)*b'}}}
        lambdify_modules : str|dict, optional
            Optional argument to pass as the 'modules' parameter of
            sympy.lambdify.
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
            reaction_term = parse_expr(
                str(v["reaction"]),
                transformations=ADR_Model.transformations)
            diffusion_term = parse_expr(
                str(v["diffusion"]),
                transformations=ADR_Model.transformations)
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
                    "args": reaction_args,
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
        """Substitutes numeric values of self.constants into a sympy expression.

        Parameters
        ----------
        expression : SymPy expression
            The expression into which the constant values are to be
            substituted.

        Returns
        -------
        SymPy expression
            The updated expression with constants replaced by numeric values.
        """
        for s in expression.free_symbols:
            constant_name = str(s)
            if constant_name in self.constants.keys():
                constant_value = self.constants[constant_name]
                expression = expression.subs(
                    sympy.symbols(constant_name),
                    constant_value)
        return expression

    def species_name(self, k):
        return self.species[k]["name"]

    def diffusion(self, k):
        return self.species[k]["diffusion"]

    def reaction_term(self, k):
        return self.species[k]["reaction"]["expression"]

    def reaction_args(self, k):
        return self.species[k]["reaction"]["args"]

    def reaction_function(self, k):
        return self.species[k]["reaction"]["function"]

    def list_species_keys(self):
        return sorted(
            list(self.species_keys),
            key=lambda k: self.species[k]['index'])

    def dependency_graph(self):
        """Creates a graph representing dependencies between species.

        Adds a node for each species.
        Adds a directed edge from species s1 to s2 for all (s1, s2) pairs
        in which s2 appears in the reaction term of s1, such that s1 is
        dependent on s2.
        """
        dependencies = nx.DiGraph()
        dependencies.add_nodes_from(self.list_species_keys())
        for k in self.species_keys:
            for s in self.species[k]["reaction"]["args"]:
                symbol_str = str(s)
                if symbol_str in self.species_keys:
                    dependencies.add_edge(k, symbol_str)
        return dependencies
