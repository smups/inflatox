#  Copyright© 2023 Raúl Wolters(1)
#
#  This file is part of Inflatox.
#
#  Inflatox is free software: you can redistribute it and/or modify it under
#  the terms of the European Union Public License version 1.2 or later, as
#  published by the European Commission.
#
#  Inflatox is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE. See the European Union Public License for more details.
#
#  You should have received a copy of the EUPL in an/all official language(s) of
#  the European Union along with Inflatox.  If not, see
#  <https://ec.europa.eu/info/european-union-public-licence_en/>.
#
#  (1) Resident of the Kingdom of the Netherlands; agreement between licensor and
#  licensee subject to Dutch law as per article 15 of the EUPL.

from typing import Literal

import numpy as np
import sympy
from joblib import Parallel, cpu_count, delayed
from sympy import powdenest


class SymbolicOutput:
    """Class containing the components of the projected Hesse matrix, as well as
    information about the model that was used to calculate said components.
    """

    def __init__(
        self,
        hesse_cmp: list[list[sympy.Expr]],
        basis: list[list[sympy.Expr]],
        coordinates: list[sympy.Symbol],
        potential: sympy.Expr,
        gradient_size: sympy.Expr,
        model_name: str,
        silent: bool,
    ):
        self.hesse_cmp = hesse_cmp
        self.basis = basis
        self.dim = len(hesse_cmp[0])
        self.coordinates = coordinates
        self.potential = potential
        self.model_name = model_name
        self.gradient_square = gradient_size
        self.silent = silent
        if len(hesse_cmp[0]) != len(hesse_cmp):
            raise Exception(
                "The Hesse matrix is square; the provided list was not (number of columns != number of rows)"
            )
        if len(hesse_cmp[0]) != len(basis[0]):
            raise Exception(
                "The provided Hesse Matrix and basis are of different dimensionality"
            )

    def __str__(self):
        return f"""[Symbolic Calculation Output]
    dimensionality: {self.dim} field(s)
    model name: {self.model_name}
    coordinates: {[f for f in self.coordinates]}
    potential: {self.potential}
    hesse matrix: {sympy.Matrix(self.hesse_cmp)}
    basis vectors (cntr. var.): {[sympy.Matrix(vec) for vec in self.basis]}
    """


class SymbolicCalculation:
    """This class represents the symbolic calculation of the Hesse matrix of the
    scalar field potential, projected on an orthonormal vielbein basis constructed
    from the supplied potential. Both the basis and Hesse matrix are covariant
    with respect to the supplied metric on the scalar manifold.

    ### Usage
    An instance of this `SymbolicCalculation` class may be constructed from a
    user-defined metric on the scalar manifold, a potential and a list of sympy
    Symbols indicating which symbols (used in the definition of the potential and
    metric) should be interpreted as fields.

    Two different constructors can be used:
    1. `SymbolicCalculation.new()` takes a `field_metric` argument as an instance
      of EinsteinPy's `MetricTensor` class.
    2. `SymbolicCalculation.new_from_list()` takes a `field_metric` argument as a
      normal python nested list of sympy expressions. This constructor is especially
      useful when other functionality supplied by the EinsteinPy package is not required.

    Calling the `.execute()` method on an instance of `SymbolicCalculation` with
    an appropriate starting point for constructing the vielbein basis will perform
    the calculation of the components of the covariant Hesse matrix projected
    onto the vielbein.

    ### Simplification depth
    When constructing this class, the simplification depth has to be set. The
    simplification depth represents how often intermediate answers will be
    simplified. Intermediate simplifications can take a lot of time, but may
    produce prettier and more human-readable output.

    Currently there are five levels, where each level enables all the previous
    simplifications as well as the bulleted additional ones:
      - (0) No simplification
      - (1) Only simplify the final output of the calculation
      - (2) Simplify the output of nested functions before returning.
      - (3) Simplify the output of twice nested functions before returning.
      - (>=4) Simplify all intermediate steps.
      - (>=5) Extra expensive simplification of final output
    It can be beneficial to play around with this setting to see which level works
    best.
    """

    @classmethod
    def new_from_list(
        cls,
        fields: list[sympy.Symbol],
        field_metric: list[list[sympy.Symbol]],
        potential: sympy.Expr,
        model_name: str | None = None,
        simplification_depth: int = 4,
        silent: bool = False,
        init_sympy_printing: bool = True,
        assertions: bool = False,
        simplify_for: Literal["length"] | Literal["ops"] = "ops",
    ):
        """[DEPRECATED] see SymbolicCalculation.new()"""
        return SymbolicCalculation.new(
            fields,
            field_metric,
            potential,
            model_name,
            simplification_depth,
            silent,
            init_sympy_printing,
            assertions,
            simplify_for,
        )

    @classmethod
    def new(
        cls,
        fields: list[sympy.Symbol],
        field_metric: list[list[sympy.Symbol]],
        potential: sympy.Expr,
        model_name: str | None = None,
        simplification_depth: int = 4,
        silent: bool = False,
        init_sympy_printing: bool = True,
        assertions: bool = False,
        simplify_for: Literal["length"] | Literal["ops"] = "ops",
    ):
        """Constructor for `SymbolicCalculation`

        ### Args
        - `fields` (`list[sympy.Symbol]`): list of sympy symbols that should be
          interpreted as fields (coordinates on the scalar manifold).
        - `field_metric` (`list[list[sympy.Symbol]]`): metric tensor on the scalar
          manifold.
        - `potential` (`sympy.Expr`): potential of the scalar field theory.
        - `model_name` (`str | None`, *optional*): name of the model (potential +
          scalar manifold geometry). Used for printing. Defaults to None.
        - `simplification_depth` (`int`, *optional*): sets the simplification depth
          for the calculation. See class documentation. Defaults to 4.
        - `silent` (`bool`, *optional*): if True, no console output will be produced
          during the calculation. Defaults to False.
        - `init_sympy_printing` (`bool`, *optional*): if True, pretty printing of sympy
          expressions will be initialised with default options. Set to False if you
          want to use your own settings for printing. Defaults to True.
        - `assertions` (`bool`, *optional*): if False, expensive intermediate
          assertions will be disabled. Defaults to False.
        - `simplify_for` (`Literal['length'] | Literal['ops']`): simplification
          strategy. When set to `length`, expressions will be optimized for code length.
          When set to `ops`, expressions will be optimized to minimize the number of
          operations. Defaults to `ops`.

        ### Returns
        `SymbolicCalculation`: object that can be used to perform the symbolic
        calculation by calling the `.execute()` method.
        """
        if init_sympy_printing:
            sympy.init_printing()

        return cls(
            fields,
            field_metric,
            potential,
            model_name if model_name is not None else "generic model",
            simplification_depth,
            silent,
            assertions,
            simplify_for,
        )

    def __init__(
        self,
        fields: list[sympy.Symbol],
        field_metric: list[list[sympy.Symbol]],
        potential: sympy.Expr,
        model_name: str,
        simplification_depth: int,
        silent: bool,
        assertions: bool,
        simplify_for: str,
    ):
        """Internal constructor"""
        assert len(field_metric) == len(
            field_metric[0]
        ), "field metric should be square"
        assert len(field_metric) == len(
            fields
        ), "number of fields must match dimensionality of metric tensor"

        self.coords = fields
        self.g = sympy.Matrix(field_metric)
        self.V = potential
        self.model_name = model_name
        self.simp = simplification_depth
        self.silent = silent
        self.assertions = assertions
        self.simplify_for = simplify_for

    def simplify(self, expr: sympy.Expr) -> sympy.Expr:
        """simplifies expression"""
        if self.simplify_for == "length":
            return powdenest(expr.nsimplify().cancel()).simplify()
        else:
            return expr.nsimplify().collect(self.coords).expand().powsimp()

    def print(self, msg: str) -> None:
        """prints msg to stdout if self.silent is not True"""
        if not self.silent:
            print(msg)

    def display(self, expr: sympy.Expr, lhs: str | None = None) -> None:
        """displays sympy expression if self.silent is not True. If lhs is not None,
        the result will be formatted as
          lhs = expr
        """
        if self.silent:
            return
        eq = sympy.Eq(lhs, expr, evaluate=False) if lhs is not None else expr
        try:
            from IPython.display import display

            display(eq)
        except NameError:
            sympy.pprint(eq)

    def execute(self, guesses: list[list[sympy.Expr]] | None = None) -> SymbolicOutput:
        """Performs fully symbolic calculation of the components of the covariant
        Hesse matrix of the potential with respect to the metric, which are then
        projected onto an orthonormal vielbein basis constructed (using the
        Gramm-Schmidt procedure) from:
          1. the gradient of the potential.
          2. the list of vectors supplied by the caller of this function.
        The details of this procedure are specified in the documentation of the
        individual methods of this class.

        ### Args
        `guesses` (`list[list[sympy.Expr]] | None`): list of vectors to be used to calculate
          an orthonormal vielbein basis onto which the components of the Hesse matrix
          will be projected. The supplied vectors *do not* have to be orthogonal, but
          they *must be* linearly independent. If this argument equals `None` and the model under
          consideration is only two-dimensional, the code is able to determine the basis on its own.
          Otherwise, it will throw an error.

        ### Simplification
        If the simplification depth is set to 1 or higher, this function will
        simplify its output before returning. See the docs of the constructor of this
        class for more info.

        ### Returns
        `HesseMatrix`: object containing the components of the projected Hesse matrix,
          as well as information about the model that was used to calculate said components.
        """
        dim = len(self.coords)
        if guesses is not None:
            assert (
                len(guesses) == dim - 1
            ), "number of guessed vectors must equal the number of fields minus one (n-1)"

        # (1) Calculate an orthonormal basis
        # (1a)...starting with a vector parallel to the potential gradient
        self.print("Calculating orthonormal basis...")
        basis = [self.calc_v()]
        self.display(sympy.Matrix(basis[0]), lhs=sympy.symbols("v"))

        if guesses is None and dim == 2:
            basis.append([-basis[0][1], basis[0][0]])
            self.display(sympy.Matrix(basis[-1]), lhs=sympy.symbols("w_1"))
        elif guesses is None:
            raise Exception(
                "guesses argument cannot be None if model has more than two fields"
            )
        else:
            # (1b) followed by other gramm-schmidt produced vectors
            for i, guess in enumerate(guesses):
                basis.append(self.gramm_schmidt(basis, guess))
                self.display(sympy.Matrix(basis[-1]), lhs=sympy.symbols(f"w_{i+1}"))

        # (1b) make sure the basis is orthonormal
        if self.assertions:
            for a in range(dim):
                for b in range(dim):
                    if a == b:
                        assert sympy.Eq(
                            1, self.inner_prod(basis[a], basis[b])
                        ).simplify(), "normalisation error: v•v does not equal 1"
                    else:
                        assert sympy.Eq(
                            0, self.inner_prod(basis[a], basis[b])
                        ).simplify(), "orthogonality error: v•w does not equal 0"

        # (2) Calculate the components of the covariant Hesse Matrix
        print("Calculating covariant Hesse matrix...")
        H = self.calc_hesse()
        self.display(sympy.Matrix(H), lhs=sympy.symbols("H"))

        # (3) Project Hesse matrix
        def process(a: int, b: int):
            Hab = 0
            for x in range(dim):
                for y in range(dim):
                    Hab = Hab + H[x][y] * basis[a][x] * basis[b][y]
            return (
                [a, b],
                powdenest(Hab.simplify(), force=True) if self.simp >= 5 else Hab,
            )

        print("Projecting the Hesse matrix on the vielbein basis...")
        H_proj = [[0 for _ in range(dim)] for _ in range(dim)]
        results = Parallel(n_jobs=cpu_count())(
            delayed(process)(a, b) for a in range(dim) for b in range(dim)
        )

        # (3b) print projected components of the Hesse matrix
        for idx, component in results:
            a, b = idx
            H_proj[a][b] = component
            if a == 0:
                a = "v"
            if b == 0:
                b = "v"
            self.display(component, lhs=sympy.symbols(f"H_{{{a}{b}}}"))

        # (4) Calculate the size of the gradient
        print("Calculating the norm of the gradient...")
        gradnorm = self.calc_gradient_square()

        return SymbolicOutput(
            H_proj, basis, self.coords, self.V, gradnorm, self.model_name, self.silent
        )

    def inner_prod(self, v1: list[sympy.Expr], v2: list[sympy.Expr]) -> sympy.Expr:
        """returns the inner product of v1 and v2 with respect to configured metric.

        ### Args
        - `v1` (`list[sympy.Expr]`): first vector, once contravariant
        - `v2` (`list[sympy.Expr]`): second vector, once contravariant

        ### Simplification
        If the simplification depth is set to 4 or higher, this function will
        simplify its output before returning.

        ### Returns
        `sympy.Expr`: inner product of v1 with v2 with respect to the configured
          metric tensor of the current instance.
        """
        ans = 0
        dim = len(v1)
        for a in range(dim):
            for b in range(dim):
                ans = ans + (v1[a] * v2[b] * self.g[a, b])
        return self.simplify(ans) if self.simp >= 4 else ans

    def normalize(self, vec: list[sympy.Expr]) -> list[sympy.Expr]:
        """normalizes the input vector with respect to the configured metric tensor.

        ### Args
        vec (`list[sympy.Expr]`): components of the vector to be normalised.

        ### Simplification
        If the simplification depth is set to 3 or higher, this function will
        simplify its output before returning.

        ### Returns
        `list[sympy.Expr]`: normalized components of the supplied vector vec with
          respect to the metric tensor of the current instance.
        """
        norm = sympy.sqrt(self.inner_prod(vec, vec))
        return [
            self.simplify(cmp / norm) if self.simp >= 3 else cmp / norm for cmp in vec
        ]

    def christoffels(self) -> list[list[list[sympy.Expr]]]:
        """Computes the Christoffel symbols from the metric tensor.

        Returns:
            list[list[list[sympy.Symbols]]]: Christoffel Connection Γ^a_bc, indexed as Γ[a][b][c]
        """
        # Implementation taken from Einsteinpy's christoffel.py (licensed under MIT)
        dims = len(self.coords)
        gammas = np.zeros((dims, dims, dims), dtype=int).tolist()
        mat = sympy.Matrix(self.g)
        matinv = mat.inv()
        for t in range(dims**3):
            # i,j,k each goes from 0 to (dims-1)
            # hack for codeclimate. Could be done with 3 nested for loops
            k = t % dims
            j = (int(t / dims)) % (dims)
            i = (int(t / (dims**2))) % (dims)
            if k <= j:
                tmpvar = 0
                for n in range(dims):
                    tmpvar += (matinv[i, n] / 2) * (
                        sympy.diff(mat[n, j], self.coords[k])
                        + sympy.diff(mat[n, k], self.coords[j])
                        - sympy.diff(mat[j, k], self.coords[n])
                    )
            gammas[i][j][k] = gammas[i][k][j] = tmpvar
        return gammas

    def calc_hesse(self) -> list[list[sympy.Expr]]:
        """returns the components of the covariant Hesse matrix in a twice-covariant
        form. Components for all pairs of the supplied coordinates are calculated for
        the scalar potential V using the supplied metric tensor.

        ### Precise formulation of calculated quantities
        The components of the covariant Hesse matrix are defined as:
          V_ab(ϕ) = ∇_a ∇_b V(ϕ)
        This is expanded as (using Einstein notation):
          V_ab(ϕ) = ∇_a (∂_b V(ϕ)) = ∂_a ∂_b V(ϕ) - Γ_ab^c(ϕ) ∂_c V(ϕ)
        Where Γ_ab^c(ϕ) is the standard Christoffel connection defined as:
          Γ_ab^c = 1/2 g^cd (∂_a g_bd + ∂_b g_ad - ∂_d g_ab)

        ### Simplification
        If the simplification depth is set to 2 or higher, this function will
        simplify its output before returning.

        ### Returns
        `list[list[sympy.Expr]]`: nested list of components of the covariant Hesse matrix.
        """
        dim = len(self.coords)
        # The connection has indices up-down-down (opposite order that we usually use)
        conn = self.christoffels()
        # output components of the Hesse matrix
        Vab = [[0 for _ in range(dim)] for _ in range(dim)]

        for a in range(dim):
            for b in range(dim):
                # Calculate ∂_a ∂_b V(ϕ)
                da_dbV = sympy.diff(self.V, self.coords[b], self.coords[a])
                if self.simp >= 3:
                    da_dbV = da_dbV.simplify()

                # Calculate the contraction Γ_ab^c(ϕ) ∂_c V(ϕ)
                gamma_ab = 0
                for c in range(dim):
                    # Calculate ∂_c V(ϕ)
                    Vc = sympy.diff(self.V, self.coords[c])
                    if self.simp >= 3:
                        Vc = self.simplify(Vc)

                    # Calculate the full thing
                    gamma_ab = gamma_ab + conn[c][b][a] * Vc

                # set the output components
                cmp = da_dbV - gamma_ab
                Vab[a][b] = self.simplify(cmp) if self.simp >= 2 else cmp
        return Vab

    def calc_gradient_square(self) -> sympy.Expr:
        """Calculates the size of the gradient of the potential given the metric g_ab.

        ### Precise formulation of calculated quantities
          output(ϕ) = g^ab(ϕ) ∂_a V(ϕ) ∂_b V(ϕ)
        Note that the output is actual the size of the gradient *squared*.

        ### Simplification
        If the simplification depth is set to 2 or higher, this function will
        simplify its output before returning.

        Returns:
          sympy.Expr: size of the gradient squared (V_a V^a)
        """
        dim = len(self.coords)
        # non-normalised components of grad V
        gradient = [sympy.diff(self.V, φ) for φ in self.coords]
        out = 0.0

        # contract v with the inverse of the metric tensor
        for a in range(dim):
            for b in range(dim):
                out = out + self.g.inv()[a, b] * gradient[a] * gradient[b]
        return self.simplify(out) if self.simp >= 2 else out

    def calc_v(self) -> list[sympy.Expr]:
        """calculates a normalized vector pointing in the direction of the gradient of
        the configured scalar potential of the current instance.

        ### Precise formulation of calculated quantities
        The contravariant components of the gradient of V are given by:
          (grad V)^a(ϕ) = g^ab(ϕ) ∂_b V(ϕ)

        ### Simplification
        If the simplification depth is set to 2 or higher, this function will
        simplify its output before returning.

        ### Returns
        `list[sympy.Expr]`: contravariant components of normalized gradient vector v.
        """
        dim = len(self.coords)
        # non-normalised components of grad V
        v = [sympy.diff(self.V, φ) for φ in self.coords]

        # contract v with the inverse of the metric tensor
        for a in range(dim):
            for b in range(dim):
                v[a] = v[a] + self.g.inv()[a, b] * v[a]

        # normalize v
        v = self.normalize(v)
        return [self.simplify(va) for va in v] if self.simp >= 2 else v

    def gramm_schmidt(
        self, current_basis: list[list[sympy.Expr]], guess: list[sympy.Expr]
    ) -> list[sympy.Expr]:
        """Use the Gramm-Schmidt procedure to find a new orthogonal basis vector given
        an incomplete set of orthogonal basis vectors and a third vector that is linearly
        independent from the other vectors.

        ### Args
        - `current_basis` (`list[list[sympy.Expr]]`): list of current *orthogonal*
          basisvectors. The components of the vectors should be given in
          *contravariant* form.
        - `guess` (`list[sympy.Expr]`): vector that is linearly independent from the
          (incomplete) set of current basis vectors. The components of this vector
          should be given in *contravariant* form. This vector needn't be
          normalized nor orthogonal to the set of current basis vectors.

        ### Precise formulation of calculated quantities
        The Gramm-Schmidt procedure starts with a(n incomplete) set of orthonormal
        basis vectors x_i and new vector y that is linearly independent of all x_i. We
        then simply subtract the overlap between y and each basisvector to obtain a
        vector x_i+1 that is orthogonal to all x_i:
          x_i+1 = y - Σ_a g_ij x^i_a y^j
        The final step is to normalise x_i+1

        ### Simplification
        If the simplification depth is set to 2 or higher, this function will
        simplify its output before returning.

        ### Returns
        `list[sympy.Expr]`: list of the contravariant components of an additional
          basis vector orthogonal to the supplied set of basis vectors, with respect
          to the supplied metric.
        """
        dim = len(current_basis[0])
        # make sure the supplied basis is not already complete
        assert (
            len(current_basis) < dim
        ), "current basis is already complete. No need for more vecs."

        # start with vector y
        y = guess

        # subtract the overlap of each current basis with the guessed vector from y
        for x in current_basis:
            xy = self.inner_prod(x, guess)
            for a in range(dim):
                y[a] = y[a] - xy * x[a]
        # normalize and return y
        y = self.normalize(y)
        return [self.simplify(ya) for ya in y] if self.simp >= 2 else y

    def project_hesse(
        self,
        hesse_matrix: list[list[sympy.Expr]],
        v1: list[sympy.Expr],
        v2: list[sympy.Expr],
    ) -> sympy.Expr:
        """This function calculates the projection of the Hesse matrix along the two
        supplied vectors.

        ### Args
        - `hesse_matrix` (`list[list[sympy.Expr]]`): twice-covariant components of
          the Hesse matrix.
        - `v1` (`list[sympy.Expr]`): first vector along which to project the Hesse
          matrix.
        - `v2` (`list[sympy.Expr]`): second vector along which to project the Hesse
          matrix.

        ### Precise formulation of calculated quantities
        The output H12 is given by
          H12 = H_ab v1^a v2^b
        Where v1 is the first input vector, v2 the second one and H_ab are the
        twice covariant components of the Hesse matrix. No metrics are required for
        this operation.

        ### Simplification
        If the simplification depth is set to 1 or higher, this function will
        simplify its output before returning.

        ### Returns
        `sympy.Expr`: returns the inner product of the Hesse matrix with v1 and v2.
        """
        dim = len(v1)
        V_proj = 0
        for a in range(dim):
            for b in range(dim):
                V_proj = V_proj + hesse_matrix[a][b] * v1[a] * v2[b]
        return self.simplify(V_proj) if self.simp >= 1 else V_proj
