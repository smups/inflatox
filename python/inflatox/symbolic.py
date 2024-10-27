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

import numpy as np
import sympy
from interruptingcow import timeout
from joblib import Parallel, cpu_count, delayed
from sympy.simplify import sqrtdenest
from sympy.vector import Gradient


class InflationModel:
    """This class describes a multi-field inflation model. It contains all the symbolic expressions
    needed by inflatox to compute consistency conditions, solve the equations of motion etc.

    To obtain an instance of this class, you can either:
    1. Construct it directly
        This requires manually computing a fair few expressions, which can result in much smaller
        and thus more efficient expressions.
    2. Construct it automatically using the `SymbolicCalculation` class
        Using the `SymbolicCalculation` class is much easier: you only need to provide said class
        with an expression for the field-space metric and potential and it will automatically
        derive all other quantities. The disadvantage of that approach is that it may produce
        suboptimal expressions.
    """

    def __init__(
        self,
        model_name: str,
        coordinates: list[sympy.Symbol],
        tangents: list[sympy.Symbol],
        basis: list[list[sympy.Expr]],
        eom: list[sympy.Expr],
        potential: sympy.Expr,
        gradient_square: sympy.Expr,
        hesse_cmp: list[list[sympy.Expr]],
    ):
        self.model_name = model_name
        self.hesse_cmp = hesse_cmp
        self.coordinates = coordinates
        self.coordinate_tangents = tangents
        self.dim = len(coordinates)
        self.basis = basis
        self.eom = eom
        self.potential = potential
        self.gradient_square = gradient_square
        if len(hesse_cmp[0]) != len(hesse_cmp):
            raise Exception(
                "The Hesse matrix is square; the provided list was not (number of columns != number of rows)"
            )
        if len(hesse_cmp[0]) != len(basis[0]):
            raise Exception("The provided Hesse Matrix and basis are of different dimensionality")
        if len(basis) != len(coordinates):
            raise Exception(
                "The dimension of the provided basis does not match the number of fields."
            )
        if len(coordinates) != len(tangents):
            raise Exception(
                "The number of coordinate symbols does not match the number of tangent symbols."
            )

    def __str__(self):
        return f"""[Inflatox Inflation Model]
model name: {self.model_name}
dimensionality: {self.dim} field(s)
coordinates: {[f for f in self.coordinates]}
potential: {self.potential}
basis vectors (cntr. var.): {[sympy.Matrix(vec) for vec in self.basis]}
hesse matrix: {sympy.Matrix(self.hesse_cmp)}
    """


class SimplificationTimeOut(Exception):
    """This exception is thrown when a simplification exceeds a user-defined time-out time. This
    exception is meant to be caught."""

    pass


class SymbolicCalculation:
    """This class is a helper-class for producing instances of `InflationModel`. After obtaining
    an instance of this class, an instance of `InflationModel` can be obtained with the `.execute()`
    method.

    ### Usage
    An instance of this `SymbolicCalculation` class may be constructed from a
    user-defined metric on the scalar manifold, a potential and a list of sympy
    Symbols indicating which symbols (used in the definition of the potential and
    metric) should be interpreted as fields.

    Calling the `.execute()` method on an instance of `SymbolicCalculation` with
    an appropriate starting point for constructing the vielbein basis will perform
    the calculation of the components of the covariant Hesse matrix projected
    onto the vielbein.
    """

    @classmethod
    def new(
        cls,
        fields: list[sympy.Symbol],
        field_metric: list[list[sympy.Symbol]],
        potential: sympy.Expr,
        model_name: str | None = None,
        silent: bool = False,
        init_sympy_printing: bool = True,
        assertions: bool = False,
        simplify: bool = True,
        simplify_timeout: float | None = None,
    ):
        """Constructor for `SymbolicCalculation`

        ### Note
        This class will automatically derive names for the derivatives of the fields (these are
        used by other inflatox components when solving the equations of motion).

        ### Args
        - `fields` (`list[sympy.Symbol]`): list of sympy symbols that should be
          interpreted as fields (coordinates on the scalar manifold).
        - `field_metric` (`list[list[sympy.Symbol]]`): metric tensor on the scalar
          manifold.
        - `potential` (`sympy.Expr`): potential of the scalar field theory.
        - `model_name` (`str | None`, *optional*): name of the model (potential +
          scalar manifold geometry). Used for printing. Defaults to None.
        - `silent` (`bool`, *optional*): if True, no console output will be produced
          during the calculation. Defaults to False.
        - `init_sympy_printing` (`bool`, *optional*): if True, pretty printing of sympy
          expressions will be initialised with default options. Set to False if you
          want to use your own settings for printing. Defaults to True.
        - `assertions` (`bool`, *optional*): if False, expensive intermediate
          assertions will be disabled. Defaults to False.
        - `simplify` (`bool`, *optional*): When `True` `sympy`'s simplify method will be used.
          Defaults to `True`.
        - `simplify_timeout` (`float`, *optional*): time-out time in seconds for simplification
          steps.

        ### Returns
        `SymbolicCalculation`: object that can be used to perform the symbolic
        calculation by calling the `.execute()` method.
        """
        if init_sympy_printing:
            sympy.init_printing()

        return cls(
            fields=fields,
            field_metric=field_metric,
            potential=potential,
            model_name=model_name if model_name is not None else "generic model",
            silent=silent,
            assertions=assertions,
            simplify=simplify,
            simplify_timeout=simplify_timeout if simplify_timeout is not None else 20.0,
        )

    def __init__(
        self,
        fields: list[sympy.Symbol],
        field_metric: list[list[sympy.Symbol]],
        potential: sympy.Expr,
        model_name: str,
        silent: bool,
        assertions: bool,
        simplify: bool,
        simplify_timeout: float,
    ):
        """Internal constructor"""
        assert len(field_metric) == len(field_metric[0]), "field metric should be square"
        assert len(field_metric) == len(
            fields
        ), "number of fields must match dimensionality of metric tensor"

        self.model_name = model_name
        self.dim = len(fields)
        self.fields = fields
        self.field_derivatives = sympy.symbols([f"{phi}d" for phi in fields])
        self.g = sympy.Matrix(field_metric)
        self.V = potential
        self.assertions = assertions
        self.silent = silent
        self.simplify = simplify
        self.simplify_timeout = simplify_timeout

    def simplify_expr(self, expr: sympy.Expr) -> sympy.Expr:
        """simplifies expression"""
        if not self.simplify:
            return expr
        try:
            with timeout(self.simplify_timeout, exception=SimplificationTimeOut):
                return sympy.simplify(expr, ratio=1, inverse=True)
        except SimplificationTimeOut:
            print(f"""Simplification step timed out (>{self.simplify_timeout}s)!
Consider increasing the simpliciation time-out time or turning off simplifications.""")
            return expr

    def expand_and_factor_expr(self, expr: sympy.Expr) -> sympy.Expr:
        """`sympy.expand` followed by `sympy.factor` with a time-out"""
        if not self.simplify:
            return expr
        try:
            with timeout(self.simplify_timeout, exception=SimplificationTimeOut):
                return sympy.factor(sympy.expand(expr))
        except SimplificationTimeOut:
            print(f"""Simplification step timed out (>{self.simplify_timeout}s)!
Consider increasing the simpliciation time-out time or turning off simplifications.""")
            return expr

    def sqrt_and_denest_expr(self, expr: sympy.Expr) -> sympy.Expr:
        """returns denested square root of expr (with timeout)"""
        if not self.simplify:
            return sympy.sqrt(expr)
        try:
            with timeout(self.simplify_timeout, exception=SimplificationTimeOut):
                return sqrtdenest(sympy.sqrt(expr))
        except SimplificationTimeOut:
            print(f"""Simplification step timed out (>{self.simplify_timeout}s)!
Consider increasing the simpliciation time-out time or turning off simplifications.""")
            return sympy.sqrt(expr)

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

    def execute(self, guesses: list[list[sympy.Expr]] | None = None) -> InflationModel:
        """Constructs an `InflationModel` instance.

        Performs fully symbolic calculation of the components of the covariant
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

        ### Returns
        `InflationModel`
        """
        if guesses is not None:
            assert (
                len(guesses) == self.dim - 1
            ), "number of guessed vectors must equal the number of fields minus one (n-1)"

        # Calculate an orthonormal basis
        # ...starting with a vector parallel to the potential gradient
        self.print("Calculating orthonormal basis...")
        self.print("Computing normalised potential gradient...")
        basis = [self.calc_v()]
        self.display(sympy.Matrix(basis[0]), lhs=sympy.symbols("v"))

        if guesses is None and self.dim == 2:
            basis.append([-basis[0][1], basis[0][0]])
            self.display(sympy.Matrix(basis[-1]), lhs=sympy.symbols("w_1"))
        elif guesses is None:
            raise Exception("guesses argument cannot be None if model has more than two fields")
        else:
            # followed by other gramm-schmidt produced vectors
            for i, guess in enumerate(guesses):
                basis.append(self.gramm_schmidt(basis, guess))
                self.display(sympy.Matrix(basis[-1]), lhs=sympy.symbols(f"w_{i+1}"))

        # make sure the basis is orthonormal
        if self.assertions:
            for a in range(self.dim):
                for b in range(self.dim):
                    if a == b:
                        assert sympy.Eq(
                            1, self.inner_prod(basis[a], basis[b])
                        ).simplify(), "normalisation error: v•v does not equal 1"
                    else:
                        assert sympy.Eq(
                            0, self.inner_prod(basis[a], basis[b])
                        ).simplify(), "orthogonality error: v•w does not equal 0"

        # Calculate the components of the covariant Hesse Matrix
        self.print("Calculating covariant Hesse matrix...")
        H = self.calc_hesse()
        self.display(sympy.Matrix(H), lhs=sympy.symbols("H"))

        # Project Hesse matrix
        def process(a: int, b: int):
            Hab = 0
            for x in range(self.dim):
                for y in range(self.dim):
                    Hab = Hab + H[x][y] * basis[a][x] * basis[b][y]
            return ([a, b], self.simplify_expr(Hab))

        self.print("Projecting the Hesse matrix on the vielbein basis...")
        H_proj = [[0 for _ in range(self.dim)] for _ in range(self.dim)]
        results = Parallel(n_jobs=cpu_count())(
            delayed(process)(a, b) for a in range(self.dim) for b in range(self.dim)
        )

        # print projected components of the Hesse matrix
        for idx, component in results:
            a, b = idx
            H_proj[a][b] = component
            if a == 0:
                a = "v"
            if b == 0:
                b = "v"
            self.display(component, lhs=sympy.symbols(f"H_{{{a}{b}}}"))

        # calculate the size of the gradient
        self.print("Calculating the norm of the gradient...")
        gradnorm = self.calc_gradient_square()
        self.display(gradnorm, lhs=Gradient(sympy.symbols("V")) ** 2)

        # compute the equations of motion
        self.print("Computing the equations of motion...")

        return InflationModel(
            model_name=self.model_name,
            coordinates=self.fields,
            tangents=self.field_derivatives,
            basis=basis,
            eom=[],
            potential=self.V,
            gradient_square=gradnorm,
            hesse_cmp=H_proj,
        )

    def inner_prod(self, v1: list[sympy.Expr], v2: list[sympy.Expr]) -> sympy.Expr:
        """returns the inner product of v1 and v2 with respect to configured metric.

        ### Args
        - `v1` (`list[sympy.Expr]`): first vector, once contravariant
        - `v2` (`list[sympy.Expr]`): second vector, once contravariant

        ### Returns
        `sympy.Expr`: inner product of v1 with v2 with respect to the configured
          metric tensor of the current instance.
        """
        dot = 0
        for a in range(self.dim):
            for b in range(self.dim):
                dot += self.g[a, b] * v1[a] * v2[b]
        return self.expand_and_factor_expr(dot)

    def normalize(self, vec: list[sympy.Expr]) -> list[sympy.Expr]:
        """normalizes the input vector with respect to the configured metric tensor.

        ### Args
        vec (`list[sympy.Expr]`): components of the vector to be normalised.

        ### Returns
        `list[sympy.Expr]`: normalized components of the supplied vector vec with
          respect to the metric tensor of the current instance.
        """
        # first compute the norm squared and write it as a single fraction
        normsq = 0
        for a in range(self.dim):
            for b in range(self.dim):
                normsq += self.g[a, b] * vec[a] * vec[b]
        normsq = sympy.cancel(normsq) if self.simplify else normsq

        # Try to get rid of nested square roots
        numerator, denominator = sympy.fraction(normsq)
        numerator = self.sqrt_and_denest_expr(numerator)
        denominator = self.sqrt_and_denest_expr(denominator)

        return [
            sympy.cancel(vi * denominator / numerator)
            if self.simplify
            else vi * denominator / numerator
            for vi in vec
        ]

    def christoffels(self) -> list[list[list[sympy.Expr]]]:
        """Computes the Christoffel symbols from the metric tensor.

        Returns:
            list[list[list[sympy.Symbols]]]: Christoffel Connection Γ^a_bc, indexed as Γ[a][b][c]
        """
        # Implementation taken from Einsteinpy's christoffel.py (licensed under MIT)
        gammas = np.zeros((self.dim, self.dim, self.dim), dtype=int).tolist()
        mat = sympy.Matrix(self.g)
        matinv = mat.inv()
        for t in range(self.dim**3):
            # i,j,k each goes from 0 to (self.dim-1)
            # hack for codeclimate. Could be done with 3 nested for loops
            k = t % self.dim
            j = (int(t / self.dim)) % (self.dim)
            i = (int(t / (self.dim**2))) % (self.dim)
            if k <= j:
                tmpvar = 0
                for n in range(self.dim):
                    tmpvar += (matinv[i, n] / 2) * (
                        sympy.diff(mat[n, j], self.fields[k])
                        + sympy.diff(mat[n, k], self.fields[j])
                        - sympy.diff(mat[j, k], self.fields[n])
                    )
            gammas[i][j][k] = gammas[i][k][j] = self.simplify_expr(tmpvar)
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

        ### Returns
        `list[list[sympy.Expr]]`: nested list of components of the covariant Hesse matrix.
        """
        # The connection has indices up-down-down (opposite order that we usually use)
        conn = self.christoffels()
        # output components of the Hesse matrix
        Vab = [[0 for _ in range(self.dim)] for _ in range(self.dim)]

        for a in range(self.dim):
            for b in range(self.dim):
                # Calculate ∂_a ∂_b V(ϕ)
                da_dbV = sympy.diff(self.V, self.fields[b], self.fields[a])

                # Calculate the contraction Γ_ab^c(ϕ) ∂_c V(ϕ)
                gamma_ab = 0
                for c in range(self.dim):
                    # Calculate ∂_c V(ϕ)
                    Vc = sympy.diff(self.V, self.fields[c])

                    # Calculate the full thing
                    gamma_ab = gamma_ab + conn[c][b][a] * Vc

                # set the output components
                cmp = da_dbV - gamma_ab
                Vab[a][b] = self.simplify_expr(cmp)
        return Vab

    def calc_gradient_square(self) -> sympy.Expr:
        """Calculates the size of the gradient of the potential given the metric g_ab.

        ### Precise formulation of calculated quantities
          output(ϕ) = g^ab(ϕ) ∂_a V(ϕ) ∂_b V(ϕ)
        Note that the output is actual the size of the gradient *squared*.

        Returns:
          sympy.Expr: size of the gradient squared (V_a V^a)
        """
        dim = len(self.fields)
        # non-normalised components of grad V
        gradient = [sympy.diff(self.V, φ) for φ in self.fields]
        out = 0.0

        # contract v with the inverse of the metric tensor
        for a in range(dim):
            for b in range(dim):
                out += self.g.inv()[a, b] * gradient[a] * gradient[b]
        try:
            with timeout(self.simplify_timeout, exception=SimplificationTimeOut):
                out = sympy.factor(sympy.expand(out))
        except SimplificationTimeOut:
            print(f"""Simplification step timed out (>{self.simplify_timeout}s)!
Consider increasing the simpliciation time-out time or turning off simplifications.""")
        return self.simplify_expr(out)

    def calc_v(self) -> list[sympy.Expr]:
        """calculates a normalized vector pointing in the direction of the gradient of
        the configured scalar potential of the current instance.

        ### Precise formulation of calculated quantities
        The contravariant components of the gradient of V are given by:
          (grad V)^a(ϕ) = g^ab(ϕ) ∂_b V(ϕ)

        ### Returns
        `list[sympy.Expr]`: contravariant components of normalized gradient vector v.
        """
        # first construct and normalize v, then raise it (this is often simpler)
        v = [sympy.diff(self.V, φ) for φ in self.fields]
        v = self.normalize(v)

        # contract v with the inverse of the metric tensor
        vup = [0 for _ in v]
        for a in range(self.dim):
            for b in range(self.dim):
                vup[a] += self.g.inv()[a, b] * v[b]
        return [self.simplify_expr(vupi) for vupi in vup]

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

        ### Returns
        `list[sympy.Expr]`: list of the contravariant components of an additional
          basis vector orthogonal to the supplied set of basis vectors, with respect
          to the supplied metric.
        """
        dim = len(current_basis[0])
        # make sure the supplied basis is not already complete
        assert len(current_basis) < dim, "current basis is already complete. No need for more vecs."

        # start with vector y
        y = guess

        # subtract the overlap of each current basis with the guessed vector from y
        for x in current_basis:
            xdoty = self.inner_prod(x, y)
            for a in range(dim):
                y[a] -= xdoty * x[a]
        try:
            with timeout(self.simplify_timeout, exception=SimplificationTimeOut):
                y = [sympy.factor(sympy.expand(yi)) for yi in y]
        except SimplificationTimeOut:
            print(f"""Simplification step timed out (>{self.simplify_timeout}s)!
Consider increasing the simpliciation time-out time or turning off simplifications.""")

        return [self.simplify_expr(yi) for yi in self.normalize(y)]

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

        ### Returns
        `sympy.Expr`: returns the inner product of the Hesse matrix with v1 and v2.
        """
        V_proj = 0
        for a in range(self.dim):
            for b in range(self.dim):
                V_proj = V_proj + hesse_matrix[a][b] * v1[a] * v2[b]
        return self.simplify_expr(V_proj)
