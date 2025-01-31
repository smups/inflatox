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

# External imports
import numpy as np

# Internal imports
from .compiler import CompilationArtifact
from .libinflx_rs import *

# Limit exports to these items
__all__ = ["InflationCondition", "GeneralisedAL"]


class InflationCondition:
    """Base class for all inflation conditions. Provides native methods to evaluate
    the potential and projected Hesse matrix. This base-class may be extended either
    by using these native methods, or by including your own native code that hooks
    into the Rust API or C ABI.
    """

    def __init__(self, compiled_artifact: CompilationArtifact, validate_basis: bool = True):
        """Passes compiled model to lib_inflx_rs.

        ### Args:
        - `compiled_artifact` (`CompilationArtifact`): output of `Compiler` (see its docs)
        - `validate_basis` (`bool`, optional): if `True`, lib_inflx_rs will check that the
          field-space basis defined in the `CompilationArtifact` is orthonormal
          at some number of random field-space points for with random parameter values. It will
          throw an exception if this is not the case. You may disable this if inflatox picks
          random points outside the domain of your model.
        """
        self.artifact = compiled_artifact
        self.dylib = open_inflx_dylib(compiled_artifact.shared_object_path, validate_basis)

    def calc_V(self, x: np.ndarray, args: np.ndarray) -> float:
        """calculates the scalar potential at field-space coordinates `x` with
        model-specific parameters `args`.

        ### Args
        - `x` (`np.ndarray`): field-space coordinates at which to calculate
        - `args` (`np.ndarray`): values of the model-dependent parameters. See
        `CompilationArtifact.print_sym_lookup_table()` for an overview of which
        sympy symbols were mapped to which args index.

        ### Returns
          `float`: Value of scalar potential with parameters `args` at coordinates `x`.
        """
        return self.dylib.potential(x, args)

    def calc_V_array(
        self,
        args: list[float] | np.ndarray,
        start: list[float] | np.ndarray,
        stop: list[float] | np.ndarray,
        N: list[int] | None = None,
    ) -> np.ndarray:
        """constructs an array of field space coordinates and fills it with the
        value of the scalar potential at those field space coordinates.
        The start and stop values of each axis in field-space can be specified with
        the `start` and `stop` arguments. The number of samples along each axis can
        be set with the `N` argument. It defaults to `8000` per axis.

        ### Args
        - `args` (`list[float] | np.ndarray`): values of the model-dependent
        parameters. See `CompilationArtifact.print_sym_lookup_table()` for an
        overview of which sympy symbols were mapped to which args index.
        - `start` (`list[float] | np.ndarray`): list of minimum values for
        each axis of the to-be-constructed array in field space.
        - `stop` (`list[float] | np.ndarray`): list of maximum values for each
        axis of the to-be-constructed array in field space.
        - `N` (`list[int] | None`, optional): _description_. list of the number of
        samples along each axis in field space. If set to `None`, 8000 samples will
        be used along each axis.

        ### Returns
        `np.ndarray`: value of scalar potential at specified field-space
        coordinates
        """
        n_fields = self.artifact.n_fields
        start_stop = np.array([[float(start), float(stop)] for (start, stop) in zip(start, stop)])
        N = N if N is not None else (8000 for _ in range(n_fields))
        x = np.zeros(N)
        self.dylib.potential_array(x, args, start_stop)
        return x

    def calc_H(self, x: np.ndarray, args: np.ndarray) -> np.ndarray:
        """calculates the projected covariant Hesse matrix at field-space
        coordinates `x` with model-specific parameters `args`.

        ### Args
        - `x` (`np.ndarray`): field-space coordinates at which to calculate
        - `args` (`np.ndarray`): values of the model-dependent parameters. See
        `CompilationArtifact.print_sym_lookup_table()` for an overview of which
        sympy symbols were mapped to which args index.

        ### Returns
        `np.ndarray`: Components of the projected covariant hesse matrix with
          parameters `args` at coordinates `x`.
        """
        return self.dylib.hesse(x, args)

    def calc_H_array(
        self,
        args: list[float] | np.ndarray,
        x0_start: float,
        x0_stop: float,
        x1_start: float,
        x1_stop: float,
        N: list[int] | None = None,
    ) -> np.ndarray:
        """constructs an array of field space coordinates and fills it with the
        value of the projected Hesse matrix at those field space coordinates.
        The start and stop values of each axis in field-space can be specified with
        the `start` and `stop` arguments. The number of samples along each axis can
        be set with the `N` argument. It defaults to `8000` per axis.

        ### Args
        - `args` (`list[float] | np.ndarray`): values of the model-dependent
        parameters. See `CompilationArtifact.print_sym_lookup_table()` for an
        overview of which sympy symbols were mapped to which args index.
        - `start` (`list[float] | np.ndarray`): list of minimum values for
        each axis of the to-be-constructed array in field space.
        - `stop` (`list[float] | np.ndarray`): list of maximum values for each
        axis of the to-be-constructed array in field space.
        - `N` (`list[int] | None`, optional): _description_. list of the number of
        samples along each axis in field space. If set to `None`, 8000 samples will
        be used along each axis.

        ### Returns
        `np.ndarray`: (d+2)-dimensional array for a d-dimensional field-space. The
          first two axes of this array represent the axes of the Hesse matrix itself.
          The other axes correspond to the field-space components.
        """
        n_fields = self.artifact.n_fields
        start_stop = np.array(
            [[float(x0_start), float(x0_stop)], [float(x1_start), float(x1_stop)]]
        )
        N = N if N is not None else (8000 for _ in range(n_fields))
        return self.dylib.hesse_array(np.array(n_fields, dtype=np.int64), args, start_stop)

    def validate_basis_on_domain(
        self,
        args: list[float] | np.ndarray,
        start: list[float] | np.ndarray,
        stop: list[float] | np.ndarray,
        N: list[int] | int = 100,
        accuracy: float = 1e-3,
    ) -> None:
        """checks if basis is orthonormal at all points in the interval [start, stop].

        If `start` and `stop` are arrays/lists, they will be interpreted as start/stop values along
        each axis. The number of samples along each axis is specified with the `N` argument.

        At each point, the basis defined in the model is constructed and it is verified that:
            1. All vectors are normalised in the sense that (g_ab v^a v^a - 1) < accuracy.
            2. All vectors are orthogonal in the sense that |g_ab v^a v^b| < accuracy.

        ### Args
        - `args` (`list[float] | np.ndarray`): values of the model-dependent
        parameters. See `CompilationArtifact.print_sym_lookup_table()` for an
        overview of which sympy symbols were mapped to which args index.
        - `start` (`list[float] | np.ndarray`): list of minimum values for
        each axis of the to-be-constructed array in field space.
        - `stop` (`list[float] | np.ndarray`): list of maximum values for each
        axis of the to-be-constructed array in field space.
        - `N` (`int | list[int]`, optional): list of the number of samples along each axis in field
        space. If set to a single `int`, the same number of samples will be used along each axis.
        Defaults to 100.
        - `accuracy` (float, optional): tolerance to within basis must be orthonormal. Defaults to
        1e-3.

        ### Returns
        `None`. May throw an exception if the basis is malformed.
        """
        n_fields = self.artifact.n_fields
        start_stop = np.array([[float(start), float(stop)] for (start, stop) in zip(start, stop)])
        if N is int:
            N = N * np.ones(n_fields)
        self.dylib.validate_basis_on_domain(N, args, start_stop, accuracy)


class GeneralisedAL(InflationCondition):
    """This class extends the generic `InflationCondition` with the generalised rapid-turn (ω>>ε^½)
    consistency condition from (`arXiv:2405.11628`).

    In addition, related quantities that may be estimated from the potential and field-space metric,
    such as the slow-roll parameters εH, η_||, the angle δ and turn-rate ω can also be computed.
    The rapid-turn limit of the consistency condition derived in the 2022 Anguelova and Lazaroiu 2022
    (`arXiv:2210.00031v2`) is also available.

    ### Usage
    To construct an instance of this class, a `CompilationArtifact` is required.
    Such an artifact can be obtained by running an instance of `inflatox.Compiler`
    with a specific model (fieldspace metric + scalar potential). For more info on
    how to use the `Compiler`, see its documentation.

    The artifact contains all the necessary information to evaluate the consistency
    condition and can be used to construct an instance of this class.

    To run evaluate the consistency condition for various model parameters and
    regions of field-space, use the `.complete_analysis()` method on an instance
    of this class with the appropriate methods. For more info, see the
    `.complete_analysis()` method.
    """

    def __init__(self, compiled_artifact: CompilationArtifact):
        super().__init__(compiled_artifact)

    def complete_analysis(
        self,
        args: np.ndarray,
        x0_start: float,
        x0_stop: float,
        x1_start: float,
        x1_stop: float,
        N_x0: int = 1_000,
        N_x1: int = 1_000,
        progress: bool = True,
        threads: None | int = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This function performs a complete analysis of possible slow-roll (rapid)
        turn trajectories using the methods described in (paper), based on the AL
        consistency condition. It returns six arrays filled with:
          1. Let rhs and lhs denote the left-hand side and right-hand side of the AL
            consistency condition. This first array is populated with the normalised
            difference between the rhs and lhs:
              out = ||lhs| - |rhs||/(|lhs| + |rhs|)
            Note that out = 0 corresponds to the consistency condition holding
            perfectly. Values larger than zero indicate it does not hold perfectly.
          2. ε_V (first potential slow-roll parameter)
          3. ε_H (first dynamical slow-roll parameter), calculated assuming that the
            AL condition holds.
          4. η_|| (second dynamical slow-roll parameter), calculated assuming that the
            AL condition holds.
          5. δ (characteristic angle), calculated assuming that the AL condition holds.
          6. ω (relative turn rate), calculated assuming that the AL condition holds.
        Using (1) and (3), slow-roll trajectories with |ε_H|, |η_||| << 1 can be
        identified. See (paper) for a more complete discussion.

        ### *NOTE*
        The consistency condition or ε_H *on their own* are usually insufficient to discriminate
        determine the location of the rapid-turn attractor. Always verify that:
        1. The consistency condition holds AND
        2. ε_H << 1 AND
        3. ε_H > 0

        ### Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x0_start` (`float`): minimum value of first field `x[0]`.
        - `x0_stop` (`float`): maximum value of first field `x[0]`.
        - `x1_start` (`float`): minimum value of second field `x[1]`.
        - `y_stop` (`float`): maximum value of second field `x[1]`.
        - `N_x` (`int`, optional): number of steps along `x[0]` axis. Defaults to 10_000.
        - `x1_stop` (`int`, optional): number of steps along `x[1]` axis. Defaults to 10_000.
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `threads` (`None | int`, optional): number of threads to use for calculation.
          When set to `None`, inflatox will choose the optimum number (usually equal
          to the number of CPU's). When set to 1, a single-threaded implementation
          will be used.

        ### Returns:
        `np.ndarray` (✕6): arrays filled with the following quantities (in order):
          1. Consistency condition ||lhs| - |rhs||/(|lhs| + |rhs|)
          2. ε_V (first potential slow-roll parameter)
          3. ε_H (first dynamical slow-roll parameter)
          4. η_|| (speed-up parameter)
          5. δ (characteristic angle)
          6. ω (relative turn rate)
        """

        # set up args for anguelova's condition
        out = np.zeros((N_x0, N_x1, 6), dtype=float)

        start_stop = np.array(
            [[float(x0_start), float(x0_stop)], [float(x1_start), float(x1_stop)]]
        )

        # calculate
        threads = threads if threads is not None else 0

        # evaluate and return
        complete_analysis(self.dylib, args, out, start_stop, progress, threads)
        return (
            out[:, :, 0],
            out[:, :, 1],
            out[:, :, 2],
            out[:, :, 3],
            out[:, :, 4],
            out[:, :, 5],
        )

    def consistency(
        self,
        args: np.ndarray,
        x0_start: float,
        x0_stop: float,
        x1_start: float,
        x1_stop: float,
        N_x0: int = 1_000,
        N_x1: int = 1_000,
        progress: bool = True,
        threads: None | int = None,
    ) -> np.ndarray:
        """returns array filled with the normalised difference between one and the
        quotient of the left-hand-side (lhs) and right-hand-side (rhs) of the slow-
        roll turn consistency condition.

        ### Exact formulation of calculated quantities
        This function returns:
          ||lhs| - |rhs||/(|lhs| + |rhs|)
        Where
          lhs = Vww/V
          rhs = 3 + 3 (Vvw/Vvv)² + (Vvv/V) (Vvw/Vvv)²

        ### Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x0_start` (`float`): minimum value of first field `x[0]`.
        - `x0_stop` (`float`): maximum value of first field `x[0]`.
        - `x1_start` (`float`): minimum value of second field `x[1]`.
        - `y_stop` (`float`): maximum value of second field `x[1]`.
        - `N_x` (`int`, optional): number of steps along `x[0]` axis. Defaults to 10_000.
        - `x1_stop` (`int`, optional): number of steps along `x[1]` axis. Defaults to 10_000.
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `threads` (`None | int`, optional): number of threads to use for calculation.
          When set to `None`, inflatox will choose the optimum number (usually equal
          to the number of CPU's). When set to 1, a single-threaded implementation
          will be used.

        ### Returns:
        `np.ndarray`: array filled with slow-roll (intermediate) turn consistency
          condition from (paper)
        """

        # set up args for anguelova's condition
        out = np.zeros((N_x0, N_x1), dtype=float)

        start_stop = np.array(
            [[float(x0_start), float(x0_stop)], [float(x1_start), float(x1_stop)]]
        )

        # calculate
        threads = threads if threads is not None else 0

        # evaluate and return
        consistency_only(self.dylib, args, out, start_stop, progress, threads)
        return out

    def epsilon_v(
        self,
        args: np.ndarray,
        x0_start: float,
        x0_stop: float,
        x1_start: float,
        x1_stop: float,
        N_x0: int = 1_000,
        N_x1: int = 1_000,
        progress: bool = True,
        threads: None | int = None,
    ) -> np.ndarray:
        """returns array filled with the potential first-order slow-roll parameter
        ε_V

        ### Exact formulation of calculated quantities
        This function returns:
          ε_V = 1/2 (∇V/V)²

        ### Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x0_start` (`float`): minimum value of first field `x[0]`.
        - `x0_stop` (`float`): maximum value of first field `x[0]`.
        - `x1_start` (`float`): minimum value of second field `x[1]`.
        - `y_stop` (`float`): maximum value of second field `x[1]`.
        - `N_x` (`int`, optional): number of steps along `x[0]` axis. Defaults to 10_000.
        - `x1_stop` (`int`, optional): number of steps along `x[1]` axis. Defaults to 10_000.
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `threads` (`None | int`, optional): number of threads to use for calculation.
          When set to `None`, inflatox will choose the optimum number (usually equal
          to the number of CPU's). When set to 1, a single-threaded implementation
          will be used.

        ### Returns:
        `np.ndarray`: array filled with slow-roll (intermediate) turn consistency
          condition from (paper)
        """

        # set up args for anguelova's condition
        out = np.zeros((N_x0, N_x1), dtype=float)

        start_stop = np.array(
            [[float(x0_start), float(x0_stop)], [float(x1_start), float(x1_stop)]]
        )

        # calculate
        threads = threads if threads is not None else 0

        # evaluate and return
        epsilon_v_only(self.dylib, args, out, start_stop, progress, threads)
        return out

    def consistency_rapidturn(
        self,
        args: np.ndarray,
        x0_start: float,
        x0_stop: float,
        x1_start: float,
        x1_stop: float,
        N_x0: int = 1_000,
        N_x1: int = 1_000,
        progress: bool = True,
        threads: None | int = None,
    ) -> np.ndarray:
        """returns array filled with the normalised difference between one and the
        quotient of the left-hand-side (lhs) and right-hand-side (rhs) of
        Anguelova & Lazaroiu's original consistency condition (`arXiv:2210.00031v2`).

        ### Exact formulation of calculated quantities
        This function returns:
          ||lhs| - |rhs||/(|lhs| + |rhs|)
        Where
          lhs = Vww/V
          rhs = 3 (Vvw/Vvv)²

        ### Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x0_start` (`float`): minimum value of first field `x[0]`.
        - `x0_stop` (`float`): maximum value of first field `x[0]`.
        - `x1_start` (`float`): minimum value of second field `x[1]`.
        - `y_stop` (`float`): maximum value of second field `x[1]`.
        - `N_x` (`int`, optional): number of steps along `x[0]` axis. Defaults to 10_000.
        - `x1_stop` (`int`, optional): number of steps along `x[1]` axis. Defaults to 10_000.
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `threads` (`None | int`, optional): number of threads to use for calculation.
          When set to `None`, inflatox will choose the optimum number (usually equal
          to the number of CPU's). When set to 1, a single-threaded implementation
          will be used.

        ### Returns:
        `np.ndarray`: array filled with Anguelova & Lazaroiu's original consistency
          condition.
        """

        # set up args for anguelova's condition
        out = np.zeros((N_x0, N_x1), dtype=float)

        start_stop = np.array(
            [[float(x0_start), float(x0_stop)], [float(x1_start), float(x1_stop)]]
        )

        # calculate
        threads = threads if threads is not None else 0

        # evaluate and return
        consistency_rapidturn_only(self.dylib, args, out, start_stop, progress, threads)
        return out

    def flag_quantum_dif(
        self,
        args: np.ndarray,
        x0_start: float,
        x0_stop: float,
        x1_start: float,
        x1_stop: float,
        N_x0: int = 10_000,
        N_x1: int = 10_000,
        progress=True,
        accuracy=1e-3,
    ) -> np.ndarray:
        """returns boolean array where `True` values indicate that both components
        of the gradient of the scalar potential are smaller than the specified
        `accuracy` parameter. This is useful to identify points in the potential where
        quantum diffusion may have a large impact (saddle points in the potential).
        This calculation explicitly *does not* take into account the full inner product
        using the metric to avoid measuring where the metric goes to zero or becomes
        signular.

        Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x0_start` (`float`): minimum value of first field `x[0]`.
        - `x0_stop` (`float`): maximum value of first field `x[0]`.
        - `x1_start` (`float`): minimum value of second field `x[1]`.
        - `y_stop` (`float`): maximum value of second field `x[1]`.
        - `N_x` (`int`, optional): number of steps along `x[0]` axis. Defaults to 10_000.
        - `x1_stop` (`int`, optional): number of steps along `x[1]` axis. Defaults to 10_000.
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `accuracy` (`float`, optional):

        Returns:
        `np.ndarray`: boolean array. `True` where the absolute value of both components
          of the gradient are smaller than `accuracy`, `False` otherwise.
        """

        # set up args for anguelova's condition
        x = np.zeros((N_x0, N_x1), dtype=bool)

        start_stop = np.array(
            [[float(x0_start), float(x0_stop)], [float(x1_start), float(x1_stop)]]
        )

        # evaluate and return
        flag_quantum_dif_py(self.dylib, args, x, start_stop, progress, accuracy)
        return x

    #########################
    # On_trajectory methods #
    #########################

    def complete_analysis_ot(
        self,
        args: np.ndarray,
        x: np.ndarray,
        progress: bool = True,
        threads: None | int = None,
    ) -> np.ndarray:
        """This function performs a complete analysis of possible slow-roll (rapid)
        turn trajectories using the methods described in (paper), based on the AL
        consistency condition. It returns six arrays filled with:
          1. Let rhs and lhs denote the left-hand side and right-hand side of the AL
            consistency condition. This first array is populated with the normalised
            difference between the rhs and lhs:
              out = ||lhs| - |rhs||/(|lhs| + |rhs|)
            Note that out = 0 corresponds to the consistency condition holding
            perfectly. Values larger than zero indicate it does not hold perfectly.
          2. ε_V (first potential slow-roll parameter)
          3. ε_H (first dynamical slow-roll parameter), calculated assuming that the
            AL condition holds.
          4. η_|| (second dynamical slow-roll parameter), calculated assuming that the
            AL condition holds.
          5. δ (characteristic angle), calculated assuming that the AL condition holds.
          6. ω (relative turn rate), calculated assuming that the AL condition holds.
        Using (1) and (3), slow-roll trajectories with |ε_H|, |η_||| << 1 can be
        identified. See (paper) for a more complete discussion.

        ### *NOTE*
        The consistency condition or ε_H *on their own* are usually insufficient to discriminate
        determine the location of the rapid-turn attractor. Always verify that:
        1. The consistency condition holds AND
        2. ε_H << 1 AND
        3. ε_H > 0

        ### Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x` (`np.ndarray`): array of field-space points specifying the trajectory
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `threads` (`None | int`, optional): number of threads to use for calculation.
          When set to `None`, inflatox will choose the optimum number (usually 1).
          When set to 1, a single-threaded implementation will be used.

        ### Returns:
        `np.ndarray` (✕6): arrays filled with the following quantities (in order):
          1. Consistency condition ||lhs| - |rhs||/(|lhs| + |rhs|)
          2. ε_V (first potential slow-roll parameter)
          3. ε_H (first dynamical slow-roll parameter)
          4. η_|| (speed-up parameter)
          5. δ (characteristic angle)
          6. ω (relative turn rate)
        """

        # set up args for anguelova's condition
        out = np.zeros((x.shape[0], 6), dtype=float)

        # Single-threaded default is more appropriate for smaller number of iterations
        threads = threads if threads is not None else 1

        # evaluate and return
        complete_analysis_on_trajectory(self.dylib, args, x, out, progress, threads)
        return np.split(out, 6, 1)

    def consistency_ot(
        self,
        args: np.ndarray,
        x: np.ndarray,
        progress: bool = True,
        threads: None | int = None,
    ) -> np.ndarray:
        """returns array filled with the normalised difference between one and the
        quotient of the left-hand-side (lhs) and right-hand-side (rhs) of the slow-
        roll turn consistency condition.

        ### Exact formulation of calculated quantities
        This function returns:
          ||lhs| - |rhs||/(|lhs| + |rhs|)
        Where
          lhs = Vww/V
          rhs = 3 + 3 (Vvw/Vvv)² + (Vvv/V) (Vvw/Vvv)²

        ### Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x` (`np.ndarray`): array of field-space points specifying the trajectory
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `threads` (`None | int`, optional): number of threads to use for calculation.
          When set to `None`, inflatox will choose the optimum number (usually equal
          to the number of CPU's). When set to 1, a single-threaded implementation
          will be used.

        ### Returns:
        `np.ndarray`: array filled with slow-roll (intermediate) turn consistency
          condition from (paper)
        """
        # set up args for anguelova's condition
        out = np.zeros((x.shape[0]), dtype=float)

        # Single-threaded default is more appropriate for smaller number of iterations
        threads = threads if threads is not None else 1

        # evaluate and return
        consistency_only_on_trajectory(self.dylib, args, x, out, progress, threads)
        return out

    def consistency_rapidturn_ot(
        self,
        args: np.ndarray,
        x: np.ndarray,
        progress: bool = True,
        threads: None | int = None,
    ) -> np.ndarray:
        """returns array filled with the normalised difference between one and the
        quotient of the left-hand-side (lhs) and right-hand-side (rhs) of the slow-
        roll turn consistency condition.

        ### Exact formulation of calculated quantities
        This function returns:
          ||lhs| - |rhs||/(|lhs| + |rhs|)
        Where
          lhs = Vww/V
          rhs = 3 + 3 (Vvw/Vvv)² + (Vvv/V) (Vvw/Vvv)²

        ### Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x` (`np.ndarray`): array of field-space points specifying the trajectory
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `threads` (`None | int`, optional): number of threads to use for calculation.
          When set to `None`, inflatox will choose the optimum number (usually equal
          to the number of CPU's). When set to 1, a single-threaded implementation
          will be used.

        ### Returns:
        `np.ndarray`: array filled with slow-roll (intermediate) turn consistency
          condition from (paper)
        """
        # set up args for anguelova's condition
        out = np.zeros((x.shape[0]), dtype=float)

        # Single-threaded default is more appropriate for smaller number of iterations
        threads = threads if threads is not None else 1

        # evaluate and return
        consistency_rapidturn_only_on_trajectory(self.dylib, args, x, out, progress, threads)
        return out

    def epsilon_v_ot(
        self,
        args: np.ndarray,
        x: np.ndarray,
        progress: bool = True,
        threads: None | int = None,
    ) -> np.ndarray:
        """returns array filled with the normalised difference between one and the
        quotient of the left-hand-side (lhs) and right-hand-side (rhs) of
        Anguelova & Lazaroiu's original consistency condition (`arXiv:2210.00031v2`).

        ### Exact formulation of calculated quantities
        This function returns:
          ||lhs| - |rhs||/(|lhs| + |rhs|)
        Where
          lhs = Vww/V
          rhs = 3 (Vvw/Vvv)²

        ### Args:
        - `args` (`np.ndarray`): values of the model-dependent parameters.
        - `x` (`np.ndarray`): array of field-space points specifying the trajectory
        - `progress` (`bool`, optional): whether to render a progressbar or not. Showing the
          progressbar may slightly degrade performance. Defaults to True.
        - `threads` (`None | int`, optional): number of threads to use for calculation.
          When set to `None`, inflatox will choose the optimum number (usually equal
          to the number of CPU's). When set to 1, a single-threaded implementation
          will be used.

        ### Returns:
        `np.ndarray`: array filled with Anguelova & Lazaroiu's original consistency
          condition.
        """
        # set up args for anguelova's condition
        out = np.zeros((x.shape[0]), dtype=float)

        # Single-threaded default is more appropriate for smaller number of iterations
        threads = threads if threads is not None else 1

        # evaluate and return
        epsilon_v_only_on_trajectory(self.dylib, args, x, out, progress, threads)
        return out
