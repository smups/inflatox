def test_doc_example():
    #import inflatox
    import inflatox
    import sympy as sp
    import numpy as np
    sp.init_printing()

    #define model
    φ, θ, L, m, φ0 = sp.symbols('φ θ L m φ0')
    fields = [φ, θ]

    V = (1/2*m**2*(φ-φ0)**2).nsimplify()
    g = [
      [1, 0],
      [0, L**2 * sp.sinh(φ/L)**2]
    ]

    #symbolic calculation
    calc = inflatox.SymbolicCalculation.new(fields, g, V)
    hesse = calc.execute()

    #run the compiler
    out = inflatox.Compiler(hesse).compile()

    #evaluate the compiled potential and Hesse matrix
    from inflatox.consistency_conditions import GeneralisedAL
    anguelova = GeneralisedAL(out)

    params = np.array([1.0, 1.0, 1.0])
    x = np.array([2.0, -2.0])
    assert anguelova.calc_V(x, params) == 0.5
    assert np.allclose(anguelova.calc_H(x, params), np.array([[ 1., 0.], [0., 13.6449586]]))

    extent = [-1., 1., -1., 1.]
    consistency_condition, epsilon_V, epsilon_H, eta_H, delta, omega = anguelova.complete_analysis(params, *extent)