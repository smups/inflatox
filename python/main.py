import sympy
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor

def calc_vtt(coords, g_fm, V):
  #We have to calculate covariant derivatives using the covariant derivative of
  #V using the christoffel symbols of g_fm
  conn = ChristoffelSymbols.from_metric(g_fm).tensor()
  dim = conn[0][0][0]
  Vab = [[0 for _ in range(dim)] for _ in range(dim)]
  for a in range(dim):
    for b in range(dim):
      da_dbV = sympy.diff(V, coords[b], coords[a])
      gamma_ab = 0
      for c in range(dim):
        gamma_ab += conn[a][b][c]*sympy.diff(V, coords[c])
      Vab[a][b] = da_dbV - gamma_ab
  return Vab