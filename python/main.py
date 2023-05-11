import sympy
from sympy import powdenest
from IPython.display import display
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols

def calc_hesse(coords: list, g_fm: MetricTensor, V):
  """returns the components of the covariant Hesse matrix in a twice-covariant
  form. Components for all pairs of the supplied coordinates are calculated for
  the scalar potential V using the supplied metric tensor.
  
  The components of the covariant Hesse matrix are defined as:
    V_ab(ϕ) = ∇_a ∇_b V(ϕ)
  This is expanded as (using Einstein notation):
    V_ab(ϕ) = ∇_a (∂_b V(ϕ)) = ∂_a ∂_b V(ϕ) - Γ_ab^c(ϕ) ∂_c V(ϕ)
  Where Γ_ab^c(ϕ) is the standard Christoffel connection defined as:
    Γ_ab^c = 1/2 g^cd (∂_a g_bd + ∂_b g_ad - ∂_d g_ab)

  Args:
    coords (list[sympy symbols]): coordinates (scalar fields) with respect to
      which the components of the Hesse matrix are defined
    g_fm (MetricTensor): metric tensor to be used to raise/lower indices and define
      the covariant derivative
    V (sympy expression): expression for the scalar potential of the inflaton
      fields.

  Returns:
    list[list[sympy expressions]]: nested list of components of the Hesse matrix
  """
  dim = len(coords)
  #The connection has indices up-down-down (opposite order that we usually use)
  conn = ChristoffelSymbols.from_metric(g_fm).tensor()
  #output components of the Hesse matrix
  Vab = [[0 for _ in range(dim)] for _ in range(dim)]
  
  for a in range(dim):
    for b in range(dim):
      #Calculate ∂_a ∂_b V(ϕ)
      da_dbV = sympy.diff(V, coords[b], coords[a]).simplify()
      #Calculate the contraction Γ_ab^c(ϕ) ∂_c V(ϕ)
      gamma_ab = 0
      for c in range(dim):
        gamma_ab = (gamma_ab + conn[c][b][a]*sympy.diff(V, coords[c])).simplify()
      #set the output components
      Vab[a][b] = (da_dbV - gamma_ab).simplify()
  return Vab

def inner_prod(vec1: list, vec2: list, metric: MetricTensor):
  """returns the inner product of vec1 and vec2 with respect to the supplied metric

  Args:
    vec1 (list): first vector, once contravariant
    vec2 (list): second vector, once contravariant
    metric (MetricTensor): metric tensor, twice covariant

  Returns:
    symbolic sympy expression: inner product of vec1 with vec2 with respect to
    the specified metric tensor
  """
  ans = 0
  for a in range(len(vec1)):
    for b in range(len(vec2)):
      ans = ans + vec1[a]*vec2[b]*metric.arr[a][b]
  return powdenest(ans.simplify(), force=True)

def normalize(vec: list, metric: MetricTensor):
  norm = sympy.sqrt(inner_prod(vec, vec, metric))
  return [powdenest((cmp / norm).simplify(), force=True) for cmp in vec]

def calc_v(coords: list, g_fm: MetricTensor, V):
  dim = len(coords)
  norm = (1 / sympy.sqrt(sympy.Matrix(g_fm.arr).det())).simplify()
  
  #Non-normalised components of grad V
  va = [0 for _ in range(dim)]
  for a in range(dim):
    va[a] = powdenest((sympy.diff(V, coords[a]) * norm).simplify(), force=True)
  return normalize(va, g_fm)

def calc_next_w(current_basis: list, guess: list, g: MetricTensor):
  """Use the Gramm-Schmidt procedure to find a new orthogonal basis vector given
  an incomplete set of orthogonal basis vectors and a third vector that is linearly
  independent from the other vectors.

  Args:
    current_basis (list(list)): list of current *orthogonal* basisvectors. The
      components of the vectors should be given in *contravariant* form.
    guess (list): vector that is linearly independent from the (incomplete) set of
      current basis vectors. The components of this vector should be given in
      *contravariant* form. This vector needn't be normalized nor orthogonal to
      the set of current basis vectors.
    g (MetricTensor): metric tensor (twice covariant) used to define inner products
      for the Gramm-Schmidt procedure

  Returns:
    (list): list of the contravariant components of an additional basis vector
      orthogonal to the supplied set of basis vectors, with respect to the supplied
      metric.
  """
  dim = len(current_basis[0])
  #make sure the supplied basis is not already complete
  assert(len(current_basis) < dim)
  ans = guess
  for vec in current_basis:
    overlap = inner_prod(vec, guess, g)
    for a in range(dim):
      ans[a] = (ans[a] - overlap*vec[a]).simplify()
  return normalize(ans, g)

def project_hesse(hesse_matrix: list, vec1: list, vec2: list):
  dim = len(vec1)
  assert(len(vec1) == len(vec2))
  V_proj = 0
  for a in range(dim):
    for b in range(dim):
      V_proj = V_proj + hesse_matrix[a][b]*vec1[a]*vec2[b]
  return powdenest(V_proj.simplify(), force=True)