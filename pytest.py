import numpy as np

def calc_h10(coords: list, params: list):
    φ = coords[0]
    ψ = coords[1]

    m = params[0]
    φ0 = params[1]
    ψ0 = params[2]

    return 2.0*ψ0*(φ - φ0)*(2.0*pow(m, 4.0)*pow(φ, 2.0)*(φ - φ0)*(φ*(pow(φ, 2.0) + 1.0) + 2.0*φ - 2.0*φ0 - (φ - φ0)*(pow(φ, 2.0) + 1.0)) - pow(ψ0, 2.0)*pow(pow(φ, 2.0) + 1.0, 2.0))/(pow(φ, 2.0)*(4.0*pow(m, 4.0)*pow(φ, 2.0)*pow(φ - φ0, 2.0) + pow(ψ0, 2.0)*pow(pow(φ, 2.0) + 1.0, 2.0))*(pow(m, 2.0)*pow(φ - φ0, 2.0) - 2.0*ψ*ψ0)*abs(φ - φ0))
  
params = [10.0, 2.0, 1.0]
step = 10.0
r = 1000.0

accumulator = 0.0

for i in np.arange(-r, r, step):
  for j in np.arange(-r, r, step):
    result = calc_h10([i, j], params)
    accumulator += result
    #print(f"({i},{j})={result}")

print(result)