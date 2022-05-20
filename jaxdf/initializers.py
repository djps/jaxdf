from functools import wraps

import numpy as np

from jaxdf.discretization import *


def filter_input(func, to_remove: list):
    # Remove unwanted parameters
  @wraps(func)
  def _f(**kwargs):
    for key in to_remove:
      kwargs.pop(key)
    return func(**kwargs)
  return _f


def get_kvec(x: FourierSeries, stagger = [0]) -> dict:
  r'''Returns the k-vectors of a Fourier series,
  representing the frequency axis.

  Functionally equivalent to `x._freq_axis`.

  Args:
    x (FourierSeries): Input field
    stagger (List[int]): Staggering of the k-vectors.

  Returns:
    dict: Dictionary of the type `{'k_vec': List[freq_axis]}`
  '''
  dx = np.asarray(x.domain.dx)
  k_vec = x._freq_axis
  if len(stagger) == 1:
    stagger = [dx[i]*stagger[0] for i in range(len(k_vec))]
  else:
    stagger = [dx[i]*stagger[i] for i in range(len(k_vec))]
  k_vec = [
    1j * k * jnp.exp(1j * k * stagger)
    for k, delta in zip(k_vec, stagger)
  ]

  return  {'k_vec': k_vec}

def _bubble_sort_gridpoints(grid_points):
    # Sorts by distance from 0
    # [-3, -2, -1, 0, 1, 2, 3] -> [0, 1, -1, 2, -2, 3, -3]
    # [0.5, 1.5, -0.5, 2.5, -1.5, -2.5] -> [0.5, -0.5, 1.5, -1.5, 2.5, -2.5]
    for i in range(len(grid_points)):
        for j in range(0, len(grid_points) - i - 1):
            magnitude_condition = abs(grid_points[j]) > abs(grid_points[j + 1])
            same_mag_condition = abs(grid_points[j]) == abs(grid_points[j + 1])
            sign_condition = np.sign(grid_points[j]) < np.sign(grid_points[j + 1])
            if  magnitude_condition or (same_mag_condition and sign_condition):
                temp = grid_points[j]
                grid_points[j] = grid_points[j+1]
                grid_points[j+1] = temp

    return grid_points

def _fd_coefficients_fornberg(order, grid_points, x0 = 0):
  # from Generation of Finite Difference Formulas on Arbitrarily Spaced Grids
  # Bengt Fornberg, 1998
  # https://web.njit.edu/~jiang/math712/fornberg.pdf
  M = order
  N = len(grid_points) - 1

  # Sort the grid points
  alpha = _bubble_sort_gridpoints(grid_points)
  delta = dict() # key: (m,n,v)
  delta[(0,0,0)] = 1.
  c1 = 1.

  for n in range(1, N+1):
    c2 = 1.
    for v in range(n):
      c3 = alpha[n] - alpha[v]
      c2 = c2 * c3
      if n < M:
        delta[(n,n-1,v)] = 0.
      for m in range(min([n, M])+1):
        d1 = delta[(m,n-1,v)] if (m,n-1,v) in delta.keys() else 0.
        d2 = delta[(m-1, n-1, v)] if (m-1,n-1,v) in delta.keys() else 0.
        delta[(m,n,v)] = ((alpha[n] - x0)*d1 - m*d2)/c3

    for m in range(min([n,M])+1):
      d1 = delta[(m-1, n-1, n-1)] if (m-1,n-1,n-1) in delta.keys() else 0.
      d2 = delta[(m,n-1,n-1)] if (m,n-1,n-1) in delta.keys() else 0.
      delta[(m,n,n)] = (c1/c2)*(m*d1 - (alpha[n-1] - x0)*d2)
    c1 = c2

  # Extract the delta with m = M and n = N
  coeffs = [None]*(N+1)
  for key in delta:
    if key[0] == M and key[1] == N:
      coeffs[key[2]] = delta[key]

  # sort coefficeient and alpha by alpha
  idx = np.argsort(alpha)
  alpha = np.take_along_axis(np.asarray(alpha),idx, axis=-1)
  coeffs = np.take_along_axis(np.asarray(coeffs),idx, axis=-1)

  return coeffs, alpha

def _get_fd_coefficients(x: FiniteDifferences, order=1, stagger = 0):
  # Check that all the values of stagger are in [0, 0.5, -0.5]
  assert stagger in [0, -0.5, 0.5], 'Staggering must be in [0, 0.5, -0.5] for finite differences'
  dx = np.asarray(x.domain.dx)
  accuracy = x.accuracy
  points = np.arange(-accuracy//2, accuracy//2+1)
  if stagger > 0:
    points = (points + stagger)[:-1]
  elif stagger < 0:
    points = (points + stagger)[1:]

  # get coefficients
  coeffs = _fd_coefficients_fornberg(order, points, x0 = 0)[0].tolist()

  # Append zero if a coefficient has been removed
  if stagger > 0:
    coeffs = coeffs + [0.]
  else:
    coeffs = [0.] + coeffs

  return np.asarray(coeffs)

def fd_derivative_init(
  x: FiniteDifferences,
  axis=0,
  stagger = 0
):
  kernel = _get_fd_coefficients(x, order = 1, stagger=stagger)

  if len(x.domain.ndim) > 1:
    for _ in range(len(x.domain.ndim) - 1):
      kernel = np.expand_dims(kernel, axis=0)
    # Move kernel to the correct axis
    kernel = np.moveaxis(kernel, 0, axis)

  # Add dx
  kernel = kernel / x.domain.dx[axis]

  return {'fd_kernel': kernel}

def ft_diag_jacobian_init(
  x: FiniteDifferences,
  stagger = [0]
):
  if len(stagger) != x.domain.ndim:
    stagger = [stagger[0] for _ in range(x.domain.ndim)]

  kernels = []
  for i in range(x.domain.ndim):
    kernels.append(fd_derivative_init(x, axis=i, stagger=stagger[i])['fd_kernel'])

  return {'fd_diag_jacobian': kernels}
