import os

import jax
import numpy as np
from jax import grad, jit
from jax import numpy as jnp

from jaxdf import *

ATOL = 1e-6

## Fixtures and setup funcs
def load_test_data(filename):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  filename = dir_path + '/test_data/' + filename
  return np.loadtxt(filename, dtype=np.float32, delimiter=',')

def load_to_fourier(filename):
  data = load_test_data(filename)
  N = data.shape
  dx = (1.0, 1.0)
  domain = geometry.Domain(N, dx)
  field = FourierSeries(jnp.expand_dims(data, -1), domain)
  return field

def load_to_fourier_vector(filename):
  data = load_test_data(filename)
  # Split the data
  data = np.stack(
    [data[:,:data.shape[0]], data[:,data.shape[0]:]],
    axis=-1
  )
  N = data.shape[:-1]
  dx = (1.0, 1.0)
  domain = geometry.Domain(N, dx)
  field = FourierSeries(data, domain)
  return field

def continuous_quadratic():
  N = (1,1)
  dx = (0.1, 0.1)
  domain = geometry.Domain(N, dx)

  params = jnp.asarray([2.0, 3.0])
  def f(p, x):
    return p[0]*(x[0]**2) + p[1]*(x[1]**2)

  return Continuous(params, domain, f)

## Continuous tests
def test_continuous_gradient():
  u = continuous_quadratic()
  x0 = jnp.asarray([1., 0.])
  x1 = jnp.asarray([0., 1.])

  # Computing the gradient
  del_u = operators.gradient(u)
  assert del_u.get_field(x0)[0] == u.params[0]*2.0
  assert del_u.get_field(x1)[1] == u.params[1]*2.0

  # Testing jitting
  @jit
  def jit_gradient(u, x0):
    del_u = operators.gradient(u)
    return del_u.get_field(x0)

  assert jit_gradient(u, x0)[0] == u.params[0]*2.0
  assert jit_gradient(u, x1)[1] == u.params[1]*2.0

  # Test jitting with field output
  @jit
  def jit_gradient_field(u):
    del_u = operators.gradient(u)
    return del_u
  del_u_jit = jit_gradient_field(u)

  assert del_u_jit.get_field(x0)[0] == u.params[0]*2.0
  assert del_u_jit.get_field(x1)[1] == u.params[1]*2.0


## Fourier tests
def test_fourier_gradient():
  # Loading data and ground truth
  u = load_to_fourier('CROSS_IMG.txt')
  dudx_r = load_to_fourier('CROSS_IMG_FOURIER_DX.txt').on_grid[...,0]
  dudy_r = load_to_fourier('CROSS_IMG_FOURIER_DY.txt').on_grid[...,0]

  # Computing the gradient
  del_u = operators.gradient(u).on_grid
  dudx = del_u[...,0]
  dudy = del_u[...,1]

  # Checking they are similar
  assert jnp.allclose(dudx, dudx_r, atol=ATOL)
  assert jnp.allclose(dudy, dudy_r, atol=ATOL)

  # Testing jitting
  @jit
  def jit_gradient(u):
    del_u = operators.gradient(u).on_grid
    dudx = del_u[...,0]
    dudy = del_u[...,1]
    return dudx, dudy
  dudx_jit, dudy_jit = jit_gradient(u)

  assert jnp.allclose(dudx_jit, dudx_r, atol=ATOL)
  assert jnp.allclose(dudy_jit, dudy_r, atol=ATOL)

  # Checking paramters initializer
  params = operators.gradient.init_params(u)

  # Checking jitting
  @jit
  def jit_gradient_init(u):
    params = operators.gradient.init_params(u)

    # Update parameters
    for k, v in params.items():
      params[k] = [a + 1 for a in v]
    return params

  params_jit = jit_gradient_init(u)

  # Test jitting with custom parameters
  @jit
  def jit_gradient_custom(u, params):
    del_u = operators.gradient(u, params=params).on_grid
    dudx = del_u[...,0]
    dudy = del_u[...,1]
    return dudx, dudy

  dudx_jit, dudy_jit = jit_gradient_custom(u, params_jit)


def test_fourier_laplacian():
  # Loading data and ground truth
  u = load_to_fourier('CROSS_IMG.txt')
  nabla_u_r = load_to_fourier('CROSS_IMG_FOURIER_LAPLACIAN.txt').on_grid[...,0]

  # Computing the laplacian
  nabla_u = operators.laplacian(u).on_grid[...,0]

  # Checking they are similar
  assert jnp.allclose(nabla_u, nabla_u_r, atol=ATOL)

  # Testing jitting
  @jit
  def jit_laplacian(u):
    nabla_u = operators.laplacian(u).on_grid[...,0]
    return nabla_u
  nabla_u_jit = jit_laplacian(u)

  assert jnp.allclose(nabla_u_jit, nabla_u_r, atol=ATOL)

  # Checking paramters initializer
  params = operators.laplacian.init_params(u)

  # Checking jitting
  @jit
  def jit_laplacian_init(u):
    params = operators.laplacian.init_params(u)

    # Update parameters
    for k, v in params.items():
      params[k] = [a + 1 for a in v]
    return params

  params_jit = jit_laplacian_init(u)

  # Test jitting with custom parameters
  @jit
  def jit_laplacian_custom(u, params):
    nabla_u = operators.laplacian(u, params=params).on_grid[...,0]
    return nabla_u

  nabla_u_jit = jit_laplacian_custom(u, params_jit)

  # Testing gradient
  @jit
  @grad
  def jit_laplacian_grad(u):
    Lu = operators.laplacian(u)
    return jnp.sum(jnp.where(Lu.on_grid > 0.5, Lu.on_grid, 0))

  laplacian_grad_jit = jit_laplacian_grad(u)


def test_fourier_diag_jacobian():
  # Loading data and ground truth
  u = load_to_fourier_vector('CROSS_IMG_VEC.txt')
  nabla_dag_u = load_to_fourier_vector('CROSS_IMG_VEC_FOURIER_DAG.txt').on_grid

  # Computing the operator
  nabla_u = operators.diag_jacobian(u).on_grid

  # Checking they are similar
  assert jnp.allclose(nabla_u, nabla_dag_u, atol=ATOL)

  # Testing jitting
  @jit
  def jit_diag_jacobian(u):
    nabla_u = operators.diag_jacobian(u).on_grid
    return nabla_u
  nabla_u_jit = jit_diag_jacobian(u)

  assert jnp.allclose(nabla_u_jit, nabla_dag_u, atol=ATOL)

  # Checking paramters initializerload_to_fourier
  # Checking jitting
  @jit
  def jit_diag_jacobian_init(u):
    params = operators.diag_jacobian.init_params(u)

    # Update parameters
    for k, v in params.items():
      params[k] = [a + 1 for a in v]
    return params

  params_jit = jit_diag_jacobian_init(u)

  # Test jitting with custom parameters
  @jit
  def jit_diag_jacobian_custom(u, params):
    nabla_u = operators.diag_jacobian(u, params=params).on_grid
    return nabla_u

  nabla_u_jit = jit_diag_jacobian_custom(u, params_jit)


if __name__ == '__main__':
  with jax.checking_leaks():
    test_fourier_laplacian()
