import numpy as np
import pytest
from jax import jit

from jaxdf import *

ATOL=1e-6

## Fixtures
@pytest.fixture
def init_geometry():
  N = (64,64)
  dx = (1.0, 1.0)
  domain = geometry.Domain(N, dx)
  return domain, N

@pytest.fixture
def fourier_scalar_field(init_geometry):
  domain, N = init_geometry
  N_new = tuple(list(N) + [1,])
  params = jnp.ones(N_new)
  return FourierSeries(params, domain)

@pytest.fixture
def continuous_scalar_field(init_geometry):
  domain, N = init_geometry
  def f(p, x):
    return jnp.expand_dims(jnp.sum(p*(x**2)), -1)
  params = 5.0
  return Continuous(params, domain, f)

## Tests

# Fourier with Fourier
@pytest.mark.parametrize(
  "op", [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y,
    lambda x, y: x ** y,
    ]
  )
def test_if_arithmetic_runs_fourier(fourier_scalar_field, op):
  u = fourier_scalar_field
  v = fourier_scalar_field
  # Testing without jitting
  z = op(u, v)

  # Testing with jitting
  @jit
  def jit_op(x, y):
    return op(x, y)
  z_jit = jit_op(u, v)

  # Check if they give the same answer
  assert np.allclose(z.on_grid, z_jit.on_grid)

# Fourier with number
@pytest.mark.parametrize(
  "op", [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y,
    lambda x, y: x ** y,
    ]
  )
def test_if_arithmetic_runs_fourier_num(fourier_scalar_field, op):
  u = fourier_scalar_field
  v = 1.0
  # Testing without jitting
  z = op(u, v)
  q = op(v, u)

  # Testing with jitting
  @jit
  def jit_op(x, y):
    return op(x, y)
  z_jit = jit_op(u, v)
  q_jit = jit_op(v, u)

  # Check if they give the same answer
  assert np.allclose(z.on_grid, z_jit.on_grid)
  assert np.allclose(q.on_grid, q_jit.on_grid)

# Continuous with Continuous
@pytest.mark.parametrize(
  "op", [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y,
    lambda x, y: x ** y,
    ]
  )
def test_if_arithmetic_runs_continuous(continuous_scalar_field, op):
  u = continuous_scalar_field
  v = continuous_scalar_field

  # Testing without jitting
  z = op(u, v)

  # Testing with jitting
  @jit
  def jit_op(x, y):
    return op(x, y)
  z_jit = jit_op(u, v)

  # Check if they give the same answer
  assert np.allclose(z.on_grid, z_jit.on_grid)

# Continuous with number
@pytest.mark.parametrize(
  "op", [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y,
    lambda x, y: x ** y,
    ]
  )
def test_if_arithmetic_runs_continuous_num(init_geometry, op):
  domain, N = init_geometry
  def f(p, x):
    return jnp.expand_dims(jnp.sum(p*(x**2)), -1)
  params = 5.0
  u = Continuous(params, domain, f)

  v = 1.0
  # Testing without jitting
  z = op(u, v)
  q = op(v, u)

  # Testing with jitting
  @jit
  def jit_op(x, y):
    return op(x, y)
  z_jit = jit_op(u, v)
  q_jit = jit_op(v, u)

  # Check if they give the same answer
  assert np.allclose(z.on_grid, z_jit.on_grid)
  assert np.allclose(q.on_grid, q_jit.on_grid)

if __name__ == "__main__":
  test_if_arithmetic_runs_continuous_num()
