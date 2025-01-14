import jax.numpy as jnp
import pytest
from jax import jit

from jaxdf.discretization import FourierSeries
from jaxdf.geometry import Domain


@pytest.mark.parametrize("N", [(33,), (33,33), (33, 33, 33)])
@pytest.mark.parametrize("jitting", [True, False])
@pytest.mark.parametrize("out_dims", [1, 3])
def test_call(N, out_dims, jitting):
  domain = Domain(N, dx=[1.]*len(N))
  true_size = list(N) + [out_dims]
  params = jnp.zeros(true_size)

  delta_position = [x//2 for x in N]
  if len(N) == 1:
    params = params.at[delta_position[0], :].set(1.)
  elif len(N) == 2:
    params = params.at[
      delta_position[0], delta_position[1], :].set(1.)
  elif len(N) == 3:
    params = params.at[
      delta_position[0], delta_position[1], delta_position[2], :].set(1.)

  value = jnp.asarray([1.]*out_dims)
  x = jnp.asarray([0.]*len(N))

  def get(params, x):
    field = FourierSeries(params, domain)
    return field(x)

  get = jit(get) if jitting else get

  field_value = get(params, x)
  if jitting:
    field_value =  get(params, x)

  assert jnp.allclose(field_value, value)

if __name__ == '__main__':
  pass
