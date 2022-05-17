import colored_traceback
from jax import jit
from jax import numpy as jnp

from jaxdf import *

colored_traceback.add_hook()

if __name__ == "__main__":

  N = (64,64)
  dx = (1, 1)
  domain = geometry.Domain(N, dx)

  def f(p, x):
    p = jnp.sum(p)
    return jnp.expand_dims(jnp.sum(p*(x**2)), -1)
  params = jnp.asarray([[5], [6], [7]])
  u = Continuous(params, domain, f)

  @operator
  def add(x:float, y:float, params=None):
    return x + y

  @operator
  def add(x:int, y:int, params=None):
    return x - y

  @operator
  def add(x:object, y:object, params=None):
    return x + y

  @operator
  def add(x:Continuous, y:object, params=None):
    get_x = x.aux['get_field']
    def get_fun(p, coords):
      return get_x(p[0], coords) + p[1]
    return Continuous([x.params, y], x.domain, get_fun)

  @jit
  def f(x,y):
    return add(x, y, params=1.0)

  @jit
  def g(x,y):
    z = add(x, y, params=1.0)
    return z.get_field(jnp.asarray([0,0]))

  #print(f(1.0, 2.0))
  #print(f(1,   2))
  y = f(u,   2.0)
  #print(u.get_field(jnp.asarray([0,0])))
  #print(y.get_field(jnp.asarray([0,0])))
  #print(y.params)
  #print(jit(y.get_field)(jnp.asarray([0,0])))
  #print(jax.make_jaxpr(f)(u,   2.0))
  #print(jax.make_jaxpr(g)(u,   2.0))
  #print(jit(y.get_field).to_string(option))
  #print(jit(y.get_field)(jnp.asarray([0,0])))

  @jit
  def h(y):
    return g(u,y)

  h_low = h.lower(2.0)
  #print(h_low._xla_computation().as_hlo_text())
  h_comp = h_low.compile()
  print(h_comp._xla_executable().hlo_modules()[0].to_string())
