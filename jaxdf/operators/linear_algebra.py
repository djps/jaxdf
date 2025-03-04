from jaxdf.core import operator
from jaxdf.discretization import *
from jaxdf.discretization import OnGrid

from .functions import compose


@operator
def dot_product(x: OnGrid, y: OnGrid):
  r'''Computes the dot product of two fields.
  '''
  x_conj = compose(x)(jnp.conj)
  return jnp.sum((x_conj * y).on_grid)
