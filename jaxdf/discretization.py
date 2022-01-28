from typing import Callable
from jax.random import PRNGKey
from jax import eval_shape, vmap
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax import tree_util
from jax import numpy as jnp
from jaxdf.core import operator, Field, new_discretization

@new_discretization
class Linear(Field):
  r'''This discretization assumes that the field is a linear function of the
  parameters contained in `Linear.params`.'''
  def __init__(
    self,
    params,
    domain,
    dims=1,
    aux = None,
  ):
    super().__init__(params, domain, dims, aux)

@register_pytree_node_class
class Continuous(Field):
  r'''A continous discretization, which is defined via a `get_field` function stored
  in the `aux` parameters. This is the most general form of a discretization, and its
  operation are implemented using function composition and autograd.
  
  Attributes:
    params (PyTree): The parameters of the discretization. This must be a pytree
      and is transformed by JAX transformations such as `jit` and `grad`.
    domain (Domain): The domain of the discretization.
    dims (int): The dimensionality of the field.
    aux (dict): A dictionary of auxiliary data, containing the `get_field`
      key. This is a function that takes a parameter vector and a point in
      the domain and returns the field at that point. The signature of this
      function is `get_field(params, x)`.
      
  An object of this class can be called as a function, returning the field at a
  desired point.
  '''
  def __init__(
    self,
    params,
    domain,
    get_fun: Callable
  ):
    r'''Initializes a continuous discretization.
    
    Args:
      params (PyTree): The parameters of the discretization.
      domain (Domain): The domain of the discretization.
      get_fun (Callable): A function that takes a parameter vector and a point in
      the domain and returns the field at that point. The signature of this
      function is `get_field(params, x)`.
      
    Returns:
      Continuous: A continuous discretization.
    '''  
    aux = {"get_field": get_fun}
    x = domain.origin
    dims = eval_shape(get_fun, params, x).shape
    super().__init__(params, domain, dims, aux)
    
  def tree_flatten(self):
    children = (self.params,)
    aux_data = (self.dims, self.domain, self.aux["get_field"])
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    params = children[0]
    dims, domain, get_fun = aux_data
    a = cls(params, domain=domain, get_fun=get_fun)
    return a
  
  def replace_params(self, new_params):
    r'''Replaces the parameters of the discretization with new ones. The domain
    and `get_field` function are not changed.
    
    Args:
      new_params (PyTree): The new parameters of the discretization.
      
    Returns:
      Continuous: A continuous discretization with the new parameters.
    '''
    return self.__class__(new_params, self.domain, self.aux["get_field"])
  
  def update_fun_and_params(
    self,
    params,
    get_field
  ):
    r'''Updates the parameters and the function of the discretization.
    
    Args:
      params (PyTree): The new parameters of the discretization.
      get_field (Callable): A function that takes a parameter vector and a point in
        the domain and returns the field at that point. The signature of this
        function is `get_field(params, x)`.
    
    Returns:
      Continuous: A continuous discretization with the new parameters and function.
    '''
    return self.__class__(params, self.domain, get_field)
  
  @classmethod
  def from_function(
    cls, 
    domain,
    init_fun: Callable, 
    get_field: Callable,
    seed
  ):
    r'''Creates a continuous discretization from a `get_field` function.
    
    Args:
      domain (Domain): The domain of the discretization.
      init_fun (Callable): A function that initializes the parameters of the
        discretization. The signature of this function is `init_fun(rng, domain)`.
      get_field (Callable): A function that takes a parameter vector and a point in
        the domain and returns the field at that point. The signature of this
        function is `get_field(params, x)`.
      seed (int): The seed for the random number generator.
    
    Returns:
      Continuous: A continuous discretization.
    '''
    params = init_fun(seed, domain)
    return cls(params, domain=domain, get_fun=get_field)
  
  def __call__(self, x):
    r'''Same as the `get_field` function.
    
    !!! example
        ```python
        a = Continuous.from_function(init_params, domain, get_field)
        
        # Querying the field at the coordinate $`x=1.0`$
        a(1.0)
        ```
    '''
    return self.get_field(x)
    
  def get_field(self, x):
    return self.aux["get_field"](self.params, x)
  
  @property
  def on_grid(self):
    '''The field on the grid points of the domain.'''
    fun = self.aux["get_field"]
    ndims = len(self.domain.N)
    for _ in range(ndims):
        fun = vmap(fun, in_axes=(None, 0))
        
    return fun(self.params, self.domain.grid)

@register_pytree_node_class
class OnGrid(Linear):
  r'''A linear discretization on the grid points of the domain.'''
  def __init__(
    self,
    params,
    domain
  ):
    r'''Initializes a linear discretization on the grid points of the domain.
    
    Args:
      params (PyTree): The parameters of the discretization.
      domain (Domain): The domain of the discretization.
      
    Returns:
      OnGrid: A linear discretization on the grid points of the domain.
    '''
    dims = params.shape[-1]
    super().__init__(params, domain, dims, None)
    
  def tree_flatten(self):
    children = (self.params,)
    aux_data = (self.domain,)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    params = children[0]
    domain = aux_data[0]
    a = cls(params, domain=domain)
    return a
  
  @classmethod
  def empty(cls, domain, dims=1):
    r'''Creates an empty OnGrid field (zero field).'''
    N = tuple(list(domain.N) + [1,])
    return cls(jnp.zeros(N), domain)
  
  @property
  def ndim(self):
    r'''The number of dimensions of the field.'''
    return len(self.params.shape) - 1
  
  @property
  def is_field_complex(self):
    r'''Whether the field is complex.'''
    return self.params.dtype == jnp.complex64 or self.params.dtype == jnp.complex128
  
  @property
  def real(self):
    r'''Whether the field is real.'''
    return not self.is_field_complex
  
  @classmethod
  def from_grid(cls, grid_values, domain):
    r'''Creates an OnGrid field from a grid of values.
    
    Args:
      grid_values (ndarray): The grid of values.
      domain (Domain): The domain of the discretization.
    '''
    return cls(grid_values, domain)
  
  def replace_params(self, new_params):
    r'''Replaces the parameters of the discretization with new ones. The domain
    is not changed.
    
    Args:
      new_params (PyTree): The new parameters of the discretization.
    
    Returns:
      OnGrid: A linear discretization with the new parameters.
    '''
    return self.__class__(new_params, self.domain)
  
  @property
  def on_grid(self):
    r'''The field on the grid points of the domain.'''
    return self.params
  
  
@register_pytree_node_class
class FourierSeries(OnGrid):
  r'''A Fourier series field defined on a collocation grid.'''
  
  @property
  def _freq_axis(self):
    r'''Returns the frequency axis of the grid'''
    if self.is_field_complex:
      def f(N, dx):
        return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    else:
      def f(N, dx):
        return jnp.fft.rfftfreq(N, dx) * 2 * jnp.pi

    k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]
    return k_axis

  @property
  def _cut_freq_axis(self):
    r'''Same as _freq_axis, but last frequency axis is relative to a real FFT.
    Those frequency axis match with the ones of the rfftn function
    '''
    def f(N, dx):
      return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi

    k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]
    if not self.is_field_complex:
      k_axis[-1] = (
          jnp.fft.rfftfreq(self.domain.N[-1], self.domain.dx[-1]) * 2 * jnp.pi
      )
    return k_axis

  @property
  def _cut_freq_grid(self):
      return jnp.stack(jnp.meshgrid(*self._cut_freq_axis, indexing="ij"), axis=-1)
    
  @property
  def _freq_grid(self):
    return jnp.stack(jnp.meshgrid(*self._freq_axis, indexing="ij"), axis=-1)
  
@register_pytree_node_class
class FiniteDifferences(OnGrid):
  r'''A Finite Differences field defined on a collocation grid.'''
  def __init__(
    self,
    params,
    domain
  ):
    r'''Initializes a Finite Differences field on a collocation grid.
    
    Args:
      params (PyTree): The parameters of the discretization.
      domain (Domain): The domain of the discretization.
    
    Returns:
      FiniteDifferences: A Finite Differences field on a collocation grid.
    '''
    super().__init__(params, domain)
    
  def tree_flatten(self):
    children = (self.params,)
    aux_data = (self.domain,)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    params = children[0]
    domain = aux_data[0]
    a = cls(params, domain=domain)
    return a