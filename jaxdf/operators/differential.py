import jax
from findiff import coefficients as findif_coeff
from jax import numpy as jnp
from jax import scipy as jsp

from jaxdf.core import operator
from jaxdf.discretization import *
from jaxdf.initializers import *
from jaxdf.util import get_ffts


def _convolve_with_pad(
  kernel: jnp.ndarray,
  array: jnp.ndarray,
  axis: int
) -> jnp.ndarray:
  r'''Convolves an array with a kernel, using reflection padding.

  The kernel is supposed to be with the same number of indices as the array,
  but the only dimension different than 1 corresponds to the axis. Padding
  is only applied to such axis.

  Parameters:
    kernel (jnp.ndarray): The kernel to convolve with.
    array (jnp.ndarray): The array to convolve.

  Returns:
    jnp.ndarray: The convolved array.
  '''
  # Reflection padding the array where appropriate
  pad_size = max(kernel.shape)//2
  extra_pad = (pad_size,pad_size)
  pad = [(0, 0)] * x.ndim
  pad[axis] = extra_pad
  f = jnp.pad(array, pad, mode="wrap")

  # Apply kernel
  out = jsp.signal.convolve(f, kernel, mode="valid")

  return out

################## Derivative ##################
@operator(init_params=fd_derivative_init)
def derivative(
  x: FiniteDifferences,
  *,
  axis=0,
  stagger = [0],
  params=None
):
  kernel = params

  # Getting data
  array = x.on_grid[...,0]

  # Apply kernel
  out = _convolve_with_pad(kernel, array, axis)
  out = jnp.expand_dims(out, axis=-1)

  # Make it a field again
  return x.replace_params(out)

################## Diagonal of Jacobian ##################
@operator(init_params=get_kvec)
def diag_jacobian(
  x: FourierSeries,
  *,
  stagger = [0],
  params=None
) -> FourierSeries:
  r'''Returns the diagonal of the Jacobian of a Fourier series.

  Args:
    x (FourierSeries): Input field
    params (dict, optional): Dictionary of the type `{'k_vec': List[freq_axis]}`

  Params initializer: `get_kvec`

  Returns:
    FourierSeries: Gradient of the input field
  '''
  # Checking inputs
  assert x.domain.ndim == x.dims # Diagonal jackobian only works on vector fields of the same dimension as the domain

  # Choosing the FFT
  ffts = get_ffts(x)

  # Numerics
  k_vec = params["k_vec"]
  new_params = jnp.zeros_like(x.params)

  # Gradient on a single direction / axis
  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = 1j * Fx * k_vec[axis]
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  # Loop for all dimensions
  for ax in range(x.ndim):
    new_params = new_params.at[..., ax].set(single_grad(ax, x.params[..., ax]))

  return FourierSeries(new_params, x.domain)


@operator(init_params=ft_diag_jacobian_init)
def diag_jacobian(
  x: FiniteDifferences,
  *,
  stagger = [0],
  params = None
) -> FiniteDifferences:
  # Checking inputs
  assert x.domain.ndim == x.dims # Diagonal jackobian only works on vector fields of the same dimension as the domain

  kernels = params
  array = x.on_grid

  # Apply the corresponding kernel to each dimension
  outs = [_convolve_with_pad(kernels[i], array[...,i]) for i in range(x.ndim)]
  new_params = jnp.stack(outs, axis=-1)

  return x.replace_params(new_params)

################## Gradient ##################
@operator
def gradient(x: Continuous, *, params=None):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    v = f_jac(p, coords)[0]
    return v
  return x.update_fun_and_params(x.params, grad_fun)

@operator(init_params=get_kvec)
def gradient(
  x: FourierSeries,
  *,
  stagger = [0],
  params=None
) -> FourierSeries:
  r'''Returns the gradient of a Fourier series.

  Args:
    x (FourierSeries): Input field
    params (dict, optional): Dictionary of the type `{'k_vec': List[freq_axis]}`

  Params initializer: `get_kvec`

  Returns:
    FourierSeries: Gradient of the input field
  '''
  # Checking inputs
  assert x.dims == 1 # Gradient only defined for scalar fields

  # Choosing the FFT
  ffts = get_ffts(x)

  # Exracting numerics
  k_vec = params['k_vec']
  u = x.params[...,0]

  # Gradient on a single direction
  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = 1j * Fx * k_vec[axis]
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  # V-mapping on all axis
  new_params = jnp.stack([single_grad(i, u) for i in range(x.ndim)], axis=-1)

  return FourierSeries(new_params, x.domain)

################## Laplacian ##################
@operator(init_params=get_kvec)
def laplacian(
  x: FourierSeries,
  *,
  stagger = [0],
  params=None
) -> FourierSeries:
  r'''Returns the Laplacian of a Fourier series.

  Args:
    x (FourierSeries): Input field
    params (dict, optional): Dictionary of the type `{'k_vec': List[freq_axis]}`

  Params initializer: `get_kvec`

  Returns:
    FourierSeries: Laplacian of the input field
  '''
  # Checking inputs
  assert x.dims == 1 # Laplacian only defined for scalar fields

  # Choosing the FFT
  ffts = get_ffts(x)

  # Exracting numerics
  k_vec = params['k_vec']
  u = x.params[...,0]

  # Gradient on a single direction
  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = -Fx * k_vec[axis] ** 2
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  # V-mapping on all axis
  new_params = jnp.sum(
        jnp.stack([single_grad(i, u) for i in range(x.ndim)], axis=-1),
        axis=-1,
        keepdims=True,
    )

  return FourierSeries(new_params, x.domain)


if __name__ == "__main__":


  ## derivative
  @operator
  def derivative(x: Continuous, axis=0, params=None):
    get_x = x.aux['get_field']
    def grad_fun(p, coords):
      f_jac = jax.jacfwd(get_x, argnums=(1,))
      return jnp.expand_dims(f_jac(p, coords)[0][0][axis], -1)
    return Continuous(x.params, x.domain, grad_fun), None


  def _fd_coefficients(
    order: int = 1,
    accuracy: int = 2,
    staggered: str = 'center'
  ):
    fd_kernel = findif_coeff(order, accuracy)[staggered]
    coeffs = fd_kernel['coefficients'].tolist()
    offsets = fd_kernel['offsets']

    # Add zeros if needed, to make it work with padding
    if staggered == 'forward':
      coeffs = [0.,]*offsets[-1] + coeffs
    elif staggered == 'backward':
      coeffs = coeffs + [0.,]*(-offsets[0])

    return jnp.asarray(coeffs)

  @operator
  def gradient(x: FiniteDifferences, params=None, accuracy=2, staggered='center'):
    if params is None:
      params = _fd_coefficients(1, accuracy, staggered)

    kernel = params
    new_params = _convolve_kernel(x, kernel)
    return FiniteDifferences(new_params, x.domain), params


  # diag_jacobian
  @operator
  def diag_jacobian(x: Continuous, params=None):
    get_x = x.aux['get_field']
    def diag_fun(p, coords):
      f_jac = jax.jacfwd(get_x, argnums=(1,))
      return jnp.diag(f_jac(p, coords)[0])
    return x.update_fun_and_params(x.params, diag_fun), None

  @operator
  def diag_jacobian(x: FiniteDifferences, params=None, accuracy=2, staggered='center'):
    if params is None:
      params = _fd_coefficients(1, accuracy, staggered)

    outs = []
    img = x.params
    kernel = params

    # Make kernel the right size
    extra_pad = (len(kernel) // 2, len(kernel) // 2)
    for ax in range(x.ndim-1):
      kernel = jnp.expand_dims(kernel, axis=0)  # Kernel on the last axis

    for ax in range(x.ndim):
      img_shifted = jnp.moveaxis(img[...,ax], ax, -1)
      pad = [(0, 0)] * x.ndim
      pad[-1] = extra_pad
      f = jnp.pad(img_shifted, pad, mode="constant")
      out = jsp.signal.convolve(f, kernel, mode="valid")/x.domain.dx[ax]
      out = jnp.moveaxis(out, -1, ax)
      outs.append(out)

    outs = jnp.stack(outs, axis=-1)

    return FiniteDifferences(outs, x.domain), params

  # laplacian
  @operator
  def laplacian(x: Continuous, params=None):
    get_x = x.aux['get_field']
    def grad_fun(p, coords):
      hessian = jax.hessian(get_x, argnums=(1,))(p,coords)[0][0][0]
      return jnp.diag(hessian)
    return x.update_fun_and_params(x.params, grad_fun), None


  @operator
  def laplacian(x: FiniteDifferences, params=None, accuracy=4):
    if params == None:
      coeffs = {
        2: [1, -2, 1],
        4: [-1 / 12, 4 / 3, -5/2, 4 / 3, -1 / 12],
        6: [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
      }
      params = {"laplacian_kernel": jnp.asarray(coeffs[accuracy])}

    kernel = params["laplacian_kernel"]
    new_params = _convolve_kernel(x, kernel)
    return FiniteDifferences(new_params, x.domain), params


  if __name__ == '__main__':
    from jaxdf.util import _get_implemented

    funcs = [
      derivative, diag_jacobian, gradient, laplacian,
    ]

    print('differential.py:')
    print('----------------')
    for f in funcs:
      _get_implemented(f)
    print('\n')
