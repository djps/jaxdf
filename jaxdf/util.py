from typing import Callable, Tuple

from jax import numpy as jnp
from jax import scipy as jsp
from jax.numpy import expand_dims, ndarray


def append_dimension(x: ndarray):
  return expand_dims(x, -1)

def update_dictionary(old: dict, new_entries: dict):
  r'''Update a dictionary with new entries.

  Args:
    old (dict): The dictionary to update
    new_entries (dict): The new entries to add to the dictionary

  Returns:
    dict: The updated dictionary
  '''
  for key, val in zip(new_entries.keys(), new_entries.values()):
    old[key] = val
  return old

def get_ffts(x) ->Tuple[Callable, Callable]:
  r'''Returns the appropriate fft and ifft functions
  depending on the type of the input.'''
  if x.real:
    ffts = [jnp.fft.rfft, jnp.fft.irfft]
  else:
    ffts = [jnp.fft.fft, jnp.fft.ifft]
  return ffts


def _convolve_kernel_1d(
  x: jnp.ndarray,
  kernel: jnp.ndarray,
  mode: str = 'valid',
  pad_mode: str = "constant",
  pad_values: float = 0.0,
) -> jnp.ndarray:
  r'''Convolves a 1D kernel with a field over all possible
  dimensions.

  Arguments:
    x (ndarray): The field to convolve. The last dimension size should be 1.
    kernel (ndarray): The kernel to convolve with.
    mode (str): The mode of the convolution.
    pad_mode (str): The mode of the padding.
    pad_values (float): The value of the padding (for 'constant' mode).

  Returns:
    ndarray: The convolved field.
  '''
  # Make kernel the right size
  extra_pad = (len(kernel) // 2, len(kernel) // 2)
  for ax in range(x.ndim-1):
    kernel = jnp.expand_dims(kernel, axis=0)  # Kernel on the last axis

  # Convolve in each dimension
  outs = []
  img = x.params[...,0]
  for i in range(x.ndim):
    k = jnp.moveaxis(kernel, -1, i)

    pad = [(0, 0)] * x.ndim
    pad[i] = extra_pad
    f = jnp.pad(img, pad, mode="constant", constant_values=pad_values)

    out = jsp.signal.convolve(f, k, mode="valid")/x.domain.dx[i]
    outs.append(out)

  new_params = jnp.stack(outs, -1)
  return new_params

def _get_implemented(f):
  r'''Prints the implemented methods of a function. For internal use.

  Arguments:
    f (function): The function to get the implemented methods of.

  Returns:
    None

  '''
  from inspect import signature

  # TODO: Why there are more instances for the same types?

  print(f.__name__ + ':')
  instances = []
  a = f.methods.values()
  for f_instance in a:
    instances.append(signature(f_instance[0]).__repr__()[11:-1])

  instances = set(instances)
  for instance in instances:
    # if `self` is in the signature, skip
    if 'self' in instance:
      continue
    # Remove `jaxdf.discretization` from instance, wherever
    # it is in the string
    instance = instance.replace('jaxdf.discretization.', '')
    print(' ─ ' + instance)
