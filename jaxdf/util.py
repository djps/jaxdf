from typing import Callable, Tuple

from jax import numpy as jnp
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
