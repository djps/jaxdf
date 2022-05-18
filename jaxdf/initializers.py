from jaxdf.discretization import *


def get_kvec(x: FourierSeries) -> dict:
  r'''Returns the k-vectors of a Fourier series,
  representing the frequency axis.

  Functionally equivalent to `x._freq_axis`.

  Args:
    x (FourierSeries): Input field

  Returns:
    dict: Dictionary of the type `{'k_vec': List[freq_axis]}`
  '''
  return  {'k_vec': x._freq_axis}
