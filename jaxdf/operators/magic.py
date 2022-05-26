from jaxdf.core import operator, params_map
from jaxdf.discretization import *

"""
This file contains the operators that are
bound with the magic functions of fields
"""


## __add__
@operator
def __add__(x: Linear, y: Linear, *, params=None):
  new_params = params_map(lambda x,y: x+y, x.params, y.params)
  return x.replace_params(new_params)

@operator(precedence=-1)
def __add__(x: OnGrid, y: object, *, params=None):
  new_params = params_map(lambda x: x+y, x.params)
  return x.replace_params(new_params)

@operator
def __add__(x: Continuous, y: Continuous, *, params=None):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) + get_y(p[1], coords)
  return Continuous([x.params, y.params], x.domain, get_fun)

@operator
def __add__(x: Continuous, y: object, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p['params'], coords) + p['constant']
  new_params = {'params': x.params, 'constant': y}
  return Continuous(new_params, x.domain, get_fun)

## __bool__
@operator
def __bool__(x: OnGrid, *, params=None):
  new_params = params_map(lambda x: bool(x), x.params)
  return x.replace_params(new_params)


## __divmod__
@operator
def __divmod__(x: OnGrid, y: OnGrid, *, params=None):
  new_params = params_map(lambda x,y: divmod(x, y), x.params, y.params)
  return x.replace_params(new_params)

@operator
def __divmod__(x: Linear, y, *, params=None):
  new_params = params_map(lambda x: divmod(x, y), x.params)
  return x.replace_params(new_params)


## __mul__
@operator
def __mul__(x: OnGrid, y: OnGrid, *, params=None):
  new_params = params_map(lambda x,y: x*y, x.params,y.params)
  return x.replace_params(new_params)

@operator(precedence=-1)
def __mul__(x: Linear, y, *, params=None):
  new_params = params_map(lambda x: x*y, x.params)
  return x.replace_params(new_params)

@operator
def __mul__(x: Continuous, y: Continuous, *, params=None):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) * get_y(p[1], coords)
  return x.update_fun_and_params([x.params, y.params], get_fun)

@operator
def __mul__(x: Continuous, y, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p['params'], coords) * p['constant']
  new_params = {'params': x.params, 'constant': y}
  return x.update_fun_and_params(new_params, get_fun)

## __neg__
@operator
def __neg__(x: Linear, *, params=None):
  new_params = params_map(lambda x: -x, x.params)
  return x.replace_params(new_params)

@operator
def __neg__(x: Continuous, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return -get_x(p, coords)
  return Continuous(x.params, x.domain, get_fun)


## __pow__
@operator
def __pow__(x: OnGrid, y: OnGrid, *, params=None):
  new_params = params_map(lambda x,y: x**y, x.params,y.params)
  return x.replace_params(new_params)

@operator(precedence=-1)
def __pow__(x: OnGrid, y: object, *, params=None):
  new_params = params_map(lambda x: x**y, x.params)
  return x.replace_params(new_params)

@operator
def __pow__(x: Continuous, y: Continuous, *, params=None):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) ** get_y(p[1], coords)
  return x.update_fun_and_params([x.params, y.params], get_fun)

@operator
def __pow__(x: Continuous, y: object, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p['params'], coords) ** p['constant']
  new_params = {'params': x.params, 'constant': y}
  return Continuous(new_params, x.domain, get_fun)


## __radd__
@operator(precedence=-1)
def __radd__(x: OnGrid, y: object, *, params=None):
  return x + y


@operator(precedence=-1)
def __radd__(x: Continuous, y: object, *, params=None):
  return x + y


## __rmul__
@operator(precedence=-1)
def __rmul__(x: Field, y: object, *, params=None):
  return x * y



## __rpow__
@operator(precedence=-1)
def __rpow__(x: OnGrid, y: object, *, params=None):
  new_params = params_map(lambda x: x**y, x.params)
  return x.replace_params(new_params)

@operator(precedence=-1)
def __rpow__(x: Continuous, y: object, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return p['constant'] ** get_x(p['params'], coords)
  new_params = {'params': x.params, 'constant': y}
  return Continuous(new_params, x.domain, get_fun)


## __rsub__
@operator
def __rsub__(x: Linear, y: object, *, params=None):
  return (-x) + y

@operator(precedence=-1)
def __rsub__(x: Continuous, y: object, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return p['constant'] - get_x(p['params'], coords)
  new_params = {'params': x.params, 'constant': y}
  return Continuous(new_params, x.domain, get_fun)


## __rtruediv__
@operator
def __rtruediv__(x: OnGrid, y: object, *, params=None):
  new_params = params_map(lambda x: y/x, x.params)
  return x.replace_params(new_params)

@operator(precedence=-1)
def __rtruediv__(x: Continuous, y: object, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return p['constant'] / get_x(p['params'], coords)
  new_params = {'params': x.params, 'constant': y}
  return Continuous(new_params, x.domain, get_fun)

## __sub__
@operator
def __sub__(x: Linear, y: Linear, *, params=None):
  new_params = params_map(lambda x,y: x-y, x.params,y.params)
  return x.replace_params(new_params)

@operator(precedence=-1)
def __sub__(x: OnGrid, y: object, *, params=None):
  new_params = params_map(lambda x: x-y, x.params)
  return x.replace_params(new_params)

@operator
def __sub__(x: Continuous, y: Continuous, *, params=None):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) - get_y(p[1], coords)
  return Continuous([x.params, y.params], x.domain, get_fun)

@operator
def __sub__(x: Continuous, y: object, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p['params'], coords) - p['constant']
  new_params = {'params': x.params, 'constant': y}
  return Continuous(new_params, x.domain, get_fun)

## __truediv__
@operator
def __truediv__(x: Continuous, y: Continuous, *, params=None):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) / get_y(p[1], coords)
  return Continuous([x.params, y.params], x.domain, get_fun)

@operator
def __truediv__(x: Continuous, y: object, *, params=None):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p["params"], coords) / p["constant"]
  new_params = {'params': x.params, 'constant': y}
  return Continuous(new_params, x.domain, get_fun)

@operator
def __truediv__(x: OnGrid, y: OnGrid, *, params=None):
  new_params = params_map(lambda x,y: x/y, x.params,y.params)
  return x.replace_params(new_params)

@operator(precedence=-1)
def __truediv__(x: Linear, y, *, params=None):
  new_params = params_map(lambda x: x/y, x.params)
  return x.replace_params(new_params)

## inverse
@operator
def inverse(x: OnGrid, *, params=None):
  new_params = params_map(lambda x: 1/x, x.params)
  return x.replace_params(new_params)

if __name__ == '__main__':
  from jaxdf.util import _get_implemented

  magic = [
    __add__, __bool__, __divmod__,
    __mul__, __neg__, __pow__, __radd__,
    __rmul__, __rpow__, __rsub__, __rtruediv__,
    __sub__, __truediv__, inverse
  ]

  print('magic.py:')
  print('----------------')
  for f in magic:
    _get_implemented(f)
  print('\n')
