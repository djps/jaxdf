from jaxdf import operator


def test_init_parameters():

  @operator
  def f(a: float, b: float, params=None):
    return a + b, params

  def params_constructor(a: int, b: int):
    return a - b

  @operator(init_params=params_constructor)
  def f(a: int, b: int, params=None):
    return a + b + params

  print(f(1.0, 2.0))
  print(f.init_params(1.0, 2.0))

  print(f(1, 2))
  print(f.init_params(1, 2))

if __name__ == "__main__":
  test_init_parameters()
