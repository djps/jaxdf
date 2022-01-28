import jax.numpy as jnp
from jax import jit
from functools import partial
import jax
from typing import Callable

def _unroll(x):
  return tuple([x.replace_params(y) for y in x.params])

def _identity(x):
    return x

def euler_integration(f, x0, dt, output_steps):
  r"""Integrates the differential equation

  ```math
  \dot x = f(x,t)
  ```
  using a [first-order Euler method](https://en.wikipedia.org/wiki/Euler_method).
  The solution $`x`$ is given by

  ```math
  x^{(n+1)} = x^{(n)} + f\left(x^{(n)}, t^{(n)}\right)dt\cdot\kappa
  ```
  The structure of $`x`$ is inferred by $`x^{(0)}`$, which can be any pytree or Field.
  The output of $`f`$ should be compatible with $`x^{(0)}`$ (must have the same
  [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) structure)

  `output_steps` must be the index of the steps to save.

  The stepsize of the euler integration is given by $`\kappa\cdot dt`$.

  Args:
      f (function): Differential equation to integrate
      x0 (pytree): Initial point
      dt (float): Time-step
      output_steps ([int]): Iterations of the euler solver to save

  Returns:
      [pytree]: List of pytree, one for each value of `output_steps`. The
          structure of each pytree is the same as $`x^{(0)}`$.

  !!! example
      ```python
      from jaxdf.ode import euler_integration

      # Simulation settings
      g = 9.81
      dt = 0.01
      output_steps = jnp.array([0,5])/dt # Must be in steps, not seconds

      # Newton law of motion
      f = lambda (x, v), t: v, 0.5*g

      # Initial conditions
      x0 = 1.
      v0 = 0.

      # Calculate endpoint
      x_end, v_end = euler_integration(f, (x0,v0), dt, output_steps)
      ```
  """
  
  def euler_step(i, x):
    dx_dt = f(x, i * dt)
    return x + dt*dx_dt

  def euler_jump(x_t, i):
    x = x_t[0]
    start = x_t[1]
    end = start + i

    y = jax.lax.fori_loop(start, end, euler_step, x)
    return (y, end), y

  jumps = jnp.diff(output_steps)

  _, ys = jax.lax.scan(euler_jump, (x0, 0.0), jumps)
  return _unroll(ys)

def generalized_semi_implicit_euler(
    f: Callable,
    g: Callable,
    measurement_operator: Callable,
    alpha: jnp.ndarray,
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    dt: float,
    output_steps: jnp.ndarray,
    backprop=False,
    checkpoint=True,
):
    r"""This functions works in the same way as the
    [`semi_implicit_euler`](#jaxdf.ode.semi_implicit_euler) integrator,
    with the difference that the update function accepts an extra
    parameter $`\alpha`$ with the same pytree-structure as $`x`$ and 
    $`y`$. 

    Variable update is performed as 
    ```math
    \begin{dcases}
        x^{(n+1)} &= \alpha\left[\alpha x^{(n)}   + f\left(y^{(n)}, t^{(n)}\right)dt\cdot\right] \\
        y^{(n+1)} &= \alpha\left[\alpha y^{(n)} + g\left(x^{(n+1)}, t^{(n)}\right)dt\cdot\right] \\
        r^{(n)} &= M(x^{(n+1)}, y^{(n+1)})
    \end{dcases}
    ```

    $`M(x,y)`$ is an arbitrary measurement operator that is applied at
    the end of each timestep, for example it could evaluate the pressure intensity
    or the field value at some specific locations. If `None`, defaults to the identity 
    operator. 
    The vector of measurements $`r`$=`r` is returned. 

    !!! warning
        Calling this method with `backprop=True` allows to perform backpropagation.
        However, this requires storing the entire forward pass history and is therefore
        memory demanding. Alternatively, `backprop=False` allows to calculate
        derivatives using forward-propagation, or jacobian-vector products
        with memory cost independent of the simulation length. 
        Combined with `jax.vmap` or `jax.jaxfwd`, this makes easy to calculate
        gradients for functions that have tall jacobians, such as simulations 
        that depends on a small amount of parameters (e.g. delays, steering angle, 
        etc)

    Args:
        f (Callable): 
        g (Callable): 
        alpha (jnp.ndarray): 
        x0 (jnp.ndarray): 
        y0 (jnp.ndarray): 
        dt (float): [description]
        output_steps (jnp.ndarray): 
        measurement_operator ([type], optional): Defaults to `None`
        backprop (bool, optional): If true, the `vjp` operator can be evaluated, but requires
            a much larger memory footprint (all forward fields must be stored)
        checkpoint (bool, optional): If true, checkpointing is applied at each timestep to
            save memory during backpropagation. Defaults to `True`.

    !!! example
        ```python
        # Integrating the equations of motions for a planet around a star
        M_sun = 2.0  # kg
        p0 = jnp.array([0.0, 3.0])  # m
        v0 = jnp.array([1.0, 0.0])  # m/s
        G = 1
        dt = 0.1
        t_end = 200.0
        output_steps = (jnp.arange(0, t_end, 10 * dt) / dt).round()

        # Equations of motion
        f_1 = lambda v, t: v
        f_2 = lambda p, t: newton_grav_law(G=1, M=M_sun, r=p)

        M = lambda x: x # Identity operator, could have been `None`

        # Integrate
        trajectory , _ = generalized_semi_implicit_euler(
            f = f_1, 
            g = f_2, 
            measurement_operator = M, 
            alpha=0.0,
            x0 = p0,
            y0 = v0,
            dt = dt,
            output_steps = output_steps,
            backprop=False
        )
        ```
    """

    # Create vectors of (positive) indices to return
    # assert any(map(lambda x: x >= 0, output_steps))
    if measurement_operator is None:
        measurement_operator = _identity

    if backprop:
        return _generalized_semi_implicit_euler_with_vjp(
            f,
            g,
            measurement_operator,
            alpha,
            x0,
            y0,
            dt,
            output_steps,
            checkpoint,
        )
    else:
        return _generalized_semi_implicit_euler(
            f, g, measurement_operator, alpha, x0, y0, dt, output_steps
        )


def variable_update_with_pml(x, dx_dt, k, dt):
    x = k * (x * k + dt * dx_dt)
    return x


@partial(jit, static_argnums=(1, 2, 3))
def _generalized_semi_implicit_euler(
    f, g, measurement_operator, k, x0, y0, dt, output_steps
):
    def euler_step(i, conj_variables):
        x, y = conj_variables
        dx_dt = f(y, i * dt)
        x = variable_update_with_pml(x, dx_dt, k, dt)
        dy_dt = g(x, i * dt)
        y = variable_update_with_pml(y, dy_dt, k, dt)
        return (x, y)

    def euler_jump(x_t, i):
        x, start = x_t
        end = start + i

        y = jax.lax.fori_loop(
            start, end, jax.named_call(euler_step, name="euler_step"), x
        )
        return (y, end), measurement_operator(y)

    jumps = jnp.concatenate([jnp.diff(output_steps), jnp.array([1])])

    _, ys = jax.lax.scan(euler_jump, ((x0, y0), 0.0), jumps)
    return _unroll(ys)


def _generalized_semi_implicit_euler_with_vjp(
    params, f, g, measurement_operator, k, x0, y0, dt, output_steps, checkpoint
):
    def step_without_measurements(carry, t):
        x, y = carry
        dx_dt = f(params, y, t * dt)
        x = variable_update_with_pml(x, dx_dt, k, dt)
        dy_dt = g(params, x, t * dt)
        y = variable_update_with_pml(y, dy_dt, k, dt)
        return (x, y)

    def single_step(carry, t):
        fields = step_without_measurements(carry, t)
        return fields, measurement_operator(fields)

    if checkpoint:
        single_step = jax.checkpoint(single_step)

    _, ys = jax.lax.scan(single_step, (x0, y0), output_steps)

    return _unroll(ys)


if __name__ == "__main__":
    pass
