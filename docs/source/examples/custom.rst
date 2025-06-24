.. _ksd-example:

Customized Using Interface
--------------------------------

Here is a complete code example demonstrating how to use Customized target distribution and base kernel function.

.. code-block:: python
   :linenos:

   import jax
   from jax import numpy as jnp
   from jaxtyping import Array, Float

   from ksd_metric.kernel import KernelJax
   from ksd_metric.stein import KernelSteinDiscrepancyJax
   from ksd_metric.target import TargetDistributionInterface

   key = jax.random.PRNGKey(42)
   dim = 2
   mean = jnp.zeros(dim)
   cov = jnp.eye(dim)
   N = 10_000
   x = jax.random.multivariate_normal(key, mean, cov, shape=(N,))


   class CustomTargetDistribution(TargetDistributionInterface):
       def log_target_pdf(self, x: Float[Array, "num dim"]) -> Float[Array, "num dim"]:
           return -0.5 * x @ x

       def grad_log_target_pdf(
           self, x: Float[Array, "num dim"]
       ) -> Float[Array, "num dim"]:
           return -x


   def custom_kernel(
       x: Float[Array, "num"],
       y: Float[Array, "num"],
   ) -> Float[Array, "num"]:
       dim = len(x)
       diff = x - y
       return (1.0 + (diff @ jnp.eye(dim) @ diff)) ** (-0.5)


   # Define the target distribution and kernel
   target = CustomTargetDistribution()
   kernel = KernelJax(custom_kernel)
   ksd = KernelSteinDiscrepancyJax(target=target, kernel=kernel)

   # Calculate kernel Stein discrepancy
   res = ksd.kernel_stein_discrepancy(x)
   print(res)
