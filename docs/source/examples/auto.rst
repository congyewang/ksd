.. _ksd-example:

Quick Start
--------------------------------

Here is a complete code example demonstrating how to use `KernelSteinDiscrepancyJax`:

.. code-block:: python
   :linenos:
   :emphasize-lines: 20,21,22

   import jax
   from jax import numpy as jnp
   from jax.scipy.stats import multivariate_normal

   from ksd_metric.kernel import KernelJax
   from ksd_metric.stein import KernelSteinDiscrepancyJax
   from ksd_metric.target import TargetDistributionJax
   from ksd_metric.utils import JaxKernelFunction

   # Example usage of KernelSteinDiscrepancyJax
   key = jax.random.PRNGKey(42)
   dim = 2
   mean = jnp.zeros(dim)
   cov = jnp.eye(dim)
   N = 10_000
   x = jax.random.multivariate_normal(key, mean, cov, shape=(N,))

   # Define the target distribution and kernel
   log_target_pdf = lambda x: multivariate_normal.logpdf(x, mean=mean, cov=cov)
   target = TargetDistributionJax(log_target_pdf=log_target_pdf)
   kernel = KernelJax(lambda x, y: JaxKernelFunction.imq(x, y, jnp.eye(dim)))
   ksd = KernelSteinDiscrepancyJax(target=target, kernel=kernel)

   # Calculate kernel Stein discrepancy
   res = ksd.kernel_stein_discrepancy(x)
   print(res)
