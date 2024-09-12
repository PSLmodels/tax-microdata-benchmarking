import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# Example input: large sparse matrix A and vector b
m, n = 500, 200000  # Dimensions of A: m x n
density = 0.01  # Sparsity of A

# Create a random sparse matrix A and a dense vector b
A_scipy = csr_matrix(np.random.rand(m, n) * (np.random.rand(m, n) < density))
b = np.random.rand(m)
lambd = 0.1  # Regularization parameter

# Convert SciPy sparse matrix to JAX-compatible BCOO format
A_jax = BCOO.from_scipy_sparse(A_scipy)


# Define the residual function using JAX
def residual_function(x, A, b, lambd):
    A_dot_x = A @ x  # JAX sparse matrix-vector multiplication
    residual = A_dot_x - b
    regularization = jnp.sqrt(lambd) * (x - 1)  # Regularization term
    return jnp.concatenate([residual, regularization])


# Objective function for minimization (sum of squared residuals)
def objective_function(x, A, b, lambd):
    res = residual_function(x, A, b, lambd)
    return jnp.sum(jnp.square(res))


# Function to compute the JVP using JAX
def jvp_residual_function(x, A, b, lambd, v):
    # Computes the Jacobian-vector product (JVP) without forming the full Jacobian
    _, jvp = jax.jvp(lambda x: residual_function(x, A, b, lambd), (x,), (v,))
    return jvp


# Define gradient using JAX autodiff
def gradient_function(x, A, b, lambd):
    grad = jax.grad(objective_function)(x, A, b, lambd)
    return np.asarray(grad)


# Initial guess for x
x0 = np.ones(n)

# Set up bounds for non-negative constraints
bounds = [(0, None) for _ in range(n)]

# Call minimize with L-BFGS-B
result = minimize(
    fun=objective_function,  # Objective function
    x0=x0,  # Initial guess
    jac=gradient_function,  # Gradient (JVP-based)
    args=(A_jax, b, lambd),  # Additional arguments
    method="L-BFGS-B",  # Use L-BFGS-B solver for large-scale problems
    bounds=bounds,  # Non-negative bounds
    options={"disp": True},  # Display convergence info
)

# Extract the optimized solution
x_opt = result.x
print(f"Optimized x: {x_opt}")
print(np.quantile(result.x, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))

diff = A_scipy @ result.x - b
pdiff = diff / b
print(np.quantile(pdiff, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))
