import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import least_squares
import jax
import jax.numpy as jnp

# Example input: large sparse matrix A and vector b
m, n = 500, 200000  # Dimensions of A: m x n
density = 0.001     # Sparsity of A

# Create a random sparse matrix A and a dense vector b
A = csr_matrix(np.random.rand(m, n) * (np.random.rand(m, n) < density))
b = np.random.rand(m)
lambd = 0.1  # Regularization parameter

# Define the residual function using JAX
def residual_function(x, A, b, lambd):
    A_dot_x = A @ x  # Sparse matrix-vector multiplication
    residual = A_dot_x - b
    regularization = jnp.sqrt(lambd) * (x - 1)  # Regularization term
    return jnp.concatenate([residual, regularization])

# Function to compute the JVP using JAX
def jvp_residual_function(x, A, b, lambd, v):
    # Computes the Jacobian-vector product (JVP) without forming the full Jacobian
    _, jvp = jax.jvp(lambda x: residual_function(x, A, b, lambd), (x,), (v,))
    return jvp

# Wrapper function to be passed to least_squares that computes the residual
def least_squares_residual(x, A, b, lambd):
    return np.asarray(residual_function(x, A, b, lambd))  # Convert to NumPy array

# Wrapper function to compute the JVP for least_squares
def least_squares_jvp(x, A, b, lambd):
    # Define a function for JVP with a dummy vector v
    def jvp(v):
        return np.asarray(jvp_residual_function(x, A, b, lambd, v))
    return jvp

# Initial guess for x
x0 = np.ones(n)

# Run least_squares using JVP without forming full Jacobian
result = least_squares(
    least_squares_residual,  # The residual function
    x0,                      # Initial guess for x
    jac_sparsity=None,        # Do not compute full Jacobian, use JVP instead
    args=(A, b, lambd),       # Arguments for the residual function
    bounds=(0, np.inf),       # Non-negative bounds for x
    method='trf',             # Trust Region Reflective method for large-scale problems
    verbose=2,                # Verbosity level for detailed output
)

# Extract the optimized solution
x_opt = result.x
print(f"Optimized x: {x_opt}")
