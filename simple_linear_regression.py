import numpy as np

"""

Simple Linear Regression via Least Squares using Gradient Descent

"""

def compute_cost(x, y, w, b):
    m = x.shape[0] #  set m as the number of training examples
    cost = 0 # initialize cost as 0

    # accumulate squared errors over all examples
    for i in range(m):
        f_wb = w * x[i] + b                 # prediction for x[i]
        cost = cost + (f_wb - y[i]) ** 2    # squared error cost for i-th example

    # return Half-MSE
    total_cost = cost / (2 * m)

    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0] # set m as the number of training examples

    # initialize gradient accumulators (∂J/∂w, ∂J/∂b)
    dj_dw = 0.0
    dj_db = 0.0

     # accumulate per-example gradient contributions
    for i in range(m):
        f_wb = w * x[i] + b               # prediction 
        dj_dw_i = (f_wb - y[i]) * x[i]    # compute gradient ∂J/∂w at i-th iteration
        dj_db_i = f_wb - y[i]             # compute gradient ∂J/∂b at i-th iteration

        # update the gradients
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    # average over m (number of training examples) -> follows from cost function definition
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, grad_fn):
    # initialize b to b_in (mostlikely 0) and w to w_in (mostlikely 0)
    b = b_in
    w = w_in

    # initialize J_history and p_history as empty lists
    J_history = []
    p_history = []

    # perform gradient descent updates
    for i in range(num_iters):
        dj_dw, dj_db = grad_fn(x, y, w, b) # calls compute_gradient function to get gradients.

        # now that we have the gradients, move parameters w,b downhill (negative gradient direction)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # keep track of cost and parameters each iteration
        J_history.append(compute_cost(x, y, w, b))
        p_history.append([w, b])

        # repeat num_iters times (i.e 1000 iterations)

    return w, b, J_history, p_history