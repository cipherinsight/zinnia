import json

from zinnia import *


@zk_circuit
def verify_solution(
    training_x: NDArray[float, 10, 2],
    training_y: NDArray[float, 10],
    testing_x: NDArray[float, 2, 2],
    testing_y: NDArray[float, 2]
):
    # Tiny MLP (1 hidden layer, quadratic activation φ(h)=h^2)
    # Fixed sizes to keep the circuit static:
    #   input_dim = 2, hidden_dim = 3, output_dim = 1
    input_dim = 2
    hidden_dim = 3
    train_m = 10
    test_m = 2
    # hyperparams
    steps = 10
    lr = 0.02  # smaller step

    # smaller init
    W1 = [[0.02, -0.01, 0.016],
          [0.014, 0.004, -0.006]]
    b1 = [0.0, 0.0, 0.0]
    W2 = [0.05, -0.03, 0.02]
    b2 = 0.0

    # Training loop (full-batch gradient descent; all bounds static)
    for _ in range(steps):
        # Forward pass on training set (explicit loops, no nested defs)
        H = [[0.0, 0.0, 0.0] for _i in range(train_m)]  # pre-activations
        A = [[0.0, 0.0, 0.0] for _i in range(train_m)]  # activations h^2
        preds = [0.0 for _i in range(train_m)]

        for i in range(train_m):
            # Compute hidden pre-activations and activations
            for k in range(hidden_dim):
                h = 0.0
                # x · W1 + b1
                h += training_x[i][0] * W1[0][k]
                h += training_x[i][1] * W1[1][k]
                h += b1[k]
                H[i][k] = h
                A[i][k] = h * h  # φ(h)=h^2

            # Output: yhat = A · W2 + b2
            out = 0.0
            out += A[i][0] * W2[0]
            out += A[i][1] * W2[1]
            out += A[i][2] * W2[2]
            out += b2
            preds[i] = out

        # Errors
        errors = [0.0 for _i in range(train_m)]
        for i in range(train_m):
            errors[i] = preds[i] - training_y[i]

        # Gradients (MSE with factor (2/m))
        inv_m2 = 2.0 / float(train_m)

        # dW2, db2
        dW2 = [0.0, 0.0, 0.0]
        db2 = 0.0
        for i in range(train_m):
            e = errors[i]
            db2 += e
            dW2[0] += e * A[i][0]
            dW2[1] += e * A[i][1]
            dW2[2] += e * A[i][2]
        dW2[0] *= inv_m2
        dW2[1] *= inv_m2
        dW2[2] *= inv_m2
        db2 *= inv_m2

        # Hidden layer grads via chain rule; dφ/dh = 2h
        dW1 = [[0.0, 0.0, 0.0],  # for input 0
               [0.0, 0.0, 0.0]]  # for input 1
        db1g = [0.0, 0.0, 0.0]
        for i in range(train_m):
            e = errors[i]
            # For each hidden unit k
            # dL/dh_{ik} = (2/m) * e * W2[k] * (2 * H[i][k]) = (4/m) * e * W2[k] * H[i][k]
            # Note (4/m) = 2 * (2/m) = 2 * inv_m2
            for k in range(hidden_dim):
                dh = (2.0 * inv_m2) * e * W2[k] * (2.0 * H[i][k]) * 0.5  # simplified to (4/m)*e*W2[k]*H[i][k]
                # (Using algebraic equivalence to keep arithmetic simple)
                # Actually compute as:
                dh = (4.0 / float(train_m)) * e * W2[k] * H[i][k]
                dW1[0][k] += dh * training_x[i][0]
                dW1[1][k] += dh * training_x[i][1]
                db1g[k] += dh

        # Parameter updates
        for k in range(hidden_dim):
            W2[k] -= lr * dW2[k]
            b1[k] -= lr * db1g[k]
            W1[0][k] -= lr * dW1[0][k]
            W1[1][k] -= lr * dW1[1][k]
        b2 -= lr * db2

    # Forward pass on testing set (no helper methods)
    test_preds = [0.0, 0.0]
    for i in range(test_m):
        # hidden
        h0 = testing_x[i][0] * W1[0][0] + testing_x[i][1] * W1[1][0] + b1[0]
        h1 = testing_x[i][0] * W1[0][1] + testing_x[i][1] * W1[1][1] + b1[1]
        h2 = testing_x[i][0] * W1[0][2] + testing_x[i][1] * W1[1][2] + b1[2]
        a0 = h0 * h0
        a1 = h1 * h1
        a2 = h2 * h2
        yhat = a0 * W2[0] + a1 * W2[1] + a2 * W2[2] + b2
        test_preds[i] = yhat

    # Test MSE
    se_sum = 0.0
    for i in range(test_m):
        diff = test_preds[i] - testing_y[i]
        se_sum += diff * diff
    test_mse = se_sum / float(test_m)

    # Require modest generalization
    assert test_mse <= 50


if __name__ == '__main__':
    # Synthetic quadratic target:
    #   y = (x1 + 0.5*x2 + 1.0)^2 + 0.5
    def f(x1, x2):
        t = x1 + 0.5 * x2 + 1.0
        return t * t + 0.5

    training_x = [
        [-2.0, -1.0],
        [-2.0,  1.0],
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 0.0,  0.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [ 2.0, -1.0],
        [ 2.0,  1.0],
        [ 2.0,  0.0],
    ]
    training_y = [f(x[0], x[1]) for x in training_x]

    testing_x = [
        [-1.0, 0.0],
        [ 1.0, 2.0],
    ]
    testing_y = [f(x[0], x[1]) for x in testing_x]

    assert verify_solution(training_x, training_y, testing_x, testing_y)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(training_x, training_y, testing_x, testing_y)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
