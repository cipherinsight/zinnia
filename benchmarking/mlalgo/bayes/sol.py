import json

from zinnia import *


@zk_circuit
def verify_solution(
        training_x: NDArray[float, 10, 2],
        training_y: NDArray[float, 10],
        testing_x: NDArray[float, 2, 2],
        testing_y: NDArray[float, 2]
):
    # Bernoulli Naive Bayes with Laplace smoothing on a tiny, static dataset
    # - Classes: {0,1}
    # - Features: 2 binary features in {0,1}
    # - Training set size: 10
    # - Test set size: 2

    n_train = 10
    n_features = 2
    n_classes = 2
    alpha = 1.0  # Laplace smoothing

    # Count per-class examples and per-class per-feature "1" counts
    count_c0 = 0.0
    count_c1 = 0.0
    # counts of feature==1 given class
    count1_c0_f0 = 0.0
    count1_c0_f1 = 0.0
    count1_c1_f0 = 0.0
    count1_c1_f1 = 0.0

    for i in range(n_train):
        yi = training_y[i]  # 0.0 or 1.0
        x0 = training_x[i, 0]
        x1 = training_x[i, 1]

        if yi == 0.0:
            count_c0 += 1.0
            if x0 >= 0.5:
                count1_c0_f0 += 1.0
            if x1 >= 0.5:
                count1_c0_f1 += 1.0
        else:
            count_c1 += 1.0
            if x0 >= 0.5:
                count1_c1_f0 += 1.0
            if x1 >= 0.5:
                count1_c1_f1 += 1.0

    # Class priors with Laplace smoothing: (count_c + alpha) / (n_train + n_classes*alpha)
    prior0 = (count_c0 + alpha) / (n_train + n_classes * alpha)
    prior1 = (count_c1 + alpha) / (n_train + n_classes * alpha)

    # Feature likelihoods θ_{c,j} = P(x_j = 1 | y=c) with Laplace smoothing over {0,1}
    # denominator = count_c + 2*alpha (Bernoulli)
    denom0 = count_c0 + 2.0 * alpha
    denom1 = count_c1 + 2.0 * alpha

    theta0_f0 = (count1_c0_f0 + alpha) / denom0
    theta0_f1 = (count1_c0_f1 + alpha) / denom0
    theta1_f0 = (count1_c1_f0 + alpha) / denom1
    theta1_f1 = (count1_c1_f1 + alpha) / denom1

    # Predict on the 2 test points using product of likelihoods (no logs to keep it simple)
    # Score_c(x) = prior_c * Π_j [ θ_{c,j}^xj * (1-θ_{c,j})^(1-xj) ]
    preds = [0.0, 0.0]
    for i in range(2):
        x0 = testing_x[i, 0]
        x1 = testing_x[i, 1]

        # For binary features held as floats in {0.0,1.0}
        # term(j) = θ if xj==1 else (1-θ)
        t0_f0 = theta0_f0 if x0 >= 0.5 else (1.0 - theta0_f0)
        t0_f1 = theta0_f1 if x1 >= 0.5 else (1.0 - theta0_f1)
        score0 = prior0 * t0_f0 * t0_f1

        t1_f0 = theta1_f0 if x0 >= 0.5 else (1.0 - theta1_f0)
        t1_f1 = theta1_f1 if x1 >= 0.5 else (1.0 - theta1_f1)
        score1 = prior1 * t1_f0 * t1_f1

        # Argmax over two classes
        pred = 1.0 if score1 >= score0 else 0.0
        preds[i] = pred

    # Verify predictions match provided testing_y exactly
    assert preds[0] == testing_y[0]
    assert preds[1] == testing_y[1]


if __name__ == '__main__':
    # A tiny, linearly separable toy dataset (binary features), kept minimal
    # Class 0 mostly zeros, Class 1 mostly ones
    training_x = [
        [0.0, 0.0],  # y=0
        [0.0, 1.0],  # y=0
        [0.0, 0.0],  # y=0
        [0.0, 0.0],  # y=0
        [0.0, 1.0],  # y=0
        [1.0, 1.0],  # y=1
        [1.0, 1.0],  # y=1
        [1.0, 0.0],  # y=1
        [1.0, 1.0],  # y=1
        [1.0, 1.0],  # y=1
    ]
    training_y = [
        0.0, 0.0, 0.0, 0.0, 0.0,  # first five are class 0
        1.0, 1.0, 1.0, 1.0, 1.0   # last five are class 1
    ]

    testing_x = [
        [0.0, 0.0],  # expect class 0
        [1.0, 1.0],  # expect class 1
    ]
    testing_y = [0.0, 1.0]

    assert verify_solution(training_x, training_y, testing_x, testing_y)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(training_x, training_y, testing_x, testing_y)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
