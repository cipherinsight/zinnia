# npbench_deep_learning_resnet_resnet

Category: `needs_gadget: sqrt`

Batch normalization needs standard deviation, another sqrt(std+eps), and real division; cannot omit normalization or supply its values as witnesses.

as_written: needs gadget

kernel.py:28: std = np.std(x, axis=0, keepdims=True). Batch normalization needs standard deviation, another sqrt(std+eps), and real division; cannot omit normalization or supply its values as witnesses.

rewrite: needs gadget

Batch normalization needs standard deviation, another sqrt(std+eps), and real division; cannot omit normalization or supply its values as witnesses.

