## Import the model with iree-import-tf
```shell
iree-import-tf --tf-import-type=savedmodel_v1 --tf-savedmodel-exported-names=predict /home/shenghao/iree/saved_model/mobilenet_v2/ -o iree_input.mlir
```

## Results or benchmark

- [x] MNIST (28x28), BATCH_SIZE=32, ITERS=50, 2 Dense layers 128-relu-10-softmax
Noop model
Predict Time (Noop): 0.016204414367675782
Training Time (Noop): 0.013857049942016602

Compiled model
Predict Time (Compiled): 0.001243915557861328
Training Time (Compiled): 0.0019531869888305665

----

