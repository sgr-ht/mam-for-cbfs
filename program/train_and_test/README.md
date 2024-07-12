## Note: Training process
* [Models](../../model) were trained with preprocessed "Dataset-1" from [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity) on Ubuntu 20.04 on an Amazon EC2 P3 instance (p3.2xlarge).
* We changed a random seed in [config.py](./core/config.py) and [graph_factory_training.py](./core/graph_factory_training.py) when training models. For example, we set random seed 12 in [config.py](./core/config.py) and [graph_factory_training.py](./core/graph_factory_training.py) when training GNN+BoW on random seed 12.
* If you change early stopping patience, please change the "MAX_PATIENCE" in [gnn_model.py](./core/gnn_model.py).
