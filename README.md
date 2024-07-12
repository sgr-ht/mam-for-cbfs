# MAM-for-CBFS

This repository contains the code, the testing dataset for the multi-architecture models
and some trained multi-architecture models for the following paper.
> Suguru Horimoto, Keane Lucas, and Lujo Bauer. Approach for the optimization of machine learning models for calculating binary function similarity. In Proceedings of the 21st Conference on Detection of Intrusions and Malware & Vulnerability Assessment (DIMVA '24), 2024.

The code, the testing dataset and the models are available under the MIT License, and are dependent on the following repositories:
* "fastText for Python" (v0.9.2) from **[fastText](https://github.com/facebookresearch/fastText)** (Copyright (c) 2016-present, Facebook, Inc.)
* Code and preprocessed datasets from **[Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity)** (Copyright (c) 2019-2022 Cisco Talos)
* Binaries of "Normal dataset" from **[BinKit 2.0](https://github.com/SoftSec-KAIST/BinKit?tab=readme-ov-file)** (Copyright (c) 2020 Dongkwan Kim)

gnn.py in [train_and_test](./program/train_and_test) and Python programs in [core](./program/train_and_test/core) excluding mrr_recall_evaluation.py contain part of the code licensed under Apache License 2.0.
For more information, please refer to the license information in the Python programs.


## How to recreate a GNN+fastText model and run the testing
The easiest way to run the code is to install Docker, build docker images provided by [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity)
and download preprocessed "Dataset-1" from [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity).

At first, please create the directory structure of [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity). For example, you can create the directory structure by running the following script.
```
git clone https://github.com/Cisco-Talos/binary_function_similarity.git
```


* **recreate a fastText model for multi-architecture models**
	1. build "gnn-preprocessing" docker image based on the "README.md" under `Models/GGSNN-GMN/` of [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity)
	2. run the following script to run a docker container
		```
		docker run --name fastText_gnn_pre_forxarch \
		 -v $(pwd)/../../DBs:/input \
		 -v $(pwd)/Preprocessing/fastText_Dataset-1_forXarch:/output \
		 -v $(pwd)/Preprocessing/fastText_Dataset-1_forXarch:/training_data \
		-it gnn-preprocessing bash
		```
		
	3. install "fastText for Python" (v0.9.2) based on [fastText](https://github.com/facebookresearch/fastText) and pandas (1.3.5) in the docker container

	4. replace "gnn_preprocessing.py" under `/code/` in the docker container with [gnn_preprocessing_fastText.py](./program/preprocessing/gnn_preprocessing_fastText.py) 

	5. train a fastText model using the following script
		```
		docker exec /code/fastText_gnn_pre_forxarch gnn_preprocessing_fastText.py \
		-i /input/Dataset-1/features/training/acfg_disasm_Dataset-1_training \
		--training --t_mode fastText \
		-n 200 \
		-o /training_data/Dataset-1_training
		```

	6. preprocess functions using the following script
	   ### for training
	   ```	
	   docker exec fastText_gnn_pre_forxarch /code/gnn_preprocessing_fastText.py \
	   -i /input/Dataset-1/features/training/acfg_disasm_Dataset-1_training \
	   --t_mode fastText \
	   -n 200 \
	   -d /training_data/Dataset-1_training/fastText_model_dim200 \
	   -o /output/Dataset-1_training
	   ```	

	   ### for validation
	   ```	
	   docker exec fastText_gnn_pre_forxarch /code/gnn_preprocessing_fastText.py \
	   -i /input/Dataset-1/features/validation/acfg_disasm_Dataset-1_validation \
	   --t_mode fastText \
	   -n 200 \
	   -d /training_data/Dataset-1_training/fastText_model_dim200 \
	   -o /output/Dataset-1_validation
	   ```	

	   ### for testing
	   ```	
	   docker exec fastText_gnn_pre_forxarch /code/gnn_preprocessing_fastText.py \
	   -i /input/Dataset-1/features/testing/acfg_disasm_Dataset-1_testing \
	   --t_mode fastText \
	   -n 200 \
	   -d /training_data/Dataset-1_training/fastText_model_dim200 \
	   -o /output/Dataset-1_testing
	   ```	


* **recreate a GNN+fastText model as a multi-architecture model**
	1. build "gnn-neuralnetwork" docker image based on the "README.md" under `Models/GGSNN-GMN/` of [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity)
	2. run the following script to run a docker container
	   ```	
	   docker run --name gnn_forxarch \
	   -v $(pwd)/../../DBs:/input \
	   -v $(pwd)/NeuralNetwork:/output \
	   -v $(pwd)/Preprocessing:/preprocessing \
	   -it gnn-neuralnetwork bash
	   ```	
	
	3. replace Python programs under `/code/` in the docker container with [Python programs](./program/train_and_test) (gnn.py and Python programs in [core](./program/train_and_test/core))
	
	4. put [csv files and a directory](./program/train_and_test/validation_pairs_dataset/) into `DBs/Dataset-1/pairs/validation`

	5. run the following script to train a GNN+fastText model
	   ```	
	   docker exec  \
	   gnn_forxarch /code/gnn.py --train --num_epochs 200 \
	   --model_type embedding --training_mode pair \
	   --featuresdir /preprocessing/fastText_Dataset-1_forXarch \
	   --features_type fastText --dataset one \
	   -c /output/model_checkpoint_forXarch_fastText \
	   -o /output/Dataset-1_forXarch_fastText_$(date +'%Y-%m-%d') \
	   -f graph_func_fastText_dim_200.pickle
	   ```	



* **run the MRR10 and Recall@1 testing for multi-architecture models**
	1. put unzipped [csv files](./testing_pair_dataset) on Dataset-1 into `DBs/Dataset-1/pairs/testing/`
	
	2. run the following script for a GNN+fastText model
	   ```	
	   docker exec \
	   gnn_forxarch /code/gnn.py --test \
	   --model_type embedding --training_mode pair \
	   --featuresdir /preprocessing/fastText_Dataset-1_forXarch \
	   --features_type fastText --dataset one \
	   -c /output/model_checkpoint_forXarch_fastText \
	   -o /output/Dataset-1_forXarch_fastText_testing_$(date +'%Y-%m-%d') \
	   -f graph_func_fastText_dim_200.pickle	
	   ```
    
	3. the result is saved in "mrr_recall.csv" in the output directory, such as Dataset-1_forXarch_fastText_testing_$(date +'%Y-%m-%d')
	
	

* **run the AUC testing for multi-architecture models**
	1. put unzipped [csv files](./testing_pair_dataset) on Dataset-1 into `DBs/Dataset-1/pairs/testing/`

	2. run the following script to run a docker container
	   ```
	   docker run --name gnn_forxarch_AUC \
	   -v $(pwd)/../../DBs:/input \
	   -v $(pwd)/NeuralNetwork:/output \
	   -v $(pwd)/Preprocessing:/preprocessing \
	   -it gnn-neuralnetwork bash
	   ```	
	
	3. replace Python programs under `/code/` in the docker container with [Python programs](./program/train_and_test) (gnn.py and Python programs in [core](./program/train_and_test/core)).
	   After that, save "gnn_model_AUC_testing.py" as "gnn_model.py", and "config_AUC_testing.py" as "config.py" for AUC testing.

	4. run the following script for a GNN+fastText model
	   ```	
	   docker exec \
	   gnn_forxarch_AUC /code/gnn.py --test \
	   --model_type embedding --training_mode pair \
	   --featuresdir /preprocessing/fastText_Dataset-1_forXarch \
	   --features_type fastText --dataset one \
	   -c /output/model_checkpoint_forXarch_fastText \
	   -o /output/Dataset-1_forXarch_fastText_testing_$(date +'%Y-%m-%d') \
	   -f graph_func_fastText_dim_200.pickle	
	   ```		

	5. create `data/Dataset-1/` under the `Results` directory.

  	6. put "neg_testing_Dataset-1_sim.csv" and "pos_testing_Dataset-1_sim.csv" saved in the output directory, such as Dataset-1_forXarch_fastText_testing_$(date +'%Y-%m-%d'), into `Results/data/Dataset-1/`

	7. put [revised_AUC_and_similarity_plots.ipynb](./program/train_and_test/revised_AUC_and_similarity_plots.ipynb) into `Results/notebooks/`

  	8. run [revised_AUC_and_similarity_plots.ipynb](./program/train_and_test/revised_AUC_and_similarity_plots.ipynb)







## References

If you use the code in this repository, please consider 
citing the following papers:

```
@article{Bojanowski2017,
  author={Piotr Bojanowski and Edouard Grave and Armand Joulin and Tomas Mikolov},
  title={Enriching Word Vectors with Subword Information},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  pages={135--146}
}

@inproceedings{Marcelli2022,
  author={Andrea Marcelli and Mariano Graziano and Xabier Ugarte-Pedrero and Yanick Fratantonio and Mohamad Mansouri and Davide Balzarotti},
  title={How Machine Learning Is Solving the Binary Function Similarity Problem},
  booktitle={31st USENIX Security Symposium (USENIX Security 22)},
  year={2022},
  pages={2099--2116}
}

@article{kim2023,
  author={Dongkwan Kim and Eunsoo Kim and Sang Kil Cha and Sooel Son and Yongdae Kim},
  title={Revisiting Binary Code Similarity Analysis Using Interpretable Feature Engineering and Lessons Learned}, 
  journal={IEEE Transactions on Software Engineering}, 
  volume={49},
  number={4},
  year={2023},
  pages={1661-1682}
}

@inproceedings{Horimoto2024,
  author={Suguru Horimoto and Keane Lucas and Lujo Bauer},
  title={Approach for the optimization of machine learning models for calculating binary function similarity},
  booktitle={Proceedings of the 21st Conference on Detection of Intrusions and Malware & Vulnerability Assessment (DIMVA)},
  year={2024},
  pages={}
}
```

