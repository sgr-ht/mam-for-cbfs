- *Note*:  GNN+BoW models, GMN+BoW models, GNN+fastText models and GMN+fastText models were trained with preprocessed "Dataset-1" from [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity) on Ubuntu 20.04 on an Amazon EC2 P3 instance (p3.2xlarge).


## How to use GNN+fastText models and GMN+fastText models
**Note: The link to the repository on KiltHub will be available soon.**
1. Please download model_testingdataset.zip from [KiltHub](https://kilthub.cmu.edu/), which contains the fastText model, GNN+fastText models and GMN+fastText models as follows:

```
File name 			SHA256
fastText_model_dim200 		c5163a9a976324c730924e40e2828b667aa00513934b92954385e7f70eb4efed
GNN+fastText.zip		59c267b850ab9b8a449f03a10def2774a2bffd1db9598fb345b2e1a90853e55b
GMN+fastText.zip		9b7f4f3d1ccbe6204e49709b2e7a142dbc824593f0b7775982e18517ebfeb248
```

2. unzip GNN+fastText.zip and GMN+fastText.zip, which contain models trained with random seed 11 to 15

3. run the following steps for testing on Dataset-1, and run the steps in ["**Run testing on Dataset-BINKIT**"](../testing_pair_dataset) for testing on Dataset-BINKIT
	

## Run testing on Dataset-1
**Note**: In order to run testing on Dataset-1, please follow these steps after reading or executing [**How to recreate a GNN+fastText model and run the testing**](../).

* **run testing by GNN+fastText models**

	1. create a docker container of "gnn-preprocessing" and preprocess testing functions in Dataset-1 based on ["**recreate a fastText model for multi-architecture models**"](../)

	2. build "gnn-neuralnetwork" docker image based on [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity)

	3. run the following script to run a docker container
	   ```	
	   docker run --name gnn_forxarch \
	   -v $(pwd)/../../DBs:/input  \
	   -v $(pwd)/NeuralNetwork:/output \
	   -v $(pwd)/Preprocessing:/preprocessing \
	   -it gnn-neuralnetwork bash
	   ```
	4. put a model directory (e.g., model_checkpoint_forXarch_epoch200_seed11_fastText_embedding_EP15MRR) into `Models/GGSNN-GMN/NeuralNetwork/`

	5. run testing based on ["**run the MRR10 and Recall@1 testing for multi-architecture models**"](../) or ["**run the AUC testing for multi-architecture models**"](../). The following script is an example of running the MRR10 and Recall@1 testing for multi-architecture models.
	   ```	
	   docker exec \
	   gnn_forxarch /code/gnn.py --test \
	   --model_type embedding --training_mode pair \
	   --featuresdir /preprocessing/fastText_Dataset-1_forXarch \
	   --features_type fastText --dataset one \
	   -c /output/model_checkpoint_forXarch_epoch200_seed11_fastText_embedding_EP15MRR \
	   -o /output/Dataset-1_forXarch_seed11_fastText_embedding_EP15MRR_testing_$(date +'%Y-%m-%d') \
	   -f graph_func_fastText_dim_200.pickle	
	   ```	


* **run testing by GNN+BoW models**

  	1. create a docker container using the following script after following steps i to v in "**run testing by GNN+fastText models**"
		```
		docker run --name bow_gnn_pre_forxarch \
		-v $(pwd)/../../DBs:/input  \
		-v $(pwd)/Preprocessing/BoW_Dataset-1_forXarch:/output \
		-v $(pwd)/Preprocessing/BoW_Dataset-1_forXarch:/training_data \
		-it gnn-preprocessing bash
		```
	2. replace "gnn_preprocessing.py" under `/code/` in the docker container with ["gnn_preprocessing_fastText.py"](../program/preprocessing/gnn_preprocessing_fastText.py)

  	3. create a "Dataset-1_training" directory in `Preprocessing/BoW_Dataset-1_forXarch/`
  	 
  	4. put "opcodes_dict.json" in [GNN+BoW](./GNN+BoW) into `Preprocessing/BoW_Dataset-1_forXarch/Dataset-1_training/`

	5. run the following script to preprocess testing functions in Dataset-1
	   ```	
	   docker exec bow_gnn_pre_forxarch /code/gnn_preprocessing_fastText.py \
	   -i /input/Dataset-1/features/testing/acfg_disasm_Dataset-1_testing \
	   --t_mode BoW \
	   -n 200 \
	   -d /training_data/Dataset-1_training/opcodes_dict.json \
	   -o /output/Dataset-1_testing
	   ```
    
	6. put a model directory (e.g., model_checkpoint_forXarch_epoch200_seed11_embedding_EP15MRR in [GNN+BoW](./GNN+BoW)) into `Models/GGSNN-GMN/NeuralNetwork/`

	7. for the MRR10 and Recall@1 testing, please run the following script, which is an example of running the MRR10 and Recall@1 testing for multi-architecture models.
	   ```	
	   docker exec \
	   gnn_forxarch /code/gnn.py --test \
	   --model_type embedding --training_mode pair \
	   --featuresdir /preprocessing/BoW_Dataset-1_forXarch \
	   --features_type opc --dataset one \
	   -c /output/model_checkpoint_forXarch_epoch200_seed11_embedding_EP15MRR \
	   -o /output/Dataset-1_forXarch_seed11_BoW_embedding_EP15MRR_testing_$(date +'%Y-%m-%d') \
	   -f graph_func_dict_opc_200.pickle
	   ```	

	8. the result of the MRR10 and Recall@1 testing is saved in "mrr_recall.csv" in the output directory, such as Dataset-1_forXarch_seed11_BoW_embedding_EP15MRR_testing_$(date +'%Y-%m-%d')

  	9. for the AUC testing, please follow the steps i to iii in ["**run the AUC testing for multi-architecture models**"](../), and run the following script, which is an example of running the AUC testing for multi-architecture models.   
	   ```	
	   docker exec \
	   gnn_forxarch_AUC /code/gnn.py --test \
	   --model_type embedding --training_mode pair \
	   --featuresdir /preprocessing/BoW_Dataset-1_forXarch \
	   --features_type opc --dataset one \
	   -c /output/model_checkpoint_forXarch_epoch200_seed11_embedding_EP15MRR \
	   -o /output/Dataset-1_forXarch_seed11_BoW_embedding_EP15MRR_testing_$(date +'%Y-%m-%d') \
	   -f graph_func_dict_opc_200.pickle
	   ```	

	10. To run [revised_AUC_and_similarity_plots.ipynb](../program/train_and_test/revised_AUC_and_similarity_plots.ipynb), please follow the steps v to viii in ["**run the AUC testing for multi-architecture models**"](../)






