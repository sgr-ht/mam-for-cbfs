- *Note*: When using [models](../model), Dataset-1_testing.zip and Dataset-BINKIT_testing.zip, you can reproduce the results of GNN+BoW, GNN+fastText, GMN+BoW and GMN+fastText in Table 1 through 4 in the following paper:
> Suguru Horimoto, Keane Lucas, and Lujo Bauer. Approach for the optimization of machine learning models for calculating binary function similarity. In Proceedings of the 21st Conference on Detection of Intrusions and Malware & Vulnerability Assessment (DIMVA '24), 2024.



## Download testing datasets

Please download model_testingdataset.zip from [KiltHub](https://kilthub.cmu.edu/articles/dataset/Trained_models_and_testing_datasets_used_in_Approach_for_the_optimization_of_machine_learning_models_for_calculating_binary_function_similarity_/26042788), which contains zip files for testing on Dataset-1 and Dataset-BINKIT as follows:

```
File name                              SHA256
Dataset-1_testing.zip                  5cfda2acf3245f041720430f4995e7b843c63d1d15c60176ba19663512420a3a
Dataset-BINKIT_testing.zip             f76ef8c93564393bd4b3909154afb356e7d4b61fc8e9470ffc567b270e1b0996
```


## Run testing on Dataset-1
**Note**: In order to run testing on Dataset-1, please follow these steps after reading or executing [**How to recreate a GNN+fastText model and run the testing**](../).
1. unzip Dataset-1_testing.zip, which contains csv files as follows:
```
File name                              SHA256
neg_rank_testing_Dataset-1.csv         4755b5975e4c39851fedfd5da8b11f41ac342e20129d55e44a034c74fa3bbeef    
neg_testing_Dataset-1.csv              3cfa98e587e6b3143d5b37d5477f6cc74232e5f332f0fe7193dcdd34abca6b6e
pos_rank_testing_Dataset-1.csv         fad1355cb7a790ba7efd87babd8962225c468616b3fb355db7d859e836c18af4
pos_testing_Dataset-1.csv              9bf82887007c14f7c1cb38f039c2591b8dbcc8ac1a4fce6aacec641fd1425bc0
```

2. put the unzipped csv files into `DBs/Dataset-1/pairs/testing/`

3. run testing based on ["**run the MRR10 and Recall@1 testing for multi-architecture models**"](../) or ["**run the AUC testing for multi-architecture models**"](../) 


## Run testing on Dataset-BINKIT
**Note**: In order to run testing on Dataset-BINKIT, please follow these steps after reading or executing [**How to recreate a GNN+fastText model and run the testing**](../) .

- **preprocessing functions on Dataset-BINKIT**

1. unzip Dataset-BINKIT_testing.zip, which contains csv files and a json file as follows:
```
File name                              SHA256
neg_rank_testing_Dataset-BINKIT.csv    803928d547c58b42b7b52cbe82b91c6eadefc51dc4f807974b9f1581e2e70888
neg_testing_Dataset-BINKIT.csv         0ab4787def5f072c9e6a8d365e3c8e951cd419e04972209004bb9a23f3fa3b44
pos_rank_testing_Dataset-BINKIT.csv    ec873488fd02c3d953dca7fa5fa9c7ddb4c8ca94d01fd19e7c3bbdee2d9e6d96
pos_testing_Dataset-BINKIT.csv         cf955cfbeb09f74be017b478127a27b0ddf478361c8586cdf81eec0e9ad481e1
selected_testing_Dataset-BINKIT.json   061598fe04185eb15d293f69ed94999e0888d6c0dc32e61fd65ebd4e17157f0e
testing_Dataset-BINKIT.csv             b8ed66df92cb62d7befaf5f1c22bf6c5904e6b3f2c63e0f783fd41e36d8104d4
```

2. put testing_Dataset-BINKIT.csv into `DBs/Dataset-1/`, and put the other csv files into `DBs/Dataset-1/pairs/testing-BINKIT/`

3. put "selected_testing_Dataset-BINKIT.json" into `DBs/Dataset-1/features/testing-BINKIT/`

4. download binaries of "Normal dataset" from [BinKit 2.0](https://github.com/SoftSec-KAIST/BinKit?tab=readme-ov-file). As for binaries compiled for MIPS architecture, please use binaries compiled for **MIPS (Big Endian)**.
    ```
    File name            SHA256
    normal_dataset.7z    a2a4ceb717d0a1f8b0324b7b231a85be254d5c8619f4674408b6038eb3cbc497        
    ```

5. unzip "normal_dataset.7z" and put binaries into `Binaries/`. Please put binaries in the following directory structure:
    ```
    Binaries/BinKit2/package-name/binary-name
    ```
   For example, please put binaries of "gsl" as follows:
    ```
    Binaries/BinKit2/gsl/gsl-2.5_clang-5.0_arm_32_O0_libgsl.so.23.1.0
    ```

6. generate IDB files from the binaries in `DBs/Dataset-1/testing_Dataset-BINKIT.csv`. You can generate IDB files automatically using "generate_idbs.py" from [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity).

7. put the IDB files in the following directory structure:
    ```
    IDBs/BinKit2/package-name/binary-name
    ```
    For example, please put IDB files of "gsl" as follows:
    ```
    IDBs/BinKit2/gsl/gsl-2.5_clang-5.0_arm_32_O0_libgsl.so.23.1.0.i64
    ```

8. preprocess the functions of the IDB files using "cli_acfg_disasm.py", "IDA_acfg_disasm.py" from [Binary Function Similarity](https://github.com/Cisco-Talos/binary_function_similarity) and `DBs/Dataset-1/features/testing-BINKIT/selected_testing_Dataset-BINKIT.json`. After using the Python programs, please make sure that "acfg_disasm_Dataset-BINKIT_testing" directory is made in "DBs/Dataset-1/features/testing-BINKIT/".

9. preprocess functions in `DBs/Dataset-1/features/testing-BINKIT/acfg_disasm_Dataset-BINKIT_testing/` using a docker container from "gnn-preprocessing" (e.g., fastText_gnn_pre_forxarch). For example, the following script is used for preprocessing the functions using fastText_gnn_pre_forxarch.
    ```	
    docker exec fastText_gnn_pre_forxarch /code/gnn_preprocessing_fastText.py \
    -i /input/Dataset-1/features/testing-BINKIT/acfg_disasm_Dataset-BINKIT_testing \
	--t_mode fastText \
	-n 200 \
	-d /training_data/Dataset-1_training/fastText_model_dim200 \
	-o /output/Dataset-BINKIT_testing
    ```


- **run the MRR10 and Recall@1 testing on Dataset-BINKIT**
1. run the following script to run a docker container
    ```
	docker run --name gnn_forxarch_BINKIT \
	-v $(pwd)/../../DBs:/input  \
	-v $(pwd)/NeuralNetwork:/output \
	-v $(pwd)/Preprocessing:/preprocessing \
	-it gnn-neuralnetwork bash
	```	
	
2. replace Python programs in the docker container with [gnn.py and Python programs in core](../program/train_and_test). After that, save "config_BINKIT_testing.py" as "config.py" for the MRR10 and Recall@1 testing on Dataset-BINKIT.

3. run the following script for a GNN+fastText model
    ```	
	docker exec \
	gnn_forxarch_BINKIT /code/gnn.py --test \
	--model_type embedding --training_mode pair \
	--featuresdir /preprocessing/fastText_Dataset-1_forXarch \
	--features_type fastText --dataset one \
	-c /output/model_checkpoint_forXarch_fastText \
	-o /output/Dataset-BINKIT_forXarch_fastText_testing_$(date +'%Y-%m-%d') \
	-f graph_func_fastText_dim_200.pickle
	```		
4. the result is saved in "mrr_recall.csv" in the output directory, such as Dataset-BINKIT_forXarch_fastText_testing_$(date +'%Y-%m-%d')
	





- **run the AUC testing on Dataset-BINKIT**

1. run the following script to run a docker container
    ```
	docker run --name gnn_forxarch_BINKIT_AUC \
	-v $(pwd)/../../DBs:/input  \
	-v $(pwd)/NeuralNetwork:/output \
	-v $(pwd)/Preprocessing:/preprocessing \
	-it gnn-neuralnetwork bash
	```	
	
2. replace Python programs in the docker container with [gnn.py and Python programs in core](../program/train_and_test). After that, save "gnn_model_AUC_testing.py" as "gnn_model.py", and "config_BINKIT_AUC_testing.py" as "config.py" for the AUC testing on Dataset-BINKIT.


3. run the following script for a GNN+fastText model
```	
docker exec \
gnn_forxarch_BINKIT_AUC /code/gnn.py --test \
--model_type embedding --training_mode pair \
--featuresdir /preprocessing/fastText_Dataset-1_forXarch \
--features_type fastText --dataset one \
-c /output/model_checkpoint_forXarch_fastText \
-o /output/Dataset-BINKIT_forXarch_fastText_testing_$(date +'%Y-%m-%d') \
-f graph_func_fastText_dim_200.pickle	
```

4. create `data/Dataset-BINKIT/` under the `Results` directory.

5. put "neg_testing_Dataset-BINKIT_sim.csv" and "pos_testing_Dataset-BINKIT_sim.csv" saved in the output directory, such as Dataset-BINKIT_forXarch_fastText_testing_$(date +'%Y-%m-%d'), into `Results/data/Dataset-BINKIT/`

6. put [revised_AUC_and_similarity_plots.ipynb](./program/train_and_test/revised_AUC_and_similarity_plots.ipynb) into `Results/notebooks/`

7. run [revised_AUC_and_similarity_plots.ipynb](./program/train_and_test/revised_AUC_and_similarity_plots.ipynb)
