##############################################################################
#                                                                            #
#  Code for the USENIX Security '22 paper:                                   #
#  How Machine Learning Is Solving the Binary Function Similarity Problem.   #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2019-2022 Cisco Talos                                       #
#                                                                            #
#  Permission is hereby granted, free of charge, to any person obtaining     #
#  a copy of this software and associated documentation files (the           #
#  "Software"), to deal in the Software without restriction, including       #
#  without limitation the rights to use, copy, modify, merge, publish,       #
#  distribute, sublicense, and/or sell copies of the Software, and to        #
#  permit persons to whom the Software is furnished to do so, subject to     #
#  the following conditions:                                                 #
#                                                                            #
#  The above copyright notice and this permission notice shall be            #
#  included in all copies or substantial portions of the Software.           #
#                                                                            #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,           #
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        #
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     #
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE    #
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION    #
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION     #
#  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           #
#                                                                            #
#  Gated Graph Sequence Neural Networks (GGSNN) and                          #
#    Graph Matching Networks (GMN) models implementation.                    #
#                                                                            #
#  This implementation contains code from:                                   #
#  https://github.com/deepmind/deepmind-research/blob/master/                #
#    graph_matching_networks/graph_matching_networks.ipynb                   #
#    licensed under Apache License 2.0                                       #
#                                                                            #
##############################################################################

# some parts of this code has been changed by sgr-ht in 2023-2024

import collections
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import time

from .build_dataset import *
from .build_model import *
from .model_evaluation import *
from .mrr_recall_evaluation import *

import logging
log = logging.getLogger('gnn')


def _it_check_condition(it_num, threshold):
    """
    Utility function to make the code cleaner.

    Args:
        it_num: the iteration number.
        threshold: threshold at which the condition must be verfied.

    Return:
        True if it_num +1 is a multiple of the threshold.
    """
    return (it_num + 1) % threshold == 0


class GNNModel:

    def __init__(self, config):
        """
        GNNModel initialization

        Args:
            config: global configuration
        """
        self._config = config
        self._model_name = self._get_model_name()

        # Set random seeds
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed + 1)
        return

    def _get_model_name(self):
        """Return the name of the model based to the configuration."""
        model_type = self._config['model_type']
        training_mode = self._config['training']['mode']
        model_name = "graph-{}-{}".format(model_type, training_mode)
        return model_name

    def _get_debug_str(self, accumulated_metrics):
        """Return a string with the mean of the input values"""
        metrics_to_print = {k: np.mean(v)
                            for k, v in accumulated_metrics.items()}
        info_str = ', '.join([' %s %.4f' % (k, v)
                              for k, v in metrics_to_print.items()])
        return info_str

    #def _create_network(self, batch_generator):
    def _create_network(self, batch_generator, is_training):
        """Build the model and set _tensors, _placeholders and _model."""
        # Automatically infer the node and edge features dim.
        ### Choose among pair or triplet training
        #if self._config['training']['mode'] == 'pair':
        #    training_it = batch_generator.pairs()
        #    first_batch_graphs, _ = next(training_it)
        #else:
        #    training_it = batch_generator.triplets()
        #    first_batch_graphs = next(training_it)
        
        _it = None
        if is_training and self._config['training']['mode'] == 'triplet':
            _it = batch_generator.triplets()
        else:
            _it = batch_generator.pairs()

        first_batch_graphs = None
        if is_training and self._config['training']['mode'] == 'pair':
            first_batch_graphs, _ = next(_it)
        else:
            first_batch_graphs = next(_it)


        # Set the feature dimensions
        node_feature_dim = first_batch_graphs.node_features.shape[-1]
        edge_feature_dim = first_batch_graphs.edge_features.shape[-1]
        log.info("node_feature_dim: %d", node_feature_dim)
        log.info("edge_feature_dim: %d", edge_feature_dim)

        self._tensors, self._placeholders, self._model = build_model(
            self._config, node_feature_dim, edge_feature_dim)
        return

    #def _model_initialize(self, batch_generator):
    def _model_initialize(self, batch_generator, is_training=True):
        """Create TF session, build the model, initialize TF variables"""
        tf.compat.v1.reset_default_graph()

        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

        self._session = tf.compat.v1.Session(config=session_conf)

        # Note: tf.compat.v1.set_random_seed sets the graph-level TF seed.
        # Results will be still different from one run to the other because
        # tf.random operations relies on an operation specific seed.
        tf.compat.v1.set_random_seed(self._config['seed'] + 2)

        # Create the TF NN
        #self._create_network(batch_generator)
        self._create_network(batch_generator, is_training=is_training)

        # Initialize all the variables
        init_ops = (tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
        self._session.run(init_ops)
        return

    def _create_tfsaver(self):
        """Create a TF saver for model checkpoint"""
        self._tf_saver = tf.compat.v1.train.Saver(max_to_keep = 100)
        checkpoint_dir = self._config['checkpoint_dir']
        self._checkpoint_path = os.path.join(checkpoint_dir, self._model_name)
        return

    def _restore_model(self):
        """Restore the model from the latest checkpoint"""
        checkpoint_dir = self._config['checkpoint_dir']
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        log.info("Loading trained model from: {}".format(latest_checkpoint))
        self._tf_saver.restore(self._session, latest_checkpoint)
        return

    def _run_evaluation(self, batch_generator):
        """
        Common operations for the dataset evaluation.
        Used with validation and testing.

        Args:
            batch_generator: provides batches of input data.
        """
        evaluation_metrics = evaluate(
            self._session,
            self._tensors['metrics']['evaluation'],
            self._placeholders,
            batch_generator)

        log.info("pair_auc (avg over batches): %.4f" %
                 evaluation_metrics['pair_auc'])

        # Print the AUC for each DB type
        xm_auc = 0
        for item in evaluation_metrics['pair_auc_list']:
            log.info("\t\t%s - AUC: %.4f", item[0], item[1])
            if item[0] == 'XM':
                xm_auc = item[1]

        return evaluation_metrics['pair_auc'], xm_auc

    def _initialize_bg(self, batch_generator):
        """Re-initialize the batch-generator"""
        if self._config['training']['mode'] == 'pair':
            return batch_generator.pairs()
        return batch_generator.triplets()

    def model_train(self, restore):
        """Run model training"""

        # Create a training and validation dataset
        training_set, validation_set = \
            build_train_validation_generators(self._config)

        # Model initialization
        self._model_initialize(training_set)

        # Model restoring
        self._create_tfsaver()
        if restore:
            self._restore_model()

        # Logging
        print_after = self._config['training']['print_after']

        log.info("Starting model training!")

        t_start = time.time()

        best_val_auc = 0
        accumulated_metrics = collections.defaultdict(list)

        # Iterates over the training data.
        it_num = 0


        
        # Create a val_mrr_recall dataset
        pos_val_mrr_recall_generator = build_val_mrr_recall_generator(
            self._config,
            self._config['validation_mrr_recall']['full_val_mrr_recall_inputs'][1])
        neg_val_mrr_recall_generator = build_val_mrr_recall_generator(
            self._config,
            self._config['validation_mrr_recall']['full_val_mrr_recall_inputs'][0])
            
        df_pos_val_mrr_recall = pd.read_csv(self._config['validation_mrr_recall']['full_val_mrr_recall_inputs'][1], index_col=0)
        df_neg_val_mrr_recall = pd.read_csv(self._config['validation_mrr_recall']['full_val_mrr_recall_inputs'][0], index_col=0)

        # calculate similarity for Early Stopping
        df_pos_sim, df_neg_sim = self._calculate_sim_for_earlystopping(pos_val_mrr_recall_generator, neg_val_mrr_recall_generator, df_pos_val_mrr_recall, df_neg_val_mrr_recall)      
        # calculate mrr and recall for Early Stopping
        mrr10, recall_1, recall_5, recall_10, _ = compute_mrr_and_recall(df_pos_val_mrr_recall, df_neg_val_mrr_recall, df_pos_sim, df_neg_sim)
        log.info('mrr10: %.4f, recall_1: %.4f, recall_5: %.4f, recall_10: %.4f', mrr10, recall_1, recall_5, recall_10)  

        # Let's check the starting values
        self._run_evaluation(validation_set)

        # Early Stopping parameter
        best_mrr10 = 0
        patience = 0
        MAX_PATIENCE = 20 

        for epoch_counter in range(self._config['training']['num_epochs']):
            log.info("Epoch %d", epoch_counter)

            # Batch generator in triplet or pair mode.
            training_batch_generator = self._initialize_bg(training_set)

            for training_batch in training_batch_generator:
                # TF Training step
                _, train_metrics = self._session.run([
                    self._tensors['train_step'],
                    self._tensors['metrics']['training']],
                    feed_dict=fill_feed_dict(
                        self._placeholders,
                        training_batch))

                # Accumulate over minibatches to reduce variance
                for k, v in train_metrics.items():
                    accumulated_metrics[k].append(v)

                # Logging
                if _it_check_condition(it_num, print_after):

                    # Print the AVG for each metric
                    info_str = self._get_debug_str(accumulated_metrics)
                    elapsed_time = time.time() - t_start
                    log.info('Iter %d, %s, time %.2fs' %
                             (it_num + 1, info_str, elapsed_time))

                    # Reset
                    accumulated_metrics = collections.defaultdict(list)

                it_num += 1

            # Run the evaluation at the end of each epoch:
            log.info("End of Epoch %d (elapsed_time %.2fs)",
                     epoch_counter, elapsed_time)

            log.info("Validation set")
            val_auc, xm_auc = self._run_evaluation(validation_set)
 
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                log.warning("best_val_auc: %.4f", best_val_auc)
       
            # calculate similarity for Early Stopping
            df_pos_sim, df_neg_sim = self._calculate_sim_for_earlystopping(pos_val_mrr_recall_generator, neg_val_mrr_recall_generator, df_pos_val_mrr_recall, df_neg_val_mrr_recall)      
            # calculate mrr and recall for Early Stopping
            mrr10, recall_1, recall_5, recall_10, _ = compute_mrr_and_recall(df_pos_val_mrr_recall, df_neg_val_mrr_recall, df_pos_sim, df_neg_sim)
            log.info('mrr10: %.4f, recall_1: %.4f, recall_5: %.4f, recall_10: %.4f', mrr10, recall_1, recall_5, recall_10)            
            
            # Early Stopping
            if mrr10 > best_mrr10:
                best_mrr10 = mrr10
                patience = 0 
                # Save the model when current mrr10 outperforms the best one
                self._tf_saver.save(
                   self._session,
                   self._checkpoint_path,
                   global_step=it_num)
                log.info("Model saved: {}".format(self._checkpoint_path))
            else:
                patience += 1
                if patience == MAX_PATIENCE:
                    log.warning("Early Stopping! Best Epoch: %d, BestMRR10: %.4f", epoch_counter - MAX_PATIENCE, best_mrr10)
                    break



        if self._session:
            self._session.close()
        return

    def model_validate(self):
        """Run model validation"""


        # Model initialization
        #self._model_initialize(training_set)
        self._model_initialize(training_set, is_training=False)

        # Model restoring
        self._create_tfsaver()
        self._restore_model()

        # Evaluate the validation set
        self._run_evaluation(validation_set)

        if self._session:
            self._session.close()
        return

    def model_test(self):
        """Testing the GNN model on a single CSV with function pairs"""

        batch_generator = build_testing_generator(
             self._config,
             self._config['testing']['full_tests_inputs'][0])

        self._model_initialize(batch_generator, is_training=False)

        # Model restoring
        self._create_tfsaver()
        self._restore_model()

        # Evaluate the full testing dataset
        for df_input_path, df_output_path in \
            zip(self._config['testing']['full_tests_inputs'],
                self._config['testing']['full_tests_outputs']):

            df = pd.read_csv(df_input_path, index_col=0)

            batch_generator = build_testing_generator(
                self._config,
                df_input_path)

            similarity_list = evaluate_sim(
                self._session,
                self._tensors['metrics']['evaluation'],
                self._placeholders,
                batch_generator)

            # Save the cosine similarity
            df['sim'] = similarity_list[:df.shape[0]]

            # Save the result to CSV
            #df.to_csv(df_output_path)
            #log.info("Result CSV saved to {}".format(df_output_path))

            if "neg_rank_testing_Dataset" in df_input_path:
                df_neg_rank_sim = df.copy()
                df_neg_rank_input = pd.read_csv(df_input_path, index_col=0) 
            elif "pos_rank_testing_Dataset" in df_input_path:
                df_pos_rank_sim = df.copy()
                df_pos_rank_input = pd.read_csv(df_input_path, index_col=0)


        # calculate MRR and Recall for testing dataset
        # extract model_name from outputdir
        model_name = os.path.basename(os.path.dirname(self._config['testing']['full_tests_outputs'][0]))
        # calculate MRR and Recall
        mrr10, recall_1, recall_5, recall_10, _ = compute_mrr_and_recall(df_pos_rank_input, df_neg_rank_input, df_pos_rank_sim, df_neg_rank_sim)
        # create dataframe for mrr_recall.csv
        columns = ['model_name',  'Recall@1',  'Recall@5',  'Recall@10', 'MRR@10' ]
        df_all_result = pd.DataFrame(columns=columns)
        new_row = [model_name, recall_1, recall_5, recall_10, mrr10]
        df_all_result.loc[len(df_all_result)] = new_row
        # save mrr_recall.csv
        output_path = os.path.dirname(self._config['testing']['full_tests_outputs'][0])
        df_all_result.to_csv(os.path.join(output_path, "mrr_recall.csv"))
        log.info("Result CSV saved to {}".format(output_path))




        if self._session:
            self._session.close()
        return



    def _calculate_sim_for_earlystopping(self, pos_val_mrr_recall_generator, neg_val_mrr_recall_generator, df_pos_val_mrr_recall, df_neg_val_mrr_recall):

        # create df_pos_sim
        similarity_list = evaluate_sim(
            self._session,
            self._tensors['metrics']['evaluation'],
            self._placeholders,
            pos_val_mrr_recall_generator)

        # Save the cosine similarity
        df_pos_sim = df_pos_val_mrr_recall.copy()
        df_pos_sim['sim'] = similarity_list[:df_pos_sim.shape[0]]


        # create df_neg_sim
        similarity_list = evaluate_sim(
            self._session,
            self._tensors['metrics']['evaluation'],
            self._placeholders,
            neg_val_mrr_recall_generator)

        # Save the cosine similarity
        df_neg_sim = df_neg_val_mrr_recall.copy()
        df_neg_sim['sim'] = similarity_list[:df_neg_sim.shape[0]]

        
        return df_pos_sim, df_neg_sim
        

