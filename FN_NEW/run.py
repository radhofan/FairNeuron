import argparse
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

from utils.transform_dataset import transform_dataset,transform_dataset_credit,transform_dataset_census
from Evaluate import get_metrics, train_and_evaluate
from FairNeuron import Fixate_with_val, Fixate_with_val_rand


class DataClass():
    def __init__(self,df,dataset) -> None:

        if dataset=='compas':
            df_binary, Y, S, Y_true = transform_dataset(df)
            Y = Y.to_numpy()    
            self.l_tensor = tf.convert_to_tensor(Y_true.to_numpy().reshape(-1, 1).astype(np.float32))
            self.threshold=4
        elif dataset=='credit':
            df_binary, Y, S, Y_true = transform_dataset_credit(df)
            self.l_tensor = tf.convert_to_tensor(Y.reshape(-1, 1).astype(np.float32))
            self.threshold=0.5
        else:
            df_binary, Y, S, Y_true = transform_dataset_census(df)
            self.l_tensor = tf.convert_to_tensor(Y.reshape(-1, 1).astype(np.float32))
            self.threshold=0.5
        self.x_tensor = tf.convert_to_tensor(df_binary.to_numpy().astype(np.float32))
        self.y_tensor = tf.convert_to_tensor(Y.reshape(-1, 1).astype(np.float32))
        self.s_tensor = tf.convert_to_tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray().astype(np.float32))
        
        # Create dataset as tuple of tensors
        self.dataset = (self.x_tensor, self.y_tensor, self.l_tensor, self.s_tensor)
        
        base_size = len(self.x_tensor) // 10
        split = [7 * base_size, 1 * base_size, len(self.x_tensor) - 8 * base_size]
        
        # Manual split for TensorFlow
        train_indices = slice(0, split[0])
        val_indices = slice(split[0], split[0] + split[1])
        test_indices = slice(split[0] + split[1], None)
        
        self.train_dataset = (
            self.x_tensor[train_indices],
            self.y_tensor[train_indices],
            self.l_tensor[train_indices],
            self.s_tensor[train_indices]
        )
        
        self.val_dataset = (
            self.x_tensor[val_indices],
            self.y_tensor[val_indices],
            self.l_tensor[val_indices],
            self.s_tensor[val_indices]
        )
        
        self.test_dataset = (
            self.x_tensor[test_indices],
            self.y_tensor[test_indices],
            self.l_tensor[test_indices],
            self.s_tensor[test_indices]
        )
        
        self.x_train_tensor = self.train_dataset[0]
        self.y_train_tensor = self.train_dataset[1]
        self.l_train_tensor = self.train_dataset[2]
        self.s_train_tensor = self.train_dataset[3]
        self.global_results=[]


def run(dataset,inputpath,outputpath,epoch,BATCH_SIZE,rand):
    device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
    BATCH_SIZE=128
    file_name='{}_epoch{}_{}'.format(dataset,epoch,int(time.time()))
    print(os.path.join(outputpath,file_name))
    if dataset=='credit':
        df=pd.read_csv(inputpath,sep=' ')
    else:
        df=pd.read_csv(inputpath)
    data_class = DataClass(df,dataset)

    # Create TensorFlow data loaders
    train_loader = tf.data.Dataset.from_tensor_slices(data_class.train_dataset).batch(BATCH_SIZE).shuffle(buffer_size=1000)
    val_loader = tf.data.Dataset.from_tensor_slices(data_class.val_dataset).batch(BATCH_SIZE)
    test_loader = tf.data.Dataset.from_tensor_slices(data_class.test_dataset).batch(BATCH_SIZE)

    ori_start=time.time()
    threshold = 4

    net, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=data_class.x_tensor.shape[1],
                                        grl_lambda=50)

    ori_end=time.time()
    ori_cost_time=ori_end-ori_start
    print('time costs:{} s'.format(ori_cost_time))

    result = get_metrics(results, threshold, 0)
    data_class.global_results.append(result)
    net_nodrop, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=data_class.x_tensor.shape[1],
                                        grl_lambda=0,dataset='compas_nodrop')
    result = get_metrics(results, threshold, 0)
    data_class.global_results.append(result)

    if rand:
        Fixate_with_val_rand(net,data_class,epoch=epoch,BATCH_SIZE=BATCH_SIZE)
    else:
        Fixate_with_val(net,data_class,epoch=epoch,BATCH_SIZE=BATCH_SIZE)
    res = pd.DataFrame(data_class.global_results)
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    res.to_csv(os.path.join(outputpath,file_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices={'compas','census','credit'},default='compas')
    parser.add_argument('--epoch',default=10)
    parser.add_argument('--batch-size',default=128,dest='batchsize')
    parser.add_argument('--input-path',default='../data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv',dest='inputpath')
    parser.add_argument('--save-dir',default='./results',dest='outputpath')
    parser.add_argument('--rand',action='store_true',dest='rand')
    args=parser.parse_args()

    run(dataset=args.dataset,inputpath=args.inputpath,outputpath=args.outputpath,epoch=args.epoch,BATCH_SIZE=args.batchsize,rand=args.rand)