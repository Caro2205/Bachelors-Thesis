"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import numpy as np
from numpy.core.numeric import Infinity
import torch 
from torch import nn

import sys
sys.path.append('...')      
# Before run: replace ... with current directory path

from CoreLSTM.test_core_lstm import LSTM_Tester
from CoreLSTM.train_core_lstm_gestalten_mir_bd import LSTM_Trainer
from Data_Compiler.data_preparation import Preprocessor


"""
    Setup to train LSTM for walker. 

    Training method on class 'train_core_lstm_gestalten_mir_bd.py'
    
"""

def main():
    
    # LSTM parameters
    frame_samples = 1000
    train_window = 10
    testing_size = 100
    hidden_num = 100

    num_features = 15 

    # num_dimensions = 3
    num_dimensions = 6
    # num_dimensions = 7

    # noise = None
    # noise = 0.0005
    # noise = 0.001
    # noise = 0.0001
    noise = 0.00002

    # layerNorm = True
    layerNorm = False

    # Training parameters
    epochs = 2000
    mse=nn.MSELoss()
    # l1 = nn.SmoothL1Loss(reduction='sum', beta=0.0001)
    l1 = nn.SmoothL1Loss(reduction='sum', beta=0.001)
    loss_function=mse
    learning_rate=0.01
    weight_decay=0.0
    # weight_decay=0.0000001
    betas=(0.9, 0.999)
    # betas=(0.3, 0.4)

    # Init tools
    paths = []
    # Init tools
    data_asf_path = 'Data_Compiler/samples/S35T07.asf'
    data_amc_path = 'Data_Compiler/samples/S35T07.amc'
    
    # define model path
    model_name = model_save_path = f'{num_dimensions}_{num_features}_{hidden_num}_{loss_function}_{learning_rate}_{betas}_{weight_decay}_{train_window}_{epochs}'
    model_name += f'_nse{noise}' if noise is not None else ''
    model_name += '_lNORM' if layerNorm else ''
    model_name += '_corrVel'

    model_save_path = f'CoreLSTM/models/mod_{model_name}.pt'
    
    prepro = Preprocessor(num_features=num_features, num_dimensions=num_dimensions)
    trainer = LSTM_Trainer(
        loss_function, 
        learning_rate, 
        betas, 
        weight_decay, 
        train_window, 
        hidden_num,
        layerNorm,
        num_dimensions,
        num_features
    )
    tester = LSTM_Tester(mse, num_dimensions,
        num_features)

    
    with torch.no_grad():

        io_seq_s = []
        dt_test = []
        test_input = []

        # Preprocess data
        io_seq_p, dt_train_p, dt_test_p = prepro.get_LSTM_data_gestalten(
            data_asf_path, 
            data_amc_path, 
            frame_samples, 
            testing_size,
            train_window,
            noise
        )
        
        io_seq_s += [io_seq_p]

        test_input.append(dt_train_p[0,-train_window:])
        dt_test.append(dt_test_p)

    # Train LSTM
    losses = trainer.train(epochs, io_seq_s, model_save_path, prepro, noise)
    loss_path = f"CoreLSTM/testing_predictions/train_loss/{model_name}"
    trainer.plot_losses(losses, loss_path)
    torch.save(losses, f'{loss_path}.pt')


    # Test LSTM
    plot_path = f"CoreLSTM/testing_predictions/"
    tester.test(
        testing_size, 
        model_save_path, 
        plot_path, 
        model_name,
        test_input[0], 
        dt_test[0], 
        train_window,
        hidden_num, 
        layerNorm
    )    


if __name__ == "__main__":
    main()

