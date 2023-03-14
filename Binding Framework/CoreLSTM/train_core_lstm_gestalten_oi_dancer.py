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
    Setup to train LSTM for dancer. 

    Training method on class 'train_core_lstm_gestalten_mir_bd.py'
    
"""

def main():
    
    # LSTM parameters
    frame_samples = 10000
    train_window = 20
    testing_size = 1000
    hidden_num = 300

    illusion = 'dancer'
    num_features = 15 

    mirrored = True
    # mirrored = False

    turn = 'turn35'

    both_directions = True
    # both_directions = False

    # num_dimensions = 3
    num_dimensions = 6
    # num_dimensions = 7

    # noise = None
    # noise = 0.0005
    # noise = 0.001
    noise = 0.0001
    # noise = 0.00001

    # layerNorm = True
    layerNorm = False

    # Training parameters
    epochs = 400
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
    dir_path = f"Data_Compiler/optical_illusions/{illusion}_data/"
    
    paths.append(dir_path + "frames10000_rot['0.9990', '0.0000', '0.0350', '0.0000']_2022Mar23104115_flatback_asarmr180" + ".pt")
    if both_directions:
        paths.append(dir_path + "frames10000_rot['0.9990', '0.0000', '-0.0350', '0.0000']_2022Mar23104121_flatback_asarmr180" + ".pt")

    if mirrored: 
        paths.append(dir_path + "frames10000_rot['0.9990', '0.0000', '0.0350', '0.0000']_2022Mar23104107_mirrored_flatback_asarmr180" + ".pt")
        if both_directions:  
            paths.append(dir_path + "frames10000_rot['0.9990', '0.0000', '-0.0350', '0.0000']_2022Mar23104100_mirrored_flatback_asarmr180" + ".pt")

    
    # define model path
    model_name = model_save_path = f'{num_dimensions}_{num_features}_{hidden_num}_{loss_function}_{learning_rate}_{betas}_{weight_decay}_{train_window}_{epochs}'
    model_name += f'_nse{noise}' if noise is not None else ''
    model_name += '_lNORM' if layerNorm else ''
    model_name += '_mirr' if mirrored else ''
    model_name += turn

    # model_name = "asarot" + model_name
    # model_name = "asarhi" + model_name
    # model_name = "asarse" + model_name
    # model_name = model_name + "asar180"

    # model_name = model_name + "_re"
    # model_name = model_name + "at2"
    model_name = model_name + "asar"
    # model_name = model_name + "attr_sep"
    # model_name = model_name + "testing"
    model_name = model_name + "difvel"
    # model_name = model_name + "_2"

    model_save_path = f'CoreLSTM/models/optical_illusions/{illusion}/mod_{model_name}.pt'
    
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

        for p in paths: 
            io_seq_p, dt_train_p, dt_test_p = prepro.get_LSTM_OI_data_gestalten(
                p, 
                frame_samples, 
                testing_size,
                train_window, 
                noise
            )
        
            io_seq_s += [io_seq_p]

            test_input.append(dt_train_p[0,-train_window:])
            dt_test.append(dt_test_p)

    # # Train LSTM
    losses = trainer.train(epochs, io_seq_s, model_save_path, prepro, noise)
    loss_path = f"CoreLSTM/testing_predictions/optical_illusions/{illusion}/train_loss/{model_name}"
    trainer.plot_losses(losses, loss_path)
    torch.save(losses, f'{loss_path}.pt')


    # Test LSTM
    plot_path = f"CoreLSTM/testing_predictions/optical_illusions/{illusion}/"
    if both_directions:
        tester.test(testing_size, model_save_path, plot_path, f'{model_name}_dir1', test_input[0], dt_test[0], train_window, hidden_num, layerNorm)
        tester.test(testing_size, model_save_path, plot_path, f'{model_name}_dir2', test_input[1], dt_test[1], train_window, hidden_num, layerNorm)
        if mirrored:
            tester.test(testing_size, model_save_path, plot_path, f'{model_name}_dir1_mirrored', test_input[2], dt_test[2], train_window, hidden_num, layerNorm)
            tester.test(testing_size, model_save_path, plot_path, f'{model_name}_dir2_mirrored', test_input[3], dt_test[3], train_window, hidden_num, layerNorm)       
    else:
        tester.test(testing_size, model_save_path, plot_path, model_name, test_input[0], dt_test[0], train_window, hidden_num, layerNorm)
        if mirrored:
            tester.test(testing_size, model_save_path, plot_path, model_name+'_mirrored', test_input[1], dt_test[1], train_window, hidden_num, layerNorm)

        


if __name__ == "__main__":
    main()


