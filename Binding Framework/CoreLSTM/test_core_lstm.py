"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import torch
from torch import nn
import matplotlib.pyplot as plt

import sys
sys.path.append('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')
# Before run: replace ... with current directory path

from CoreLSTM.core_lstm import CORE_NET
from Data_Compiler.data_preparation import Preprocessor
from Data_Compiler.skeleton_renderer import SKEL_RENDERER
from Data_Compiler.optical_illusions.cube_renderer import CUBE_RENDERER


class LSTM_Tester(): 

    """
        Class to test core LSTM model. 

        Call function test with the following parameters:
            > num_predictions: How many time steps should be predicted?
            > model_path: Where to find the model to test?
            > plot_path: Where to save the testing graphics?
            > model_name: What's the name of the model to test?
            > test_input: Initial time steps to perform teacher forcing on
            > test_target: Target values for prediction time steps
            > train_window: length of test_input
            > hidden_num, layer_norm: model parameters
    """

    def __init__(self, loss_function, num_dimensions,num_features):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'DEVICE TrainM: {self.device}')

        self._loss_function = loss_function
        self.num_dimensions = num_dimensions
        self.num_features = num_features
        self.renderer = SKEL_RENDERER()


    def predict(self, num_predictions, model, test_input, test_target, train_window): 
        prediction_error = []
        state = model.init_hidden(train_window)
        state_scale = 0.9
        for i in range(num_predictions):
            seq = test_input[-train_window:]

            with torch.no_grad():
                if i>0:
                    loss = self._loss_function(test_input[-1], test_target[0,i]).item()
                    prediction_error.append(loss.to('cpu'))

                state = (state[0] * state_scale, state[1] * state_scale)
                new_prediction, state = model(seq, state)
                test_input = torch.cat((test_input, new_prediction[-1].reshape(1,45)), 0)

        predictions = test_input[-num_predictions:].reshape(num_predictions, 15, 3)
        # self.renderer.render(predictions)

        return predictions, prediction_error

    
    def predict(self, num_predictions, model, plot_path, model_name, test_input, test_target, train_window): 
        test_input = test_input.to(self.device)
        test_target = test_target.to(self.device)
        prediction_error = []
        prediction_error_position = []
        prediction_error_direction = []
        prediction_error_magnitude = []
        state = model.init_hidden(train_window)
        # state_scale = 0.9

        for i in range(num_predictions):
            seq = test_input[-train_window:]

            with torch.no_grad():
                if i>0:
                    loss = self._loss_function(test_input[-1], test_target[0,i]).item()
                    prediction_error.append(loss)

                    last_test_input = test_input[-1].view(self.num_features,self.num_dimensions)
                    current_test_target = test_target[0,i].view(self.num_features,self.num_dimensions)

                    loss = self._loss_function(last_test_input[:,:3], current_test_target[:,:3]).item()
                    prediction_error_position.append(loss)

                    if self.num_dimensions >3:
                        loss = self._loss_function(last_test_input[:,3:6], current_test_target[:,3:6]).item()
                        prediction_error_direction.append(loss)

                        if self.num_dimensions >6:
                            loss = self._loss_function(last_test_input[:,-1], current_test_target[:,-1]).item()
                            prediction_error_magnitude.append(loss)


                # state = (state[0] * state_scale, state[1] * state_scale)
                new_prediction, state = model(seq, state)
                test_input = torch.cat((test_input, new_prediction[-1].reshape(1,self.num_dimensions*self.num_features)), 0)

        predictions = test_input[-num_predictions:].reshape(num_predictions, self.num_features, self.num_dimensions)
        torch.save(predictions, f'{plot_path}/predictions/dim{model_name}.pt')

        predictions = predictions.to('cpu')

        if self.num_dimensions == 3:
            self.renderer.render(predictions, None, None, False)

        else:    
            pos = predictions[:,:,:3]
            dir = predictions[:,:,3:6]
            if self.num_dimensions > 6:
               mag = predictions[:,:,-1]
            else: 
               mag = torch.ones(pos.size()[0], pos.size()[1], 1)
            
            self.renderer.render(pos, dir, mag, True)

        return predictions, prediction_error, prediction_error_position, prediction_error_direction, prediction_error_magnitude

    
    def plot_pred_error(self, errors, plot_path, model_name, gest):
        fig = plt.figure(figsize=(6, 4))
        axes = fig.add_axes([0.13, 0.12, 0.8, 0.8]) 
        axes.plot(errors, 'r')
        axes.grid(False)
        axes.set_xlabel('time steps in testing', size=14)
        axes.set_ylabel('prediction error', size=14)
        axes.set_title('Prediction error (testing)', size = 15, fontweight='bold')
        plt.savefig(f'{plot_path}plots/dim{model_name}_{gest}.png')
        plt.savefig(f'{plot_path}plots/dim{model_name}_{gest}.pdf')
        plt.close()

    
    def test(self, num_predictions, model_path, plot_path, model_name, test_input, test_target, train_window, hidden_num, layer_norm):
        model = CORE_NET(input_size=self.num_features*self.num_dimensions, hidden_layer_size=hidden_num, layer_norm=layer_norm)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        print(model)

        pred, pred_err, pred_err_pos, pred_err_dir, pred_err_mag  = self.predict(num_predictions, model, plot_path, model_name, test_input, test_target, train_window)

        self.plot_pred_error(pred_err_pos, plot_path, model_name, 'position')
        if self.num_dimensions >=6:
            self.plot_pred_error(pred_err, plot_path, model_name, 'overall performance')
            if self.num_dimensions == 6:
                self.plot_pred_error(pred_err_dir, plot_path, model_name, 'velocity')
            elif self.num_dimensions == 7:
                self.plot_pred_error(pred_err_dir, plot_path, model_name, 'direction')
                self.plot_pred_error(pred_err_mag, plot_path, model_name, 'magnitude')


###############################################################################
###############################################################################

"""
    Setup to test LSTM for walker. 
    
"""


def main():
    # LSTM parameters
    frame_samples = 1000
    train_window = 10
    testing_size = 100
    hidden_num = 200

    num_features = 15
    num_dimensions = 6
    # num_dimensions = 7
    # num_dimensions = 3

    noise = None
    # noise = 0.0001
    
    # layerNorm = True
    layerNorm = False

    # Training parameters
    epochs = 2
    mse=nn.MSELoss()
    loss_function=nn.MSELoss()
    learning_rate=0.01
    momentum=0.9
    weight_decay=0.0

    # Init tools
    data_asf_path = 'Data_Compiler/samples/S35T07.asf'
    data_amc_path = 'Data_Compiler/samples/S35T07.amc'

    # define model path
    model_name = model_save_path = f'{num_dimensions}_{num_features}_{hidden_num}_{loss_function}_{learning_rate}_{weight_decay}_{train_window}_{epochs}'
    model_name += f'_nse{noise}' if noise is not None else ''
    model_name += '_lNORM' if layerNorm else ''

    # model_name = "testing" + model_name

    model_save_path = f'CoreLSTM/models/mod_{model_name}.pt'

    # Init tools
    prepro = Preprocessor(num_features=num_features, num_dimensions=num_dimensions)
    tester = LSTM_Tester(loss_function, num_dimensions,
        num_features)
    
    with torch.no_grad():
        # Preprocess data
        io_seq, dt_train, dt_test = prepro.get_LSTM_data_gestalten(
            data_asf_path, 
            data_amc_path, 
            frame_samples, 
            testing_size,
            train_window,
            noise
        )

    test_input = dt_train[0,-train_window:]

    # Test LSTM
    plot_path = f"CoreLSTM/testing_predictions/"
    tester.test(
        testing_size, 
        model_save_path, 
        plot_path, 
        model_name,
        test_input, 
        dt_test, 
        train_window,
        hidden_num, 
        layerNorm
    )


if __name__ == "__main__":
    main()
