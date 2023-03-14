"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

from torch import nn

from Testing.testing_module.general_tester import TESTER
from Testing.testing_module.TESTING_statistical_evaluation_abstract import TEST_STATISTICS


"""
    Class for defining parameters of experiment for walker data. 

    In here, all basic parameters are set to their default values. 
    If a parameter is not defined as 'changed_parameter', this default value is used in all 
    experiment trials. 
    If a parameter is defined as 'changed_parameter', its value is changed for all 
    experiment trials, i.e. in every trial its value is set to the next value in tested_values. 
    Parameters for which a testing is implemented can be set as 'changed_parameter').    

    Method 'perform_experiment' is to call from separate experiment classes. 
        -> defines default parameters
        -> performes all trials for different values for 'changed_parameter'
        -> evaluates experiment by creating comparing plots over all trials 

"""


class EXPERIMENT_INTERFACE(TESTER): 

    def __init__(self, num_features, num_observations, num_dimensions, experiment_name):
        super().__init__(num_features, num_observations, num_dimensions, experiment_name)
        self.stats = TEST_STATISTICS(self.num_features, self.num_observations, self.num_dimensions)
        print('Initialized experiment.')


    def perform_experiment(self, 
            sample_nums, 
            modification, 
            structure,
            changed_parameter, 
            tested_values, 
            rotation_type,
            distractors, 
            temperature): 


        #########################################################################################
        #### Experiment parameters ####
        #########################################################################################

        # NOTE: set manually, for all parameters that are not tested

        ### Tuning parameters       #############################################################
        # length of tuning horizon:
        tuning_length = 10
        # number of tuning cycles performed for every frame
        num_tuning_cycles = 1


        ### Tuning structure        #############################################################
        structure = structure


        ### Initial Binding Matrix  #############################################################
        set_BM = False
        

        ### Tuning loss             #############################################################
        # loss function for active tuning:
        at_loss_function = nn.SmoothL1Loss(reduction='sum', beta=0.0001)
        loss_parameters = [('beta', 0.0001), ('reduction', 'sum')]


        ### Binding parameters      #############################################################
        # scaler: 
        scaler = 'rcwSM'        # either: 'unscaled', 'sigmoid' or 'rcwSM'

        # prescaling of Softmax:
        pres = 'clamp'     # either: None, 'clamp', or 'tanh_fixed'
        sigma = 7
        

        ### Temperature parameters  #############################################################
        temperature_params = []
        ## If temperature is NOT turned up:
        if temperature == 'fixed':
            temperature_params = (2,2)             # format: (temp_row, temp_col)
            # temperature_params = (3,5)             # format: (temp_row, temp_col)

        elif temperature == "smooth_turn_down":
            # temp_max = ((550, 4.7),(550, 4.7))    # distractors
            temp_max = ((530, 4.6),(530, 4.6))    # no distractors
            # temp_max = (200,200)    # no distractors

            temperature_params = [
                temp_max
                ]

            # number of frames after which the temperature is turned up:
            temp_range_col = 1      # columns
            temp_range_row = 1      # rows

            # number of frames after which the temperature is set back to 0: 
            temp_reset_frame = 800 + tuning_length

        ## If temperature is turned up: 
        elif temperature == 'turn_up': 

            # maximum temperature:
            # temp_max = (2,2)
            # temp_max = (1,1)
            temp_max = (3,5)

            # value by which the temperature is turned up in every step:
            temp_step_col = 0.01    # column
            temp_step_row = 0.01    # rows
            # temp_step_col = 0.01    # column
            # temp_step_row = 0.01    # rows
            # temp_step_col = 0.1    # column
            # temp_step_row = 0.1    # rows

            # number of frames of which the gradients are consideres for temperature turn-up:
            temp_grad_range_col = 100
            temp_grad_range_row = 100

            # number of frames after which the temperature is turned up:
            temp_range_col = 1      # columns
            temp_range_row = 1      # rows
            # temp_range_col = 6      # columns
            # temp_range_row = 6      # rows
            
            # function to determine values of temperature:
            temp_fct = 'linear'                    # either: 'sigmoid' or 'linear'
            temp_fct_relative = 'lpfilter_deriv'    # either: 'lpfilter_deriv', 'mean' or 'sum_of_abs_diff'

            # number of frames after which the temperature is set back to 0: 
            temp_reset_frame = 800        

            # collect important parameters for setup
            temperature_params = [
                temp_max, 
                temp_step_col, 
                temp_step_row, 
                temp_fct, 
                temp_fct_relative
                ]

        else: 
            print("ERROR: Invalid temperature variant.")
            exit()
            
            
        ### NxM Binding parameters  #############################################################
        # more observations than the LSTM has input:
        nxm_bool = False

        # index of additional observation features:
        index_additional_features = []

        # how to initialize outcast line:
        initial_value_outcast_line = 0.01

        # pre-binding enhancement for outcast line:
        nxm_enhance = 'square'  # either: 'square', 'squareroot', 'log10'

        # pre-binding scaling for outcast line:
        nxm_outcast_line_scaler = 0.1

        # check for distractors and in case of existence increase number of observations
        if distractors is not None: 
            idx = self.num_features-1
            for (_, num_add_feat) in distractors: 
                i_num_add_feat = len(num_add_feat)
                self.num_observations += i_num_add_feat
                for i in range(i_num_add_feat):
                    idx += 1
                    index_additional_features.append(idx)


        ### Rotation parameters     #############################################################
        rot_type = rotation_type    # either: 'eulrotate' or 'qrotate'


        ### Learning rates          #############################################################
        # at_learning_rate_binding = 1.0
        # at_learning_rate_rotation =  0.0005
        # at_learning_rate_translation = 0.005

        at_learning_rate_binding = 1.0
        at_learning_rate_rotation =  0.01
        at_learning_rate_translation = 0.05

        at_learning_rate_state = 0.0


        ### Momenta                 #############################################################
        # at_momentum_binding = 0.95
        at_momentum_binding = 0.9
        at_momentum_rotation = 0.4
        at_momentum_translation = 0.4


        ### Signdamping values      #############################################################
        # at_signdamp_binding = 0.8
        # at_signdamp_rotation = 0.6
        # at_signdamp_translation = 0.7

        # at_signdamp_binding = 0.0
        # at_signdamp_rotation = 0.0
        # at_signdamp_translation = 0.0

        at_signdamp_binding = 0.8
        # at_signdamp_binding = 0.7
        at_signdamp_rotation = 0.9
        at_signdamp_translation = 0.9

        # at_signdamp_binding = 0.4
        # at_signdamp_rotation = 0.9
        # at_signdamp_translation = 0.9


        ### Gradient calculation    #############################################################
        # how to calculate the gradient over the tuning horizon
        grad_calc_binding = 'meanOfTunHor'
        grad_calc_rotation = 'meanOfTunHor'
        grad_calc_translation = 'meanOfTunHor'
        grad_calculations = [grad_calc_binding, grad_calc_rotation, grad_calc_translation]


        ### Gradient weighting      #############################################################
        # bias for gradient weighting (not important for 'meanOfTunHor')
        grad_bias_binding = 1.4
        grad_bias_rotation = 1.1
        grad_bias_translation = 1.1 
        grad_biases = [grad_bias_binding, grad_bias_rotation, grad_bias_translation]


        

        experiment_results = []

        ## experiment performed for all values of the tested parameter
        for val in tested_values: 

            #####################################################################################
            #### Set-up experiment ####
            #####################################################################################

            ### Tested parameter    #############################################################

            if changed_parameter == 'dimensions':
                self.set_dimensions(val)
                print(f'CHANGED gestalt!\nNew value for dimension: {val}')

            elif changed_parameter == 'prescale':
                pres = val
                print(f'CHANGED prescaling!\nNew prescale: {val}')
            
            elif changed_parameter == 'sigma':
                sigma = val
                print(f'CHANGED prescaling!\nNew sigma: {val}')

            elif changed_parameter == 'structure':
                structure = val
                print(f'CHANGED structure type!\nNew structure: {val}')

            elif changed_parameter == 'temperature_fixed':
                temp = val
                print(f'CHANGED temperature!\nNew value for temperature: {val}')

            elif changed_parameter == 'temperature_turn_up':
                (temp_max, 
                    temp_step_col, 
                    temp_step_row, 
                    temp_range_col, 
                    temp_range_row, 
                    temp_fct, 
                    temp_reset_frame)  = val

                temperature_params = [
                    temp_max, 
                    temp_step_col, 
                    temp_step_row, 
                    temp_fct, 
                    temp_fct_relative
                ]

                print(f'CHANGED temperature!\nNew value for temperature: {val}')

            else: 
                print(f'ERROR: Invalid value for changed_parameter: {changed_parameter}')
                exit()


            self.set_BPAT_structure(structure)
            self.preprocessor.set_parameters(
                self.num_features, 
                self.num_observations, 
                self.num_dimensions)


            ### LSTM model          #############################################################
            layerNorm = False
            self.BPAT.set_hidden_num(100)
            if self.num_dimensions == 7:
                # model_path = 'CoreLSTM/models/LSTM_gest.pt'
                # model_path = 'CoreLSTM/models/mod_7_15_200_MSELoss()_0.01_0.9_0.0_10_1500_nse2e-05_layerNORM_3.pt'
                model_path = 'CoreLSTM/models/mod_7_15_200_MSELoss()_0.01_0.9_0.0_10_1000_nse0.0005_layerNORM_3.pt'
                # model_path = 'CoreLSTM/models/mod_7_15_200_MSELoss()_0.01_0.9_0.0_10_1000_nse5e-05_layerNORM_3.pt'
                # model_path = 'CoreLSTM/models/mod_7_15_200_MSELoss()_0.01_0.9_0.0_10_2000_nse5e-05_layerNORM_3.pt'
                # model_path = 'CoreLSTM/models/mod_7_15_200_MSELoss()_0.01_0.9_0.0_10_1400_nse5e-05_layerNORM_3.pt'
                # model_path = 'CoreLSTM/models/mod_7_15_200_MSELoss()_0.01_0.9_0.0_10_2000_nse2e-05_layerNORM_2.pt'
                # model_path = 'CoreLSTM/models/mod_7_15_200_MSELoss()_0.01_0.9_0.0_10_1000_old.pt'
            elif self.num_dimensions == 6:
                # model_path = 'CoreLSTM/models/LSTM_vel.pt'
                # model_path = 'CoreLSTM/models/mod_6_15_200_MSELoss()_0.01_(0.9, 0.999)_0.0_10_2000_nse2e-05.pt'
                # model_path = 'CoreLSTM/models/mod_6_15_200_MSELoss()_0.01_0.9_0.0_10_2000_nse2e-05_layerNORM.pt'
                # model_path = 'CoreLSTM/models/mod_6_15_200_MSELoss()_0.01_0.9_0.0_10_1000_old.pt'
                ## correct Velocity: 
                # model_path = 'CoreLSTM/models/mod_6_15_200_MSELoss()_0.01_(0.9, 0.999)_0.0_40_2000_nse2e-05_corrVel.pt'
                model_path = 'CoreLSTM/models/mod_6_15_100_MSELoss()_0.01_(0.9, 0.999)_0.0_10_2000_nse2e-05_corrVel.pt'
                # model_path = 'CoreLSTM/models/mod_6_15_100_MSELoss()_0.01_(0.9, 0.999)_0.0_40_600_nse2e-05_corrVel.pt'

            elif self.num_dimensions == 3: 
                # model_path = 'CoreLSTM/models/LSTM_pos.pt'
                model_path = 'CoreLSTM/models/mod_3_15_200_MSELoss()_0.01_(0.9, 0.999)_0.0_10_2000_nse2e-05.pt'
                # model_path = 'CoreLSTM/models/mod_3_15_200_MSELoss()_0.01_0.9_0.0_10_2000_nse2e-05_layerNORM.pt'
                # model_path = 'CoreLSTM/models/mod_3_15_200_MSELoss()_0.01_0.9_0.0_10_1000_old.pt'
            else: 
                print('ERROR: Unvalid number of dimensions!\nPlease use 3, 6, or 7.')
                exit()
         

            ### BPAT parameters   #############################################################

            self.BPAT.set_binding_prescale(pres)
            self.BPAT.set_binding_sigma(sigma)           
            self.BPAT.set_gradient_calculation(grad_calculations)
            self.BPAT.set_weighted_gradient_biases(grad_biases)  
            self.BPAT.set_rotation_type(rot_type)    
            self.BPAT.set_scale_mode(scaler)
            self.BPAT.set_init_axis_angle(0)
            self.BPAT.set_binding_prescale(pres)
            
            ## temperature ##
            self.BPAT.set_temperature_parameters(temperature_params)
            if structure == 'parallel_temp_turnup_relative':
                self.BPAT.set_relative_temperature_grad_range(temp_grad_range_col, temp_grad_range_row)
            if temperature == 'turn_up' or temperature == 'smooth_turn_down':
                self.BPAT.set_range_temperature_turnup(temp_range_col, temp_range_row)
                self.BPAT.set_temp_reset(temp_reset_frame)

            if nxm_bool:
                self.BPAT.set_distractors(distractors)
                self.preprocessor.set_distractors(distractors)
                self.BPAT.set_additional_features(index_additional_features)
                self.BPAT.set_outcast_init_value(initial_value_outcast_line)
                self.BPAT.set_nxm_enhancement(nxm_enhance)
                self.BPAT.set_nxm_last_line_scale(nxm_outcast_line_scaler) 
            

            ## Set initial binding matrix
            self.BPAT.set_binding_matrix_init(set_BM, None)
            

            ###################################################################################
            #### Run experiment ####
            ###################################################################################
            sample_names, result_names, results = super().run(
                changed_parameter+"_"+str(val)+"/",
                modification,
                sample_nums, 
                model_path, 
                layerNorm,
                tuning_length, 
                num_tuning_cycles, 
                at_loss_function,
                loss_parameters,
                [at_learning_rate_binding, at_learning_rate_rotation, at_learning_rate_translation], 
                at_learning_rate_state, 
                [at_momentum_binding, at_momentum_rotation, at_momentum_translation],
                [at_signdamp_binding, at_signdamp_rotation, at_signdamp_translation]
            )

            experiment_results += [results]

        #########################################################################################
        #### Experiment evaluation ####
        #########################################################################################

        self.stats.set_parameters(
            self.num_features, 
            self.num_observations, 
            self.num_dimensions
            )

        dfs = self.stats.load_csvresults_to_dataframe(
            self.prefix_res_path, 
            changed_parameter, 
            tested_values, 
            sample_names, 
            result_names
            )

        self.stats.plot_histories(
            dfs, 
            self.prefix_res_path, 
            changed_parameter, 
            result_names, 
            result_names
        )

        self.stats.plot_value_comparisons(
            dfs, 
            self.prefix_res_path, 
            changed_parameter, 
            result_names, 
            result_names
        )


        #########################################################################################
        print("Terminated experiment.")




### Previous Parameters

        # at_learning_rate_binding = 0.1
        # at_learning_rate_rotation =  0.1
        # at_learning_rate_translation = 0.1
        # at_learning_rate_state = 0.0

        # at_momentum_binding = 0.9
        # at_momentum_rotation = 0.8
        # at_momentum_translation = 0.8

        # at_learning_rate_binding = 0.01
        # at_learning_rate_rotation =  0.0001
        # at_learning_rate_translation = 0.1
        # at_learning_rate_state = 0.0

        # at_momentum_binding = 0.3
        # at_momentum_rotation = 0.5
        # at_momentum_translation = 0.3

        # old models
        # at_learning_rate_binding = 0.1
        # at_learning_rate_rotation =  0.001
        # at_learning_rate_translation = 0.1
        # at_learning_rate_state = 0.0

        # at_momentum_binding = 0.95
        # at_momentum_rotation = 0.9
        # at_momentum_translation = 0.9

        # at_learning_rate_binding = 0.05
        # at_learning_rate_rotation =  0.001
        # at_learning_rate_translation = 0.05
        # at_learning_rate_state = 0.0

        # at_momentum_binding = 0.95
        # at_momentum_rotation = 0.9
        # at_momentum_translation = 0.9