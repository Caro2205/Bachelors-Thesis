"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import torch 

import sys
sys.path.append('...')      
# Before run: replace ... with current directory path

from testing_module.experiment_interface_opt_illusions import EXPERIMENT_INTERFACE_OPT_ILLUSIONS


class EXPERIMENT_TEMPLATE(EXPERIMENT_INTERFACE_OPT_ILLUSIONS):     # use EXPERIMENT_INTERFACE for walker

    def __init__(self, num_features, num_observations, num_dimensions, illusion, edge_points):

        experiment_name = ... # NOTE: enter experiment name (will be used for folder name)

        super().__init__(
            num_features, 
            num_observations, 
            num_dimensions, 
            experiment_name, 
            illusion, 
            edge_points)


    def perform_experiment(self, 
            sample_nums,
            modification, 
            structure, 
            tested_values, 
            rotation_type
        ): 
        
        changed_parameter = ...         # name of parameter that should be tested
        tested_values = tested_values   
        distractors = ...               # either None or list of distractors (see HowToText.txt)
        temperature = ...               # either: fixed, smooth_turn_down, or turn_up

        super().perform_experiment(
            sample_nums, 
            modification, 
            structure,
            changed_parameter, 
            tested_values,
            rotation_type, 
            distractors, 
            temperature)



                  
def main(): 
    # set the following parameters
    num_observations = ...      # {15, 16, 17}
    num_input_features = ...    # {15}
    num_dimensions = ...        # {3, 6, 7}
    
    rotation_type = 'qrotate'   # either: 'eulrotate' or 'qrotate'

    illusion = 'dancer'         # either: 'necker_cube' or 'dancer'
    edge_points = None          # only needed for 'necker_cube' illusion

    ### Tuning structure        #############################################################
        # either: 'parallel', 'sequential', 
        #  'parallel_temp_turnup'
        #  'parallel_temp_turnup_BMforce'
        # 'necker_cube_static_bind'
    structure = ...       

    test = EXPERIMENT_TEMPLATE(
        num_input_features, 
        num_observations, 
        num_dimensions, 
        illusion,
        edge_points) 

    
    # NOTE: The following variables must be set according to the performed experiment. 

    modification = [
        ## EXAMPLE
        # 1 #
        # ('bind', None, None),             # -> binding will be infered
        # ('rotate', 'det', rotation_type)  # -> rotation will be infered and modified deterministically
                                            # -> translation is optimal and thus not infered

        # 2 #
        # ('bind', None, None),             # -> binding will be infered
        # ('rotate', 'det', rotation_type), # -> rotation will be infered and modified deterministically
        # ('translate', 'det', range(5))    # -> translation will be infered and modified randomly in the given range
        

        ## PREVIOUS VALUES ##
        ## quaternion values ##
        # ('rotate', 'det', rotation_type)
        # ('rotate', 'set', (rotation_type, torch.Tensor([ 1, 0, 0, 0 ]))) # axang 0
        # ('rotate', 'set', (rotation_type, torch.Tensor([ 0.7071068 , 0.4082483, 0.4082483, 0.4082483 ]))) # axang 90
        # ('rotate', 'set', (rotation_type, torch.Tensor([ 0.953717, 0.1736126, 0.1736126, -0.1736126 ]))) # axang 35
        # ('rotate', 'set', (rotation_type, torch.Tensor([ 0.9396926 , 0.1974654, 0.1974654, 0.1974654 ]))) # axang 40
        # ('rotate', 'set', (rotation_type, torch.Tensor([ -0.4617486, 0.5121159, 0.5121159, -0.5121159 ]))) # axang 235
        # ('rotate', 'rand', rotation_type)
        
        ## translation values ##
        # ('translate', None, None)
        # ('translate', 'det', None
        # ('translate', 'set', torch.Tensor([-1,0.5,-0.8]))
        # ('translate', 'set', torch.Tensor([ 0.2, -0.8,  0.4]))
        # ('translate', 'rand', [-2, 4])
    ]

    sample_nums = ...           # determines how many samples to test from which motion capture example
    tested_values = ...     

    test.perform_experiment(
        sample_nums, 
        modification, 
        structure,
        tested_values, 
        rotation_type)



if __name__ == "__main__":
    main()