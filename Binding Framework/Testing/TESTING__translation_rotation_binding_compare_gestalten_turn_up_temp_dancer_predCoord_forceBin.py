"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import torch 
import os

import sys
# Before run: replace ... with current directory path
sys.path.append('/Users/MartinButz2/Documents/CODE/Python/BindingDancersAndCubes/BindingAndPerspectiveTaking-Research-main')

from testing_module.experiment_interface_opt_illusions import EXPERIMENT_INTERFACE_OPT_ILLUSIONS


class COMP_GEST_ROTATION_BINDING(EXPERIMENT_INTERFACE_OPT_ILLUSIONS):

    def __init__(self, num_features, num_observations, num_dimensions, illusion, edge_points):
        experiment_name = f"compare_gestalten_bin_tempturnup_predCoord_foreBM_{illusion}"
        super().__init__(num_features, num_observations, num_dimensions, experiment_name, illusion, edge_points)


    def perform_experiment(self, sample_nums, modification, structure, dimension_values, rotation_type): 
        
        changed_parameter = 'dimensions'
        tested_values = dimension_values
        distractors = None
        temperature = 'smooth_turn_down'        # either: fixed, smooth_turn_down, or turn_up

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
    os.chdir('..')
    # set the following parameters
    num_observations = 15
    num_input_features = 15
    num_dimensions = 3
    
    rotation_type = 'qrotate'   # either: 'eulrotate' or 'qrotate'

    illusion = 'dancer'    # either: 'necker_cube' or 'dancer'
    edge_points = None

### Tuning structure        #############################################################
        # either: 'parallel', 'sequential', 
        #  'parallel_temp_turnup'
        #  'parallel_temp_turnup_BMforce'
    structure = 'parallel_temp_turnup_BMforce' 

    test = COMP_GEST_ROTATION_BINDING(
        num_input_features, 
        num_observations, 
        num_dimensions, 
        illusion,
        edge_points) 

    
    modification = [
        ('bind', None, None)
    ]


    # sample_nums = [600, 600, 600, 600] 
    sample_nums = [400]

    tested_dimensions = [6]

    test.perform_experiment(
        sample_nums, 
        modification, 
        structure,
        tested_dimensions, 
        rotation_type)



if __name__ == "__main__":
    main()
