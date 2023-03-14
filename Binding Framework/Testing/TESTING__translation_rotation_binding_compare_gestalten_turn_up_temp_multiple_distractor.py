"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import torch 

import sys

import os
sys.path.append('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')
os.chdir('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')
# Before run: replace ... with current directory path

from testing_module.experiment_interface_walker import EXPERIMENT_INTERFACE


class COMP_GEST_ROTATION_BINDING(EXPERIMENT_INTERFACE): 

    def __init__(self, num_features, num_observations, num_dimensions):
        experiment_name = "compare_gestalten_trans_rot_bin_tempturnup_multiple_dist"
        super().__init__(num_features, num_observations, num_dimensions, experiment_name)


    def perform_experiment(self, 
            sample_nums, 
            modification, 
            structure,
            dimension_values, 
            distractors, 
            rotation_type): 

        changed_parameter = 'dimensions'
        tested_values = dimension_values
        temperature = 'smooth_turn_down'    # either: fixed, smooth_turn_down, or turn_up

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
    num_observations = 15
    num_input_features = 15
    num_dimensions = 3

    rotation_type = 'qrotate'   # either: 'eulrotate' or 'qrotate'


    ### Tuning structure        #############################################################
            # either: 'parallel', 'sequential', 
            #  'parallel_temp_turnup'
            #  'parallel_temp_turnup_BMforce'
    structure = 'parallel_temp_turnup'     


    test = COMP_GEST_ROTATION_BINDING(
        num_input_features, 
        num_observations, 
        num_dimensions) 

    
    modification = [
        ('bind', None, None)      
        ,
        # ('rotate', 'det', rotation_type)
        # ('rotate', 'set', (rotation_type, torch.Tensor([ 1, 0, 0, 0 ]))) # axang 0
        # ('rotate', 'set', (rotation_type, torch.Tensor([ 0.7071068 , 0.4082483, 0.4082483, 0.4082483 ]))) # axang 90
        ('rotate', 'set', (rotation_type, torch.Tensor([ 0.953717, 0.1736126, 0.1736126, -0.1736126 ]))) # axang 35
        # ('rotate', 'set', (rotation_type, torch.Tensor([ 0.9396926 , 0.1974654, 0.1974654, 0.1974654 ]))) # axang 40
        # ('rotate', 'set', (rotation_type, torch.Tensor([ -0.4617486, 0.5121159, 0.5121159, -0.5121159 ]))) # axang 235
        # ('rotate', 'rand', rotation_type)
        ,
        # ('translate', None, None)
        # ('translate', 'det', None
        ('translate', 'set', torch.Tensor([-1,0.5,-0.8]))
        # ('translate', 'set', torch.Tensor([ 0.2, -0.8,  0.4]))
        # ('translate', 'rand', [-2, 4])
    ]

    distractors = [
        # ('thaichi', [4])
        ('thaichi', [4,7,12])
        # ('thaichi', [4,6,8,10,12,14])
        ,
        # ('modern', [3,5,7,9,11,13])
        ('modern', [0,6,11])
        # ('modern', [3])
        ]

    # sample_nums = [100, 100, 100]
    sample_nums = [1600, 910, 430, 450]

    tested_dimensions = [6]
    # tested_dimensions = [3,6]

    test.perform_experiment(
        sample_nums, 
        modification, 
        structure,
        tested_dimensions, 
        distractors,
        rotation_type)



if __name__ == "__main__":
    main()