#	HowToTest BPAT

## General procedure

For performing an experiment with the BPAT model, one must simply define the experimental parameters 
in an experiment python file as described below. 

During an experiment, only one defined hypeparameter is changed. 
For all tested values of this parameter, the inference is performed and the results are 
stored in the automatically created experiment directory in the folder Grafics/.
No additional files or directories need to be created manually. 

Important results are listed below:

parameter				|	explaination		
------------------------|------------------------------------
at_loss_history			|	loss of prediction during active tuning inference
predicion_errors		|	prediction error for overall observation with final parameters
determinante_history	|	determinante of binding matrix during active tuning inference
fbe_history				|	feature binding error during active tuning inference
bin_grads				|	gradients for all entires of the binding matrix (before sign damping)
final_binding_matrix	|	last computed binding matrix
final_binding_activations |	last binding activations used to compute final binding matrix
rotmat_loss				|	rotation error during active tuning inference
rot_grads				|	gradients for all entries in the quaternion used in rotation
transba_loss 			|	translation error during active tuning inference 
trans_grads				|	gradients for all entries in the translation bias
prediction_errors_dim	|	prediction error for every coordinate predicted in closed loop -* for PosVel-variant: suffix _pos or _vel is added


Of course, binding, rotation, and translation parameters are only evaluated if the inference should solve 
	the respective task. Dimentional prediction errors are only computed in the respective BPAT subclass.
After the experiment has been completed for all tested values, comparsion and history plots are 
	generated over all trials out of the previously stored csv files and stored in the experiment directory. 


## Setup experiment


To perform an existing experiments, one needs to adapt the experiment template accordingly. 
All variables which are assigned to '...' in the template must be given an appropriate value. 
The following gives an overview over these variables and their possible values. 

In __init__: 
* experiment_name: name of the folder in which experiment results will be stored

In __perform_experiment__: 
* changed_parameter: name of the parameter that should be tested in this experiment. Should be set according to the name in the experiment interface.
* distractors: If experiment does not test distractors, set them here. 
	* Distractors are motion joints selected from the motion capture data for thaichi or modern dance. 
	* Set distractors as a list of tupels, indicating the motion for the distractor and the joint which should be selected.
	* e.g.: \
		distractors = [\
			('thaichi', [4,7,12])		-> select the 4th, 7th, and 12th joint from the thaichi data\
			,\
			('modern', [3])				-> select the 3rd joint from the modern dance data\
			]\
* temperature: either 'fixed' or 'turn_up', temperature parameters are to be set in the experiment interface

In __main()__: 
* num_observations, num_input_features, num_dimensions 
* modifications: how should the observations be modified with respect to the task
	* modification = [(task_1, type_1, specify), ... ] -* task is one out of {'bind', 'rotate', 'translat'}
	* For binding, set ('bind', None, None) if the binding should be inferred.
	* For rotation and translation, there are four different modification types possible:
		* None : no modification 
		* det  : deterministic modification as defined in the general_tester
		* rand : random modification 
		* set  : set the modification by defining it in the experimental file
	* specify depends on the task: 
		* 'bind'
			* None
		* 'rotate' 
			* None
			* rotation_type
			* if type is 'set': (rotation_type, rotation quaternion as tensor)
		* 'translate' 
			* None 
			* translation bias as tensor 
			* if type is 'rand', range of translation values

* sample nums: list of number of used frames per sample.
	* walker : order of sample sequence is fixed and every sample has a maximum frame number: \
			1) S35T07 | 1000\
			2) S05T01 | 550\
			3) S06T01 | 450\
			4) S08T02 | 300\
			5) S07T02 | 250\
		However, frames are repeated if a number in the list exceeds another. \
		e.g.:\
			1) sample_nums = [500, 500, 450] -> the run is performed for samples 1), 2), 3) on 500 frames each\
			2) sample_nums = [2000, 990] -> the run is performed for sample 1), which is repeated after 990 frames for 2000 frames\
	* dancer : order of sample sequences is fixed. All samples have 10000 frames and correspond to a version of the dancer:\
			1) normal dancer, rotating clockwise\
			2) mirrored dancer, rotating clockwise\
			3) normal dancer, rotating counter-clockwise\
			4) mirrored dancer, rotating counter-clockwise\
		e.g.: sample_nums = [1000, 1000] -> the run is performed on 1000 frames of the dancer data rotating clockwise.
	* necker cube : order of sample sequences is fixed. Only two options - rotation clockwise or counter-clockwise.

* parameter values: values of tested parameters that should be change between inferences




The template offers a complete structure to change all of these variables. 
Of course, it can be adapted according to the experiment. 


## Already implemented

The BPAT model can be tested in different experimental setups. 
Here, the following experiments are implemented: 

* experiments to test difference between Gestalt-variants in different setups:
	* TESTING__translation_rotation_binding_compare_gestalten_turn_up_temp_dancer_predCoord
	* TESTING__translation_rotation_binding_compare_gestalten_turn_up_temp_dancer
	* TESTING__translation_rotation_binding_compare_gestalten_turn_up_temp_multiple_distractor
	* TESTING__translation_rotation_binding_compare_gestalten_turn_up_temp_neckercube
	* TESTING__translation_rotation_binding_compare_gestalten_turn_up_temp
	* TESTING__translation_rotation_binding_illusion_multiple_distractors
* example experiments to test different values for an inference parameter
	* TESTING__translation_rotation_binding_compare_temperature
	* TESTING__translation_rotation_binding_illusion_compare_sigma



Author: Franziska Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de

