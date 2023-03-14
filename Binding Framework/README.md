# BindingAndPerspectiveTaking-Research


****************************************************************************************************
	BPAT - Binding and Perspective taking through Active Tuning
****************************************************************************************************

The presented code implements the BPAT model, as well as corresponding experiments. 
To perform experiments, please follow the description in Testing/HowToTest.txt. 


Before running the code the respective path must be set in the used files.
For this, simply replace the <...> in the following line of every python file:
	sys.path.append('...') 


In general, the current version has implementaions to perform BPAT on a simple walker or on the optical
	illusions of the rotating dancer or the necker cube.


Throughout the code, the following general parameters are constantly used:

parameter		|	explaination					|	values
------------------------|-------------------------------------------------------|-------------------------------------------------------------
num_observations	|	number of observed motion features		|	walker and dancer: 15,16,17; neckar cube: depending on number of edge points
num_features		|	number of input features of LSTM 		|	15
num_dimensions		|	indicator for used gestalt variant, equalls total number of values describing a feature	|	Pos: 3; PosVel: 6; PosDirMag: 7


The actual BPAT model relies on multiple hyper-parameters which all have a default value, 
	specified in experiment_interface (for walker) and experiment_interface_oi (for optical illusions).
Throughout an experiment, these values are kept constant. The only exception is the parameter that should be tested 
	during the current experimental run. This parameter is specified by 'changed_parameter'. The values this parameter 
	should take in the different experiment trials are specified in tested_values. 

The Testing/testing_module contains the organisational classes which perform the setup and evaluation of the experiment. 
BPAT_class_combination_gestalten holds general methods for initializing and performing the BPAT inference. It has different 
	subclasses which all implement a different version of the BPAT algorithm (specified in class-comments). 
	BPAT inference is then performed by run_inferenence() in the respective subclasses.



NOTE: 

(1)
Quaternion implementaions are partially adopted from 
Dario Pavllo, David Grangier, and Michael Auli. QuaterNet: A Quaternion-based Recurrent Model for Human Motion. In British Machine Vision Conference (BMVC), 2018.
[https://github.com/facebookresearch/QuaterNet]


Author: 
Franziska Kaltenberger
franziska.kaltenberger@t-online.de





