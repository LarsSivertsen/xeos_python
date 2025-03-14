All parameters and variables used are defined in the input_parameters_main.txt

When main.py is run it creates a new folder with name equal to the run_number variable 

In every run folder, for example run_3000, there are 3 files: parameters.txt, filenames.txt and parameters_w_class.txt
The files are explained below:
-parameters.txt has the parameters B, Delta, m_s and c for all the computed eos
-filenames.txt have the filenames of all the eos that are valid (have a transition to quark matter), with each 
filename corresponding to the line number in parameters.txt
-parameters_w_class.txt have all the filenames and a boolean variable that tells if a valid transition happened for 
those parameters. This is used to make a classifier for the valid eos. 

To create mass radius plots one can go to the "xeos_pyhton" folder and open the "plot_all_mass_radius_relations.py" file, 
change the run number to the desired run number, and run it.

Tests:
The "tests_new.py" has the code for different tests. They can be run by opening the "run_tests.py" and running the desired test. 
The "run_tests.py" file should be self explanatory. 

