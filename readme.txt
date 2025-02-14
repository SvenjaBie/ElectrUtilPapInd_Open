The files in this folder contain the code used in the research article "The impact of energy prices on the electrification of utility systems in industries with fluctuating energy demand" by Svenja Bielefeld, Brendon de Raad, Marit van Lieshout, Lydia Stougie, Milos Cvetkovic and Andrea Ramirez.

A gurobi (or other solver's) licence is required to run the optimisation. 
The optimisation is started from the "Modelruns.py" script and calls the functions in "functions.py". 
The results are stored in pickle files which can be converted into csv files using the "Postprocessing.py" script.
The "environment.yaml" file indicates the required python packages and respective versions which need to be installed before running the code.

The energy price input data is stored in the "input_data" folder. The energy demand data file is filled with ones due to confidentiality reasons. To run a real case, the demand data has to be replaced.