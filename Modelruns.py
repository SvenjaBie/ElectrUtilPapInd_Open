import pyomo.environ as pm
import pandas as pd
import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from pathlib import Path
from functions import optimisation_run_PI_CHP, optimisation_run_benchmark_CHP


# IMPORT INPUT DATA
reporoot_dir = Path(__file__).resolve().parent
# heat demand data
heat_demand_orig = pd.read_csv(os.path.join(reporoot_dir, r'input_data/demand_data/Steamconsumption_15122023.csv'))
heat_demand_data_dict = {'110': pd.read_csv(os.path.join(reporoot_dir, r'input_data/demand_data'
                                                                       r'/Steamconsumption_110_20122023.csv')),
                         '120': pd.read_csv(os.path.join(reporoot_dir, r'input_data/demand_data'
                                                                       r'/Steamconsumption_120_20122023.csv')),
                         '130': pd.read_csv(os.path.join(reporoot_dir, r'input_data/demand_data'
                                                                       r'/Steamconsumption_130_20122023.csv')),
                         '140': pd.read_csv(os.path.join(reporoot_dir, r'input_data/demand_data'
                                                                       r'/Steamconsumption_140_20122023.csv')),
                         '150': pd.read_csv(os.path.join(reporoot_dir, r'input_data/demand_data'
                                                                       r'/Steamconsumption_150_20122023.csv')),
                         '160': pd.read_csv(os.path.join(reporoot_dir, r'input_data/demand_data'
                                                                       r'/Steamconsumption_160_20122023.csv')),
                         'sum': pd.read_csv(
                             os.path.join(reporoot_dir, r'input_data/demand_data/Steamconsumption_15122023'
                                                        r'.csv'))}

# define factor by which volatility should be amplified
amp_values = []  # , 1.3, 1.4
variability_values = ['original']
# define operational hours of the plant (=length of the optimisation)
hours = 8000

# define minimal load factor of CHP
GT_min_load = 0.3  # minimal load factor, [% of Pnom]

# define scenarios:
HP_integration_scenarios = ['PlugIn']
el_price_scenarios = [
    'MeanLow-VarLow',
    'MeanHigh-VarLow',
    'MeanLow-VarHigh',
    'MeanHigh-VarHigh'
]
gas_use_cost_scenarios = [
    'MeanLow-VarLow-EGR1.6',
    'MeanLow-VarLow-EGR1',
    'MeanHigh-VarLow-EGR1.6', 'MeanHigh-VarLow-EGR1',
    'MeanLow-VarHigh-EGR1.6',
    'MeanLow-VarHigh-EGR1',
    'MeanHigh-VarHigh-EGR1.6',
    'MeanHigh-VarHigh-EGR1'
]
capex_scenarios = ['HighHP-LowRest', 'LowHP-HighRest']

# load required energy price data:
all_electricity_prices = {
    'MeanLow-VarLow': pd.read_excel(os.path.join(reporoot_dir,
                                                 r'input_data/Model_price_data_python_v5.xlsx'),
                                    sheet_name='MEANlow_VARlow', header=0, index_col=0,
                                    usecols='A,F'),
    'MeanHigh-VarLow': pd.read_excel(os.path.join(reporoot_dir,
                                                  r'input_data/Model_price_data_python_v5.xlsx'),
                                     sheet_name='MEANhigh_VARlow', header=0, index_col=0,
                                     usecols='A,F'),
    'MeanLow-VarHigh': pd.read_excel(os.path.join(reporoot_dir,
                                                  r'input_data/Model_price_data_python_v5.xlsx'),
                                     sheet_name='MEANlow_VARhigh', header=0, index_col=0,
                                     usecols='A,F'),
    'MeanHigh-VarHigh': pd.read_excel(os.path.join(reporoot_dir,
                                                   r'input_data/Model_price_data_python_v5.xlsx'),
                                      sheet_name='MEANhigh_VARhigh', header=0, index_col=0,
                                      usecols='A,F')
}

all_gas_prices = {
    'MeanLow-VarLow-EGR1.6': pd.read_excel(os.path.join(reporoot_dir,
                                                         r'input_data/Model_price_data_python_v5.xlsx'),
                                            sheet_name='MEANlow_VARlow', header=0, index_col=0,
                                            usecols='A,K'),
    'MeanLow-VarLow-EGR1': pd.read_excel(os.path.join(reporoot_dir,
                                                       r'input_data/Model_price_data_python_v5.xlsx'),
                                          sheet_name='MEANlow_VARlow', header=0, index_col=0,
                                          usecols='A,L'),
    'MeanHigh-VarLow-EGR1.6': pd.read_excel(os.path.join(reporoot_dir,
                                                          r'input_data/Model_price_data_python_v5.xlsx'),
                                             sheet_name='MEANhigh_VARlow', header=0, index_col=0,
                                             usecols='A,K'),
    'MeanHigh-VarLow-EGR1': pd.read_excel(os.path.join(reporoot_dir,
                                                        r'input_data/Model_price_data_python_v5.xlsx'),
                                           sheet_name='MEANhigh_VARlow', header=0, index_col=0,
                                           usecols='A,L'),
    'MeanLow-VarHigh-EGR1.6': pd.read_excel(os.path.join(reporoot_dir,
                                                           r'input_data/Model_price_data_python_v5.xlsx'),
                                              sheet_name='MEANlow_VARhigh', header=0, index_col=0,
                                              usecols='A,K'),
    'MeanLow-VarHigh-EGR1': pd.read_excel(os.path.join(reporoot_dir,
                                                         r'input_data/Model_price_data_python_v5.xlsx'),
                                            sheet_name='MEANlow_VARhigh', header=0, index_col=0,
                                            usecols='A,L'),
    'MeanHigh-VarHigh-EGR1.6': pd.read_excel(os.path.join(reporoot_dir,
                                                           r'input_data/Model_price_data_python_v5.xlsx'),
                                              sheet_name='MEANhigh_VARhigh', header=0, index_col=0,
                                              usecols='A,K'),
    'MeanHigh-VarHigh-EGR1': pd.read_excel(os.path.join(reporoot_dir,
                                                         r'input_data/Model_price_data_python_v5.xlsx'),
                                            sheet_name='MEANhigh_VARhigh', header=0, index_col=0,
                                            usecols='A,L')
}
# converting index into Datetime index
for gas_use_cost_scenario in gas_use_cost_scenarios:
    all_gas_prices[gas_use_cost_scenario].index = pd.to_datetime(all_gas_prices[gas_use_cost_scenario].index,
                                                                 dayfirst=True,
                                                                 format='mixed')  # format='%d.%m.%Y %H:%M'

# define technology cost data
all_capex_data = {
    'PI': {'HighHP-LowRest': {'ElB': 30000, 'Bat': 180e3, 'TES': 15000, 'HP': 500e3, 'H2E': 760e3, 'H2B': 35000,
                              'H2S': 10000},
           'LowHP-HighRest': {'ElB': 30000, 'Bat': 320e3, 'TES': 40000, 'HP': 300e3, 'H2E': 980e3, 'H2B': 35000,
                              'H2S': 10000}}}

# define dict for looping scenario runs and storing results:
scenario_dict = {HP_integration_scenario: {el_price_scenario: {gas_price_scenario: {capex_scenario: {}
                                                                                    for capex_scenario in
                                                                                    capex_scenarios}
                                                               for gas_price_scenario in gas_use_cost_scenarios}
                                           for el_price_scenario in el_price_scenarios}
                 for HP_integration_scenario in HP_integration_scenarios}

# starting the model runs
for HP_integration_scenario in ['PlugIn']:
    print("Started: " + HP_integration_scenario)
    for i, el_price_scenario in enumerate(el_price_scenarios):
        print("Started: " + el_price_scenario)
        price_el_hourly = all_electricity_prices[el_price_scenario]
        for j in [2 * i, 2 * i + 1]:
            gas_use_cost_scenario = gas_use_cost_scenarios[j]
            print("Started: " + gas_use_cost_scenario)
            price_NG_use = all_gas_prices[gas_use_cost_scenario]
            for capex_scenario in capex_scenarios:
                print("Started: " + capex_scenario)
                # store starting time
                begin = time.time()
                if HP_integration_scenario == 'PlugIn':
                    capex_data = all_capex_data['PI'][capex_scenario]
                    scenario_dict[HP_integration_scenario][el_price_scenario][gas_use_cost_scenario][capex_scenario] = \
                        optimisation_run_PI_CHP(heat_demand_orig, price_el_hourly, price_NG_use, amp_values,
                                                variability_values, GT_min_load, hours, capex_data)
                    # storing the results of the individual scenario run
                    prefix = 'PI_' + gas_use_cost_scenario + '_' + capex_scenario + '_minload30%_opt005'
                    timestamp_format = "{:%Y%m%dT%H%M}"
                    timestamp = timestamp_format.format(datetime.now())
                    output_filename = f"{prefix}__{timestamp}.pickle"
                    with open(output_filename, 'wb') as handle:
                        pickle.dump(scenario_dict[HP_integration_scenario][el_price_scenario][gas_use_cost_scenario]
                                    [capex_scenario], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Finished saving the individual scenario run")
                    # stop counting the time
                    time.sleep(1)
                    # store end time
                    end = time.time()

                    # total time taken
                    print(f"Total runtime of the program is {end - begin}")

                else:
                    print("Invalid HP integration scenario.")


# save the results of all model runs
prefix = 'outputs_with_CHP_minload30%_opt005'
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(scenario_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving complete scenario dict")

# save the input data
prefix = 'elprices'
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(all_electricity_prices, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving el prices")

prefix = 'NGprices'
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(all_gas_prices, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving NG prices")

prefix = 'technologycostdata'
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(all_capex_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving TC data")

# Run benchmark system optimisation
benchmark_scenario_dict = {el_price_scenario: {gas_price_scenario: {} for gas_price_scenario in gas_use_cost_scenarios}
                           for el_price_scenario in el_price_scenarios}

# starting the model runs
for i, el_price_scenario in enumerate(el_price_scenarios):
    print("Started: " + el_price_scenario)
    price_el_hourly = all_electricity_prices[el_price_scenario]
    for j in [2 * i, 2 * i + 1]:
        gas_use_cost_scenario = gas_use_cost_scenarios[j]
        print("Started: " + gas_use_cost_scenario)
        price_NG_use = all_gas_prices[gas_use_cost_scenario]
        benchmark_scenario_dict[el_price_scenario][gas_use_cost_scenario] = \
            optimisation_run_benchmark_CHP(heat_demand_orig, price_el_hourly, price_NG_use, amp_values,
                                           variability_values, GT_min_load, hours)

# save the results
prefix = 'outputs_benchmark_with_CHP_minload30%'
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(benchmark_scenario_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving complete scenario dict")

# save the input data
prefix = 'elprices_benchmark'
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(all_electricity_prices, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving el prices")

prefix = 'NGprices_benchmark'
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(all_gas_prices, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving NG prices")

# ------------------------------------ END of script -------------------------------------------------------------------
print("End of the script")
