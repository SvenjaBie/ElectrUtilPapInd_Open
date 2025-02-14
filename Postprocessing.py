import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

# load required energy price data:
reporoot_dir = Path(__file__).resolve().parent
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
    'MeanLow-VarLow-GP1.6to1': pd.read_excel(os.path.join(reporoot_dir,
                                                          r'input_data/Model_price_data_python_v5.xlsx'),
                                             sheet_name='MEANlow_VARlow', header=0, index_col=0,
                                             usecols='A,K'),
    'MeanLow-VarLow-GP1to1': pd.read_excel(os.path.join(reporoot_dir,
                                                        r'input_data/Model_price_data_python_v5.xlsx'),
                                           sheet_name='MEANlow_VARlow', header=0, index_col=0,
                                           usecols='A,L'),
    'MeanHigh-VarLow-GP1.6to1': pd.read_excel(os.path.join(reporoot_dir,
                                                           r'input_data/Model_price_data_python_v5.xlsx'),
                                              sheet_name='MEANhigh_VARlow', header=0, index_col=0,
                                              usecols='A,K'),
    'MeanHigh-VarLow-GP1to1': pd.read_excel(os.path.join(reporoot_dir,
                                                         r'input_data/Model_price_data_python_v5.xlsx'),
                                            sheet_name='MEANhigh_VARlow', header=0, index_col=0,
                                            usecols='A,L'),
    'MeanLow-VarHigh-GP1.6to1': pd.read_excel(os.path.join(reporoot_dir,
                                                           r'input_data/Model_price_data_python_v5.xlsx'),
                                              sheet_name='MEANlow_VARhigh', header=0, index_col=0,
                                              usecols='A,K'),
    'MeanLow-VarHigh-GP1to1': pd.read_excel(os.path.join(reporoot_dir,
                                                         r'input_data/Model_price_data_python_v5.xlsx'),
                                            sheet_name='MEANlow_VARhigh', header=0, index_col=0,
                                            usecols='A,L'),
    'MeanHigh-VarHigh-GP1.6to1': pd.read_excel(os.path.join(reporoot_dir,
                                                            r'input_data/Model_price_data_python_v5.xlsx'),
                                               sheet_name='MEANhigh_VARhigh', header=0, index_col=0,
                                               usecols='A,K'),
    'MeanHigh-VarHigh-GP1to1': pd.read_excel(os.path.join(reporoot_dir,
                                                          r'input_data/Model_price_data_python_v5.xlsx'),
                                             sheet_name='MEANhigh_VARhigh', header=0, index_col=0,
                                             usecols='A,L')
}
# converting index into Datetime index
for gas_use_cost_scenario in [
    'MeanLow-VarLow-GP1.6to1', 'MeanLow-VarLow-GP1to1',
    'MeanHigh-VarLow-GP1.6to1',
    'MeanHigh-VarLow-GP1to1',
    'MeanLow-VarHigh-GP1.6to1', 'MeanLow-VarHigh-GP1to1',
    'MeanHigh-VarHigh-GP1.6to1', 'MeanHigh-VarHigh-GP1to1'
]:
    all_gas_prices[gas_use_cost_scenario].index = pd.to_datetime(all_gas_prices[gas_use_cost_scenario].index,
                                                                 dayfirst=True,
                                                                 format='mixed')  # format='%d.%m.%Y %H:%M'
# define scenarios:
HP_integration_scenarios = ['FullyIntegrated', 'PlugIn']
el_price_scenarios = [
    'MeanLow-VarLow',
    'MeanHigh-VarLow',
    'MeanLow-VarHigh',
    'MeanHigh-VarHigh'
]
gas_use_cost_scenarios = [
    'MeanLow-VarLow-GP1.6to1', 'MeanLow-VarLow-GP1to1',
    'MeanHigh-VarLow-GP1.6to1', 'MeanHigh-VarLow-GP1to1',
    'MeanLow-VarHigh-GP1.6to1',
    'MeanLow-VarHigh-GP1to1',
    'MeanHigh-VarHigh-GP1.6to1', 'MeanHigh-VarHigh-GP1to1'
]
capex_scenarios = ['HighHP-LowRest', 'LowHP-HighRest']

hours = 8000

# ---------------------- Access the results and export them to csv files (post processing) -----------------------------
# open pickle file with results dictionary
# insert name of pickle file which should be converted
for filename in ['outputs_with_CHP_minload0%_opt005__20241129T0759']:
    with open(filename + '.pickle', 'rb') as handle:
        all_scenarios_dict = pickle.load(handle)

# create the dataframe for files containing both heat pump scenarios
results_df = pd.DataFrame.from_records(
    [
        (HPtype, ELscenario, NGscenario, CAPEXscenario, system, amp, parameter, value)
        for HPtype, ELscenario_dict in all_scenarios_dict.items()
        for ELscenario, NGscenario_dict in ELscenario_dict.items()
        for NGscenario, CAPEXscenario_dict in NGscenario_dict.items()
        for CAPEXscenario, system_dict in CAPEXscenario_dict.items()
        for system, amp_dict in system_dict.items()
        for amp, parameter_dict in amp_dict.items()
        for parameter, value in parameter_dict['results'].items()
        if
        parameter in ['Optimal result', 'CAPEX', 'OPEX', 'scope 1 emissions', 'required space',
                      'CHP excess heat gen [MWh]', 'Heat pump size [MW]', 'ElB size [MW]',
                      'Battery size [MWh]', 'TES size [MWh]', 'electrolyser size [MW]',
                      'Hydrogen boiler size [MW]', 'Hydrogen storage size [MWh]',
                      'Simultaneous charging and discharging hours battery',
                      'Simultaneous charging and discharging hours TES',
                      'Simultaneous charging and discharging hours H2S',
                      'GT excess electricity gen [MWh]', 'GT electricity gen to grid [MWh]',
                      'total natural gas consumption [MWh]', 'grid to process [MWh]',
                      ]
    ],
    columns=(['HPtype', 'ELscenario', 'NGscenario', 'CAPEXscenario', 'system', 'amp', 'parameter', 'value'])
)
# store the dataframe as csv file
filename = 'results_' + \
           filename \
           + '.csv'
results_csv_data = results_df.to_csv(filename, index=False)
print('Saved csv file containing the results.')

# create the dataframe for files containing just ONE SCENARIO
# insert name of pickle file which should be converted
# for filename in ['PI_MeanLow-VarHigh-GP1.6to1_HighHP-LowRest_minload30__opt005__20241015T1746',
#                  'PI_MeanLow-VarHigh-GP1.6to1_LowHP-HighRest_minload30__opt005__20241016T0017',
#                  'PI_MeanLow-VarHigh-GP1to1_HighHP-LowRest_minload30%_opt005__20241014T2111',
#                  'PI_MeanLow-VarHigh-GP1to1_LowHP-HighRest_minload30__opt005__20241016T0747']:
#     with open(filename +
#               '.pickle', 'rb') as handle:
#   all_scenarios_dict = pickle.load(handle)
# results_df = pd.DataFrame.from_records(
#     [
#         (system, amp, parameter, value)
#         for system, amp_dict in all_scenarios_dict.items()
#         for amp, parameter_dict in amp_dict.items()
#         for parameter, value in parameter_dict['results'].items()
#         if parameter in ['Optimal result', 'CAPEX', 'OPEX', 'scope 1 emissions',
#                          'required space',
#                          'CHP excess heat gen [MWh]', 'CHP excess electricity gen [MWh]',
#                          'CHP electricity gen to grid [MWh]', 'Heat pump size [MW]', 'ElB size [MW]',
#                          'Battery size [MWh]', 'TES size [MWh]', 'electrolyser size [MW]',
#                          'Hydrogen boiler size [MW]', 'Hydrogen storage size [MWh]',
#                          'Simultaneous charging and discharging hours battery',
#                          'Simultaneous charging and discharging hours TES',
#                          'Simultaneous charging and discharging hours H2S',
#                          'GT excess electricity gen [MWh]', 'GT electricity gen to grid [MWh]',
#                          ]
#     ],
#     columns=(['system', 'amp', 'parameter', 'value'])
# )
# #print(results_df)
#
# filename = 'results_' + \
#            filename \
#            + '.csv'
# results_csv_data = results_df.to_csv(filename, index=False)
# print('Saved csv file containing the results.')

# create the AGGREGATED ENERGY FLOWS dataframe
results_flows_df = pd.DataFrame.from_records(
    [
        (HPtype, ELscenario, NGscenario, CAPEXscenario, system, amp, parameter, value)
        for HPtype, ELscenario_dict in all_scenarios_dict.items()
        for ELscenario, NGscenario_dict in ELscenario_dict.items()
        for NGscenario, CAPEXscenario_dict in NGscenario_dict.items()
        for CAPEXscenario, system_dict in CAPEXscenario_dict.items()
        for system, amp_dict in system_dict.items()
        for amp, parameter_dict in amp_dict.items()
        for parameter, value in parameter_dict['results'].items()
        if parameter in ['CHP heat gen to CP [MWh]', 'GT electricity gen to process [MWh]',
                         'CHP heat gen to TES [MWh]', 'CHP excess heat gen [MWh]', 'GT electricity gen to HP [MWh]',
                         'GT electricity gen to battery [MWh]', 'GT electricity gen to ElB [MWh]',
                         'GT excess electricity gen [MWh]',
                         'GT electricity gen to H2E [MWh]',
                         'GT excess electricity gen [MWh]', 'GT electricity gen to grid [MWh]',
                         'total natural gas consumption [MWh]', 'grid to process [MWh]', 'grid to battery [MWh]',
                         'grid to electric boiler [MWh]', 'grid to electrolyser [MWh]', 'grid to HP [MWh]',
                         'ElB gen to CP [MWh]', 'ElB gen to TES [MWh]', 'battery to ElB [MWh]',
                         'battery to electrolyser [MWh]', 'battery to HP [MWh]',
                         'battery to grid [MWh]', 'battery to process [MWh]', 'TES to CP [MWh]',
                         'H2 from electrolyser to boiler [MWh]',
                         'H2 from electrolyser to storage [MWh]', 'Hydrogen boiler to CP [MWh]',
                         'H2 from storage to boiler [MWh]', 'Heat from HP to CP [MWh]',
                         'Heat from HP to TES [MWh]',
                         ]
    ],
    columns=(['HPtype', 'ELscenario', 'NGscenario', 'CAPEXscenario', 'system', 'amp', 'parameter', 'value'])
)
# print(results_flows_df)
filename_flows = 'aggregated_energyflows_' + \
                 filename \
                 + '.csv'
results_csv_data = results_flows_df.to_csv(filename_flows, index=False)
print('Saved csv file containing the results.')

# create the AGGREGATED ENERGY FLOWS dataframe for ONE scenario
# insert name of pickle file which should be converted
# for filename in ['PI_MeanLow-VarHigh-GP1.6to1_HighHP-LowRest_minload30__opt005__20241015T1746',
#                  'PI_MeanLow-VarHigh-GP1.6to1_LowHP-HighRest_minload30__opt005__20241016T0017',
#                  'PI_MeanLow-VarHigh-GP1to1_HighHP-LowRest_minload30%_opt005__20241014T2111',
#                  'PI_MeanLow-VarHigh-GP1to1_LowHP-HighRest_minload30__opt005__20241016T0747']:
#     with open(filename +
#               '.pickle', 'rb') as handle:
# all_scenarios_dict = pickle.load(handle)
# results_flows_df = pd.DataFrame.from_records(
#     [
#         (system, amp, parameter, value)
#         for system, amp_dict in all_scenarios_dict.items()
#         for amp, parameter_dict in amp_dict.items()
#         for parameter, value in parameter_dict['results'].items()
#         if parameter in ['CHP heat gen to CP [MWh]', 'GT electricity gen to process [MWh]',
#                          'CHP heat gen to TES [MWh]', 'CHP excess heat gen [MWh]', 'GT electricity gen to HP [MWh]',
#                          'GT electricity gen to battery [MWh]', 'GT electricity gen to ElB [MWh]',
#                          'GT excess electricity gen [MWh]',
#                          'GT electricity gen to H2E [MWh]',
#                          'GT excess electricity gen [MWh]', 'GT electricity gen to grid [MWh]',
#                          'total natural gas consumption [MWh]', 'grid to process [MWh]', 'grid to battery [MWh]',
#                          'grid to electric boiler [MWh]', 'grid to electrolyser [MWh]', 'grid to HP [MWh]',
#                          'ElB gen to CP [MWh]', 'ElB gen to TES [MWh]', 'battery to ElB [MWh]',
#                          'battery to electrolyser [MWh]', 'battery to HP [MWh]',
#                          'battery to grid [MWh]', 'battery to process [MWh]', 'TES to CP [MWh]',
#                          'H2 from electrolyser to boiler [MWh]',
#                          'H2 from electrolyser to storage [MWh]', 'Hydrogen boiler to CP [MWh]',
#                          'H2 from storage to boiler [MWh]', 'Heat from HP to CP [MWh]',
#                          'Heat from HP to TES [MWh]',
#                          ]
#     ],
#     columns=(['system', 'amp', 'parameter', 'value'])
# )
# #print(results_flows_df)
# filename_flows = 'aggregated_energyflows_' + \
#                  filename \
#                  + '.csv'
#
# results_csv_data = results_flows_df.to_csv(filename_flows, index=False)
# print('Saved csv file containing the results.')

# create the HOURLY ENERGY FLOWS dataframe for (plug in of fully integrated) heat pump scenarios
for HP_integration_scenario in ['PlugIn']:
    for i, el_price_scenario in enumerate(el_price_scenarios):
        price_el_hourly = all_electricity_prices[el_price_scenario]
        for j in [2 * i, 2 * i + 1]:
            gas_use_cost_scenario = gas_use_cost_scenarios[j]
            price_NG_use = all_gas_prices[gas_use_cost_scenario]
            for capex_scenario in capex_scenarios:
                hourly_energyflows = all_scenarios_dict[HP_integration_scenario][el_price_scenario][
                    gas_use_cost_scenario][capex_scenario]['new system']['original']['energy flows']

                filename_hourly_flows = 'hourly_enflows_' + HP_integration_scenario + gas_use_cost_scenario + \
                                        capex_scenario + \
                                        '_SAminload_20241129T0759' \
                                        + '.csv'
                results_csv_data = hourly_energyflows.to_csv(filename_hourly_flows, index=True)
                print('Saved csv file containing the results.')

# create the HOURLY ENERGY FLOWS dataframe for one (plug in of fully integrated) heat pump scenario
# insert name of pickle file which should be converted
# for filename in ['PI_MeanLow-VarHigh-GP1.6to1_HighHP-LowRest_minload30__opt005__20241015T1746',
#                  'PI_MeanLow-VarHigh-GP1.6to1_LowHP-HighRest_minload30__opt005__20241016T0017',
#                  'PI_MeanLow-VarHigh-GP1to1_HighHP-LowRest_minload30%_opt005__20241014T2111',
#                  'PI_MeanLow-VarHigh-GP1to1_LowHP-HighRest_minload30__opt005__20241016T0747']:
#     with open(filename +
#               '.pickle', 'rb') as handle:
#         all_scenarios_dict = pickle.load(handle)
#     hourly_energyflows = \
#         all_scenarios_dict['new system']['original']['energy flows']
#
#     filename_hourly_flows = 'hourly_energyflows_' + \
#                             filename \
#                             + '.csv'
#     results_csv_data = hourly_energyflows.to_csv(filename_hourly_flows, index=True)
#     print('Saved csv file containing the results.')

# # create the dataframe for the benchmark model
# results_df_benchmark = pd.DataFrame.from_records(
#     [
#         (ELscenario, NGscenario, system, amp, parameter, value)
#         for ELscenario, NGscenario_dict in all_scenarios_dict.items()
#         for NGscenario, system_dict in NGscenario_dict.items()
#         for system, amp_dict in system_dict.items()
#         for amp, parameter_dict in amp_dict.items()
#         for parameter, value in parameter_dict['results'].items()
#         if parameter in ['Optimal result', 'CAPEX', 'OPEX', 'scope 1 emissions', 'scope 2 emissions',
#                          'CHP excess heat gen [MWh]', 'CHP excess electricity gen [MWh]',
#                          'CHP electricity gen to grid [MWh]']
#     ],
#     columns=(['ELscenario', 'NGscenario', 'system', 'amp', 'parameter', 'value'])
# )
# print(results_df_benchmark)
# filename_benchmark = 'results_' + \
#                      'benchmark_with_CHP_minload30%__20240828T1509' \
#                      + '.csv'
# results_csv_data = results_df_benchmark.to_csv(filename_benchmark, index=False)
# print('Saved csv file containing the results.')
#
# # create the ENERGY FLOWS dataframe for the benchmark model
# results_df_benchmark = pd.DataFrame.from_records(
#     [
#         (ELscenario, NGscenario, system, amp, parameter, value)
#         for ELscenario, NGscenario_dict in all_scenarios_dict.items()
#         for NGscenario, system_dict in NGscenario_dict.items()
#         for system, amp_dict in system_dict.items()
#         for amp, parameter_dict in amp_dict.items()
#         for parameter, value in parameter_dict['results'].items()
#         if parameter in ['CHP heat gen to CP [MWh]', 'GT electricity gen to process [MWh]',
#                          'CHP excess heat gen [MWh]', 'GT excess electricity gen [MWh]',
#                          'GT electricity gen to grid [MWh]', 'total natural gas consumption [MWh]',
#                          'grid to process [MWh]']
#     ],
#     columns=(['ELscenario', 'NGscenario', 'system', 'amp', 'parameter', 'value'])
# )
# print(results_df_benchmark)
# filename_benchmark = 'energyflows_' + \
#                      filename_benchmark \
#                      + '.csv'
# results_csv_data = results_df_benchmark.to_csv(filename_benchmark, index=False)
# print('Saved csv file containing the results.')
