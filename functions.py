import pyomo.environ as pm
import pandas as pd
import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# ________________________________________ Optimisation with plug-in heat pump _________________________________________
def optimisation_run_PI_CHP(heat_demand_orig, price_el_hourly, price_NG_use, amp_values, variability_values,
                            GT_min_load, hours, capex_data):
    print("Started optimisation with PI heat pump.")

    # ------------------------------------- input DATA preperation ---------------------------------------------------
    # define the resolution of the optimisation problem in hours
    time_step = 0.5  # in hours

    # resample natural gas price data to get prices with the same resolution of the optimisation problem
    # original data contains one price per day
    price_NG_use_hourly = price_NG_use.resample('{}H'.format(1)).ffill()
    price_NG_use_half_hourly = price_NG_use_hourly.resample('30T').ffill()  # resample to get half-hourly prices

    # prepare electricity price data
    el_row_NaN = price_el_hourly[price_el_hourly.isna().any(axis=1)]  # indicates row with NaN value
    price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
    price_el_hourly.index = price_NG_use_hourly.index # use index from natural gas price data
    price_el_half_hourly = price_el_hourly.resample('30T').ffill()  # resample data to get half-hourly prices
    # 'cut' the data and replace full data sets by short data sets
    price_el_hourly_short = pd.DataFrame(price_el_hourly.iloc[0:7768])
    price_el_hourly = price_el_hourly_short
    price_el_half_hourly_short = pd.DataFrame(price_el_half_hourly.iloc[0:hours * 2])  # 8000 operational hours per year
    price_el_half_hourly = price_el_half_hourly_short

    # manipulate ELECTRICITY price data to increase the amplitude of the price variation
    # get average price of original price data
    price_el_half_hourly_mean = price_el_half_hourly.mean()
    # sort them to plot price duration-curve
    price_el_hourly_sorted = price_el_hourly['EP [EUR/MWh]'].sort_values(ascending=False)
    price_el_hourly_sorted_df = pd.DataFrame(price_el_hourly_sorted)
    price_el_half_hourly_sorted = price_el_half_hourly['EP [EUR/MWh]'].sort_values(ascending=False)
    price_el_half_hourly_sorted_df = pd.DataFrame(price_el_half_hourly_sorted)
    # generate new price profiles and sort their values from high to low to plot price duration curves
    if len(amp_values) > 0:
        for k in amp_values:
            print("Current k is: ", k)
            colname = ("amp " + "%.3f") % k
            # add new price data as additional columns to dataframe
            price_el_half_hourly[str(colname)] = price_el_half_hourly_mean.iloc[0] + \
                                                 k * (price_el_half_hourly['EP [EUR/MWh]'] -
                                                      price_el_half_hourly_mean.iloc[0])

            # sort values from high to low and add new column to dataframe
            price_el_half_hourly_sorted[str(colname)] = price_el_half_hourly[str(colname)].sort_values(ascending=False)
            price_el_half_hourly_sorted_df[str(colname)] = price_el_half_hourly_sorted[str(colname)]
            # remove the index
            price_el_half_hourly_sorted_df = price_el_half_hourly_sorted_df.reset_index(drop=True)
        price_el_half_hourly_mean_df = pd.DataFrame(price_el_half_hourly.mean())
        # calculate new mean values
        for k in amp_values:
            print("Current k is: ", k)
            colname = ("amp " + "%.3f") % k
            # removing negative prices
            price_el_half_hourly.loc[price_el_half_hourly[str(colname)] < 0, str(colname)] = 0
            # calculate new mean values
            price_el_half_hourly_mean_df[str(colname)] = price_el_half_hourly[str(colname)].mean()

    # ---------------- Preparation of dictionaries in which results are stored for each model run ----------------------
    # for electricity
    price_el_hourly.rename(columns={'EP [EUR/MWh]': 'original'}, inplace=True)
    price_el_half_hourly.rename(columns={'EP [EUR/MWh]': 'original'}, inplace=True)

    # create respective dictionary and define the variable over which the model runs loop
    looping_variable = variability_values
    el_price_scenario_dict = {'new system': {amp: {'results': {}, 'energy flows': {}} for amp in looping_variable}}
    # print(el_price_scenario_dict)

    # -------------------------------------- Start the model runs (in loop) --------------------------------------------
    for count, amp in enumerate(looping_variable):
        print("Current scenario is: ", amp)
        current_process_dict = el_price_scenario_dict['new system'][amp]
        # get the heat demand data
        H_dem = heat_demand_orig[0:hours * 2]
        H_dem_max = heat_demand_orig[0:hours * 2].max().iloc[0]
        # define power demand data
        P_dem = 0.1 * heat_demand_orig[0:hours * 2] # Power demand is 10% of the heat demand
        P_dem.rename(columns={'Heat demand [MW]': 'Power demand [MW]'}, inplace=True)

        # ------------------ START OPTIMISATION --------------------------------------------------------------------
        # Definitions of constraints

        def heat_balance(m, time):  # heat demand at t has to be equal to the sum of the heat produced at t
            return float(H_dem.iloc[time]) == m.H_ElB_CP[time] + m.H_CHP_CP[time] + m.H_TES_CP[time] + m.H_H2B_CP[time] \
                   + m.H_HP_CP[time]

        def power_balance(m, time):  # power demand at t has to be equal to the sum of the power produced or bought at t
            return float(P_dem.iloc[time]) == m.P_gr_process[time] + m.P_GT_process[time] + m.P_bat_process[time]

        def ElB_balance(m, time):  # energy conversion of electric boiler
            return m.H_ElB_CP[time] + m.H_ElB_TES[time] == (m.P_gr_ElB[time] + m.P_bat_ElB[time] + m.P_GT_ElB[time]) \
                   * eta_ElB

        def ElB_size(m, time):  # definition of boiler capacity
            return m.H_ElB_CP[time] + m.H_ElB_TES[time] <= m.ElB_cap

        def GT_ng_P_conversion(m, time):  # gas to power conversion of gas turbine
            return m.NG_GT_in[time] * eta_GT_el == m.P_GT_excess[time] + m.P_GT_bat[time] + m.P_GT_ElB[time] + \
                   m.P_GT_H2E[time] + m.P_GT_HP[time] + m.P_GT_process[time] + m.P_GT_gr[time]

        def CHP_ng_H_conversion(m, time):  # gas to heat conversion of gas turbine and re-boiler
            return (m.NG_GT_in[time] * eta_GT_th + m.NG_GB_in[time]) * eta_GB == \
                   m.H_CHP_CP[time] + m.H_CHP_TES[time] + m.H_CHP_excess[time]

        def GT_cap_rule(m, time):  # gas intake cannot exceed capacity
            return m.NG_GT_in[time] <= GT_cap / eta_GT_th # * eta_GB)  #GT_cap = H_dem_max / eta_GB #outflow GT

        def GB_cap_rule(m, time):  # gas intake can not exceed capacity
            return m.NG_GB_in[time] <= 0.2 * GT_cap / eta_GB  # GB_cap = 0.2 * GT_cap  #outflow GB

        def GT_min_load_rule(m, time):  # operation has to be above the minimal load
            return m.NG_GT_in[time] >= GT_cap / eta_GT_th * GT_min_load

        def bat_soe(m, time):  # calculating the state of energy of the battery
            if time == 0:
                return m.bat_soe[time] == 0
            else:
                return m.bat_soe[time] == m.bat_soe[time - 1] + \
                       (m.P_gr_bat[time - 1] + m.P_GT_bat[time - 1]) * eta_bat * time_step - \
                       (m.P_bat_ElB[time - 1] + m.P_bat_H2E[time - 1] + m.P_bat_HP[time - 1] + \
                        m.P_bat_process[time - 1] + m.P_bat_gr[time - 1]) / eta_bat * time_step  # / m.bat_cap[time]

        def bat_in(m, time):  # limiting the charging with c-rate and use binary to prevent simultaeous charging and discharging
            return (m.P_gr_bat[time] + m.P_GT_bat[time]) * eta_bat <= m.bat_cap * crate_bat / time_step * m.b1[time]

        def bat_out_maxP(m, time):  # limiting the discharging with c-rate and use binary to prevent simultaeous charging and discharging
            if time == 0:
                return (m.P_bat_ElB[time] + m.P_bat_H2E[time] + m.P_bat_HP[time] + m.P_bat_process[time] +
                        m.P_bat_gr[time]) / eta_bat == 0
            else:
                return (m.P_bat_ElB[time] + m.P_bat_H2E[time] + m.P_bat_HP[time] + m.P_bat_process[time] +
                        m.P_bat_gr[time]) / eta_bat <= m.bat_cap * crate_bat / time_step * (1 - m.b1[time])

        def bat_soe_max(m, time):  # define battery capacity
            return m.bat_soe[time] <= m.bat_cap

        def TES_soe(m, time):  # calculating the state of energy of the TES
            if time == 0:
                return m.TES_soe[time] == 0
            else:
                return m.TES_soe[time] == m.TES_soe[time - 1] \
                       + ((m.H_CHP_TES[time - 1] + m.H_ElB_TES[time - 1] + m.H_HP_TES[time - 1]) * eta_TES
                          - m.H_TES_CP[time - 1]) * time_step

        def TES_in(m, time):  # limiting the charging with c-rate and use binary to prevent simultaeous charging and discharging
            return (m.H_CHP_TES[time] + m.H_ElB_TES[time] + m.H_HP_TES[
                time]) * eta_TES <= m.TES_cap * crate_TES / time_step * m.b2[time]


        def TES_out(m, time):  # limiting the discharging with c-rate and use binary to prevent simultaeous charging and discharging
            if time == 0:
                return m.H_TES_CP[time] == 0
            else:
                return m.H_TES_CP[time] <= m.TES_cap * crate_TES / time_step * (1 - m.b2[time])

        def TES_size(m, time):  # define TES capacity
            return m.TES_soe[time] <= m.TES_cap

        def HP_balance(m, time):  # power to heat conversion of the heat pump
            return m.H_HP_CP[time] + m.H_HP_TES[time] == (m.P_gr_HP[time] + m.P_bat_HP[time] + m.P_GT_HP[time]) \
                   * HP_COP_carnot * eta_HP

        def HP_size(m, time):  # defining HP capacity
            return m.H_HP_CP[time] + m.H_HP_TES[time] <= m.HP_cap

        def H2S_soe(m, time):  # calculate state of energy of the hydrogen storage
            if time == 0:
                return m.H2S_soe[time] == 0
            else:
                return m.H2S_soe[time] == m.H2S_soe[time - 1] + (m.H2_H2E_H2S[time - 1] * eta_H2S -
                                                                 m.H2_H2S_H2B[time - 1]) * time_step

        def H2S_in(m, time):  # implement binary to prevent simultaneous charging and discharging
            return m.H2_H2E_H2S[time] * eta_H2S <= m.H2S_cap / time_step * m.b4[time]

        def H2S_out(m, time):  # implement binary to prevent simultaneous charging and discharging
            if time == 0:
                return m.H2_H2S_H2B[time] == 0
            else:
                return m.H2_H2S_H2B[time] <= m.H2S_soe[time] / time_step * (1 - m.b4[time])

        def H2S_size(m, time):  # define hydrogen storage capacity
            return m.H2S_soe[time] <= m.H2S_cap

        def H2B_balance(m, time):  # hydrogen to heat conversion of hyrogen boiler
            return (m.H2_H2E_H2B[time] + m.H2_H2S_H2B[time]) * eta_H2B == m.H_H2B_CP[time]

        def H2B_size(m, time):  # define hydrogen boiler size
            return m.H_H2B_CP[time] <= m.H2B_cap

        def H2E_balance(m, time):  # power to hydrogen conversion of electrolyzer
            return (m.P_gr_H2E[time] + m.P_GT_H2E[time] + m.P_bat_H2E[time]) * eta_H2E == m.H2_H2E_H2B[time] + \
                   m.H2_H2E_H2S[time]

        def H2E_size(m, time):  # define capacity of electrolyzer
            return m.P_gr_H2E[time] + m.P_GT_H2E[time] + m.P_bat_H2E[time] <= m.H2E_cap

        def max_grid_power_in(m, time):  # limit power inflow through the grid connection and use binary to prevent simultaneous bidirectional use
            return m.P_gr_ElB[time] + m.P_gr_HP[time] + m.P_gr_bat[time] + m.P_gr_H2E[time] + m.P_gr_process[time] <= \
                   gr_connection * m.b3[time]  # total power flow from grid to plant is limited to x MW

        def max_grid_power_out(m, time):  # limit power outflow through the grid connection and use binary to prevent simultaneous bidirectional use
            return m.P_GT_gr[time] + m.P_bat_gr[time] <= gr_connection * (
                    1 - m.b3[time])  # total power flow from grid to plant is limited to x MW

        def minimize_total_costs(m, time):  # define the total cost of the system
            return sum(price_el_half_hourly.iloc[time, count] * time_step * (m.P_gr_ElB[time] + m.P_gr_bat[time]
                                                                             + m.P_gr_H2E[time] + m.P_gr_HP[time]
                                                                             + m.P_gr_process[time]
                                                                             - (m.P_GT_gr[time] + m.P_bat_gr[time]))
                       + (m.NG_GT_in[time] + m.NG_GB_in[time]) * time_step * price_NG_use_half_hourly.iloc[time, 0]
                       for time in m.T) + \
                   m.bat_cap * c_bat * if_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                   m.ElB_cap * c_ElB * if_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                   m.TES_cap * c_TES_C * if_TES * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                   m.HP_cap * c_HP * if_HP * disc_rate / (1 - (1 + disc_rate) ** -HP_lifetime) + \
                   m.H2E_cap * c_H2E * if_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                   m.H2B_cap * c_H2B * if_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                   m.H2S_cap * c_H2S * if_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)

        m = pm.ConcreteModel()

        # define SETS
        m.T = pm.RangeSet(0, hours * 2 - 1)  # time steps

        # define CONSTANTS
        disc_rate = 0.1  # 10%, assumption
        EF_ng = 0.2  # emission factor natural gas, tCO2/MWh(CH4)
        gr_connection = 30  # [MW] Grid connection capacity
        # ElB constants
        c_ElB = capex_data['ElB']  # CAPEX for electric boiler, 60000 eur/MW
        ElB_lifetime = 20  # lifetime of electric boiler, years
        ElB_spatialreq = 70  # spatial requirements, m^2/MW
        eta_ElB = 0.99  # Conversion ratio electricity to steam for electric boiler [%]
        if_ElB = 2  # installation factor, [-]
        # CHP constants
        eta_GT_el = 0.3  # Electric efficiency of GT [%]
        eta_GT_th = 0.6  # Thermal efficiency of GT [%]
        eta_GB = 0.82
        GT_cap = H_dem_max / eta_GB  # Thermal capacity (LPS) GT, [MW]
        # Battery constants
        eta_bat = 0.95  # Battery (dis)charging efficiency
        c_bat = capex_data['Bat']  # CAPEX for battery per eur/MWh, 386e3 USD --> 385.5e3 eur (12.07.23)
        bat_lifetime = 20  # lifetime of battery
        bat_spatialreq = 11  # spatial requirement, [m^2/MWh]
        crate_bat = 0.7  # C rate of battery, 70% of nominal capacity, [-]
        if_bat = 2.5  # installation factor, [-]
        # TES constants
        c_TES_C = capex_data['TES']  # CAPEX for latent heat storage, including installation factor [eur/MWh]
        TES_lifetime = 25  # heat storage lifetime, [years]
        eta_TES = 0.95  # discharge efficiency [-]
        TES_spatialreq = 7  # spatial requirement TES (configuration B), [m^2/MWh]
        crate_TES = 0.5  # C rate of TES, 50% of nominal capacity, [-]
        if_TES = 2.5  # installation factor, [-]
        # Heat pump constants
        c_HP = capex_data['HP']
        HP_lifetime = 20
        HP_spatialreq = 1
        HP_COP_carnot = (160 + 273) / (160 - 55)
        eta_HP = 0.5
        if_HP = 3  # installation factor, [-]
        # Hydrogen equipment constants
        eta_H2S = 0.9  # charge efficiency hydrogen storage [-]
        eta_H2B = 0.9  # conversion efficiency hydrogen boiler [-]
        eta_H2E = 0.69  # conversion efficiency electrolyser [-]
        c_H2S = capex_data['H2S']  # CAPEX for hydrogen storage per MWh, [eur/MWh]
        c_H2B = capex_data['H2B']  # CAPEX for hydrogen boiler per MW, [eur/MW]
        c_H2E = capex_data['H2E']  # CAPEX for electrolyser per MW, [eur/MW]  # From ISPT
        H2S_lifetime = 23  # lifetime hydrogen storage, [years]
        H2B_lifetime = 20  # lifetime hydrogen boiler, [years]
        H2E_lifetime = 14  # lifetime electrolyser, [years]
        H2E_spatialreq = 105  # spatial requirement electrolyser, [m^2/MW]
        H2B_spatialreq = 70  # spatial requirement hydrogen boiler, [m^2/MW]
        H2S_spatialreq = 8.4  # spatial requirement hydrogen storage, [m^2/MWh]
        if_H2S = 4  # installation factor, [-]
        if_H2B = 2  # installation factor, [-]
        if_H2E = 1  # installation factor, [-]

        param_NaN = math.isnan(sum(m.component_data_objects(ctype=type)))

        # define VARIABLES
        m.NG_GT_in = pm.Var(m.T, bounds=(0, None))  # natural gas intake of gas turbine, MWh
        m.NG_GB_in = pm.Var(m.T, bounds=(0, None))  # natural gas intake of gas boiler, MWh
        m.P_GT_bat = pm.Var(m.T, bounds=(0, None))  # Power from CHP to battery, MW
        m.P_GT_HP = pm.Var(m.T, bounds=(0, None))  # Power from CHP to heat pump, MW
        m.P_GT_ElB = pm.Var(m.T, bounds=(0, None))  # Power from CHP to electric boiler, MW
        m.P_GT_H2E = pm.Var(m.T, bounds=(0, None))  # Power from CHP to electrolyser, MW
        m.P_GT_process = pm.Var(m.T, bounds=(0, None))  # Power from CHP to process, MW
        m.P_GT_excess = pm.Var(m.T, bounds=(0, None))  # Excess power from CHP, MW
        m.P_GT_gr = pm.Var(m.T, bounds=(0, None))  # Power from CHP to grid, MW
        m.H_CHP_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from CHP (natural gas), MW
        m.H_CHP_TES = pm.Var(m.T, bounds=(0, None))  # Heat from CHP to TES, MW
        m.H_CHP_excess = pm.Var(m.T, bounds=(0, None))  # Excess heat from CHP, MW
        m.P_gr_ElB = pm.Var(m.T, bounds=(0, None))  # grid to el. boiler, MW
        m.P_gr_bat = pm.Var(m.T, bounds=(0, None))  # max charging power batter, MW
        m.P_gr_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from grid to electrolyser, MW
        m.P_gr_HP = pm.Var(m.T, bounds=(0, None))  # power flow from grid to heat pump, MW
        m.P_gr_process = pm.Var(m.T, bounds=(0, None))  # power flow from grid to process, MW
        m.P_bat_ElB = pm.Var(m.T, bounds=(0, None))  # discharging power batter to electric boiler, MW
        m.P_bat_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from battery to electrolyser, MW
        m.P_bat_HP = pm.Var(m.T, bounds=(0, None))  # power flow from battery to heat pump, MW
        m.P_bat_process = pm.Var(m.T, bounds=(0, None))  # power flow from battery to process, MW
        m.P_bat_gr = pm.Var(m.T, bounds=(0, None))  # power flow from battery to gr, MW
        m.H_ElB_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from electricity, MW
        m.H_ElB_TES = pm.Var(m.T, bounds=(0, None))  # Heat from electric boiler to TES, MW
        m.H_TES_CP = pm.Var(m.T, bounds=(0, None))  # Heat from TES to core process, MW
        m.H_H2B_CP = pm.Var(m.T, bounds=(0, None))  # Heat flow from hydrogen boiler to core process, MW
        m.H_HP_CP = pm.Var(m.T, bounds=(0, None))  # Heat flow from heat pump to core process, MW
        m.H_HP_TES = pm.Var(m.T, bounds=(0, None))  # Heat from heat pump to TES, MW
        m.H2_H2E_H2S = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen storage, MWh
        m.H2_H2S_H2B = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from hydrogen storage to hydrogen boiler, MWh
        m.H2_H2E_H2B = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen boiler, MWh
        m.TES_soe = pm.Var(m.T, bounds=(0, None))  # state of energy TES, MWh
        m.bat_soe = pm.Var(m.T, bounds=(0, None))  # State of energy of battery
        m.H2S_soe = pm.Var(m.T, bounds=(0, None))  # state of energy hydrogen storage, MWh
        m.bat_cap = pm.Var(bounds=(0, None))  # Battery capacity, MWh
        m.ElB_cap = pm.Var(bounds=(0, None))  # electric boiler capacity, MW
        m.TES_cap = pm.Var(bounds=(0, None))  # TES capacity, MWh
        m.HP_cap = pm.Var(bounds=(0, None))  # heat pump capacity, MW
        m.H2S_cap = pm.Var(bounds=(0, None))  # hydrogen storage capacity, MWh
        m.H2B_cap = pm.Var(bounds=(0, None))  # hydrogen boiler capacity, MW
        m.H2E_cap = pm.Var(bounds=(0, None))  # electrolyser capacity, MW
        m.b1 = pm.Var(m.T, within=pm.Binary)  # binary variable battery
        m.b2 = pm.Var(m.T, within=pm.Binary)  # binary variable TES
        m.b3 = pm.Var(m.T, within=pm.Binary)  # binary variable grid connection
        m.b4 = pm.Var(m.T, within=pm.Binary)  # binary variable H2S

        # add CONSTRAINTS to the model
        # balance supply and demand
        m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
        m.power_balance_constraint = pm.Constraint(m.T, rule=power_balance)
        # CHP constraints
        m.CHP_ng_H_conversion_constraint = pm.Constraint(m.T, rule=CHP_ng_H_conversion)
        m.GT_ng_P_conversion_constraint = pm.Constraint(m.T, rule=GT_ng_P_conversion)
        m.GT_cap_constraint = pm.Constraint(m.T, rule=GT_cap_rule)
        m.GB_cap_constraint = pm.Constraint(m.T, rule=GB_cap_rule)
        m.GT_min_load_constraint = pm.Constraint(m.T, rule=GT_min_load_rule)
        # electric boiler constraint
        m.ElB_size_constraint = pm.Constraint(m.T, rule=ElB_size)
        m.ElB_balance_constraint = pm.Constraint(m.T, rule=ElB_balance)
        # heat pump constraints
        m.HP_size_constraint = pm.Constraint(m.T, rule=HP_size)
        m.HP_balance_constraint = pm.Constraint(m.T, rule=HP_balance)
        # battery constraints
        m.bat_soe_constraint = pm.Constraint(m.T, rule=bat_soe)
        m.bat_out_maxP_constraint = pm.Constraint(m.T, rule=bat_out_maxP)
        m.bat_in_constraint = pm.Constraint(m.T, rule=bat_in)
        m.SOE_max_constraint = pm.Constraint(m.T, rule=bat_soe_max)
        # TES constraints
        m.TES_discharge_constraint = pm.Constraint(m.T, rule=TES_out)
        m.TES_charge_constraint = pm.Constraint(m.T, rule=TES_in)
        m.TES_soe_constraint = pm.Constraint(m.T, rule=TES_soe)
        m.TES_size_constraint = pm.Constraint(m.T, rule=TES_size)
        # hydrogen constraints
        m.H2S_soe_constraint = pm.Constraint(m.T, rule=H2S_soe)
        m.H2S_in_constraint = pm.Constraint(m.T, rule=H2S_in)
        m.H2S_out_constraint = pm.Constraint(m.T, rule=H2S_out)
        m.H2B_balance_constraint = pm.Constraint(m.T, rule=H2B_balance)
        m.H2E_balance_constraint = pm.Constraint(m.T, rule=H2E_balance)
        m.H2S_size_constraint = pm.Constraint(m.T, rule=H2S_size)
        m.H2B_size_constraint = pm.Constraint(m.T, rule=H2B_size)
        m.H2E_size_constraint = pm.Constraint(m.T, rule=H2E_size)
        # grid constraint
        m.max_grid_power_in_constraint = pm.Constraint(m.T, rule=max_grid_power_in)
        m.max_grid_power_out_constraint = pm.Constraint(m.T, rule=max_grid_power_out)

        # add OBJECTIVE FUNCTION
        m.objective = pm.Objective(rule=minimize_total_costs,
                                   sense=pm.minimize,
                                   doc='Define objective function')

        # Solve optimization problem
        opt = pm.SolverFactory('gurobi')  # use gurobi solvers
        opt.options["MIPGap"] = 0.0005  # define optimality gap
        results = opt.solve(m, tee=True)  # solve the problem

        # ------------------ OPTIMISATION END --------------------------------------------------------------------------

        # Collect results in dataframe
        result = pd.DataFrame(index=price_NG_use_half_hourly.index[0:hours * 2])
        result['Heat demand core process'] = list(H_dem['Heat demand [MW]'])
        result['Power demand core process'] = list(P_dem['Power demand [MW]'])
        result['Natural gas consumption GT [MW]'] = pm.value(m.NG_GT_in[:])
        result['Natural gas consumption GB [MW]'] = pm.value(m.NG_GB_in[:])
        result['Power from GT to battery'] = pm.value(m.P_GT_bat[:])
        result['Power from GT to heat pump'] = pm.value(m.P_GT_HP[:])
        result['Power from GT to electric boiler'] = pm.value(m.P_GT_ElB[:])
        result['Power from GT to electrolyser'] = pm.value(m.P_GT_H2E[:])
        result['Power from GT to process'] = pm.value(m.P_GT_process[:])
        result['Power excess from GT'] = pm.value(m.P_GT_excess[:])
        result['Power from GT to grid'] = pm.value(m.P_GT_gr[:])
        result['Heat from CHP to core process'] = pm.value(m.H_CHP_CP[:])
        result['Heat from CHP to TES'] = pm.value(m.H_CHP_TES[:])
        result['Heat excess from CHP'] = pm.value(m.H_CHP_excess[:])
        result['Power from grid to electric boiler'] = pm.value(m.P_gr_ElB[:])
        result['Power from grid to battery'] = pm.value(m.P_gr_bat[:])
        result['Power from grid to electrolyser'] = pm.value(m.P_gr_H2E[:])
        result['Power from grid to heat pump'] = pm.value(m.P_gr_HP[:])
        result['Power from grid to process'] = pm.value(m.P_gr_process[:])
        result['Battery to electric boiler'] = pm.value(m.P_bat_ElB[:])
        result['Battery to electrolyser'] = pm.value(m.P_bat_H2E[:])
        result['Battery to heat pump'] = pm.value(m.P_bat_HP[:])
        result['Battery to process'] = pm.value(m.P_bat_process[:])
        result['Battery to grid'] = pm.value(m.P_bat_gr[:])
        result['Heat from electric boiler to core process'] = pm.value(m.H_ElB_CP[:])
        result['Heat from electric boiler to TES'] = pm.value(m.H_ElB_TES[:])
        result['TES to CP'] = pm.value(m.H_TES_CP[:])
        result['Heat from hydrogen boiler to core process'] = pm.value(m.H_H2B_CP[:])
        result['Heat from heat pump to core process'] = pm.value(m.H_HP_CP[:])
        result['Heat from heat pump to TES'] = pm.value(m.H_HP_TES[:])
        result['H2E to H2S'] = pm.value(m.H2_H2E_H2S[:])
        result['H2S to H2B'] = pm.value(m.H2_H2S_H2B[:])
        result['H2E to H2B'] = pm.value(m.H2_H2E_H2B[:])
        result['TES SOE'] = pm.value(m.TES_soe[:])
        result['Battery SOE'] = pm.value(m.bat_soe[:])
        result['H2S SOE'] = pm.value(m.H2S_soe[:])

        grid_P_out = result['Power from grid to electric boiler'] + result['Power from grid to battery'] + \
                     result['Power from grid to electrolyser'] + result['Power from grid to heat pump'] + \
                     result['Power from grid to process']
        grid_P_out_max = max(grid_P_out)
        Grid_gen = result['Power from grid to electric boiler'].sum() + result['Power from grid to battery'].sum() \
                   + result['Power from grid to electrolyser'].sum() + result['Power from grid to heat pump'].sum() \
                   + result['Power from grid to process'].sum()
        # calculate CO2 emissions from natural gas consumption
        CO2_emissions = (result['Natural gas consumption GT [MW]'].sum() +
                         result[
                             'Natural gas consumption GB [MW]'].sum()) * EF_ng * time_step  # [MW]*[ton/MWh]*[h] = [ton]
        # control: H_CP==H_dem?
        control_H = sum(result['Heat demand core process'] - (result['Heat from electric boiler to core process']
                                                              + result['Heat from CHP to core process']
                                                              + result['TES to CP']
                                                              + result['Heat from hydrogen boiler to core process']
                                                              + result['Heat from heat pump to core process']))

        print("control_H =", control_H)
        # display total cost and installed capacities
        print("Total cost = ", pm.value(m.objective))
        print("HP capacity =", pm.value(m.HP_cap))
        print("Battery capacity =", pm.value(m.bat_cap))
        print("Electric boiler capacity =", pm.value(m.ElB_cap))
        print("TES capacity =", pm.value(m.TES_cap))
        print("electrolyser capacity =", pm.value(m.H2E_cap))
        print("Hydrogen boiler capacity =", pm.value(m.H2B_cap))
        print("Hydrogen storage capacity =", pm.value(m.H2S_cap))
        # display grid connection use
        print("Grid capacity: ", gr_connection, "Max. power flow from grid: ", grid_P_out_max)

        # IF battery capacity is installed, how many hours does the battery charge and discharge simultaneously?
        if pm.value(m.bat_cap) > 0:
            battery_discharge_sum = result['Battery to electrolyser'] + \
                                    result['Battery to electric boiler'] + \
                                    result['Battery to heat pump'] + result['Battery to grid'] + \
                                    result['Battery to process']
            battery_charge_sum = result['Power from grid to battery'] + \
                                 result['Power from GT to battery']

            battery_hours_with_simultaneous_charging_and_discharging = pd.Series(index=battery_charge_sum.index)
            for i in range(0, len(battery_charge_sum)):
                if battery_charge_sum[i] > 0:
                    if battery_discharge_sum[i] > 0:
                        battery_hours_with_simultaneous_charging_and_discharging[i] = battery_charge_sum[i] + \
                                                                                      battery_discharge_sum[i]
            print("Number of times of simultaneous battery charging and discharging: ",
                  len(battery_hours_with_simultaneous_charging_and_discharging[
                          battery_hours_with_simultaneous_charging_and_discharging > 0]))

        # IF TES capacity is installed, how many hours does the TES charge and discharge simultaneously?
        if pm.value(m.TES_cap) > 0:
            TES_discharge_sum = result['TES to CP']
            TES_charge_sum = + result['Heat from CHP to TES'] + \
                             result['Heat from electric boiler to TES']
            TES_hours_with_simultaneous_charging_and_discharging = pd.Series(index=TES_charge_sum.index)
            for i in range(0, len(TES_charge_sum)):
                if TES_charge_sum[i] > 0:
                    if TES_discharge_sum[i] > 0:
                        TES_hours_with_simultaneous_charging_and_discharging[i] = TES_charge_sum[i] + \
                                                                                  TES_discharge_sum[i]
            print("Number of times of simultaneous TES charging and discharging: ",
                  len(TES_hours_with_simultaneous_charging_and_discharging[
                          TES_hours_with_simultaneous_charging_and_discharging > 0]))

        # IF H2S capacity is installed, how many hours does the H2S charge and discharge simultaneously?
        if pm.value(m.H2S_cap) > 0:
            H2S_discharge_sum = result['H2S to H2B']
            H2S_charge_sum = result['H2E to H2S']
            H2S_hours_with_simultaneous_charging_and_discharging = pd.Series(index=H2S_charge_sum.index)
            for i in range(0, len(H2S_charge_sum)):
                if H2S_charge_sum[i] > 0:
                    if H2S_discharge_sum[i] > 0:
                        H2S_hours_with_simultaneous_charging_and_discharging[i] = H2S_charge_sum[i] + \
                                                                                  H2S_discharge_sum[i]
            print("Number of times of simultaneous H2S charging and discharging: ",
                  len(H2S_hours_with_simultaneous_charging_and_discharging[
                          H2S_hours_with_simultaneous_charging_and_discharging > 0]))

        # Check if grid connection is simultaneously used in a bidirectional manner
        hours_with_simultaneous_gridcon_use = pd.Series(index=grid_P_out.index)
        for i in range(0, len(grid_P_out)):
            if grid_P_out[i] > 0.00001:  # because using 0 led to rounding errors
                if pm.value(m.P_GT_gr[i]) + pm.value(
                        m.P_bat_gr[i]) > 0.00001:  # because using 0 led to rounding errors
                    hours_with_simultaneous_gridcon_use[i] = grid_P_out[i] + pm.value(m.P_GT_gr[i]) + \
                                                             pm.value(m.P_bat_gr[i])
        print("Number of hours of simultaneous bidirectional grid connection use: ",
              len(hours_with_simultaneous_gridcon_use[hours_with_simultaneous_gridcon_use > 0]))


        # Add aggregated data to dictionary containing the results of all model runs
        el_price_scenario_dict['new system'][amp]['results']['Optimal result'] = pm.value(m.objective)
        el_price_scenario_dict['new system'][amp]['results']['CAPEX'] = \
            pm.value(m.bat_cap) * c_bat * if_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
            pm.value(m.ElB_cap) * c_ElB * if_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
            pm.value(m.TES_cap) * c_TES_C * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
            pm.value(m.H2E_cap) * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
            pm.value(m.H2B_cap) * c_H2B * if_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
            pm.value(m.H2S_cap) * c_H2S * if_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime) + \
            pm.value(m.HP_cap) * c_HP * if_HP * disc_rate / (1 - (1 + disc_rate) ** -HP_lifetime)
        el_price_scenario_dict['new system'][amp]['results']['OPEX'] = \
            el_price_scenario_dict['new system'][amp]['results']['Optimal result'] - \
            el_price_scenario_dict['new system'][amp]['results']['CAPEX']
        el_price_scenario_dict['new system'][amp]['results']['scope 1 emissions'] = CO2_emissions
        el_price_scenario_dict['new system'][amp]['results']['required space'] = \
            pm.value(m.bat_cap) * bat_spatialreq + pm.value(m.ElB_cap) * ElB_spatialreq + \
            pm.value(m.TES_cap) * TES_spatialreq + pm.value(m.H2E_cap) * H2E_spatialreq + \
            pm.value(m.H2B_cap) * H2B_spatialreq + pm.value(m.H2S_cap) * H2S_spatialreq + pm.value(
                m.HP_cap) * HP_spatialreq
        el_price_scenario_dict['new system'][amp]['results']['grid connection cap'] = gr_connection
        el_price_scenario_dict['new system'][amp]['results']['discount rate'] = disc_rate
        el_price_scenario_dict['new system'][amp]['results']['max. power flow from grid [MW]'] = grid_P_out_max
        el_price_scenario_dict['new system'][amp]['results'][
            'Simultaneous bidirectional use of grid connection [hours]'] \
            = len(hours_with_simultaneous_gridcon_use[
                      hours_with_simultaneous_gridcon_use > 0])
        el_price_scenario_dict['new system'][amp]['results']['CHP heat gen to CP [MWh]'] = \
            result['Heat from CHP to core process'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['CHP heat gen to TES [MWh]'] = \
            result['Heat from CHP to TES'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['CHP excess heat gen [MWh]'] = \
            result['Heat excess from CHP'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['GT electricity gen to HP [MWh]'] = \
            result['Power from GT to heat pump'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['GT electricity gen to battery [MWh]'] = \
            result['Power from GT to battery'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['GT electricity gen to ElB [MWh]'] = \
            result['Power from GT to electric boiler'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['GT electricity gen to H2E [MWh]'] = \
            result['Power from GT to electrolyser'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['GT excess electricity gen [MWh]'] = \
            result['Power excess from GT'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['GT electricity gen to process [MWh]'] = \
            result['Power from GT to process'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['GT electricity gen to grid [MWh]'] = \
            result['Power from GT to grid'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['total natural gas consumption [MWh]'] = \
            result['Natural gas consumption GT [MW]'].sum() * time_step + \
            result['Natural gas consumption GB [MW]'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['total grid consumption [MWh]'] = Grid_gen * time_step
        el_price_scenario_dict['new system'][amp]['results']['grid to battery [MWh]'] = \
            result['Power from grid to battery'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['grid to electric boiler [MWh]'] = \
            result['Power from grid to electric boiler'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['grid to electrolyser [MWh]'] = \
            result['Power from grid to electrolyser'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['grid to HP [MWh]'] = \
            result['Power from grid to heat pump'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['grid to process [MWh]'] = \
            result['Power from grid to process'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['ElB gen to CP [MWh]'] = \
            result['Heat from electric boiler to core process'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['ElB gen to TES [MWh]'] = \
            result['Heat from electric boiler to TES'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['ElB size [MW]'] = pm.value(m.ElB_cap)
        el_price_scenario_dict['new system'][amp]['results']['Battery size [MWh]'] = pm.value(m.bat_cap)
        el_price_scenario_dict['new system'][amp]['results']['battery to ElB [MWh]'] = \
            result['Battery to electric boiler'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['battery to electrolyser [MWh]'] = \
            result['Battery to electrolyser'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['battery to HP [MWh]'] = \
            result['Battery to heat pump'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['battery to process [MWh]'] = \
            result['Battery to process'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['battery to grid [MWh]'] = \
            result['Battery to grid'].sum() * time_step
        if pm.value(m.bat_cap) > 0:
            el_price_scenario_dict['new system'][amp]['results']['Simultaneous charging and discharging hours battery'] \
                = len(battery_hours_with_simultaneous_charging_and_discharging[
                          battery_hours_with_simultaneous_charging_and_discharging > 0])
        el_price_scenario_dict['new system'][amp]['results']['TES size [MWh]'] = pm.value(m.TES_cap)
        el_price_scenario_dict['new system'][amp]['results']['TES to CP [MWh]'] = result['TES to CP'].sum() * time_step
        if pm.value(m.TES_cap) > 0:
            el_price_scenario_dict['new system'][amp]['results']['Simultaneous charging and discharging hours TES'] \
                = len(TES_hours_with_simultaneous_charging_and_discharging[
                          TES_hours_with_simultaneous_charging_and_discharging > 0])
        if pm.value(m.H2S_cap) > 0:
            el_price_scenario_dict['new system'][amp]['results']['Simultaneous charging and discharging hours H2S'] \
                = len(H2S_hours_with_simultaneous_charging_and_discharging[
                          H2S_hours_with_simultaneous_charging_and_discharging > 0])
        el_price_scenario_dict['new system'][amp]['results']['electrolyser size [MW]'] = pm.value(m.H2E_cap)
        el_price_scenario_dict['new system'][amp]['results']['H2 from electrolyser to boiler [MWh]'] = \
            result['H2E to H2B'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['H2 from electrolyser to storage [MWh]'] = \
            result['H2E to H2S'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['Hydrogen boiler size [MW]'] = pm.value(m.H2B_cap)
        el_price_scenario_dict['new system'][amp]['results']['Hydrogen boiler to CP [MWh]'] = \
            result['Heat from hydrogen boiler to core process'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['Hydrogen storage size [MWh]'] = pm.value(m.H2S_cap)
        el_price_scenario_dict['new system'][amp]['results']['H2 from storage to boiler [MWh]'] = \
            result['H2S to H2B'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['Heat pump size [MW]'] = pm.value(m.HP_cap)
        el_price_scenario_dict['new system'][amp]['results']['Heat from HP to CP [MWh]'] = \
            result['Heat from heat pump to core process'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['results']['Heat from HP to TES [MWh]'] = \
            result['Heat from heat pump to TES'].sum() * time_step
        el_price_scenario_dict['new system'][amp]['energy flows'] = result

        # return the results
        return el_price_scenario_dict


# ________________________________________ Optimisation with plug-in heat pump _________________________________________
def optimisation_run_benchmark_CHP(heat_demand_orig, price_el_hourly, price_NG_use, amp_values, variability_values,
                                   GT_min_load, hours):
    print("Started optimisation of benchmark system.")
    # ------------------------------------- input DATA pre-treatment --------------------------------------------------------
    time_step = 0.5  # in hours

    ## natural gas price data
    # resample to get hourly prices (with constant price per day)
    price_NG_use_hourly = price_NG_use.resample('{}H'.format(1)).ffill()
    price_NG_use_half_hourly = price_NG_use_hourly.resample('30T').ffill()  # resample to get half-hourly prices

    # electricity price data
    el_row_NaN = price_el_hourly[price_el_hourly.isna().any(axis=1)]  # indicates row with NaN value
    price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
    price_el_hourly.index = price_NG_use_hourly.index
    price_el_half_hourly = price_el_hourly.resample('30T').ffill()  # resample to get half-hourly prices

    # Uncomment following lines to create subsets for electricity price data
    price_el_hourly_short = pd.DataFrame(price_el_hourly.iloc[0:7768])
    price_el_half_hourly_short = pd.DataFrame(price_el_half_hourly.iloc[0:hours * 2])  # 8000 operational hours per year
    # replace full data sets by short data sets (to avoid changing code below)
    price_el_hourly = price_el_hourly_short
    price_el_half_hourly = price_el_half_hourly_short

    # manipulate ELECTRICITY price data to increase the amplitude of the price variation
    # get average price of original price data
    price_el_hourly_mean = price_el_hourly.mean()
    price_el_half_hourly_mean = price_el_half_hourly.mean()

    # sort them to plot price duration-curve
    price_el_hourly_sorted = price_el_hourly['EP [EUR/MWh]'].sort_values(ascending=False)
    price_el_hourly_sorted_df = pd.DataFrame(price_el_hourly_sorted)
    price_el_half_hourly_sorted = price_el_half_hourly['EP [EUR/MWh]'].sort_values(ascending=False)
    price_el_half_hourly_sorted_df = pd.DataFrame(price_el_half_hourly_sorted)

    # generate new price profiles and sort their values from high to low to plot price duration curves
    if len(amp_values) > 0:

        for k in amp_values:
            print("Current k is: ", k)
            colname = ("amp " + "%.3f") % k
            # add new price data as additional columns to dataframe
            price_el_half_hourly[str(colname)] = price_el_half_hourly_mean.iloc[0] + \
                                                 k * (price_el_half_hourly['EP [EUR/MWh]'] -
                                                      price_el_half_hourly_mean.iloc[0])

            # sort values from high to low and add new column to dataframe
            price_el_half_hourly_sorted[str(colname)] = price_el_half_hourly[str(colname)].sort_values(ascending=False)
            price_el_half_hourly_sorted_df[str(colname)] = price_el_half_hourly_sorted[str(colname)]
            # remove the index
            price_el_half_hourly_sorted_df = price_el_half_hourly_sorted_df.reset_index(drop=True)

        price_el_half_hourly_mean_df = pd.DataFrame(price_el_half_hourly.mean())
        # calculate new mean values
        for k in amp_values:
            print("Current k is: ", k)
            colname = ("amp " + "%.3f") % k
            # removing negative prices
            price_el_half_hourly.loc[price_el_half_hourly[str(colname)] < 0, str(colname)] = 0
            # calculate new mean values
            price_el_half_hourly_mean_df[str(colname)] = price_el_half_hourly[str(colname)].mean()

      # ----------------------------- Dictionaries to run optimisation for each process -------------------------------------
    # --------------------------------(with non-optimised and optimised values) -------------------------------------------
    # for electricity
    price_el_hourly.rename(columns={'EP [EUR/MWh]': 'original'}, inplace=True)
    price_el_half_hourly.rename(columns={'EP [EUR/MWh]': 'original'}, inplace=True)

    # create respective dictionary
    looping_variable = variability_values
    el_price_scenario_dict = {
        'benchmark system': {amp: {'results': {}, 'energy flows': {}} for amp in looping_variable}}

    # for amp in variability:
    for count, amp in enumerate(looping_variable):
        print("Current scenario is: ", amp)
        H_dem = heat_demand_orig[0:hours * 2]
        H_dem_max = heat_demand_orig[0:hours * 2].max().iloc[0]
        P_dem = 0.1 * heat_demand_orig[0:hours * 2]
        P_dem.rename(columns={'Heat demand [MW]': 'Power demand [MW]'}, inplace=True)

        # ------------------ START OPTIMISATION --------------------------------------------------------------------
        # Definitions

        def heat_balance(m, time):
            return float(H_dem.iloc[time]) == m.H_CHP_process[time]

        def power_balance(m, time):
            return float(P_dem.iloc[time]) == m.P_gr_process[time] + m.P_GT_process[time]

        def GT_ng_P_conversion(m, time):
            return m.NG_GT_in[time] * eta_GT_el == m.P_GT_excess[time] + m.P_GT_process[time] + m.P_GT_gr[time]

        def CHP_ng_H_conversion(m, time):
            return (m.NG_GT_in[time] * eta_GT_th + m.NG_GB_in[time]) * eta_GB == \
                   m.H_CHP_process[time] + m.H_CHP_excess[time]

        def GT_cap_rule(m, time):
            return m.NG_GT_in[time] <= GT_cap / eta_GT_th # * eta_GB)  #GT_cap = H_dem_max / eta_GB #outflow GT

        def GB_cap_rule(m, time):
            return m.NG_GB_in[time] <= 0.2 * GT_cap / eta_GB  # GB_cap = 0.2 * GT_cap  #outflow GB

        def GT_min_load_rule(m, time):
            return m.NG_GT_in[time] >= GT_cap / eta_GT_th * GT_min_load

        def max_grid_power_in(m, time):
            return m.P_gr_process[time] <= gr_connection * m.b1[
                time]  # total power flow from grid to plant is limited to x MW

        def max_grid_power_out(m, time):
            return m.P_GT_gr[time] <= gr_connection * (
                        1 - m.b1[time])  # total power flow from grid to plant is limited to x MW

        def minimize_total_costs(m, time):
            return sum(price_el_half_hourly.iloc[time, count] * time_step * (m.P_gr_process[time] - m.P_GT_gr[time])
                       + (m.NG_GT_in[time] + m.NG_GB_in[time]) * time_step * price_NG_use_half_hourly.iloc[time, 0]
                       for time in m.T)

        m = pm.ConcreteModel()

        # SETS
        m.T = pm.RangeSet(0, hours * 2 - 1)

        # CONSTANTS
        disc_rate = 0.1  # 10%
        EF_ng = 0.2  # emission factor natural gas, tCO2/MWh(CH4)
        gr_connection = 30  # [MW] Grid connection capacity
        eta_GT_el = 0.3  # Electric efficiency of GT [%]
        eta_GT_th = 0.6  # Thermal efficiency of GT [%]
        eta_GB = 0.82
        GT_cap = H_dem_max / eta_GB  # Thermal capacity (LPS) GT, [MW]

        # VARIABLES
        m.NG_GT_in = pm.Var(m.T, bounds=(0, None))  # natural gas intake of gas turbine, MWh
        m.NG_GB_in = pm.Var(m.T, bounds=(0, None))  # natural gas intake of gas boiler, MWh
        m.P_GT_process = pm.Var(m.T, bounds=(0, None))  # Power from CHP to process, MW
        m.P_GT_excess = pm.Var(m.T, bounds=(0, None))  # Excess power from CHP, MW
        m.P_GT_gr = pm.Var(m.T, bounds=(0, None))  # Power from CHP to grid, MW
        m.H_CHP_process = pm.Var(m.T, bounds=(0, None))  # Heat generated from CHP (natural gas), MW
        m.H_CHP_excess = pm.Var(m.T, bounds=(0, None))  # Excess heat from CHP, MW
        m.P_gr_process = pm.Var(m.T, bounds=(0, None))  # power flow from grid to process, MW
        m.b1 = pm.Var(m.T, within=pm.Binary)  # binary variable grid connection

        # CONSTRAINTS
        # balance supply and demand
        m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
        m.power_balance_constraint = pm.Constraint(m.T, rule=power_balance)
        # CHP constraints
        m.CHP_ng_H_conversion_constraint = pm.Constraint(m.T, rule=CHP_ng_H_conversion)
        m.GT_ng_P_conversion_constraint = pm.Constraint(m.T, rule=GT_ng_P_conversion)
        m.GT_cap_constraint = pm.Constraint(m.T, rule=GT_cap_rule)
        m.GB_cap_constraint = pm.Constraint(m.T, rule=GB_cap_rule)
        m.GT_min_load_constraint = pm.Constraint(m.T, rule=GT_min_load_rule)

        # grid constraint
        m.max_grid_power_in_constraint = pm.Constraint(m.T, rule=max_grid_power_in)
        m.max_grid_power_out_constraint = pm.Constraint(m.T, rule=max_grid_power_out)

        # OBJECTIVE FUNCTION
        m.objective = pm.Objective(rule=minimize_total_costs,
                                   sense=pm.minimize,
                                   doc='Define objective function')

        # Solve optimization problem
        opt = pm.SolverFactory('gurobi')
        opt.options["MIPGap"] = 0.0005
        results = opt.solve(m, tee=True)

        # ------------------ OPTIMISATION END --------------------------------------------------------------------------
        # Todo: Change? stopped changing script here
        # Collect results
        result = pd.DataFrame(index=price_NG_use_half_hourly.index[0:hours * 2])
        result['Heat demand core process'] = list(H_dem['Heat demand [MW]'])
        result['Power demand core process'] = list(P_dem['Power demand [MW]'])
        result['Natural gas consumption GT [MW]'] = pm.value(m.NG_GT_in[:])
        result['Natural gas consumption GB [MW]'] = pm.value(m.NG_GB_in[:])
        result['Power from GT to process'] = pm.value(m.P_GT_process[:])
        result['Power excess from GT'] = pm.value(m.P_GT_excess[:])
        result['Power from GT to grid'] = pm.value(m.P_GT_gr[:])
        result['Heat from CHP to process'] = pm.value(m.H_CHP_process[:])
        result['Heat excess from CHP'] = pm.value(m.H_CHP_excess[:])
        result['Power from grid to process'] = pm.value(m.P_gr_process[:])

        Grid_gen = result['Power from grid to process'].sum()
        CO2_emissions = (result['Natural gas consumption GT [MW]'].sum() +
                         result[
                             'Natural gas consumption GB [MW]'].sum()) * EF_ng * time_step  # [MW]*[ton/MWh]*[h] = [ton]

        # control: H_CP==H_dem?
        control_H = sum(result['Heat demand core process'] - result['Heat from CHP to process'])

        print("control_H =", control_H)
        print("Objective = ", pm.value(m.objective))

        # Add results for stacked bar chart "Optimal energy supply" to process dictionaries
        el_price_scenario_dict['benchmark system'][amp]['results']['Optimal result'] = pm.value(m.objective)
        el_price_scenario_dict['benchmark system'][amp]['results']['CAPEX'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['OPEX'] = \
            el_price_scenario_dict['benchmark system'][amp]['results']['Optimal result'] - \
            el_price_scenario_dict['benchmark system'][amp]['results']['CAPEX']
        el_price_scenario_dict['benchmark system'][amp]['results']['scope 1 emissions'] = CO2_emissions
        el_price_scenario_dict['benchmark system'][amp]['results']['required space'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['CHP heat gen to CP [MWh]'] = \
            result['Heat from CHP to process'].sum() * time_step
        el_price_scenario_dict['benchmark system'][amp]['results']['CHP heat gen to TES [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['CHP excess heat gen [MWh]'] = \
            result['Heat excess from CHP'].sum() * time_step
        el_price_scenario_dict['benchmark system'][amp]['results']['GT electricity gen to HP [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['GT electricity gen to battery [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['GT electricity gen to ElB [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['GT electricity gen to H2E [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['GT excess electricity gen [MWh]'] = \
            result['Power excess from GT'].sum() * time_step
        el_price_scenario_dict['benchmark system'][amp]['results']['GT electricity gen to process [MWh]'] = \
            result['Power from GT to process'].sum() * time_step
        el_price_scenario_dict['benchmark system'][amp]['results']['GT electricity gen to grid [MWh]'] = \
            result['Power from GT to grid'].sum() * time_step
        el_price_scenario_dict['benchmark system'][amp]['results']['total natural gas consumption [MWh]'] = \
            result['Natural gas consumption GT [MW]'].sum() * time_step + \
            result['Natural gas consumption GB [MW]'].sum() * time_step
        el_price_scenario_dict['benchmark system'][amp]['results'][
            'total grid consumption [MWh]'] = Grid_gen * time_step
        el_price_scenario_dict['benchmark system'][amp]['results']['grid to battery [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['grid to electric boiler [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['grid to electrolyser [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['grid to HP [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['grid to process [MWh]'] = \
            result['Power from grid to process'].sum() * time_step
        el_price_scenario_dict['benchmark system'][amp]['results']['ElB gen to CP [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['ElB gen to TES [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['ElB size [MW]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['Battery size [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['battery to ElB [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['battery to electrolyser [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['battery to HP [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['battery to process [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['battery to grid [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['TES size [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['TES to CP [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['electrolyser size [MW]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['H2 from electrolyser to boiler [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['H2 from electrolyser to storage [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['Hydrogen boiler size [MW]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['Hydrogen boiler to CP [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['Hydrogen storage size [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['H2 from storage to boiler [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['Heat pump size [MW]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['Heat from HP to CP [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['results']['Heat from HP to TES [MWh]'] = 0
        el_price_scenario_dict['benchmark system'][amp]['energy flows'] = result

        return el_price_scenario_dict

# ------------------------------------ END of script -------------------------------------------------------------------
