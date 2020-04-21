# Using the module parameterisations module to determine turbine discharge coefficient for smooth transition between generating and sluicing
from modules.parameterisations import *


def initialise_barrage(Barrage_Number=1, time=0.):
    """
    Initialises dictionary of barrage status - this contains information about the status of the barrage

    :param Barrage_Number: Number of barrages considered.
    :return:
    """
    barrage_status = []
    for i in range(Barrage_Number):
        QTurbines, QSluices, PowerGT, SumPower, Mode, ModeDuration, DZ, rampf = 0., 0., 0., 0., 1., 0., 0., 0.,
        barrage_status.append({"m": Mode, "m_t": time/3600, "m_dt": ModeDuration, "DZ": DZ, "f_r": rampf,
                               "Q_t": QTurbines, "Q_s": QSluices, "P": PowerGT, "E": SumPower})
    return barrage_status


def input_predefined_barrage_specs(turbine_number, sluice_number, operation='two-way'):
    """
    Initialises certain control parameters depending on the general strategy to be adopted over the course of the operation.

    :param turbine_number: Number of turbines
    :param sluice_number: Number of sluice gates
    :param operation: operation options: Ebb-only generation        ==> "ebb"
                                         Ebb-pump generation        ==> "ebb-pump"
                                         Two-way generation         ==> "two-way"
                                         Two-way-pumping generation ==> "two-way-pump"
    :return: control parameter array , turbine parameters
    """

    params, input_2D = [], []
    turbine_params = {"f_g": 50, "g_p": 95, "g": 9.807, "t_d": 7.35,
                      "t_cap": 20, "dens": 1025, "h_min": 1.00,
                      "eta": [0.93, 0.83], "options": 0}

    Coed_t = turbine_parametrisation(turbine_params["h_min"],
                                     turbine_params)[1] / ((math.pi * (turbine_params["t_d"] / 2)**2)
                                                           * math.sqrt(2 * turbine_params["g"] * turbine_params["h_min"]))

    sluice_params = {"a_s": 100, "c_d": 1.0, "c_t": Coed_t, "g": turbine_params["g"]}

    params.append({"turbine_specs": turbine_params, "sluice_specs": sluice_params})

    if operation == "ebb":
        input_2D.append({"h_t": [3.5, 0.], "h_p": 2.5, "t_p": [0., 0.], "g_t": [6.0, 6.0], "tr_l": [7, -6],
                         "N_t": turbine_number, "N_s": sluice_number})
    elif operation == "ebb-pump":
        input_2D.append({"h_t": [3.5, 0.], "h_p": 2.5, "t_p": [1.0, 0.], "g_t": [6.0, 6.0], "tr_l": [7, -6],
                         "N_t": turbine_number, "N_s": sluice_number})
    elif operation == "two-way":
        input_2D.append({"h_t": [3.0, 3.0], "h_p": 2.5, "t_p": [0., 0.], "g_t": [3.0, 3.0], "tr_l": [7, -6],
                         "N_t": turbine_number, "N_s": sluice_number})
    elif operation == "two-way-pump":
        input_2D.append({"h_t": [2.0, 2.0], "h_p": 2.5, "t_p": [0.5, 0.5], "g_t": [3.0, 3.0], "tr_l": [7, -6],
                         "N_t": turbine_number, "N_s": sluice_number})

    return input_2D, params
