import math
import numpy

# modules.parameterisation contains functions for the turbine and sluice gate flow calculation
from modules.parameterisations import turbine_parametrisation, gate_sluicing, turbine_sluicing


def lagoon_operation(h_i, h_o, t, status, control, turbine_specs, sluice_specs, flux_limiter=0.2):
    """
    Operation algorithm for a tidal power plant that is called to calculate the flow exchange
    given the head difference and the status of the tidal power plant

    :param h_i: Inner (Upstream) water elevation
    :param h_o: Outer (downstream) water elevation
    :param t: time
    :param status: current status of power plant dictionary
    :param control: control parameters for operation dictionary
    :param turbine_specs: turbine specifications
    :param sluice_specs: sluice gate specifications
    :param flux_limiter: flux limiter (to avoid unrealistic instabilities)
    :return:
    """

    mod_0 = status["m"]
    status["DZ"] = h_i - h_o

    # Main Operation Algorithm for a two-way operation
    if mod_0 == 9 and status["DZ"] > 0:
        status["m"], status["m_t"] = 10, t
        if control["t_p"][0] <= 0.2:
            status["m"], status["m_t"] = 1, t
    elif mod_0 == 10 and status["m_dt"] >= control["t_p"][0]:
        status["m"], status["m_t"] = 1, t
    elif mod_0 == 1 and control["h_t"][0] <= 0.2 and h_i > h_o:
        status["m"], status["m_t"] = 4, t
    elif mod_0 == 1 and status["m_dt"] < control["h_t"][0]:
        status["m"] = 1
    elif mod_0 == 1 and status["m_dt"] >= control["h_t"][0] and status["DZ"] > turbine_specs["h_min"]:
        status["m"], status["m_t"] = 2, t
    elif mod_0 == 2 and status["DZ"] < turbine_specs["h_min"] and status["m_dt"] > 0.25:
        status["m"], status["m_t"] = 4, t
    elif mod_0 == 2 and status["m_dt"] > control["g_t"][0]:
        status["m"], status["m_t"] = 3, t
    elif mod_0 == 3 and status["DZ"] >= turbine_specs["h_min"]:
        status["m"] = 3
    elif mod_0 == 3 and status["DZ"] < turbine_specs["h_min"]:
        status["m"], status["m_t"] = 4, t
    elif mod_0 == 4 and status["DZ"] < turbine_specs["h_min"] and h_i >= h_o:
        status["m"] = 4
    elif mod_0 == 4 and h_i < h_o:
        status["m"], status["m_t"] = 5, t
        if control["t_p"][1] <= 0.2:
            status["m"], status["m_t"] = 6, t
    elif mod_0 == 5 and status["m_dt"] > control["t_p"][1]:
        status["m"], status["m_t"] = 6, t
    elif mod_0 == 6 and control["h_t"][1] <= 0.2 and h_i < h_o:
        status["m"], status["m_t"] = 9, t
    elif mod_0 == 6 and -status["DZ"] < turbine_specs["h_min"] and status["m_dt"] < control["h_t"][1]:
        status["m"] = 6
    elif mod_0 == 6 and status["m_dt"] > control["h_t"][1] and -status["DZ"] > turbine_specs["h_min"]:
        status["m"], status["m_t"] = 7, t
    elif mod_0 == 7 and -status["DZ"] < turbine_specs["h_min"] and status["m_dt"] > 0.25:
        status["m"], status["m_t"] = 9, t
    elif mod_0 == 7 and status["m_dt"] > control["g_t"][1]:
        status["m"], status["m_t"] = 8, t
    elif mod_0 == 8 and -status["DZ"] > turbine_specs["h_min"]:
        status["m"] = 8
    elif mod_0 == 8 and -status["DZ"] < turbine_specs["h_min"]:
        status["m"], status["m_t"] = 9, t
    elif mod_0 == 9 and -status["DZ"] < turbine_specs["h_min"]:
        status["m"] = 9
    else:
        status["m"] = mod_0

    # Special cases
    # 1. Check for holding modes
    if mod_0 == 1 and turbine_specs["h_min"] > -status["DZ"] > 0 and status["m_dt"] > 2.:
        status["m"] = 6
    elif mod_0 == 6 and turbine_specs["h_min"] > status["DZ"] > 0 and status["m_dt"] > 2.:
        status["m"] = 1
    # 2. Sluice if in holding mode by accident
    if mod_0 == 1 and -status["DZ"] > 0 and status["m_dt"] > 0.1:
        status["m"], status["m_t"] = 9, t
    elif mod_0 == 6 and status["DZ"] > 0 and status["m_dt"] > 0.1:
        status["m"], status["m_t"] = 4, t

    status["m_dt"] = t - status["m_t"]

    # Ramp function set-up (primarily for stability and opening/closing hydraulic structures
    if status["m"] != mod_0:
        status["f_r"] = 0

    # Special case for generating/sluicing and sluicing modes
    if status["m"] == 4 and mod_0 == 3:
        status["f_r"] = 1
    elif status["m"] == 9 and mod_0 == 8:
        status["f_r"] = 1

    # If hydraulic structures are opening still, increase status["f_r"] based on a sine function
    if mod_0 == status["m"] and status["m_dt"] < 0.2 and status["f_r"] < 1.0:
        status["f_r"] = math.sin(math.pi / 2 * ((t - status["m_t"]) / 0.2))
    elif 0.4 > status["m_dt"] >= 0.2:  # second condition added just in case.
        status["f_r"] = 1.0

    # Special case - trigger end of pumping  # empirical
    if status["m"] == 5 and h_i <= (control["tr_l"][1] + 0.50):
        status["f_r"] = math.sin(math.pi / 2 * abs(h_i - control["tr_l"][0]) / 0.5)
        if status["f_r"] <= 0.3:
            status["m"], status["m_t"], status["m_dt"] = 6, t, 0.0

    if status["m"] == 10 and h_i >= (control["tr_l"][0] - 0.50):
        status["f_r"] = math.sin(math.pi / 2 * abs(control["tr_l"][1] - h_i) / 0.5)
        if status["f_r"] <= 0.3:
            status["m"], status["m_t"], status["m_dt"] = 1, t, 0.0

    # ramping down for pumping stability
    if status["m"] == 5 and control["t_p"][1] - status["m_dt"] <= 0.2:
        status["f_r"] = math.sin(math.pi / 2 * ((control["t_p"][1] - status["m_dt"]) / 0.2))

    if status["m"] == 10 and control["t_p"][0] - status["m_dt"] <= 0.2:
        status["f_r"] = math.sin(math.pi / 2 * ((control["t_p"][0] - status["m_dt"]) / 0.2))

    # Individual mode operations#
    if status["m"] == 1 or status["m"] == 6:
        status["Q_t"], status["Q_s"], status["P"] = 0.0, 0.0, 0.0  # Holding

    if status["m"] == 2:  # Generating            HW -> LW
        status["P"] = status["f_r"] * control["N_t"] * turbine_specs["eta"][0] * turbine_parametrisation(abs(status["DZ"]),
                                                                                                         turbine_specs)[0]
        status["Q_t"] = -status["f_r"] * control["N_t"] * turbine_parametrisation(abs(status["DZ"]), turbine_specs)[1]
        status["Q_s"] = 0.0

    if status["m"] == 3:  # Generating/sluicing    HW -> LW
        status["P"] = control["N_t"] * turbine_parametrisation(abs(status["DZ"]), turbine_specs)[0] * turbine_specs["eta"][0]
        status["Q_t"] = - control["N_t"] * turbine_parametrisation(abs(status["DZ"]), turbine_specs)[1]
        status["Q_s"] = gate_sluicing(status["DZ"], status["f_r"], control["N_s"],
                                      status["Q_s"], sluice_specs, flux_limiter=flux_limiter)

    if status["m"] == 4:  # sluicing          HW -> LW
        status["P"] = 0.0
        status["Q_t"] = turbine_sluicing(status["DZ"], 1.0, control["N_t"],
                                         status["Q_t"], sluice_specs, turbine_specs, flux_limiter=flux_limiter)
        status["Q_s"] = gate_sluicing(status["DZ"], status["f_r"], control["N_s"],
                                      status["Q_s"], sluice_specs, flux_limiter=flux_limiter)

    if status["m"] == 5:  # Ebb pumping
        status["Q_t"] = - max(
            status["f_r"] * control["N_t"] * turbine_parametrisation(control["h_p"], turbine_specs)[1],
            0)  # Re-formulate the discharge coefficient for turbines! Introduce cd_t
        status["P"] = -(abs(status["Q_t"]) * turbine_specs["dens"] * turbine_specs["g"]
                        * abs(status["DZ"]) / (10 ** 6)) / min(max(0.4, 0.28409853 * numpy.log(abs(status["DZ"])) + 0.60270881), 0.9)
        status["Q_s"] = 0.0

    if status["m"] == 7:  # Generating             LW -> HW
        status["P"] = status["f_r"] * control["N_t"] * turbine_specs["eta"][1] * turbine_parametrisation(abs(status["DZ"]),
                                                                                                         turbine_specs)[0]
        status["Q_t"] = status["f_r"] * control["N_t"] * turbine_parametrisation(abs(status["DZ"]), turbine_specs)[1]
        status["Q_s"] = 0.0

    if status["m"] == 8:  # Generating / Sluicing LW -> HW
        status["P"] = control["N_t"] * turbine_specs["eta"][1] * turbine_parametrisation(abs(status["DZ"]),
                                                                                         turbine_specs)[0]
        status["Q_t"] = control["N_t"] * turbine_parametrisation(abs(status["DZ"]), turbine_specs)[1]
        status["Q_s"] = gate_sluicing(status["DZ"], status["f_r"], control["N_s"],
                                      status["Q_s"], sluice_specs, flux_limiter=flux_limiter)

    if status["m"] == 9:  # sluicing              LW -> HW
        status["P"] = 0.0
        status["Q_t"] = turbine_sluicing(status["DZ"], 1.0, control["N_t"],
                                         status["Q_t"], sluice_specs, turbine_specs, flux_limiter=flux_limiter)
        status["Q_s"] = gate_sluicing(status["DZ"], status["f_r"], control["N_s"],
                                      status["Q_s"], sluice_specs, flux_limiter=flux_limiter)

    if status["m"] == 10:  # Flood pumping
        status["Q_t"] = max(status["f_r"] * control["N_t"] * turbine_parametrisation(control["h_p"], turbine_specs)[1],
                            0.0)  # Re-formulate the discharge coefficient for turbines! Introduce cd_t
        status["P"] = -(
            abs(status["Q_t"]) * turbine_specs["dens"] * turbine_specs["g"] * abs(status["DZ"]) / (10 ** 6)) / min(
            max(0.4, 0.28409853 * numpy.log(abs(status["DZ"])) + 0.60270881), 0.9)
        status["Q_s"] = 0.0

    status["eta_d0"] = h_o  # Equate new downstream WL to old downstream WL for next iteration

    return status


def lagoon(t, Dt, h_i, h_o, status, control, params, boundaries):
    """
    Calculates based on the head differences and the operational conditions the fluxes of a normal tidal lagoon/barrage
    by calling lagoon_operation. Imposes hydraulic structure fluxes if available.

    :param t: time (s)
    :param Dt: timestep (s)
    :param h_i: upstream water level (m)
    :param h_o: downstream water level (m)
    :param status: current status parameters of lagoon/barrage
    :param control: control parameters imposed for the operation
    :param boundaries: hydraulic structure location
    :param file: output file
    :return: lagoon_operation(h_i, h_o, t, status, control, turbine_specs, sluice_specs):
    """

    status = lagoon_operation(h_i, h_o, t / 3600, status, control, params["turbine_specs"], params["sluice_specs"])
    status["E"] += status["P"] * Dt / 3600

    if boundaries != "None":
        boundaries["tb_o"].assign(status["Q_t"])
        boundaries["tb_i"].assign(-status["Q_t"])
        boundaries["sl_o"].assign(status["Q_s"])
        boundaries["sl_i"].assign(-status["Q_s"])

    return numpy.hstack(([t, h_o, h_i], [status["DZ"], status["P"], status["E"], status["m"], status["Q_t"], status["Q_s"],
                                         status["m_dt"], status["m_t"], status["f_r"]],))
