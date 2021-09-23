import math
import numpy


def hill_chart_parametrisation(h, turbine_specs):
    """
    Calculates power and flow rate through bulb turbines based on Aggidis and Feather (2012)

    f_g = grid frequency, g_p = generator poles,
    t_cap = Turbine capacity, h = head difference, dens = water density
    """
    turb_sp = 2 * 60 * turbine_specs["f_g"] / turbine_specs["g_p"]

    # Step 1: Calculate Hill Chart based on empirical equations
    n_11 = turb_sp * turbine_specs["t_d"] / math.sqrt(h)

    if n_11 < 255:
        q_11 = 0.0166 * n_11 + 0.4861
    else:
        q_11 = 4.75

    q = q_11 * (turbine_specs["t_d"] ** 2) * math.sqrt(h)
    h_efficiency = -0.0019 * n_11 + 1.2461
    # h_efficiency  = 1
    p1 = turbine_specs["dens"] * turbine_specs["g"] * q * h / (10 ** 6)
    # Step 2 - Adjust Curve according to capacity
    if p1 * h_efficiency < turbine_specs["t_cap"]:  # 97.25% Gearbox efficiency
        p2 = p1 * 0.9725 * h_efficiency
    else:
        p2 = turbine_specs["t_cap"] * 0.9725
        p1 = p2 / (h_efficiency * 0.9725)

    q = p1 * (10 ** 6) / (turbine_specs["dens"] * turbine_specs["g"] * h)

    return p2, q


def ideal_turbine_parametrisation(h, turbine_specs):
    """
    Calculates power and flow through a bulb turbine excluding efficiency loses
    """
    q = math.pi * ((turbine_specs["t_d"] / 2)**2) * math.sqrt(2 * turbine_specs["g"] * h)
    p1 = turbine_specs["dens"] * turbine_specs["g"] * q * h / (10 ** 6)

    if p1 < turbine_specs["t_cap"]:
        p2 = p1
    else:
        p2 = turbine_specs["t_cap"]

    q = p2 * (10 ** 6) / (turbine_specs["dens"] * turbine_specs["g"] * h)

    return p2, q


def turbine_parametrisation(h, turbine_specs):
    """
    Chooses between hill chart or idealised turbine parameterisation.
    """
    if turbine_specs["options"] == 0:

        p, q = hill_chart_parametrisation(h, turbine_specs)
    else:
        p, q = ideal_turbine_parametrisation(h, turbine_specs)

    return p, q


def gate_sluicing(h, ramp_f, N_s, q_s0, sluice_specs, flux_limiter=0.2):
    """
    Calculates overall flow through power plant sluice gates given the status of the operation
    """
    temp = ramp_f ** 2 * N_s * sluice_specs["c_d"] * sluice_specs["a_s"] * math.sqrt(2 * sluice_specs["g"] * abs(h))
    if ramp_f >= 0.5 and abs(temp) >= abs(q_s0) > 0.:
        q_s = -numpy.sign(h) * min(abs((1 + flux_limiter) * q_s0), abs(temp))
    elif ramp_f >= 0.5 and abs(q_s0) >= abs(temp):
        q_s = -numpy.sign(h) * max(abs((1 - flux_limiter) * q_s0), abs(temp))
    else:
        q_s = -numpy.sign(h) * temp
    return q_s


def turbine_sluicing(h, ramp_f, N_t, q_t0, sluice_specs, turbine_specs, flux_limiter=0.2):
    """
    Calculates flow through turbines operating in sluicing mode
    """
    temp = ramp_f ** 2 * N_t * sluice_specs["c_t"] * (math.pi * (turbine_specs["t_d"] / 2) ** 2) *\
        math.sqrt(2 * sluice_specs["g"] * abs(h))
    if ramp_f >= 0.5 and abs(temp) >= abs(q_t0):
        q_t = -numpy.sign(h) * min(abs((1 + flux_limiter) * q_t0), abs(temp))
    elif ramp_f >= 0.5 and abs(q_t0) >= abs(temp):
        q_t = -numpy.sign(h) * max(abs((1 - flux_limiter) * q_t0), abs(temp))
    else:
        q_t = -numpy.sign(h) * temp

    if abs(h) != 0.0 and ramp_f >= 0.95 and q_t == 0.:
        q_t = -numpy.sign(h) * temp
    return q_t
