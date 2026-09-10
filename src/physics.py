import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

MASS = 0.04593 # kg
RADIUS = 0.021335 # m
AREA = np.pi * (RADIUS ** 2) # m^2
DENSITY = 1.225  # Air density kg/m^3
G = 9.82 # m/s^2

def ball_flight_ode(t, state, Cd, Cl, spin_axis):
    """Calculates the 3D equations of motion for a golf ball in flight"""
    x, y, z, vx, vy, vz = state
    v_magnitude = np.sqrt((vx ** 2) + (vy ** 2) + (vz ** 2))

    if (v_magnitude == 0):
        return [0, 0, 0, 0, 0, 0]

    # Drag force
    f_drag_magnitude = 0.5 * DENSITY * AREA * Cd * (v_magnitude ** 2)
    ax_drag = -(f_drag_magnitude / MASS) * (vx / v_magnitude)
    ay_drag = -(f_drag_magnitude / MASS) * (vy / v_magnitude)
    az_drag = -(f_drag_magnitude / MASS) * (vz / v_magnitude)

    # Lift force (perpindicular to the velocity)
    f_lift_magnitude = 0.5 * DENSITY * AREA * Cl * (v_magnitude ** 2)
    # ASSUMPTION: Simple backspin lift vector (upward component)
    ax_lift = 0.0
    ay_lift = 0.0
    az_lift = f_lift_magnitude / MASS

    # Total acceleration
    ax = ax_drag + ax_lift
    ay = ay_drag + ay_lift
    az = az_drag + az_lift - G

    return [vx, vy, vz, ax, ay, az]


def landing_event(t, state, Cd, Cl, spin_axis):
    """Triggers when z = 0"""
    return state[2]

landing_event.terminal = True
landing_event.direction = -1


def project_trajectory(launch_state, Cd=0.25, Cl=0.15, max_time=10.0):
    """Integrates the flight path until the ball hits the ground"""

    solution = solve_ivp(
        fun=ball_flight_ode,
        t_span=(0, max_time),
        y0=launch_state,
        args=(Cd, Cl, None),
        events=landing_event,
        max_step=0.02,
    )

    # Extract apex and landing
    z_vals = solution.y[2]
    apex_idx = np.argmax(z_vals)

    apex_t = solution.t[apex_idx]
    apex_x, apex_y, apex_z = (solution.y[0][apex_idx], solution.y[1][apex_idx], solution.y[2][apex_idx])

    landing_t = solution.t[-1]
    landing_x, landing_y, landing_z = (solution.y[0][-1], solution.y[1][-1], solution.y[2][-1])

    return {
        "apex_t": apex_t,
        "apex_x": apex_x,
        "apex_y": apex_y,
        "apex_z": apex_z,
        "landing_t": landing_t,
        "landing_x": landing_x,
        "landing_y": landing_y,
        "landing_z": landing_z,
    }
