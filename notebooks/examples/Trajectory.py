#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math


def plot_traj(df):
    
    # Plot y vs x
    plt.figure(figsize=(5, 3.33))
    plt.plot(df['x'], df['y'], label='Projectile Path')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.ylim(0,180)
    plt.xlim(0,270)
    plt.title('Projectile Motion: y vs x')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[24]:


# Constants

from dataclasses import dataclass

@dataclass
class Constants:
    drag_f: callable
    g: float = 9.81       # Gravitational acceleration (m/s^2)
    dt: float = 0.01      # Time step (s)
    rho: float = 1.225    # Air density (kg/m^3)
    C_d: float = 0.047    # Drag coefficient
    A: float = 0.045      # Cross-sectional area (m^2)
    m: float = 0.145      # Mass (kg)
    

# Event function to stop integration when y < 0
def event_y_below_zero(t, state, consts: Constants ):
    return state[1]  # state[1] is y, so looking for y == 0

event_y_below_zero.terminal = True
event_y_below_zero.direction = -1 # When y == 0, look for transition from pos to neg. 


def drag(t, vx, vy):
    """Compute the drag force"""
    v = np.sqrt(vx**2 + vy**2) # Magnitude of velocity vector
    a =  math.atan2(vy, vx)
    
    F_d = 0.5 * consts.C_d * consts.rho * consts.A * v**2
    
    return F_d


def _projectile_motion( t, state, consts: Constants):
    """Calculate dy/dt for the state y. This version includes all state
    values, and the input is the same structure as the output. """
    
    x, y, theta, vx, vy, dtheta, ax, ay = state
    
    v = np.sqrt(vx**2 + vy**2) # Magnitude of velocity vector
    a =  math.atan2(vy, vx)
    
    F_d = consts.drag_f(t, vx, vy)
    
    ax = -F_d * math.cos(a) /consts.m 

    ay = -consts.g - (F_d * math.sin(a) / consts.m )
    
    dtheta = 0
    
    return (x, y, theta, vx, vy, dtheta, ax, ay )

def projectile_motion( t, yn, consts: Constants ):
    """adapt _projectile_motion"""
    
   
    if len(yn) < 8:
        yn = np.append(yn, np.array([0,0,0]))
    
    yn1 = _projectile_motion(t, yn, consts)
    
    return yn1[3:]

def throw(v0, angle, consts: Constants):
    """Solve a trajectory for an initial velocity and angle"""
    
    angle_rad = np.radians(angle)
    
    # Initial velocity components
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    
    x0 = 0    # Initial x position (m)
    y0 = 0    # Initial y position (m)
   
    # Time span
    t_max = 20
    t_span = (0, t_max)  
    t_eval = np.arange(0, t_max, consts.dt)


    y0 = [
            x0, # x 
            y0, # y
            angle_rad, # theta 
            vx0, # vx 
            vy0, # vy, 
          ]

    solution = solve_ivp(projectile_motion, t_span, 
                         y0 =  y0, # Initial State
                         t_eval=t_eval, 
                         args = (consts,), events=event_y_below_zero)

    t = solution.t
   
    x, y, theta, vx, vy  = solution.y # Unpack the states
    
    
    F_d = np.vectorize(consts.drag_f)(t, vx, vy)
    
    v = np.sqrt(vx**2 + vy**2)
    ax = -F_d * vx / (consts.m * v)
    ay = -consts.g - (F_d * vy / (consts.m * v))
    
    # Create a pandas DataFrame
    
    data = {
        't': t,
        'x': x,
        'y': y,
        'vx': vx,
        'vy': vy,
        'ax': ax,
        'ay': ay,
        'F_d': F_d
    }
    
    return pd.DataFrame(data)

# Example of how to use the dataclass
consts = Constants(drag_f=drag)


df = throw(v0=200, angle=55, consts=consts)
plot_traj(df)

