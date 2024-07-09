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


g = 9.81  # Gravitational acceleration (m/s^2)
dt = 0.01  # Time step (s)

C_d = 0.047  # Drag coefficient
rho = 1.225  # Air density (kg/m^3)
A = 0.045  # Cross-sectional area (m^2)

# Event function to stop integration when y < 0
def event_y_below_zero(t, state,m , drag_f):
    return state[1]  # state[1] is y, so looking for y == 0

event_y_below_zero.terminal = True
event_y_below_zero.direction = -1 # When y == 0, look for transition from pos to neg. 


def drag(t, vx, vy):
    """Compute the drag force"""
    v = np.sqrt(vx**2 + vy**2) # Magnitude of velocity vector
    F_d = 0.5 * C_d * rho * A * v**2
    
    return F_d


def _projectile_motion( t, state, m, drag_f):
    """Calculate dy/dt for the state y. This version includes all state
    values, and the input is the same structure as the output. """
    
    x, y, vx, vy, ax, ay = state
    
    v = np.sqrt(vx**2 + vy**2) # Magnitude of velocity vector
    a =  math.atan2(vy, vx)
    
    F_d = drag_f(t, vx, vy)

    # v_x/v is the cos of the angle of the velocity vector
    # so this is -F_d*cos(v)
    ax = -F_d * vx / (m * v)

    #v_y/v is the sin of the angle of the velocity vector
    ay = -g - (F_d * vy / (m * v))
    
    return (ax, ay, vx, vy, ax, ay )

def projectile_motion( t, yn, m, drag_f):
    """Calculate dy/dt for the state y"""
    
    print(t, end=' ')
  
    if len(yn) < 6:
        yn = np.append(yn, np.array([0,0]))
    
    yn1 = _projectile_motion(t, yn, m, drag_f)
    
    return yn1[2:]

def throw(v0, m, angle, drag_f):
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
    t_eval = np.arange(0, t_max, dt)

    solution = solve_ivp(projectile_motion, t_span, 
                         y0 =  [x0, y0, vx0, vy0], # Initial State
                         t_eval=t_eval, 
                         args = (m, drag_f), events=event_y_below_zero)

    t = solution.t
    x, y, vx, vy = solution.y # Unpack the states
    
    # Calculate drag force and accelerations for each time step
    # We have to re-calc these b/c they are hard to pass out of projectile_motion() through solve_ivp()
    # However, note that these equations are working with vectors, not scalars like they were in 
    # projectile_motion()

    # drag_v = np.vectorize(drag)
    
    F_d = drag_f(t, vx, vy)
    
    v = np.sqrt(vx**2 + vy**2)
    ax = -F_d * vx / (m * v)
    ay = -g - (F_d * vy / (m * v))
    
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

df = throw(v0=200, m = 0.145, angle=55, drag_f=drag)
plot_traj(df)

