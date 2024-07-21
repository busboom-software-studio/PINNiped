#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from vtraj.ode import *

# Example of how to use the dataclass
consts = Constants(drag_f=drag)
params = Parameters()
params.Kp_theta = .1

df = throw(v0=300, angle=45, consts=consts, params = params)
plot_traj(df)

