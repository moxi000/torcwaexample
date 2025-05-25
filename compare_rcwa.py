import torch
from torcwa.rcwa import rcwa as rcwa_old
from torcwa.new_rcwa import rcwa as rcwa_new

# Simple comparison script

def run_sim(sim):
    sim.set_incident_angle(0., 0.)
    sim.add_layer(thickness=1.0, eps=2.0, mu=1.0)
    sim.solve_global_smatrix()
    S_trans = sim.S_parameters([[0,0]], direction='forward', port='transmission')
    S_refl = sim.S_parameters([[0,0]], direction='forward', port='reflection')
    return S_trans, S_refl

sim_old = rcwa_old(freq=1.0, order=[0,0], L=[1.,1.])
sim_new = rcwa_new(freq=1.0, order=[0,0], L=[1.,1.])

old_T, old_R = run_sim(sim_old)
new_T, new_R = run_sim(sim_new)

print('Old transmission:', old_T)
print('New transmission:', new_T)
print('Old reflection:', old_R)
print('New reflection:', new_R)
