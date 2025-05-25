import torch
from torcwa.rcwa import rcwa as rcwa_old
from torcwa.new_rcwa import rcwa as rcwa_new

freq=1.0
order=[0,0]
L=[1.,1.]

sim_old=rcwa_old(freq=freq, order=order, L=L)
sim_new=rcwa_new(freq=freq, order=order, L=L)

# Set angle and add layer
sim_old.set_incident_angle(0.,0.)
sim_new.set_incident_angle(0.,0.)

sim_old.add_layer(1.0, eps=2.0, mu=1.0)
sim_new.add_layer(1.0, eps=2.0, mu=1.0)

print("Old kz_norm", sim_old.kz_norm[0])
print("New kz_norm", sim_new.kz_norm[0])

print("Old E_eigvec", sim_old.E_eigvec[0])
print("New E_eigvec", sim_new.E_eigvec[0])

sim_old.solve_global_smatrix()
sim_new.solve_global_smatrix()

print("Old S21", sim_old.layer_S21[0])
print("New S21", sim_new.layer_S21[0])

print("Old S (T)", sim_old.S_parameters([[0,0]], direction='forward', port='transmission'))
print("New S (T)", sim_new.S_parameters([[0,0]], direction='forward', port='transmission'))
