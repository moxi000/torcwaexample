import sys
import importlib
import torch
import numpy as np

# Load pip package under alias torcwa_pip
pip_site = '/root/.pyenv/versions/3.11.12/lib/python3.11/site-packages'
sys.path.insert(0, pip_site)
torcwa_pip = importlib.import_module('torcwa')
sys.modules['torcwa_pip'] = torcwa_pip
sys.path.pop(0)
sys.modules.pop('torcwa')

# Load local version
import torcwa as torcwa_local


def run_example(mod):
    sim_dtype = torch.complex64
    geo_dtype = torch.float32
    device = torch.device('cpu')
    lamb0 = 532.
    azi_ang = 0.
    substrate_eps = 1.46**2
    L = [300., 300.]
    order_N = 3
    order = [order_N, order_N]
    inc_ang = torch.linspace(0., 60., 3, dtype=geo_dtype, device=device)*(np.pi/180)
    mod.rcwa_geo.device = device
    mod.rcwa_geo.dtype = geo_dtype
    mod.rcwa_geo.Lx = L[0]
    mod.rcwa_geo.Ly = L[1]
    mod.rcwa_geo.nx = 50
    mod.rcwa_geo.ny = 50
    mod.rcwa_geo.edge_sharpness = 1000.
    mod.rcwa_geo.grid()

    r_vals = []
    for ang in inc_ang:
        sim = mod.rcwa(freq=1/lamb0, order=order, L=L, dtype=sim_dtype, device=device)
        sim.add_input_layer(eps=substrate_eps)
        sim.set_incident_angle(inc_ang=ang, azi_ang=azi_ang)
        sim.solve_global_smatrix()
        r_vals.append(sim.S_parameters(orders=[0,0], direction='forward',
                                       port='reflection', polarization='pp',
                                       ref_order=[0,0]))
    return torch.cat(r_vals)

res_local = run_example(torcwa_local)
res_pip = run_example(torcwa_pip)

diff = torch.max(torch.abs(res_local - res_pip))
print('max difference:', diff.item())
