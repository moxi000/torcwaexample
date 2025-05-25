import warnings
import torch
from typing import List, Tuple, Optional, Union, Literal
from .torch_eig import Eig

pi = torch.pi


class rcwa:
    """Simplified refactored RCWA implementation for comparison tests."""
    def __init__(self,
                 freq: float,
                 order: List[int],
                 L: List[float],
                 *,
                 dtype: torch.dtype = torch.complex64,
                 device: Optional[torch.device] = None,
                 stable_eig_grad: bool = True,
                 avoid_Pinv_instability: bool = False,
                 max_Pinv_instability: float = 0.005):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        if dtype not in [torch.complex64, torch.complex128]:
            warnings.warn('Invalid dtype. Using torch.complex64.', UserWarning)
            dtype = torch.complex64
        self._dtype = dtype
        self.stable_eig_grad = stable_eig_grad
        self.avoid_Pinv_instability = avoid_Pinv_instability
        self.max_Pinv_instability = max_Pinv_instability
        if avoid_Pinv_instability:
            self.Pinv_instability = []
        # params
        self.freq = torch.as_tensor(freq, dtype=self._dtype, device=self._device)
        self.omega = 2 * pi * self.freq
        self.L = torch.as_tensor(L, dtype=self._dtype, device=self._device)
        self.order = order
        self.order_x = torch.arange(-order[0], order[0]+1, dtype=torch.int64, device=self._device)
        self.order_y = torch.arange(-order[1], order[1]+1, dtype=torch.int64, device=self._device)
        self.order_N = len(self.order_x)*len(self.order_y)
        self.Gx_norm = 1/(L[0]*self.freq)
        self.Gy_norm = 1/(L[1]*self.freq)
        self.eps_in = torch.tensor(1., dtype=self._dtype, device=self._device)
        self.mu_in = torch.tensor(1., dtype=self._dtype, device=self._device)
        self.eps_out = torch.tensor(1., dtype=self._dtype, device=self._device)
        self.mu_out = torch.tensor(1., dtype=self._dtype, device=self._device)
        self.layer_N = 0
        self.thickness = []
        self.eps_conv = []
        self.mu_conv = []
        self.P = []
        self.Q = []
        self.kz_norm = []
        self.E_eigvec = []
        self.H_eigvec = []
        self.layer_S11 = []
        self.layer_S21 = []
        self.layer_S12 = []
        self.layer_S22 = []
        self.Cf = []
        self.Cb = []
        # Precompute identity matrices
        self._eye_N = torch.eye(self.order_N, dtype=self._dtype, device=self._device)
        self._eye_2N = torch.eye(2*self.order_N, dtype=self._dtype, device=self._device)
        self._zeros_2N = torch.zeros(2*self.order_N, 2*self.order_N, dtype=self._dtype, device=self._device)

    def add_input_layer(self, eps: float=1., mu: float=1.):
        self.eps_in = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        self.mu_in = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        self.Sin = []

    def add_output_layer(self, eps: float=1., mu: float=1.):
        self.eps_out = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        self.mu_out = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        self.Sout = []

    def set_incident_angle(self, inc_ang: float, azi_ang: float, angle_layer: str='input'):
        self.inc_ang = torch.as_tensor(inc_ang, dtype=self._dtype, device=self._device)
        self.azi_ang = torch.as_tensor(azi_ang, dtype=self._dtype, device=self._device)
        if angle_layer not in ['input','output']:
            warnings.warn('Invalid angle layer. Using input layer.', UserWarning)
            angle_layer = 'input'
        self.angle_layer = angle_layer
        self._kvectors()

    def _kvectors(self):
        if self.angle_layer == 'input':
            n_eff = torch.sqrt(self.eps_in*self.mu_in).real
        else:
            n_eff = torch.sqrt(self.eps_out*self.mu_out).real
        self.kx0_norm = n_eff*torch.sin(self.inc_ang)*torch.cos(self.azi_ang)
        self.ky0_norm = n_eff*torch.sin(self.inc_ang)*torch.sin(self.azi_ang)
        self.kx_norm = self.kx0_norm + self.order_x*self.Gx_norm
        self.ky_norm = self.ky0_norm + self.order_y*self.Gy_norm
        kx_grid, ky_grid = torch.meshgrid(self.kx_norm, self.ky_norm, indexing='ij')
        self.Kx_norm_dn = kx_grid.reshape(-1)
        self.Ky_norm_dn = ky_grid.reshape(-1)
        self.Kx_norm = torch.diag(self.Kx_norm_dn)
        self.Ky_norm = torch.diag(self.Ky_norm_dn)
        self._compute_V_matrix(1., 1., 'Vf')
        if hasattr(self, 'Sin'):
            self._compute_V_matrix(self.eps_in, self.mu_in, 'Vi')
            self._compute_interface_smatrix('Vi', 'Sin')
        if hasattr(self, 'Sout'):
            self._compute_V_matrix(self.eps_out, self.mu_out, 'Vo')
            self._compute_interface_smatrix('Vo', 'Sout')

    def _compute_V_matrix(self, eps, mu, name):
        if not isinstance(eps, torch.Tensor):
            eps = torch.tensor(eps, dtype=self._dtype, device=self._device)
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=self._dtype, device=self._device)
        Kz_norm_dn = torch.sqrt(eps*mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(Kz_norm_dn.imag<0, Kz_norm_dn.conj(), Kz_norm_dn)
        kxky = self.Kx_norm_dn*self.Ky_norm_dn
        kz = Kz_norm_dn
        V11 = torch.diag(-kxky/kz)
        V12 = torch.diag(-kz - self.Ky_norm_dn**2/kz)
        V21 = torch.diag(kz + self.Kx_norm_dn**2/kz)
        V22 = torch.diag(kxky/kz)
        V = torch.cat([torch.cat([V11,V12],dim=1), torch.cat([V21,V22],dim=1)],dim=0)
        setattr(self, name, V)

    def _compute_interface_smatrix(self, V_name, S_name):
        V = getattr(self, V_name)
        Vf = self.Vf
        V_sum = Vf + V
        V_diff = Vf - V
        S11 = 2*torch.linalg.solve(V_sum, V if V_name=='Vi' else Vf)
        S21 = (-1 if V_name=='Vi' else 1)*torch.linalg.solve(V_sum, V_diff)
        S12 = -S21
        S22 = 2*torch.linalg.solve(V_sum, Vf if V_name=='Vi' else V)
        S_list = getattr(self, S_name)
        S_list.extend([S11, S21, S12, S22])

    def add_layer(self, thickness, eps=1., mu=1.):
        is_eps_h = not isinstance(eps, torch.Tensor) or eps.numel()==1
        is_mu_h = not isinstance(mu, torch.Tensor) or mu.numel()==1
        if is_eps_h:
            val = eps if not isinstance(eps, torch.Tensor) else eps.item()
            self.eps_conv.append(val*self._eye_N)
        else:
            self.eps_conv.append(self._material_conv_optimized(eps))
        if is_mu_h:
            val = mu if not isinstance(mu, torch.Tensor) else mu.item()
            self.mu_conv.append(val*self._eye_N)
        else:
            self.mu_conv.append(self._material_conv_optimized(mu))
        self.layer_N += 1
        self.thickness.append(thickness)
        if is_eps_h and is_mu_h:
            eps_val = eps if not isinstance(eps, torch.Tensor) else eps.item()
            mu_val = mu if not isinstance(mu, torch.Tensor) else mu.item()
            self._eigen_decomposition_homogeneous(eps_val, mu_val)
        else:
            self._eigen_decomposition()
        self._solve_layer_smatrix()

    def _material_conv_optimized(self, material):
        material = material.to(dtype=self._dtype, device=self._device)
        material_N = material.shape[0]*material.shape[1]
        mat_fft = torch.fft.fft2(material)/material_N
        ox, oy = torch.meshgrid(self.order_x, self.order_y, indexing='ij')
        ox = ox.reshape(-1).to(torch.int64)
        oy = oy.reshape(-1).to(torch.int64)
        idx = torch.arange(self.order_N, device=self._device)
        idx_x, idx_y = torch.meshgrid(idx, idx, indexing='ij')
        conv_x = ox[idx_x]-ox[idx_y]
        conv_y = oy[idx_x]-oy[idx_y]
        return mat_fft[conv_x, conv_y]

    def _eigen_decomposition_homogeneous(self, eps, mu):
        P = torch.cat([
            torch.cat([torch.zeros_like(self.mu_conv[-1]), self.mu_conv[-1]], dim=1),
            torch.cat([-self.mu_conv[-1], torch.zeros_like(self.mu_conv[-1])], dim=1)
        ], dim=0) + (1/eps) * (torch.cat([self.Kx_norm, self.Ky_norm], dim=0) @
                              torch.cat([self.Ky_norm, -self.Kx_norm], dim=1))
        Q = torch.cat([
            torch.cat([torch.zeros_like(self.eps_conv[-1]), -self.eps_conv[-1]], dim=1),
            torch.cat([self.eps_conv[-1], torch.zeros_like(self.eps_conv[-1])], dim=1)
        ], dim=0) + (1/mu) * (torch.cat([self.Kx_norm, self.Ky_norm], dim=0) @
                              torch.cat([-self.Ky_norm, self.Kx_norm], dim=1))

        self.P.append(P)
        self.Q.append(Q)

        kz_norm = torch.sqrt(eps*mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz_norm = torch.where(kz_norm.imag < 0, kz_norm.conj(), kz_norm)
        kz_norm = torch.cat([kz_norm, kz_norm])
        self.kz_norm.append(kz_norm)
        self.E_eigvec.append(self._eye_2N)

    def _eigen_decomposition(self):
        eps_inv = torch.linalg.inv(self.eps_conv[-1])
        mu_inv = torch.linalg.inv(self.mu_conv[-1])
        zeros_N = torch.zeros_like(self._eye_N)
        P_base = torch.cat([torch.cat([zeros_N, -self.mu_conv[-1]], dim=1),
                            torch.cat([self.mu_conv[-1], zeros_N], dim=1)], dim=0)
        P_add = torch.cat([self.Kx_norm, self.Ky_norm], dim=0) @ eps_inv @ torch.cat([self.Ky_norm, -self.Kx_norm], dim=1)
        P = P_base + P_add
        Q_base = torch.cat([torch.cat([zeros_N, self.eps_conv[-1]], dim=1),
                            torch.cat([-self.eps_conv[-1], zeros_N], dim=1)], dim=0)
        Q_add = torch.cat([self.Kx_norm, self.Ky_norm], dim=0) @ mu_inv @ torch.cat([-self.Ky_norm, self.Kx_norm], dim=1)
        Q = Q_base + Q_add
        self.P.append(P)
        self.Q.append(Q)
        PQ = P @ Q
        if self.stable_eig_grad:
            kz_sq, E_eigvec = Eig.apply(PQ)
        else:
            kz_sq, E_eigvec = torch.linalg.eig(PQ)
        kz_norm = torch.sqrt(kz_sq)
        kz_norm = torch.where(kz_norm.imag<0, -kz_norm, kz_norm)
        self.kz_norm.append(kz_norm)
        self.E_eigvec.append(E_eigvec)

    def _solve_layer_smatrix(self):
        idx = -1
        kz_norm = self.kz_norm[idx]
        E_eigvec = self.E_eigvec[idx]
        thickness = self.thickness[idx]
        phase = torch.diag(torch.exp(1j*self.omega*kz_norm*thickness))
        Kz_norm = torch.diag(kz_norm)
        if self.P[idx] is not None:
            try:
                P_inv = torch.linalg.inv(self.P[idx])
            except:
                P_inv = torch.linalg.pinv(self.P[idx])
            H_eigvec = P_inv @ E_eigvec @ Kz_norm
        else:
            H_eigvec = E_eigvec @ Kz_norm
        self.H_eigvec.append(H_eigvec)
        # coupling
        Vf_inv = torch.linalg.inv(self.Vf)
        E_plus = E_eigvec + Vf_inv @ H_eigvec
        E_minus = E_eigvec - Vf_inv @ H_eigvec
        C_mat = torch.cat([torch.cat([E_plus, E_minus@phase], dim=1),
                           torch.cat([E_minus@phase, E_plus], dim=1)], dim=0)
        rhs_f = torch.cat([2*self._eye_2N, self._zeros_2N], dim=0)
        rhs_b = torch.cat([self._zeros_2N, 2*self._eye_2N], dim=0)
        Cf = torch.linalg.solve(C_mat, rhs_f)
        Cb = torch.linalg.solve(C_mat, rhs_b)
        self.Cf.append(Cf)
        self.Cb.append(Cb)
        N2 = 2*self.order_N
        Cf_t, Cf_b = Cf[:N2], Cf[N2:]
        Cb_t, Cb_b = Cb[:N2], Cb[N2:]
        eye = self._eye_2N
        E_phase = E_eigvec @ phase
        S11 = E_phase @ Cf_t + E_eigvec @ Cf_b
        S21 = E_eigvec @ Cf_t + E_phase @ Cf_b - eye
        S12 = E_phase @ Cb_t + E_eigvec @ Cb_b - eye
        S22 = E_eigvec @ Cb_t + E_phase @ Cb_b
        self.layer_S11.append(S11)
        self.layer_S21.append(S21)
        self.layer_S12.append(S12)
        self.layer_S22.append(S22)

    def _RS_prod_optimized(self, Sm, Sn, Cm, Cn):
        eye = self._eye_2N
        Sm11, Sm21, Sm12, Sm22 = Sm
        Sn11, Sn21, Sn12, Sn22 = Sn
        tmp1 = torch.linalg.inv(eye - Sm12 @ Sn21)
        tmp2 = torch.linalg.inv(eye - Sn21 @ Sm12)
        S11 = Sn11 @ tmp1 @ Sm11
        S21 = Sm21 + Sm22 @ tmp2 @ Sn21 @ Sm11
        S12 = Sn12 + Sn11 @ tmp1 @ Sm12 @ Sn22
        S22 = Sm22 @ tmp2 @ Sn22
        C = [[], []]
        for m in range(len(Cm[0])):
            C[0].append(Cm[0][m] + Cm[1][m] @ tmp2 @ Sn21 @ Sm11)
            C[1].append(Cm[1][m] @ tmp2 @ Sn22)
        for n in range(len(Cn[0])):
            C[0].append(Cn[0][n] @ tmp1 @ Sm11)
            C[1].append(Cn[1][n] + Cn[0][n] @ tmp1 @ Sm12 @ Sn22)
        return [S11,S21,S12,S22], C

    def solve_global_smatrix(self):
        if self.layer_N == 0:
            eye = self._eye_2N
            zeros = self._zeros_2N
            self.S = [eye, zeros, zeros, eye]
            self.C = [[], []]
            return
        S11 = self.layer_S11[0]
        S21 = self.layer_S21[0]
        S12 = self.layer_S12[0]
        S22 = self.layer_S22[0]
        C = [[self.Cf[0]], [self.Cb[0]]]
        for i in range(1, self.layer_N):
            S_curr = [S11,S21,S12,S22]
            S_next = [self.layer_S11[i], self.layer_S21[i], self.layer_S12[i], self.layer_S22[i]]
            C_next = [[self.Cf[i]],[self.Cb[i]]]
            [S11,S21,S12,S22], C = self._RS_prod_optimized(S_curr,S_next,C,C_next)
        if hasattr(self,'Sin'):
            [S11,S21,S12,S22], C = self._RS_prod_optimized(self.Sin,[S11,S21,S12,S22],[[],[]],C)
        if hasattr(self,'Sout'):
            [S11,S21,S12,S22], C = self._RS_prod_optimized([S11,S21,S12,S22],self.Sout,C,[[],[]])
        self.S = [S11,S21,S12,S22]
        self.C = C

    def _matching_indices(self, orders: torch.Tensor) -> torch.Tensor:
        orders = orders.clone()
        orders[:,0] = orders[:,0].clamp(-self.order[0], self.order[0])
        orders[:,1] = orders[:,1].clamp(-self.order[1], self.order[1])
        x_idx = orders[:,0] + self.order[0]
        y_idx = orders[:,1] + self.order[1]
        return len(self.order_y)*x_idx + y_idx

    def S_parameters(self, orders, *, direction='forward', port='transmission', polarization='xx'):
        orders = torch.as_tensor(orders, dtype=torch.int64, device=self._device).reshape(-1,2)
        order_indices = self._matching_indices(orders)
        ref_idx = self._matching_indices(torch.tensor([[0,0]], device=self._device))
        if direction=='forward' and port=='transmission':
            idx=0
        elif direction=='forward' and port=='reflection':
            idx=1
        elif direction=='backward' and port=='reflection':
            idx=2
        else:
            idx=3
        S = self.S[idx][order_indices, ref_idx]
        return S
