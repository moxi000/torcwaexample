import torch
import torch.fft

class geometry:
    def __init__(self,
            Lx:float=1.,
            Ly:float=1.,
            nx:int=100,
            ny:int=100,
            edge_sharpness:float=1000.,*,
            dtype=torch.float32,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        ):

        '''
            Geometry

            Parameters
            - Lx: x-direction Lattice constant (float)
            - Ly: y-direction Lattice constant (float)
            - x: x-axis sampling number (int)
            - y: y-axis sampling number (int)
            - edge_sharpness: sharpness of edge (float)

            Keyword Parameters
            - dtype: geometry data type (only torch.complex64 and torch.complex128 are allowed.)
            - device: geometry device (only torch.device('cpu') and torch.device('cuda') are allowed.)

        '''
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.edge_sharpness = edge_sharpness

        self.dtype = dtype
        self.device = device

    def grid(self):
        '''
            Update grid
        '''

        self.x = (self.Lx/self.nx)*(torch.arange(self.nx,dtype=self.dtype,device=self.device)+0.5)
        self.y = (self.Ly/self.ny)*(torch.arange(self.ny,dtype=self.dtype,device=self.device)+0.5)
        self.x_grid, self.y_grid = torch.meshgrid(self.x,self.y,indexing='ij')

    def _ensure_grid(self):
        if not hasattr(self, 'x_grid'):
            self.grid()

    def circle(self,R,Cx,Cy):
        '''
            R: radius
            Cx: x center
            Cy: y center
        '''

        self._ensure_grid()
        level = 1. - torch.sqrt(((self.x_grid-Cx)/R)**2 + ((self.y_grid-Cy)/R)**2)
        return torch.sigmoid(self.edge_sharpness*level)

    def circle_batch(self,R,Cx,Cy):
        '''Create multiple circles at once.

            Parameters
            ----------
            R : Tensor or float
                Radii for each circle, shape [N]
            Cx, Cy : Tensor or float
                Center coordinates for each circle, shape [N]
        '''

        self._ensure_grid()
        R = torch.as_tensor(R, dtype=self.dtype, device=self.device).reshape(-1,1,1)
        Cx = torch.as_tensor(Cx, dtype=self.dtype, device=self.device).reshape(-1,1,1)
        Cy = torch.as_tensor(Cy, dtype=self.dtype, device=self.device).reshape(-1,1,1)
        level = 1. - torch.sqrt(((self.x_grid-Cx)/R)**2 + ((self.y_grid-Cy)/R)**2)
        return torch.sigmoid(self.edge_sharpness*level)

    def ellipse(self,Rx,Ry,Cx,Cy,theta=0.):
        '''
            Rx: x direction radius
            Ry: y direction radius
            Cx: x center
            Cy: y center
        '''

        theta = torch.as_tensor(theta,dtype=self.dtype,device=self.device)

        self._ensure_grid()
        level = 1. - torch.sqrt((((self.x_grid-Cx)*torch.cos(theta)+(self.y_grid-Cy)*torch.sin(theta))/Rx)**2 + ((-(self.x_grid-Cx)*torch.sin(theta)+(self.y_grid-Cy)*torch.cos(theta))/Ry)**2)
        return torch.sigmoid(self.edge_sharpness*level)

    def square(self,W,Cx,Cy,theta=0.):
        '''
            W: width
            Cx: x center
            Cy: y center
            theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        '''

        theta = torch.as_tensor(theta,dtype=self.dtype,device=self.device)

        self._ensure_grid()
        level = 1. - (torch.maximum(torch.abs(((self.x_grid-Cx)*torch.cos(theta)+(self.y_grid-Cy)*torch.sin(theta))/(W/2.)),torch.abs((-(self.x_grid-Cx)*torch.sin(theta)+(self.y_grid-Cy)*torch.cos(theta))/(W/2.))))
        return torch.sigmoid(self.edge_sharpness*level)

    def rectangle(self,Wx,Wy,Cx,Cy,theta=0.):
        '''
            Wx: x width
            Wy: y width
            Cx: x center
            Cy: y center
            theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        '''

        theta = torch.as_tensor(theta,dtype=self.dtype,device=self.device)

        self._ensure_grid()
        level = 1. - (torch.maximum(torch.abs(((self.x_grid-Cx)*torch.cos(theta)+(self.y_grid-Cy)*torch.sin(theta))/(Wx/2.)),torch.abs((-(self.x_grid-Cx)*torch.sin(theta)+(self.y_grid-Cy)*torch.cos(theta))/(Wy/2.))))
        return torch.sigmoid(self.edge_sharpness*level)

    def rhombus(self,Wx,Wy,Cx,Cy,theta=0.):
        '''
            Wx: x diagonal
            Wy: y diagonal
            Cx: x center
            Cy: y center
            theta: rotation angle / center: [Cx, Cy] / axis: z-axis
        '''

        theta = torch.as_tensor(theta,dtype=self.dtype,device=self.device)

        self._ensure_grid()
        level = 1. - (torch.abs(((self.x_grid-Cx)*torch.cos(theta)+(self.y_grid-Cy)*torch.sin(theta))/(Wx/2.)) + torch.abs((-(self.x_grid-Cx)*torch.sin(theta)+(self.y_grid-Cy)*torch.cos(theta))/(Wy/2.)))
        return torch.sigmoid(self.edge_sharpness*level)

    def super_ellipse(self,Wx,Wy,Cx,Cy,theta=0.,power=2.):
        '''
            Wx: x width
            Wy: y width
            Cx: x center
            Cy: y center
            theta: rotation angle / center: [Cx, Cy] / axis: z-axis
            power: elliptic power
        '''

        theta = torch.as_tensor(theta,dtype=self.dtype,device=self.device)

        self._ensure_grid()
        level = 1. - (torch.abs(((self.x_grid-Cx)*torch.cos(theta)+(self.y_grid-Cy)*torch.sin(theta))/(Wx/2.))**power + torch.abs((-(self.x_grid-Cx)*torch.sin(theta)+(self.y_grid-Cy)*torch.cos(theta))/(Wy/2.))**power)**(1/power)
        return torch.sigmoid(self.edge_sharpness*level)

    def union(self,A,B):
        '''
            A U B
        '''

        return torch.maximum(A,B)

    def intersection(self,A,B):
        '''
            A n B
        '''

        return torch.minimum(A,B)

    def difference(self,A,B):
        '''
            A - B = A n Bc
        '''

        return torch.minimum(A,1.-B)



# rcwa_geo class is kept for backward compatibility
rcwa_geo = geometry
