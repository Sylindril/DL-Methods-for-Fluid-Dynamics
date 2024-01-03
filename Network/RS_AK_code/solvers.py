import numpy as np


class Stokes3D:
    def __init__(self, parameters):
        self.h  = parameters['h']
        self.Lx = parameters['Lx']
        self.Ly = parameters['Ly']
        self.Lz = parameters['Lz']
        self.Nx = parameters['N_g']
        self.Ny = parameters['N_g']
        self.Nz = parameters['N_g']
        self.a  = 1
        self.N  = parameters['N']
        h = self.h
        
        #Fourier space grid
        kx = (2*np.pi/h)*np.fft.fftfreq(self.Nx)
        ky = (2*np.pi/h)*np.fft.fftfreq(self.Ny)
        kz = (2*np.pi/h)*np.fft.fftfreq(self.Nz)
        self.kx, self.ky, self.kz = np.meshgrid(kx, ky, kz, 
                                                indexing=parameters['indexingConvention']) 
        self.ksq = self.kx*self.kx + self.ky*self.ky + self.kz*self.kz
        self.iksq = 1/(self.ksq+1e-16)
        self.iksq[0,0,0] = 0


        
        ## Real space
        self.xx=np.linspace(h, self.Lx, self.Nx);   
        self.x, self.y, self.z = np.meshgrid(self.xx, self.xx, self.xx, 
                                             indexing=parameters['indexingConvention']) 
        # self.x = self.x * h
        # self.y = self.y * h
        # self.z = self.z * h


    def volFrac(self, a=1):
        '''
        Given The radius, and dimensions of the box, returns volume fraction. By default, the box is assumed to be a cube of side length 32
        '''
        self.a = a
        return 4*np.pi*np.power(a,3)/(3*self.Lx*self.Ly*self.Lz)

    
    def vCalc(self, a=1, sigma=np.sqrt(2/np.pi), r=np.zeros((3)), F=np.zeros((3))):
        '''
        Returns the sedimentation velocity V of particles, takes the following as inputs
        a = Radius of each individual particle
        c = Volume Fraction, by default calculated in a cubic box of side 32
        N = Number of particles (1 by default)
        Nx, Ny, Nz = Number of grid points in x, y and z directions (32 by default)
        h = This factor scales the wave vectors in fourier space.
            kx = [-pi/h, pi/h]
            By default, h = 1
        r = coordinates of the particles; for a system of 3 particles, should be written thus -
            r = [x1,x2,x3,y1,y2,y3,z1,z2,z3]
            by default, r =  [Nx/2,Ny/2,Nz/2]
        F = Forces acting on the particles; for a system of 3 particles, should be written thus - 
            F = [Fx1,Fx2,Fx3,Fy1,Fy2,Fy3,Fz1,Fz2,Fz3] (Here, Fij means that force along the j'th coordinate axis acting on the i'th particle)
            By default, F = [0,0,2]
        sigma is variance in the mollifier term, by default is 1*np.sqrt(2/np.pi)*a 
        '''
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        kx, ky, kz = self.kx, self.ky, self.kz
        x, y, z =self.x, self.y, self.z
        N = self.N
        ksq = self.ksq
        iksq = self.iksq
        
        scale = ( 2 * np.pi * sigma**2 )**(- 3/2)
        arg = ( (x - r[0])**2 + (y - r[1])**2 +(z - r[2])**2 ) / (2 * sigma**2)
        fx0 =  np.exp(-arg) * scale
        Fk0 = np.fft.fftn(fx0)
        
        # #Force calculation 
        # self.F_x = fx0*F[i] 
        # self.F_y = fx0*F[i+N] 
        # self.F_z = fx0*F[i+2*N] 




        #Creating the Mollifier. 
        Mscale = ( 2 * np.pi * sigma**2 )**(- 3/2)
        Marg = ( (x - r[0])**2 + (y - r[1])**2 +(z - r[2])**2 ) / (2 * sigma**2)
        M =  np.exp(-Marg) * Mscale

        i=0; 
        self.Fk_x = Fk0*F[i] 
        self.Fk_y = Fk0*F[i+N] 
        self.Fk_z = Fk0*F[i+2*N] 

        #Force calculation 
        self.F_x = fx0*F[i] 
        self.F_y = fx0*F[i+N] 
        self.F_z = fx0*F[i+2*N] 
        
        Fdotk = self.Fk_x*kx + self.Fk_y*ky + self.Fk_z*kz
        
        self.vk_x = ( self.Fk_x - Fdotk*kx*iksq )*iksq
        self.vk_y = ( self.Fk_y - Fdotk*ky*iksq )*iksq
        self.vk_z = ( self.Fk_z - Fdotk*kz*iksq )*iksq
        
        self.vk_x[0,0,0] = 0
        self.vk_y[0,0,0] = 0
        self.vk_z[0,0,0] = 0

        self.vx = np.real(np.fft.ifftn(self.vk_x))
        self.vy = np.real(np.fft.ifftn(self.vk_y))
        self.vz = np.real(np.fft.ifftn(self.vk_z))

        vx_av = np.sum(M*self.vx)/np.sum(M)
        vy_av = np.sum(M*self.vy)/np.sum(M)
        vz_av = np.sum(M*self.vz)/np.sum(M)
        return vx_av, vy_av, vz_av
    

    def vUpdate(self, a=1, sigma=np.sqrt(2/np.pi), r=np.zeros((3)), F=np.zeros((3))):
        '''
        Returns the sedimentation velocity V of particles, takes the following as inputs
        a = Radius of each individual particle
        c = Volume Fraction, by default calculated in a cubic box of side 32
        N = Number of particles (1 by default)
        Nx, Ny, Nz = Number of grid points in x, y and z directions (32 by default)
        h = This factor scales the wave vectors in fourier space.
            kx = [-pi/h, pi/h]
            By default, h = 1
        r = coordinates of the particles; for a system of 3 particles, should be written thus -
            r = [x1,x2,x3,y1,y2,y3,z1,z2,z3]
            by default, r =  [Nx/2,Ny/2,Nz/2]
        F = Forces acting on the particles; for a system of 3 particles, should be written thus - 
            F = [Fx1,Fx2,Fx3,Fy1,Fy2,Fy3,Fz1,Fz2,Fz3] (Here, Fij means that force along the j'th coordinate axis acting on the i'th particle)
            By default, F = [0,0,2]
        sigma is variance in the mollifier term, by default is 1*np.sqrt(2/np.pi)*a 
        '''
        kx, ky, kz = self.kx, self.ky, self.kz
        x, y, z =self.x, self.y, self.z
        N = self.N
        iksq = self.iksq
        
        scale = ( 2 * np.pi * sigma**2 )**(- 3/2)
        arg = ( (x - r[0])**2 + (y - r[1])**2 +(z - r[2])**2 ) / (2 * sigma**2)
        fx0 =  np.exp(-arg) * scale
        Fk0 = np.fft.fftn(fx0)
        
        i=0; 
        self.Fk_x = Fk0*F[i] 
        self.Fk_y = Fk0*F[i+N] 
        self.Fk_z = Fk0*F[i+2*N] 

        #Force calculation 
        self.F_x = fx0*F[i] 
        self.F_y = fx0*F[i+N] 
        self.F_z = fx0*F[i+2*N] 
        
        Fdotk = self.Fk_x*kx + self.Fk_y*ky + self.Fk_z*kz
        
        self.vk_x = ( self.Fk_x - Fdotk*kx*iksq )*iksq
        self.vk_y = ( self.Fk_y - Fdotk*ky*iksq )*iksq
        self.vk_z = ( self.Fk_z - Fdotk*kz*iksq )*iksq
        
        self.vk_x[0,0,0] = 0
        self.vk_y[0,0,0] = 0
        self.vk_z[0,0,0] = 0

        self.vx = np.real(np.fft.ifftn(self.vk_x))
        self.vy = np.real(np.fft.ifftn(self.vk_y))
        self.vz = np.real(np.fft.ifftn(self.vk_z))


        return 
    
    # def fcalc(self, sigma=np.sqrt(2/np.pi), r=np.zeros((3)), F=np.zeros((3))):
    #     '''
    #     Returns the sedimentation velocity V of particles, takes the following as inputs
    #     a = Radius of each individual particle
    #     c = Volume Fraction, by default calculated in a cubic box of side 32
    #     N = Number of particles (1 by default)
    #     Nx, Ny, Nz = Number of grid points in x, y and z directions (32 by default)
    #     h = This factor scales the wave vectors in fourier space.
    #         kx = [-pi/h, pi/h]
    #         By default, h = 1
    #     r = coordinates of the particles; for a system of 3 particles, should be written thus -
    #         r = [x1,x2,x3,y1,y2,y3,z1,z2,z3]
    #         by default, r =  [Nx/2,Ny/2,Nz/2]
    #     F = Forces acting on the particles; for a system of 3 particles, should be written thus - 
    #         F = [Fx1,Fx2,Fx3,Fy1,Fy2,Fy3,Fz1,Fz2,Fz3] (Here, Fij means that force along the j'th coordinate axis acting on the i'th particle)
    #         By default, F = [0,0,2]
    #     sigma is variance in the mollifier term, by default is 1*np.sqrt(2/np.pi)*a 
    #     '''
    #     Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
    #     kx, ky, kz = self.kx, self.ky, self.kz
    #     x, y, z =self.x, self.y, self.z
    #     N = self.N
    #     ksq = self.ksq
    #     iksq = self.iksq
        
    #     scale = ( 2 * np.pi * sigma**2 )**(- 3/2)
    #     arg = ( (x - r[0])**2 + (y - r[1])**2 +(z - r[2])**2 ) / (2 * sigma**2)
    #     f0 =  np.exp(-arg) * scale
    #     Fk0= np.fft.fftn(f0)



    #     #Creating the Mollifier. 
    #     Mscale = ( 2 * np.pi * sigma**2 )**(- 3/2)
    #     Marg = ( (x - r[0])**2 + (y - r[1])**2 +(z - r[2])**2 ) / (2 * sigma**2)
    #     M =  np.exp(-Marg) * Mscale

    #     i=0; 
    #     # self.Fk_x = Fk0*F[i] 
    #     # self.Fk_y = Fk0*F[i+N] 
    #     # self.Fk_z = Fk0*F[i+2*N] 
        
    #     self.F_x = f0*F[i] 
    #     self.F_y = f0*F[i+N] 
    #     self.F_z = f0*F[i+2*N] 

        
    #     Fdotk = Fk0*F[i] *kx + Fk0*F[i+N]*ky + Fk0*F[i+2*N]*kz
    #     F_field = np.real(np.fft.ifftn(Fdotk*iksq))
    #     F_av = np.sum(M*F_field)/np.sum(M)

    #     return F_av