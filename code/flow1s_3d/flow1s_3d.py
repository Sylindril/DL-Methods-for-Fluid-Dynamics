import numpy as np
#Definitions
Nx, Ny, Nz  = 32, 32 ,32  # number of grid points.
a, Np = 2, 1     # number particles.
h = 1 #resolution

#Functions
# encapsulating the code in functions
def VolFrac(a, h = 1,  Nx = 32, Ny = 32, Nz = 32):
    '''
    Given The radius, and dimensions of the box, returns volume fraction. By default, the box is assumed to be a cube of side length 32
    '''
    Lx = Nx*h
    Ly = Ny*h
    Lz = Nz*h
    return 4*np.pi*np.power(a,3)/(3*Lx*Ly*Lz)

def Vcalc(a, Np = 1, Nx = 32, Ny = 32, Nz = 32, h = 1, r = np.array([Nx/2,Ny/2,Nz/2]), F = np.array([0,0,1]),c = VolFrac(a, h, Nx, Ny, Nz)):
    '''
    Returns the sedimentation velocity V of particles, takes the following as inputs
    a = Radius of each individual particle
    c = Volume Fraction, by default calculated in a cubic box of side 32
    Np = Number of particles (1 by default)
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
    x, y, z = np.meshgrid(range(Nx), range(Ny), range(Nz), indexing = 'xy') ;                                                                                                                       

    #r = np.zeros(3*Np); F = np.zeros(3*Np)
    fx = np.zeros((Nx, Ny, Nz))
    fy = np.zeros((Nx, Ny, Nz))
    fz = np.zeros((Nx, Ny, Nz))
    Fxk = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Fyk = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Fzk = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    #Fourier grid
    kx = 2 * np.pi*h / Nx * np.concatenate((np.arange(0, Nx/2+1,1),np.arange(-Nx/2+1, 0, 1)))
    ky = 2 * np.pi*h / Ny * np.concatenate((np.arange(0, Nx/2+1,1),np.arange(-Nx/2+1, 0, 1)))
    kz = 2 * np.pi*h / Ny * np.concatenate((np.arange(0, Nx/2+1,1),np.arange(-Nx/2+1, 0, 1)))
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing = 'xy') 

    #r[0], r[1], r[2] = Nx/2, Ny/2, Nz/2
    #F[0], F[1], F[2] = 0, 0, 2 
    sigma, a2 = 1*np.sqrt(2/np.pi)*a, a*a/6 
    scale = ( 2 * np.pi * sigma**2 )**(- 3/2)

    arg = ( (x - Nx/2)**2 + (y - Ny/2)**2 +(z - Nz/2)**2 ) / (2 * sigma**2)
    fx0 =  np.exp(-arg) * scale
    Fk0 = np.fft.fftn(fx0)

    k3 = kx*kx + ky*ky + kz*kz; k3[0, 0, 0] = 1.0


    #Creating the Mollifier. 
    Mscale = ( 2 * np.pi * sigma**2 )**(- 3/2)
    Marg = ( (x - r[0])**2 + (y - r[1])**2 +(z - r[2])**2 ) / (2 * sigma**2)
    M =  np.exp(-Marg) * Mscale


    for i in range(Np):
        kdotr = kx*(r[i] - Nx/2) + ky*(r[i + Np] -Ny/2) + kz*(r[i + 2*Np] -Nz/2) 
        Fxk += Fk0* np.exp(-1j * kdotr)* F[i]      
        Fyk += Fk0* np.exp(-1j * kdotr)* F[i + Np] 
        Fzk += Fk0* np.exp(-1j * kdotr)* F[i + 2*Np] 
    # print (np.max(np.abs(Fxk)))
    # print (np.max(np.abs(Fyk)))
    # print (np.max(np.abs(Fzk)))


    Fdotk = Fxk*kx + Fyk*ky + Fzk*kz
    vxk = (1-a2*k3)*( Fxk - Fdotk*(kx / k3) ) / k3
    vyk = (1-a2*k3)*( Fyk - Fdotk*(ky / k3) ) / k3
    vzk = (1-a2*k3)*( Fzk - Fdotk*(kz / k3) ) / k3
    vxk[0,0,0] = 0
    vyk[0,0,0] = 0
    vzk[0,0,0] = 0

    vx = np.real(np.fft.ifftn(vxk))
    vy = np.real(np.fft.ifftn(vyk))
    vz = np.real(np.fft.ifftn(vzk))

    vx_av = np.sum(M*vx)
    vy_av = np.sum(M*vy)
    vz_av = np.sum(M*vz)



    return np.sqrt(vz_av**2 +vx_av**2 +vy_av**2), vx, vy, vz, x, y, z