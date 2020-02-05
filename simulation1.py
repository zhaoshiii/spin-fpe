from mpl_toolkits.mplot3d import Axes3D
from shutil import copyfile  #Library to save intermediate files
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
from fipy import CellVariable, Gmsh2DIn3DSpace, GaussianNoiseVariable, Viewer, TransientTerm, DiffusionTerm, ConvectionTerm, DefaultSolver, VTKViewer
from fipy.tools import numerix
fig = plt.figure()
ax = fig.gca(projection='3d')

t = np.arange(0e-9,5e-9,1e-12)
t0 = 1e-9
tstep = 1e-11
trun = 2.1e-09
trise = 50e-12
alpha = 0.1
Ku = 2.245e5 #J/m3
mu0 = math.pi*4e-7 #N/A2
Temp = 300 #K
gr = 1.76e11 #rad/(s.T)
hbar = 1.054e-34 #J.s
Ms = 1257e3 #A/m
Pol = 0.4
tF = 1.7e-09 #m
e = 1.602e-19 #C
Nxx   = 0.0167
Nyy   = 0.0833
Nzz   = 1-Nxx-Nyy
K_I   = 1.295e-03 #J/m2
K_U   = 2.245e5 #J/m3
L_FL  = 30e-09 #m
W_FL  = 90e-09
T_FL  = 1.7e-09
V_FL  = L_FL*W_FL*T_FL*math.pi/4
T_MGO = 2e-09
E_c   = 50e-15
kb = 1.38e-23
uz = [0.0, 1.0, 0.0]
J = 1e-10 #A/m2

def cross(a,b):
    product = np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])
    return product

tstart = t0
tend = 1e-09 + tstart
trf = trise
Volt = 0
slope = Volt/trf
hextx_tmp = 39788.735772973836*0
Hext = np.array([hextx_tmp,0,0])

mesh = Gmsh2DIn3DSpace('''
    radius = 1;
    cellSize = 0.2;

    //create inner 1/8 shell
    Point(1) = {0, 0, 0, cellSize};
    Point(2) = {-radius, 0, 0, cellSize};
    Point(3) = {0, radius, 0, cellSize};
    Point(4) = {0, 0, radius, cellSize};
    Circle(1) = {2, 1, 3};
    Circle(2) = {4, 1, 2};
    Circle(3) = {4, 1, 3};
    Line Loop(1) = {1, -3, 2} ;
    Ruled Surface(1) = {1};
    //create remaining 7/8 inner shells
    t1[] = Rotate {{0,0,1},{0,0,0},Pi/2} {Duplicata{Surface{1};}};
    t2[] = Rotate {{0,0,1},{0,0,0},Pi} {Duplicata{Surface{1};}};
    t3[] = Rotate {{0,0,1},{0,0,0},Pi*3/2} {Duplicata{Surface{1};}};
    t4[] = Rotate {{0,1,0},{0,0,0},-Pi/2} {Duplicata{Surface{1};}};
    t5[] = Rotate {{0,0,1},{0,0,0},Pi/2} {Duplicata{Surface{t4[0]};}};
    t6[] = Rotate {{0,0,1},{0,0,0},Pi} {Duplicata{Surface{t4[0]};}};
    t7[] = Rotate {{0,0,1},{0,0,0},Pi*3/2} {Duplicata{Surface{t4[0]};}};

    //create entire inner and outer shell
    Surface Loop(100)={1,t1[0],t2[0],t3[0],t7[0],t4[0],t5[0],t6[0]};
''', order=2).extrude(extrudeFunc=lambda r: 1.1 * r) # doctest: +GMSH
Phi = CellVariable(name=r"$\Phi$", mesh=mesh) # doctest: +GMSH

Phi.setValue(GaussianNoiseVariable(mesh=mesh,mean=0.5,variance=0.01)) # doctest: +GMSH

if __name__ == "__main__":
    try:
        print("\n done") #Intermediate Print Statement
        #viewer = MayaviClient(vars=phi,datamin=0., datamax=1.,daemon_file="/home/elekrv/Documents/Scripts/CahnSphere/sphereDaemon.py") #Commented as not required anymore
        viewer = VTKViewer(vars=Phi,datamin=0., datamax=1.,xmin=-2.5, zmax=2.5) #Changed Statement
        print("\n Daemon file") #Intermediate Print Statement
    except (NameError,ImportError,SystemError,TypeError):
        viewer = VTKViewer(vars=Phi,datamin=0., datamax=1.,xmin=-2.5, zmax=2.5) #Changed Statement

diffusion = (alpha*gr*kb*Temp)/((1+alpha**2)*mu0*V_FL*Ms)

def model(m,t):
    #Voltage = volt(t)
    Voltage = 0
    Hdemag = -Ms*np.array([Nxx, Nyy, Nzz])
    E_field = Voltage/T_MGO
    Keff = K_U+(K_I-(E_c*E_field))/T_FL
    Hanis = np.array([0,0,2*Keff/(mu0*Ms)])
    heff = (Hanis+Hdemag)
    
    Heff = np.array([heff[0]*m[0], heff[1]*m[1], heff[2]*m[2]]) + Hext
    mp = [0,1,0]
    STT = (gr*hbar*J)/(2*e*mu0*Ms*tF)*np.array(cross(m,mp))
    #dmdt = (-mu0 * gr * cross(m,(Heff + alpha*cross(m,Heff)))-STT)/(1+alpha**2)
    dmdt = (-mu0 * gr * cross(m,(Heff + alpha*cross(m,Heff))))/(1+alpha**2)
    return dmdt


def volt(t):
    Voltage = np.array((0*(t<=tstart))+(0*(t>=tend))+(((t-tstart)*slope)*((t>=tstart)&(t<=tstart+trf)))+(Volt*((t>=tstart+trf)&(t<tend-trf)))+(Volt-((t-tend+trf)*slope))*((t>=tend-trf)&(t<=tend)))
    return Voltage
    

    
#m0 = [0.1736, 0, 0.9848]
#m = odeint(model, m0, t)
#vol = volt(m,t)
xx = mesh.cellCenters[0]
yy = mesh.cellCenters[1]
zz = mesh.cellCenters[2]
ll = len(xx)
gradient = numerix.zeros((3,ll))

for i in range(0,5):
    for j in range(0,ll):
        point = np.array([xx[j],yy[j],zz[j]])
        gradient[:,j] = model(point,i*1e-12)
    print(gradient)
    eq = (TransientTerm() == DiffusionTerm(coeff=diffusion) - ConvectionTerm(coeff=gradient))
    eq.solve(Phi,dt=1e-12,solver=DefaultSolver(precon=None))
    if __name__ == "__main__":#Parameters to be changed are in the seciton below.
        viewer.plot(filename="trial.vtk") #It will only save the final vtk file. You can change the name
        if not i%10 or i == 0:
            dest_name = '/home/zhaoshi/Zhao_test/img_' + str(i) + '.vtk'  #Path and name of the intermediate file. The Green Part should be changed to your path & name
            copyfile('/home/zhaoshi/trial.vtk',dest_name) #Specify the path of your trial.vtk file
   


viewer.plot()


#plt.subplot(221)
#plt.plot(t,m[:,0])
#plt.title('Mx versus time')
#plt.subplot(222)
#plt.plot(t,m[:,1])
#plt.title('My versus time')
#plt.subplot(223)
#plt.plot(t,m[:,2])
#plt.title('Mz versus time')
#plt.subplot(224)
#plt.plot(t,vol)
#plt.title('V versus time')
ax.plot(m[:,0],m[:,1],m[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
