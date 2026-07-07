# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:51:26 2015

@author: castro
"""

import numpy as np
import matplotlib.pylab as plt

def Hermite(xi):

    Nv1 = (1.-xi)**2*(2.+xi)/4.0
    Nt1 = L*(1.-xi)**2*(1.+xi)/8.0
    Nv2 = (1.+xi)**2*(2.-xi)/4.0
    Nt2 =-L*(1.+xi)**2*(1.-xi)/8.0

    return Nv1, Nt1, Nv2, Nt2

def Deformation(x_i,y_i,L,cost,sint,ubar):

    n  = 200
    xip = np.linspace(-1.0,1.0,n)

    x  = np.zeros(n)
    y  = np.zeros(n)

    u1 =  ubar[0]*cost + ubar[1]*sint
    v1 = -ubar[0]*sint + ubar[1]*cost
    t1 =  ubar[2]
    u2 =  ubar[3]*cost + ubar[4]*sint
    v2 = -ubar[3]*sint + ubar[4]*cost
    t2 =  ubar[5]

    maxv = 0.0
    for i in range(n):
        Nv1, Nt1, Nv2, Nt2 = Hermite(xip[i])
        v  = Nv1*v1 + Nt1*t1 + Nv2*v2 + Nt2*t2

        if abs(v) > abs(maxv):
            maxv = v

        xi = (xip[i]+1.0)*L/2.0
        u  = (u2-u1)/L*xi+u1

        x[i] = x_i +  cost*(xi+u) - sint*v
        y[i] = y_i +  sint*(xi+u) + cost*v

    return x,y, maxv

print('\n **** Welcome to 2D FEM beam model *** \n')
print(' ****      EUDIM  UTA  2015        *** ')
print(' ****    Cristóbal E. Castro       *** ')
print(' ****     ccastro at uta.cl        *** \n')

FileName = 'InputData_Beam_2D.dat'


f = open(FileName)
f.readline()

line = f.readline()
data = line.split()
nElem = int(data[0])
nNode = int(data[1])
nDof = nNode*3
print ('\n nElem = ',nElem, ' nNode = ',nNode,' nDof = ',nDof)
#f.close()

A = np.zeros(nElem)
E = np.zeros(nElem)
I = np.zeros(nElem)

Connectivity = np.zeros((nElem,2),dtype=int)

f.readline()
for i in range(nElem):
    line = f.readline()
    data = line.split()
    j    = data[0]
    A[i] = data[1]
    E[i] = data[2]
    #E[i] = E[i]*1.0e9
    I[i] = data[3]
    Connectivity[i,0] = int(data[4])
    Connectivity[i,1] = int(data[5])

#    print j,A[i],E[i]

x = np.zeros(nNode)
y = np.zeros(nNode)
F = np.zeros(nDof)

#print 'u=',u

# Forces or displacement
f.readline()
SolCase = int(f.readline())

if SolCase == 0:

    print ('\n Solving for displacements.')

    f.readline()
    for i in range(nNode):
        line = f.readline()
        data = line.split()
        j    = data[0]
        x[i] = data[1]
        y[i] = data[2]
        if len(data) > 3:
            F[i*3]   = data[3]
            F[i*3+1] = data[4]
            F[i*3+2] = data[5]

    print ('')
    print (' Reading boundary conditions')

    f.readline()
    nBC = int(f.readline())

    BCNode = np.zeros(nBC)
    BCDir  = np.zeros((nBC,3))
    f.readline()
    for i in range(nBC):
        line = f.readline()
        data = line.split()
        BCNode[i]   = int(data[0]) # Node index
        BCDir[i,0]  = data[1] # BC on x direction
        BCDir[i,1]  = data[2] # BC on y direction
        BCDir[i,2]  = data[3] # BC on rotation

        #print BCNode[i], BCDir[i,:]

else:

    print (' Error!! No impementation for SolCase /= 0')

# Read in maximum strain
f.readline()
Max_e = float(f.readline())
print ('\n Maximum strain = ',Max_e, ' maximum stress = E*',Max_e)



print ('\n Building beam-bar elements ')

L = np.zeros(nElem)
k = np.zeros(nElem)

print ('')

K = np.zeros((nDof,nDof))
#print K

Tloc = np.zeros((6,6))
Kloc = np.zeros((6,6))
idx = np.zeros(4,dtype=int)
# Matrix assembling
for i in range(nElem):
    #print '\n\n Element ',i+1
    ni = Connectivity[i,0]
    nj = Connectivity[i,1]
    xi = x[ni-1]; yi = y[ni-1]
    xj = x[nj-1]; yj = y[nj-1]
    #print ' Nodes  (',xi,yi,')','  ','(',xj,yj,')'

    L[i] = np.sqrt((xi-xj)**2+(yi-yj)**2)
    #print ' L = ',L[i]
    k[i] = A[i]*E[i]/L[i]

    cost = (xj - xi)/L[i]
    sint = (yj - yi)/L[i]
    #print ' cost =',cost,' sint = ',sint,' Theta = ',np.degrees(np.arccos(cost))

    Tloc[0,0] = cost; Tloc[0,1] = sint
    Tloc[1,0] =-sint; Tloc[1,1] = cost
    Tloc[2,2] = 1.0
    Tloc[3,3] = cost; Tloc[3,4] = sint
    Tloc[4,3] =-sint; Tloc[4,4] = cost
    Tloc[5,5] = 1.0
    Tlocinv = np.linalg.inv(Tloc)

    Kloc[0,0] = A[i]*E[i]/L[i]; Kloc[0,3] = -A[i]*E[i]/L[i]
    Kloc[1,1] = 12*E[i]*I[i]/L[i]**3; Kloc[1,2] = 6*E[i]*I[i]/L[i]**2; Kloc[1,4] = -12*E[i]*I[i]/L[i]**3; Kloc[1,5] = 6*E[i]*I[i]/L[i]**2;
    Kloc[2,1] =  6*E[i]*I[i]/L[i]**2; Kloc[2,2] = 4*E[i]*I[i]/L[i]**1; Kloc[2,4] =  -6*E[i]*I[i]/L[i]**2; Kloc[2,5] = 2*E[i]*I[i]/L[i]**1;
    Kloc[3,0] =-A[i]*E[i]/L[i]; Kloc[3,3] =  A[i]*E[i]/L[i]
    Kloc[4,1] =-12*E[i]*I[i]/L[i]**3; Kloc[4,2] =-6*E[i]*I[i]/L[i]**2; Kloc[4,4] =  12*E[i]*I[i]/L[i]**3; Kloc[4,5] =-6*E[i]*I[i]/L[i]**2;
    Kloc[5,1] =  6*E[i]*I[i]/L[i]**2; Kloc[5,2] = 2*E[i]*I[i]/L[i]**1; Kloc[5,4] =  -6*E[i]*I[i]/L[i]**2; Kloc[5,5] = 4*E[i]*I[i]/L[i]**1;

    # Reutilizando Kloc
    KK = np.dot(Tlocinv,np.dot(Kloc,Tloc))
    #print KK

#    KK = LocalStiffenedMatrix(cost,sint)
#    KK[:,:] = k[i]*KK[:,:]
#    print KK

    #KK = KK * A[i]*E[i]

    # We are looking at nodes ni and nj
    #print ' Nodes ',ni, nj
    idx[:] = [ni*3-2, ni*3, nj*3-2, nj*3]
    #print ' Dof ',idx

    #print ni, nj, idx
    # First local block [0:2,0:2]
    K[idx[0]-1:idx[1],idx[0]-1:idx[1]] = K[idx[0]-1:idx[1],idx[0]-1:idx[1]] + KK[0:3,0:3]
    # Second local block [0:2,2:4]
    K[idx[0]-1:idx[1],idx[2]-1:idx[3]] = K[idx[0]-1:idx[1],idx[2]-1:idx[3]] + KK[0:3,3:6]
    # Third local block [2:4,0:2]
    K[idx[2]-1:idx[3],idx[0]-1:idx[1]] = K[idx[2]-1:idx[3],idx[0]-1:idx[1]] + KK[3:6,0:3]
    # Fourth local block [2:4,2:4]
    K[idx[2]-1:idx[3],idx[2]-1:idx[3]] = K[idx[2]-1:idx[3],idx[2]-1:idx[3]] + KK[3:6,3:6]


#print K
#print np.linalg.eigvals(K)

#print K, F


print (' Aplying boundary condition \n')

for i in range(nBC):
    node = BCNode[i]

    if BCDir[i,0] == 0:
        dof = int((node-1)*3)
        K[dof,:]   = 0.0
        K[:,dof]   = 0.0
        K[dof,dof] = 1.0
        F[dof]     = 0.0

    if BCDir[i,1] == 0:
        dof = int((node-1)*3+1)
        K[dof,:]   = 0.0
        K[:,dof]   = 0.0
        K[dof,dof] = 1.0
        F[dof]     = 0.0

    if BCDir[i,2] == 0:
        dof = int((node-1)*3+2)
        K[dof,:]   = 0.0
        K[:,dof]   = 0.0
        K[dof,dof] = 1.0
        F[dof]     = 0.0

#print K

#print ''
#print ' Eigenvalues'
#eig = np.linalg.eigvals(K)
#print eig

if SolCase == 0:
    print ('')
    print (' Solution')
    u = np.linalg.solve(K,F)

    print (' Node   Displacements')
    print ('           ux            uy        theta')
    for i in range(nNode):
        print (' {:2d}    {:+.4E}   {:+.4E}   {:+.4E}'.format(i+1,u[i*3],u[i*3+1],u[i*3+2]))


ubar = np.zeros(6)
print ('')
#print ' Forces in kN \n Bar    Fx1           Fy1           Fx2           Fy2           AxialForce'
print (' Forces \n Bar    AxialForce [kN]        Tension [GPa]     max Tension [GPa]')
for i in range(nElem):

    #print '\n\n Element ',i+1
    ni = Connectivity[i,0]
    nj = Connectivity[i,1]
    xi = x[ni-1]; yi = y[ni-1]
    xj = x[nj-1]; yj = y[nj-1]
    #print ' Nodes  (',xi,yi,')','  ','(',xj,yj,')'

    L[i] = np.sqrt((xi-xj)**2+(yi-yj)**2)

    cost = (xj - xi)/L[i]
    sint = (yj - yi)/L[i]

    Tloc[0,0] = cost; Tloc[0,1] = sint
    Tloc[1,0] =-sint; Tloc[1,1] = cost
    Tloc[2,2] = 1.0
    Tloc[3,3] = cost; Tloc[3,4] = sint
    Tloc[4,3] =-sint; Tloc[4,4] = cost
    Tloc[5,5] = 1.0
    Tlocinv = np.linalg.inv(Tloc)

    Kloc[0,0] = A[i]*E[i]/L[i]; Kloc[0,3] = -A[i]*E[i]/L[i]
    Kloc[1,1] = 12*E[i]*I[i]/L[i]**3; Kloc[1,2] = 6*E[i]*I[i]/L[i]**2; Kloc[1,4] = -12*E[i]*I[i]/L[i]**3; Kloc[1,5] = 6*E[i]*I[i]/L[i]**2;
    Kloc[2,1] =  6*E[i]*I[i]/L[i]**2; Kloc[2,2] = 4*E[i]*I[i]/L[i]**1; Kloc[2,4] =  -6*E[i]*I[i]/L[i]**2; Kloc[2,5] = 2*E[i]*I[i]/L[i]**1;
    Kloc[3,0] =-A[i]*E[i]/L[i]; Kloc[3,3] =  A[i]*E[i]/L[i]
    Kloc[4,1] =-12*E[i]*I[i]/L[i]**3; Kloc[4,2] =-6*E[i]*I[i]/L[i]**2; Kloc[4,4] =  12*E[i]*I[i]/L[i]**3; Kloc[4,5] =-6*E[i]*I[i]/L[i]**2;
    Kloc[5,1] =  6*E[i]*I[i]/L[i]**2; Kloc[5,2] = 2*E[i]*I[i]/L[i]**1; Kloc[5,4] =  -6*E[i]*I[i]/L[i]**2; Kloc[5,5] = 4*E[i]*I[i]/L[i]**1;

    # Reutilizando Kloc
    KK = np.dot(Tlocinv,np.dot(Kloc,Tloc))

    idx[:] = [ni*3-2, ni*3, nj*3-2, nj*3]

    ubar[0:3] = u[idx[0]-1:idx[1]]
    ubar[3:6] = u[idx[2]-1:idx[3]]

    #print aux1
    #print aux2
    IntForce = np.dot(KK,ubar)
    AxialForce = np.sqrt(IntForce[0]**2+IntForce[1]**2)

    ForceSign = np.arctan2(IntForce[1],IntForce[0])
    ThetaSign = np.arctan2((yi-yj),(xi-xj))


    if ForceSign==0.0:
        #print (xi-xj)/np.abs(xi-xj)
        AxialForce = AxialForce * (xi-xj)/np.abs(xi-xj)
    else:

        ForceSign = ForceSign/np.abs(ForceSign)
        ThetaSign = ThetaSign/np.abs(ThetaSign)
        AxialForce = AxialForce * ForceSign * ThetaSign


#    print '{:2d}   {:+.4E}   {:+.4E}   {:+.4E}   {:+.4E}   {:+.4E}'.format(i+1,IntForce[0]/1e3,IntForce[1]/1e3,IntForce[2]/1e3,IntForce[3]/1e3,AxialForce/1e3
    print ('{:2d}      {:+.4E}            {:.4E}        {:.4E}            '.format(i+1,AxialForce/1e3,np.abs(AxialForce)/(1e6*A[i]),Max_e*E[i]/1e9))



fig = plt.figure(1)
plt.clf()
xmin = np.min(x)
xmax = np.max(x)
ymin = np.min(y)
ymax = np.max(y)

umin = np.min(u)
umax = np.max(u)


xbar = np.zeros(2)
ybar = np.zeros(2)
plt.subplot(1,1,1)

# Plot initial state
plt.plot(x,y,'*')
for i in range(nElem):
    ni = Connectivity[i,0]
    nj = Connectivity[i,1]
    xbar[0] = x[ni-1]; ybar[0] = y[ni-1]
    xbar[1] = x[nj-1]; ybar[1] = y[nj-1]
#    print i, ni, nj, xbar, ybar
    plt.plot(xbar,ybar,'b--')


# Plot final state
#for i in range(nNode):
#    X[i] = x[i]+u[i*3]
#    Y[i] = y[i]+u[i*3+1]

print ('\n')
for i in range(nElem):
    ni = Connectivity[i,0]
    nj = Connectivity[i,1]
    xbar[0] = x[ni-1]+u[(ni-1)*3]; ybar[0] = y[ni-1]+u[(ni-1)*3+1]
    xbar[1] = x[nj-1]+u[(nj-1)*3]; ybar[1] = y[nj-1]+u[(nj-1)*3+1]

    # print final state bars
    plt.plot(xbar,ybar,'r--')

    # print final state beams
#    L = np.sqrt((xbar[0]-xbar[1])**2+(ybar[0]-ybar[1])**2)
    L = np.sqrt((x[ni-1]-x[nj-1])**2+(y[ni-1]-y[nj-1])**2)
    cost = (x[nj-1] - x[ni-1])/L
    sint = (y[nj-1] - y[ni-1])/L

    idx[:] = [ni*3-2, ni*3, nj*3-2, nj*3]

    ubar[0:3] = u[idx[0]-1:idx[1]]
    ubar[3:6] = u[idx[2]-1:idx[3]]

    #print xbar
    #print ybar

    xdef,ydef,maxv = Deformation(x[ni-1],y[ni-1],L,cost,sint,ubar)
    print (' Element {:d} has max v = {:+0.2e}'.format(i+1,maxv))
    plt.plot(xdef,ydef,'g--',linewidth=3)

#plt.xlim([xmin-abs(umin), xmax+abs(umax)])
#plt.ylim([ymin-abs(umin), ymax+abs(umax)])

#plt.axis('equal')
x1,x2,y1,y2 = plt.axis()

#print plt.axis()
#print xmin-abs(umin), xmax+abs(umax), ymin-abs(umin), ymax+abs(umax)
plt.show()
#print ''
#print ' Eigenvlues of symetric matrix'
#import sympy as sp

#K, k1, k2 = sp.symbols('K k1 k2')

#K = sp.Matrix([[k1, -k1, 0],[-k1 , k1+k2, -k2],[0, -k2, k2]])
#k1 = k[0]
#k2 = k[1]

#sp.pprint(K)

#eig = K.eigenvects()

#sp.pprint(eig)