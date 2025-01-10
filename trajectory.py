# Trajectory

import numpy as np
from scipy import linalg

def p2pTrjTrap1DOF(t,T,Pos,c):

    n = T.size - 1 # n = number of trajectories

    for j in range(n,0,-1):
        if t <= T[:,j]:
            i = j-1

    qi = Pos[i]
    qf = Pos[i+1]
    ti = t-T[:,i]
    tf = T[:,i+1]-T[:,i]
    
    D = qf - qi
    signD = np.sign(D)

    ta = tf/(2+c)
    kv = abs(D)/((c+1)*ta)
    ka = abs(D)/((c+1)*ta**2)
    
    if ti <= ta:
        q_pos = qi + 0.5*ti**2*ka*signD
    elif ti <= tf - ta:
        q_pos = qi + (ti-0.5*ta)*kv*signD
    else:
        q_pos = qf - 0.5*(tf-ti)**2*ka*signD

    return q_pos

# def minimumJerkConstrCoefficient1DOF(T,Pos,Vel,Acc):

#     n = T.size - 1 # n = number of trajectories 

#     # Waypoints' Matrix
#     R = np.zeros((6*n,6*n)) # Start with zero matrix for R, joint

#     # Positions of all waypoints
#     for i in range(n):
#         # Position
#         R[2*i+0,6*i-6:6*i] = np.array([pow(T[i],5),pow(T[i],4),pow(T[i],3),pow(T[i],2),T[i],1])
#         R[2*i+1,6*i-6:6*i] = np.array([pow(T[i+1],5),pow(T[i+1],4),pow(T[i+1],3),pow(T[i+1],2),T[i+1],1])
#         # Velocity
#         R[2*n+2*i+0,6*i-6:6*i] = np.array([5*pow(T[i],4),4*pow(T[i],3),3*pow(T[i],2),2*T[i],1,0])
#         R[2*n+2*i+1,6*i-6:6*i] = np.array([5*pow(T[i+1],4),4*pow(T[i+1],3),3*pow(T[i+1],2),2*T[i+1],1,0])
#         # Accelaration
#         R[4*n+2*i+0,6*i-6:6*i] = np.array([20*pow(T[i],3),12*pow(T[i],2),6*T[i],2,0,0])
#         R[4*n+2*i+1,6*i-6:6*i] = np.array([20*pow(T[i+1],3),12*pow(T[i+1],2),6*T[i+1],2,0,0])

#     # Boundary Conditions Matrix
#     BC = np.zeros((6*n,1))
#     BC(1,1) = Pos(:,1);      	# Position of the first waypoint
#     BC(1+2*n:1+2*n) = Vel(:,1);	# Velocity of the first waypoint
#     BC(1+4*n:1+4*n) = Acc(:,1);	# Acceleration of the first waypoint
#     if n > 1
#         PosInter = zeros(2*(n-1),1);
#         VelInter = zeros(2*(n-1),1);
#         AccInter = zeros(2*(n-1),1);
#     end
#     for i = 2:n
#         PosInter(2*i-3:2*i-2,:) = [Pos(:,i)';Pos(:,i)'];
#         VelInter(2*i-3:2*i-2,:) = [Vel(:,i)';Vel(:,i)'];
#         AccInter(2*i-3:2*i-2,:) = [Acc(:,i)';Acc(:,i)'];
#     end
#     for i = 1:2*(n-1)
#         # Position
#         BC(1+i:i+1) = PosInter(i,:);
#         # Velocity
#         BC(1+2*n+i:2*n+i+1) = VelInter(i,:);
#         # Acceelration
#         BC(1+4*n+i:4*n+i+1) = AccInter(i,:);
#     end
#     BC(1+2*n-1:2*n+0) = Pos(:,n+1); % Position of the final waypoint
#     BC(1+2*n+1:2*n+2) = Vel(:,n+1); % Velocity of the final waypoint
#     BC(1+2*n+3:2*n+4) = Acc(:,n+1); % Acceleration of the final waypoint

#     # Coefficient  Vector
#     Cj = linalg.solve(R, BC) # Cj = R\BC;
    
#     return Cj
    
def minimumJerkCoefficient1DOF(T,Pos,Vel,Acc):
    
    n = T.size - 1 # n = number of trajectories 

    # Waypoints' Matrix
    R = np.zeros((6*n,6*n)) # Start with zero matrix for R, joint

    # Positions of all waypoints
    for i in range(n):
        R[0+2*i,6*i:6+6*i] = np.array([pow(T[i],5),pow(T[i],4),pow(T[i],3),pow(T[i],2),T[i],1])
        R[1+2*i,6*i:6+6*i] = np.array([pow(T[i+1],5),pow(T[i+1],4),pow(T[i+1],3),pow(T[i+1],2),T[i+1],1])
    # Velocity boundary conditions (inital and final waypoints)        
    R[2*n,0:6]              = np.array([5*pow(T[0],4),4*pow(T[0],3),3*pow(T[0],2),2*T[0],1,0])
    R[1+2*n,6*(n-1):1+6*n]  = np.array([5*pow(T[n],4),4*pow(T[n],3),3*pow(T[n],2),2*T[n],1,0])
    # Equal Accelaration boundary conditions (initial and final waypoints)
    R[2+2*n,0:6]            = np.array([20*pow(T[0],3),12*pow(T[0],2),6*T[0],2,0,0])
    R[3+2*n,6*(n-1):1+6*n]  = np.array([20*pow(T[n],3),12*pow(T[n],2),6*T[n],2,0,0])
    #Equal velocity, accelaration , jerk, and snap at intermideate waypoints
    for i in range(n-1):
        R[i+0*(n-1)+4+2*n,6*i:6*(i+2)] = np.array([ 5*pow(T[i+1],4), 4*pow(T[i+1],3),3*pow(T[i+1],2),2*T[i+1],1,0, -5*pow(T[i+1],4), -4*pow(T[i+1],3),-3*pow(T[i+1],2),-2*T[i+1],-1,0]) # Equal velocity at intermediate waypoints
        R[i+1*(n-1)+4+2*n,6*i:6*(i+2)] = np.array([20*pow(T[i+1],3),12*pow(T[i+1],2),6*T[i+1]       ,2       ,0,0,-20*pow(T[i+1],3),-12*pow(T[i+1],2),-6*T[i+1]       ,-2       , 0,0]) # Equal acceleration at intermediate waypoints
        R[i+2*(n-1)+4+2*n,6*i:6*(i+2)] = np.array([60*pow(T[i+1],2),24*T[i+1]       ,6              ,0       ,0,0,-60*pow(T[i+1],2),-24*T[i+1]       ,-6              ,0        , 0,0]) # Equal jerk at intermediate waypoints
        R[i+3*(n-1)+4+2*n,6*i:6*(i+2)] = np.array([120*T[i+1]      ,24              ,0              ,0       ,0,0,-120*T[i+1]      ,-24              ,0               ,0        , 0,0]) # Equal snap at intermediate waypoints

    # Boundary Conditions Matrix
    BC = np.zeros((6*n,1))
    BC[0,0] = Pos[0]    # Position of the first waypoint
    if n > 0:
        PosInter = np.zeros((2*(n-1),1))
    for i in range(1,n):
        PosInter[2*i-2] = Pos[i]
        PosInter[2*i-1] = Pos[i]
    for i in range(2*(n-1)):
        BC[i+1] = PosInter[i]
    BC[2*(n-1)+1] = Pos[n]  # Position of the final waypoint
    BC[2*(n-1)+2] = Vel[0]  # initial velocity
    BC[2*(n-1)+3] = Vel[1]  # final velocity
    BC[2*(n-1)+4] = Acc[0]  # initial acceleration
    BC[2*(n-1)+5] = Acc[1]  # final acceleration
    
    # Coefficient  Vector
    Cj = linalg.solve(R, BC) # Cj = R\BC;
    
    return Cj

def minimumJerkPolynomial1DOF(t,T,Cj):

    n = T.size - 1 # n = number of trajectories
    
    for j in range(n,0,-1):
        if t <= T[j]:
            i = j-1
    
    q_pos = Cj[6*i]*pow(t,5) + Cj[1+6*i]*pow(t,4) + Cj[2+6*i]*pow(t,3) + Cj[3+6*i]*pow(t,2) + Cj[4+6*i]*t + Cj[5+6*i]

    return q_pos
