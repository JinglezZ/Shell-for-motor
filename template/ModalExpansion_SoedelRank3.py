import src.SoedelShell as sds
import src.DonnellShell as dns
import src.CylinderMesh as CM
import numpy as np

# define parameters
#----------------------------------
# besic geometry
length = 200.e-3
radius = 100.e-3

# load info
load_f = 7230
load_k = 1
load_P = 1000000

shellInfo = {'a'  : radius,
             'L'  : length,
             'h'  : 2.e-3,
             'E'  : 2.06e11,
             'rho': 7.85e3,
             'mu' : 0.3}

meshInfo = {'length'    : length,
            'radius'    : radius,
            'lDiv'      : 100,
            'thetaDiv'  : 400}

modalExpanInfo = {'f' : load_f,
                  'k' : load_k,
                  'P' : load_P,
                  'method' : 1}

# modal order
m = 50; n = 50

# mode shape kind (0, 1, 2 for Rank3)
# kind = 2
#----------------------------------

shellRank3 = sds.Rank3(**shellInfo)

shellRank3.setOrder(m, n)

shellRank3.genModalGroup()

eta = shellRank3.modalExpan_CatI(**modalExpanInfo)

x=length * 0.5
ur = 0.
for m_idx in range(m) : 
  ur += eta[m_idx, 2] * np.cos(load_k*0.) * np.sin(m_idx*np.pi*x/length)

print('ur=', ur)

# plot modal shape
mesh = CM.Hollow_CylinderMesh2D(**meshInfo)
mesh.fileName('modalExpansion')
mesh.build()

disp_C = []; disp_P = []
for coord in mesh.Node:
  theta = np.arctan2(coord[1], coord[0])
  z = coord[2]
  u1, u2, u3 = 0., 0., 0.
  for m_idx in range(m) : 
    u1 += eta[m_idx, 0] * np.cos(load_k*theta) * np.cos(m_idx*np.pi*z/length)
    u2 += eta[m_idx, 1] * np.sin(load_k*theta) * np.sin(m_idx*np.pi*z/length)
    u3 += eta[m_idx, 2] * np.cos(load_k*theta) * np.sin(m_idx*np.pi*z/length)
  disp_P.append([u1, u2, u3])
  disp_C.append([u3*np.cos(theta)-u2*np.sin(theta), u3*np.sin(theta)+u2*np.cos(theta), u1])
mesh.nodeDataAssign(np.array(disp_C))
mesh.appdenDeform(disp_C)

mesh.writeMesh()

