import src.SoedelShell as sds
import src.CylinderMesh as CM
import numpy as np

# define parameters
#----------------------------------
# besic geometry
length = 200.e-3
radius = 100.e-3

shellInfo = {'a'  : radius,
             'L'  : length,
             'h'  : 2.e-3,
             'E'  : 2.06e11,
             'rho': 7.85e3,
             'mu' : 0.3}

meshInfo = {'length'    : length,
            'radius'    : radius,
            'lDiv'      : 50,
            'thetaDiv'  : 200}

# modal order
m = 1; n = 2

# mode shape kind (0, 1, 2 for Rank3)
kind = 2
#----------------------------------

shell = sds.Rank3(**shellInfo)
shell.setOrder(m, n)

eigvals, eigvecs = shell.getModal()


# plot modal shape
mesh = CM.Hollow_CylinderMesh2D(**meshInfo)
mesh.fileName('SodelRank3')
mesh.build()
weight = 0.02

disp_C = []; disp_P = []
for coord in mesh.Node:
  theta = np.arctan2(coord[1], coord[0])
  z = coord[2]
  u1 = weight * eigvecs[kind, 0] * np.cos(m*np.pi*z/length) * np.cos(n*theta)
  u2 = weight * eigvecs[kind, 1] * np.sin(m*np.pi*z/length) * np.sin(n*theta)
  u3 = weight * eigvecs[kind, 2] * np.sin(m*np.pi*z/length) * np.cos(n*theta)
  disp_P.append([u1, u2, u3])
  disp_C.append([u3*np.cos(theta)-u2*np.sin(theta), u3*np.sin(theta)+u2*np.cos(theta), u1])
mesh.nodeDataAssign(np.array(disp_C))
mesh.appdenDeform(disp_C)

mesh.writeMesh()
