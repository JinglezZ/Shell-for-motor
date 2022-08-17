import src.SoedelShell as sds
import src.DonnellShell as dns
import src.CylinderMesh as CM
import numpy as np

# define parameters
#----------------------------------
# besic geometry
length = 100.e-3
radius = 200.e-3

shellInfo = {'a'  : radius,
             'L'  : length,
             'h'  : 10.e-3,
             'E'  : 1.3e9,
             'rho': 7.85e3,
             'mu' : 0.3}

meshInfo = {'length'    : length,
            'radius'    : radius,
            'lDiv'      : 50,
            'thetaDiv'  : 200}

# modal order
m = 1; n = 2

# mode shape kind (0, 1, 2 for Rank3)
kind = 0
#----------------------------------

shellRank3 = sds.Rank3(**shellInfo)
shellRank5 = sds.Rank5(**shellInfo)
shellRank1 = dns.Rank1(**shellInfo)

shellRank3.setOrder(m, n)
shellRank5.setOrder(m, n)
shellRank1.setOrder(m, n)

eigvals, eigvecs = shellRank3.getModal()
print(eigvals)

eigvals, eigvecs = shellRank5.getModal()
print(eigvals)

print(shellRank1.getModal())