import numpy as np

class Hollow_CylinderMesh2D:

  __NodeData_Assign = False
  __Output_FileName = 'Hollow_Cylinder2D_default.vtu'

  def __init__(self, length=1.0, radius=1.0, lDiv=10, thetaDiv=10) -> None:
    self.length = length
    self.radius = radius
    self.lDiv = lDiv
    self.thetaDiv = thetaDiv


  def __str__(self) -> str:
    return f"""
    Hallow Cylinder:
      Length : {self.length}
      Radius : {self.radius}
      lDiv : {self.lDiv}
      thetaDiv : {self.thetaDiv}
    """


  def build(self):

    self.Node = np.empty([self.thetaDiv * (self.lDiv+1), 3])
    self.Elmt = np.empty([self.thetaDiv * self.lDiv, 4], dtype=int)

    theta = np.linspace(0, 2.*np.pi, self.thetaDiv, endpoint=False)
    z_ = np.linspace(0, self.length, self.lDiv+1, endpoint=True)

    theta, z_ = np.meshgrid(theta, z_, indexing='ij')
    x_ = self.radius * np.cos(theta)
    y_ = self.radius * np.sin(theta)

    ind = 0
    for j in range(0, self.lDiv+1):
      for i in range(0, self.thetaDiv):
        self.Node[ind, 0] = x_[i, j]
        self.Node[ind, 1] = y_[i, j]
        self.Node[ind, 2] = z_[i, j]
        ind += 1

    ind = 0
    for j in range(0, self.lDiv):
      for i in range(0, self.thetaDiv):
        if i<self.thetaDiv-1:
          self.Elmt[ind, 0] = j*self.thetaDiv + i
          self.Elmt[ind, 1] = j*self.thetaDiv + i + 1
          self.Elmt[ind, 2] = (j+1)*self.thetaDiv + i + 1
          self.Elmt[ind, 3] = (j+1)*self.thetaDiv + i
        else:
          self.Elmt[ind, 0] = j*self.thetaDiv + i
          self.Elmt[ind, 1] = j*self.thetaDiv
          self.Elmt[ind, 2] = (j+1)*self.thetaDiv
          self.Elmt[ind, 3] = (j+1)*self.thetaDiv + i

        ind += 1


  def nodeDataAssign(self, data):
    if np.shape(data) == np.shape(self.Node):
      self.__NodeData_Assign = True
      self.nodedata = data
    else:
      raise ValueError('dimension mismatch')


  def appdenDeform(self, data):
    if np.shape(data) == np.shape(self.Node):
      self.Node += data
    else:
      raise ValueError('dimension mismatch')


  def fileName(self, name):
    self.__Output_FileName = name + '.vtu'


  def writeMesh(self):
    import src.vtuWritter as vtu
    file = open(self.__Output_FileName, 'w')
    vtu.head(file, self)
    vtu.node(file, self)
    vtu.elmt(file, self)
    if self.__NodeData_Assign :
      vtu.nddata(file, self, 'displacement')
    vtu.end(file, self)

    file.close() 
    