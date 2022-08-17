import numpy as np
from scipy.linalg import eigh

class Rank3:
  """
  Rank 3 type for Soedel shell (DOFs : U1, U2, U3)

  Attributes
  ----------
  Radius : radius
  Length : length
  Thick  : thickness
  Modulus: Young's modulus 
  Density: density
  Possion: Poisson's ratio
  _m     : axial order
  _n     : circumferential order

  Methods
  -------
  setOrder(m, n) : 
      set the axial circumferential orders
  getModal( ) : 
      compute the model frequency and shape
      return 1-d array with frequency and 2-d array with modal shape
  """

  _m = 1
  _n = 1

  def __init__(self, a=1.0, L=1.0, h=0.01, E=1.1e11, rho=7.85e3, mu=0.3 ) -> None:
    self.Radius = a
    self.Length = L
    self.Thick  = h
    self.Modulus = E
    self.Density = rho
    self.Possion = mu
    self._K = E*h/(1.-mu**2)
    self._D = E*h**3/(12.*(1.-mu**2))


  def __str__(self) -> str:
    return f"""
    Soedel Shell Rank 3 :
      Radius  : {self.Radius}
      Length  : {self.Length}
      Thick   : {self.Thick}
      Modulus : {self.Modulus}
      Density : {self.Density}
      Possion : {self.Possion}
    """

  def setOrder(self, m, n):
    """
    Set m, n order
    """
    self._m = m
    self._n = n


  def __genKMatrix(self):
    pi = np.pi
    K_Mtrix = np.zeros(shape=(3, 3), dtype=np.float64)

    K_Mtrix[0, 0] = self._K*( (self._m*pi/self.Length)**2 + (1.-self.Possion)/2.*(self._n/self.Radius)**2 )
    K_Mtrix[0, 1] = -self._K*(1+self.Possion)/2.*(self._m*pi/self.Length)*(self._n/self.Radius)
    K_Mtrix[0, 2] = -(self.Possion*self._K/self.Radius)*(self._m*pi/self.Length)
    K_Mtrix[1, 1] = (self._K+self._D/self.Radius**2)*( (1.-self.Possion)/2.*(self._m*pi/self.Length)**2 + (self._n/self.Radius)**2 )
    K_Mtrix[1, 2] = (self._K/self.Radius)*(self._n/self.Radius)+(self._D/self.Radius)*(self._n/self.Radius)*((self._m*pi/self.Length)**2+(self._n/self.Radius)**2)
    K_Mtrix[2, 2] = self._D*((self._m*pi/self.Length)**2+(self._n/self.Radius)**2)**2+self._K/self.Radius**2
    K_Mtrix[1, 0] = K_Mtrix[0, 1]
    K_Mtrix[2, 0] = K_Mtrix[0, 2]
    K_Mtrix[2, 1] = K_Mtrix[1, 2]

    return K_Mtrix


  def __genMMatrix(self):
    M_Mtrix = np.zeros(shape=(3, 3), dtype=np.float64)

    M_Mtrix[0, 0] = self.Density * self.Thick
    M_Mtrix[1, 1] = self.Density * self.Thick
    M_Mtrix[2, 2] = self.Density * self.Thick

    return M_Mtrix


  def getModal(self):
    eigvals, eigvecs = eigh(self.__genKMatrix(), self.__genMMatrix(), eigvals_only=False)
    
    eigvals = np.sqrt(eigvals)

    eigvecs = eigvecs.transpose()
    for vec in eigvecs:
      vec /= np.abs(vec).max()  

    return eigvals, eigvecs



class Rank5(Rank3):
  """
  Rank 5 type for Soedel shell (DOFs : U1, U2, U3, BetaX, BetaTheta)

  Attributes
  ----------
  Radius : radius
  Length : length
  Thick  : thickness
  Modulus: Young's modulus 
  Density: density
  Possion: Poisson's ratio
  _m     : axial order
  _n     : circumferential order
  _shearCoef : shear coefficent

  Methods
  -------
  setOrder(m, n) : 
      set the axial circumferential orders
  sCoef(float) :
      set shear coefficient (default value is 2./3.)
  getModal( ) : 
      compute the model frequency and shape
      return 1-d array with frequency and 2-d array with modal shape
  """
  
  _shearCoef = 2./3.

  def __init__(self, a=1, L=1, h=0.01, E=110000000000, rho=7850, mu=0.3) -> None:
    super().__init__(a, L, h, E, rho, mu)
    self._G = E/(2.*(1.+mu))

  
  def __str__(self) -> str:
    return f"""
    Soedel Shell Rank 5 :
      Radius  : {self.Radius}
      Length  : {self.Length}
      Thick   : {self.Thick}
      Modulus : {self.Modulus}
      Density : {self.Density}
      Possion : {self.Possion}
      Shear coefficient : {self._shearCoef}
    """


  def __genKMatrix(self):
    pi = np.pi
    K_Mtrix = np.zeros(shape=(5, 5), dtype=np.float64)

    K_Mtrix[0, 0] = self._K*( (self._m*pi/self.Length)**2 + (1.-self.Possion)/2.*(self._n/self.Radius)**2 )
    K_Mtrix[0, 1] = -self._K*(1+self.Possion)/2.*(self._m*pi/self.Length)*(self._n/self.Radius)
    K_Mtrix[0, 2] = -(self.Possion*self._K/self.Radius)*(self._m*pi/self.Length)
    K_Mtrix[1, 1] = self._K*( (1.-self.Possion)/2.*(self._m*pi/self.Length)**2 + (self._n/self.Radius)**2 ) + self._shearCoef*(self._G*self.Thick/self.Radius**2)
    K_Mtrix[1, 2] = (self._K/self.Radius)*(self._n/self.Radius)+(self._n/self.Radius)*(self._shearCoef*(self._G*self.Thick/self.Radius))
    K_Mtrix[1, 4] = -self._shearCoef*(self._G*self.Thick/self.Radius)
    K_Mtrix[2, 2] = self._shearCoef*self._G*self.Thick*((self._m*pi/self.Length)**2+(self._n/self.Radius)**2)+self._K/self.Radius**2
    K_Mtrix[2, 3] = self._shearCoef*self._G*self.Thick*(self._m*pi/self.Length)
    K_Mtrix[2, 4] = -self._shearCoef*self._G*self.Thick*(self._n/self.Radius)
    K_Mtrix[3, 3] = self._shearCoef*self._G*self.Thick+self._D*( (1.-self.Possion)/2.*(self._n/self.Radius)**2+(self._m*pi/self.Length)**2 )
    K_Mtrix[3, 4] = -self._D*((1.+self.Possion)/2.)*(self._n/self.Radius)*(self._m*pi/self.Length)
    K_Mtrix[4, 4] = self._shearCoef*self._G*self.Thick+self._D*( (1.-self.Possion)/2.*(self._m*pi/self.Length)**2+(self._n/self.Radius)**2 )

    K_Mtrix[1, 0] = K_Mtrix[0, 1]
    K_Mtrix[2, 0] = K_Mtrix[0, 2]
    K_Mtrix[2, 1] = K_Mtrix[1, 2]
    K_Mtrix[3, 2] = K_Mtrix[2, 3]
    K_Mtrix[4, 1] = K_Mtrix[1, 4]
    K_Mtrix[4, 2] = K_Mtrix[2, 4]
    K_Mtrix[4, 3] = K_Mtrix[3, 4]

    return K_Mtrix


  def __genMMatrix(self):
    M_Mtrix = np.zeros(shape=(5, 5), dtype=np.float64)

    M_Mtrix[0, 0] = self.Density * self.Thick
    M_Mtrix[1, 1] = self.Density * self.Thick
    M_Mtrix[2, 2] = self.Density * self.Thick
    M_Mtrix[3, 3] = self.Density * self.Thick**3 / 12.
    M_Mtrix[4, 4] = self.Density * self.Thick**3 / 12.

    return M_Mtrix


  def sCoef(self, coef):
    self._shearCoef = coef


  def getModal(self):
    eigvals, eigvecs = eigh(self.__genKMatrix(), self.__genMMatrix(), eigvals_only=False)
    
    eigvals = np.sqrt(eigvals)

    eigvecs = eigvecs.transpose()
    for vec in eigvecs:
      vec /= np.abs(vec).max()  

    return eigvals, eigvecs

