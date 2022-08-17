import numpy as np

class Rank1:
  """
  Rank 1 type for Soedel shell (DOFs : U3)

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
      compute the model frequency
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


  def __str__(self) -> str:
    return f"""
    Donnell Shell Rank 1 :
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


  def getModal(self):
    pi=np.pi

    coef = 1./self.Radius * np.sqrt(self.Modulus/self.Density)
    term1 = (self._m*pi*self.Radius/self.Length)**4/((self._m*pi*self.Radius/self.Length)**2+self._n**2)**2
    term2 = (self.Thick/self.Radius)**2/(12.*(1.-self.Possion**2))*((self._m*pi*self.Radius/self.Length)**2+self._n**2)**2
    omega = coef * np.sqrt(term1 + term2)

    return omega
