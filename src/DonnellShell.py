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
  _AF_Velo : Acoustic fluid sound velocity
  _AF_Dens : Acoustic fluid density

  Methods
  -------
  setOrder(m, n) : 
      set the axial circumferential orders
  getModal( ) : 
      compute the model frequency
  """

  _m = 1
  _n = 1
  _AF_Velo = 0.0
  _AF_Dens = 0.0

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


  def setAcousticFluidInfo(self, cf, rhof):
    """
    Set sound velocity and density of acoustic fluid
    """
    self._AF_Velo = cf
    self._AF_Dens = rhof


  def getModal(self, m=None, n=None):
    if m is None:
      m = self._m
    if n is None:
      n = self._n

    pi=np.pi

    coef = 1./self.Radius * np.sqrt(self.Modulus/self.Density)
    term1 = (m*pi*self.Radius/self.Length)**4/((m*pi*self.Radius/self.Length)**2+n**2)**2
    term2 = (self.Thick/self.Radius)**2/(12.*(1.-self.Possion**2))*((m*pi*self.Radius/self.Length)**2+n**2)**2
    omega = coef * np.sqrt(term1 + term2)

    return omega


  def getFSIFreq(self, m=None, n=None, tol=1.e-6, MaxIter=100):
    """
    Compute the natural frequencies under FSI condition for (m,n) order

    Inputs:
    -------
    m : axial order
    n : circumferential order 
     *if (m, n) not provided, they will be set to be (_m, _n) by default
    tol     : tolerance
    MaxIter : maximum iteration step

    Return values:
    --------------
    eigvals : natural frequency under FSI condition ONLY for radial mode (unit rad/s)
    """
    import scipy.special as ssf
    import src.NLModalSolver as solver

    if m is None:
      m = self._m
    if n is None:
      n = self._n

    tolerance = tol
    maxIterStep = MaxIter

    NSolver = solver.NewtonType()

    # circular frequency in uncoupled condition
    omega_0 = self.getModal(m, n)

    dHm = lambda n, z: n*ssf.hankel1(n, z)/z-ssf.hankel1(n+1,z)

    def f_ori(omega):
      term1 = (self.Modulus*self.Thick**3/(12.*(1.-self.Possion**2)))*((n/self.Radius)**2+(m*np.pi/self.Length)**2)**4
      term2 = self.Modulus*self.Thick/self.Radius**2*(m*np.pi/self.Length)**4
      # fluid load term FL
      k_n  = m*np.pi/self.Length + 0j
      k_nr = np.sqrt(omega**2/self._AF_Velo**2 - k_n**2)
      FL = -omega**2*self._AF_Dens*ssf.hankel1(n,k_nr*self.Radius)/(k_nr * dHm(n,k_nr*self.Radius)) 
      term3 = -(self.Density*self.Thick*omega**2+FL)*((n/self.Radius)**2+(m*np.pi/self.Length)**2)**2

      return term1+term2+term3


    def f_pri(omega):
      k_n  = m*np.pi/self.Length + 0j
      k_nr = np.sqrt(omega**2/self._AF_Velo**2 - k_n**2)
      #  dFL
      dFL_Part1 = -self._AF_Dens*ssf.hankel1(n,k_nr*self.Radius)/dHm(n,k_nr*self.Radius)*omega/k_nr*(1.-k_n**2/k_nr**2)
      dFL_Part2 = n*omega/(self._AF_Velo**2*k_nr**2)*( 1./(self.Radius*k_nr)*ssf.hankel1(n,k_nr*self.Radius)**2/dHm(n,k_nr*self.Radius)**2 - ssf.hankel1(n,k_nr*self.Radius)/dHm(n,k_nr*self.Radius) )
      dFL_Part3 = self.Radius*omega/(self._AF_Velo**2*k_nr)*( ssf.hankel1(n,k_nr*self.Radius)*dHm(n+1,k_nr*self.Radius)/dHm(n,k_nr*self.Radius)**2+1. )
      dFL = dFL_Part1 - self._AF_Dens*omega**2/k_nr*(dFL_Part2+dFL_Part3)
      
      fprime = -((n/self.Radius)**2+(m*np.pi/self.Length)**2)**2*(2.*self.Density*self.Thick*omega+dFL)

      return fprime

    omega = NSolver.Classical(omega_0, f_ori, f_pri, tol=tolerance, maxIter=maxIterStep)
    #omega = NSolver.Secant(omega_0, 0.99, f_ori, tol=tolerance, maxIter=maxIterStep)

    return omega