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
  _m     : max axial order for modal expansion
  _n     : max circumferential order for modal expansion
  _AF_Velo : Acoustic fluid sound velocity
  _AF_Dens : Acoustic fluid density

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
  _AF_Velo = 0.0
  _AF_Dens = 0.0
  freqGroup = []
  shapGroup = []

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
    Set _m, _n order for modal expansion
    """
    self._m = m
    self._n = n


  def setAcousticFluidInfo(self, cf, rhof):
    """
    Set sound velocity and density of acoustic fluid
    """
    self._AF_Velo = cf
    self._AF_Dens = rhof


  def __genKMatrix(self, m, n):
    pi = np.pi
    K_Mtrix = np.zeros(shape=(3, 3), dtype=np.float64)

    K_Mtrix[0, 0] = self._K*( (m*pi/self.Length)**2 + (1.-self.Possion)/2.*(n/self.Radius)**2 )
    K_Mtrix[0, 1] = -self._K*(1+self.Possion)/2.*(m*pi/self.Length)*(n/self.Radius)
    K_Mtrix[0, 2] = -(self.Possion*self._K/self.Radius)*(m*pi/self.Length)
    K_Mtrix[1, 1] = (self._K+self._D/self.Radius**2)*( (1.-self.Possion)/2.*(m*pi/self.Length)**2 + (n/self.Radius)**2 )
    K_Mtrix[1, 2] = (self._K/self.Radius)*(n/self.Radius)+(self._D/self.Radius)*(n/self.Radius)*((m*pi/self.Length)**2+(n/self.Radius)**2)
    K_Mtrix[2, 2] = self._D*((m*pi/self.Length)**2+(n/self.Radius)**2)**2+self._K/self.Radius**2
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
    omega_0 = self.getModal(m, n)[0][0]
    # K/M-matrices in uncoupled condition
    kMat = self.__genKMatrix(m, n)
    mMat = self.__genMMatrix()

    dHm = lambda n, z: n*ssf.hankel1(n, z)/z-ssf.hankel1(n+1,z)

    def AMatrix(omega):
      # A-matrix in uncoupled condition
      AMat = omega**2 * mMat-kMat
      AMat = AMat.astype(complex)
      # fluid load term FL
      k_n  = m*np.pi/self.Length + 0j
      k_nr = np.sqrt(omega**2/self._AF_Velo**2 - k_n**2)
      FL = -omega**2*self._AF_Dens*ssf.hankel1(n,k_nr*self.Radius)/(k_nr * dHm(n,k_nr*self.Radius)) 
      # A-matrix in coupled condition
      AMat[2, 2] += FL

      return AMat

    def detAMatrix(omega):
      AMat = AMatrix(omega)
      detA = np.linalg.det(AMat)

      return detA

    def dDetAMatrix(omega):
      AMat = AMatrix(omega)
      k_n  = m*np.pi/self.Length + 0j
      k_nr = np.sqrt(omega**2/self._AF_Velo**2 - k_n**2)

      dA11 = dA22 = dA33 = 2.*self.Density*self.Thick*omega
      #  dFL
      dFL_Part1 = -self._AF_Dens*ssf.hankel1(n,k_nr*self.Radius)/dHm(n,k_nr*self.Radius)*omega/k_nr*(1.-k_n**2/k_nr**2)
      dFL_Part2 = n*omega/(self._AF_Velo**2*k_nr**2)*( 1./(self.Radius*k_nr)*ssf.hankel1(n,k_nr*self.Radius)**2/dHm(n,k_nr*self.Radius)**2 - ssf.hankel1(n,k_nr*self.Radius)/dHm(n,k_nr*self.Radius) )
      dFL_Part3 = self.Radius*omega/(self._AF_Velo**2*k_nr)*( ssf.hankel1(n,k_nr*self.Radius)*dHm(n+1,k_nr*self.Radius)/dHm(n,k_nr*self.Radius)**2+1. )
      dFL = dFL_Part1 - self._AF_Dens*omega**2/k_nr*(dFL_Part2+dFL_Part3)
      dA33 += dFL
      #  integration
      ddetA = (AMat[1,1]*AMat[2,2]-AMat[1,2]*AMat[2,1])*dA11 + (AMat[0,0]*AMat[2,2]-AMat[0,2]*AMat[2,0])*dA22 + (AMat[0,0]*AMat[1,1]-AMat[0,1]*AMat[1,0])*dA33

      return ddetA

    omega = NSolver.Classical(omega_0, detAMatrix, dDetAMatrix, tol=tolerance, maxIter=maxIterStep)
    #omega = NSolver.Secant(omega_0, 0.99, detAMatrix, tol=tolerance, maxIter=maxIterStep)

    return omega


  def getModal(self, m=None, n=None):
    """
    Compute the natural frequencies and modal shapes for (m,n) order

    Inputs:
    -------
    m : axial order
    n : circumferential order 
     *if (m, n) not provided, they will be set to be (_m, _n) by default

    Return values:
    --------------
    eigvals[0:3] : containing 3 natural frequencies with increasing order (unit rad/s)
    eigvecs[0:3,0:3] : containing modal shapes ( eigvecs[0/1/2][:] ) for three types of vibration mode
    """
    if m is None:
      m = self._m
    if n is None:
      n = self._n

    eigvals, eigvecs = eigh(self.__genKMatrix(m, n), self.__genMMatrix(), eigvals_only=False)
    
    eigvals = np.sqrt(eigvals)

    eigvecs = eigvecs.transpose()
    for vec in eigvecs:
      vec /= np.abs(vec).max()  

    return eigvals, eigvecs


  def genModalGroup(self):
    """
    Generate modal frequency&shape group for Modal Expansion up to (m,n) order
    Results are stored in freqGroup and shapGroup in the following manner:
    freqGroup[0:m, 0:n, 0:3] (unit rad/s)
    shapGroup[0:m, 0:n, 0:3, 0:3]
    """

    nFreq = np.zeros(shape=(self._m+1, self._n+1, 3), dtype=np.float64)
    nShap = np.zeros(shape=(self._m+1, self._n+1, 3, 3), dtype=np.float64)

    for m_idx in range(self._m+1):
      for n_idx in range(self._n+1):
        eigvals, eigvecs = eigh(self.__genKMatrix(m_idx, n_idx), self.__genMMatrix(), eigvals_only=False)
        eigvals = np.sqrt(eigvals)
        eigvecs = eigvecs.transpose()
        for vec in eigvecs:
          vec /= np.abs(vec).max()

        nFreq[m_idx, n_idx, :] = eigvals
        nShap[m_idx, n_idx, :, :] = eigvecs

    self.freqGroup = nFreq
    self.shapGroup = nShap

  
  def modalExpan_CatI(self, f, k, P, method=1):
    """
    Modal Expansion Catgory I :
    Compute the REAL general participation factor of each m-order mode

    Note 1 : Load is assumed to be axial uniform
    Note 2 : Damping is not considered

    Input:
    ------
    f : frequency of the load (Hz)
    k : circumferential order of the load
    P : amplitude of the load (Pa)
    method : method to compute the general participation factor
              0 : involve only the radial modes
              1 : involve all the modes (default)

    Output:
    -------
    etaGeneral[0:m, 0:3] : general participation factor at omega_m for 3 displacements,
                           has non-zero value only at odd index
    """
    
    pi = np.pi

    m = self._m+1
    n = k
    omega_k = 2.*pi*f

    modeRange = 1 if method == 0 else 3
    eta = np.zeros(shape=(m,3), dtype=np.float64)
    etaGeneral = np.zeros(shape=(m,3), dtype=np.float64)

    for m_idx in range(m) :
      if m_idx%2 == 1 :
        for modeIdx in range(modeRange) :

          omega_mn = self.freqGroup[m_idx, n, modeIdx]
          coef_1 = 1./(omega_mn**2 - omega_k**2)

          C_mn = pi*self.Radius*self.Length/2.*(self.shapGroup[m_idx, n, modeIdx, 0]**2 + self.shapGroup[m_idx, n, modeIdx, 1]**2 + self.shapGroup[m_idx, n, modeIdx, 2]**2)
          coef_2 = 2.*self.Radius*self.Length/(m_idx*self.Density*self.Thick*C_mn)

          eta[m_idx, modeIdx] = coef_1 * coef_2 * P * self.shapGroup[m_idx, n, modeIdx, 2]

        etaGeneral[m_idx, 0] = np.sum(np.multiply(eta[m_idx, :],self.shapGroup[m_idx, n, :, 0])) 
        etaGeneral[m_idx, 1] = np.sum(np.multiply(eta[m_idx, :],self.shapGroup[m_idx, n, :, 1])) 
        etaGeneral[m_idx, 2] = np.sum(np.multiply(eta[m_idx, :],self.shapGroup[m_idx, n, :, 2])) 
    
    return etaGeneral



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
  _AF_Velo = 0.0
  _AF_Dens = 0.0

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


  def __genKMatrix(self, m, n):
    pi = np.pi
    K_Mtrix = np.zeros(shape=(5, 5), dtype=np.float64)

    K_Mtrix[0, 0] = self._K*( (m*pi/self.Length)**2 + (1.-self.Possion)/2.*(n/self.Radius)**2 )
    K_Mtrix[0, 1] = -self._K*(1+self.Possion)/2.*(m*pi/self.Length)*(n/self.Radius)
    K_Mtrix[0, 2] = -(self.Possion*self._K/self.Radius)*(m*pi/self.Length)
    K_Mtrix[1, 1] = self._K*( (1.-self.Possion)/2.*(m*pi/self.Length)**2 + (n/self.Radius)**2 ) + self._shearCoef*(self._G*self.Thick/self.Radius**2)
    K_Mtrix[1, 2] = (self._K/self.Radius)*(n/self.Radius)+(n/self.Radius)*(self._shearCoef*(self._G*self.Thick/self.Radius))
    K_Mtrix[1, 4] = -self._shearCoef*(self._G*self.Thick/self.Radius)
    K_Mtrix[2, 2] = self._shearCoef*self._G*self.Thick*((m*pi/self.Length)**2+(n/self.Radius)**2)+self._K/self.Radius**2
    K_Mtrix[2, 3] = self._shearCoef*self._G*self.Thick*(m*pi/self.Length)
    K_Mtrix[2, 4] = -self._shearCoef*self._G*self.Thick*(n/self.Radius)
    K_Mtrix[3, 3] = self._shearCoef*self._G*self.Thick+self._D*( (1.-self.Possion)/2.*(n/self.Radius)**2+(m*pi/self.Length)**2 )
    K_Mtrix[3, 4] = -self._D*((1.+self.Possion)/2.)*(n/self.Radius)*(m*pi/self.Length)
    K_Mtrix[4, 4] = self._shearCoef*self._G*self.Thick+self._D*( (1.-self.Possion)/2.*(m*pi/self.Length)**2+(n/self.Radius)**2 )

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


  def getModal(self, m=None, n=None):
    """
    Compute the natural frequencies and modal shapes for (m,n) order

    Inputs:
    -------
    m : axial order
    n : circumferential order 
     *if (m, n) not provided, they will be set to be (_m, _n) by default

    Return values:
    --------------
    eigvals[0:5] : containing 5 natural frequencies with increasing order
    eigvecs[0:5,0:5] : containing modal shapes ( eigvecs[0/1/2/3/4][:] ) for three types of vibration mode
    """
    if m is None:
      m = self._m
    if n is None:
      n = self._n

    eigvals, eigvecs = eigh(self.__genKMatrix(m, n), self.__genMMatrix(), eigvals_only=False)
    
    eigvals = np.sqrt(eigvals)

    eigvecs = eigvecs.transpose()
    for vec in eigvecs:
      vec /= np.abs(vec).max()  

    return eigvals, eigvecs


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
    omega_0 = self.getModal(m, n)[0][0]
    # K/M-matrices in uncoupled condition
    kMat = self.__genKMatrix(m, n)
    mMat = self.__genMMatrix()

    dHm = lambda n, z: n*ssf.hankel1(n, z)/z-ssf.hankel1(n+1,z)

    def AMatrix(omega):
      # A-matrix in uncoupled condition
      AMat = omega**2 * mMat-kMat
      AMat = AMat.astype(complex)
      # fluid load term FL
      k_n  = m*np.pi/self.Length + 0j
      k_nr = np.sqrt(omega**2/self._AF_Velo**2 - k_n**2)
      FL = -omega**2*self._AF_Dens*ssf.hankel1(n,k_nr*self.Radius)/(k_nr * dHm(n,k_nr*self.Radius)) 
      # A-matrix in coupled condition
      AMat[2, 2] += FL

      return AMat

    def detAMatrix(omega):
      AMat = AMatrix(omega)
      detA = np.linalg.det(AMat)
      return detA

    
    omega = NSolver.Secant(omega_0, 0.99, detAMatrix, tol=tolerance, maxIter=maxIterStep)

    return omega