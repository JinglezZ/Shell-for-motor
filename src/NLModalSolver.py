import numpy as np

class NewtonType:
  def Classical(self, x0, f, df, tol=1.e-6, maxIter=100, epsilon=1.e-10):
    """
    Classical Newton iterative method to find the root of a complex-type function

    Inputs:
    -------
    x0      : the initial guess
    f       : the target function
    df      : the derivative of the target function
    tol     : tolerance
    maxIter : maximum iteration step
    epsilon : threshold value of the derivative
    """
    for i in range(maxIter):
      y = f(x0)
      dy = df(x0)

      if np.abs(dy) < epsilon :
        break

      x1 = x0 - y/dy

      if np.abs(x1-x0) <= tol :
        return x1

      x0 = x1

    print('Warning: result may not converge')
    
    return x0


  def Secant(self, x1, ratio, f, tol=1.e-6, maxIter=100, epsilon=1.e-10):
    """
    Newton's secant iterative method to find the root of a complex-type function

    Inputs:
    -------
    x1      : the initial guess
    ratio   : ratio to define x1 (x1/x0)
    f       : the target function
    tol     : tolerance
    maxIter : maximum iteration step
    epsilon : threshold value of the derivative
    """
    
    x0 = x1 * ratio
    for i in range(maxIter):
      y = f(x1)
      dy = (f(x1) - f(x0)) / (x1 - x0)

      if np.abs(dy) < epsilon :
        break

      x2 = x1 - y/dy

      if np.abs(x2-x1) <= tol :
        return x2

      x0 = x1
      x1 = x2

    print('Warning: result may not converge')

    return x1