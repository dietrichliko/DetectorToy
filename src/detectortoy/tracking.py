import numpy as np
import numpy.typing as npt
from typing import Tuple

ParVector = npt.NDArray[(5,), np.float_]



class CylindricalVector:
    """

    Attributes:
        r0: reference radius
        par: Array
            0 ... Phi
            1 ... z
            2 ... theta
            3 ... beta
            4 ... kappa
    """

    r0: float
    par: ParVector
    
    def __init__(self, r0: float, par: ParVector) -> None:
        self.r0 = r0
        self.par = par

    @staticmethod
    def from_vertex(r: float, x: float, y: float, z: float ,theta: float, phi: float ,qp: float, m: float) -> "CylindricalVector":




    def planar(self) -> Tuple["PlanarVector", DerivativeMatrix]:



class ForwardVector: