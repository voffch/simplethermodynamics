"""
This file defines some commonly used functions for describing (approximating)
various thermodynamic functions.
"""

import sympy
import numpy as np
import scipy.special
import random
from .core import R

symbolic_T = sympy.Symbol('T')

#===========#
# FUNCTIONS #
#===========#
# The following functions are used for approximating
# whole thermodynamic functions or their pieces

def cv_einstein(alpha, theta, T = symbolic_T):
    """Heat capacity; Einstein function; J/mol/K

    3 * alpha * R * x**2 * exp(x) / (exp(x) - 1)**2,
    where x = theta / T

    Args:
        alpha: "number of oscillators per formula unit" / dimensionless
        theta: Einstein temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Heat capacity value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    x = theta / T
    return 3 * alpha * R * x**2 * exp(x) / (exp(x) - 1)**2

def s_einstein(alpha, theta, T = symbolic_T):
    """Entropy; Einstein function; J/mol/K

    3 * alpha * R * ((x / (exp(x) - 1)) - log(1 - exp(-x))),
    where x = theta / T

    Args:
        alpha: "number of oscillators per formula unit" / dimensionless
        theta: Einstein temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Entropy value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
        log = sympy.log
    else:
        exp = np.exp
        log = np.log
    x = theta / T
    return 3 * alpha * R * ((x / (exp(x) - 1)) - log(1 - exp(-x)))

def u_einstein(alpha, theta, T = symbolic_T):
    """Internal energy; Einstein function; J/mol

    3 * alpha * R * theta / (exp(x) - 1),
    where x = theta / T

    Args:
        alpha: "number of oscillators per formula unit" / dimensionless
        theta: Einstein temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Internal energy value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    x = theta / T
    return 3 * alpha * R * theta / (exp(x) - 1)

def cp_einstein_beta(alpha, theta, beta, T = symbolic_T):
    """Heat capacity; Modified Einstein function; J/mol/K

    The Einstein function, modified in accordance with [1],
    which includes the anharmonic effect correction term 'beta'.

    (1 / (1 - beta*T)) * cp_einstein(alpha, theta, T)

    Warning: this function cannot be integrated symbolically.

    References:
        1. Martin CA. Simple treatment of anharmonic effects on the specific 
        heat. Journal of Physics: Condensed Matter. 1991;3(32):5967. 
        doi:10.1088/0953-8984/3/32/005

    Args:
        alpha: "number of oscillators per formula unit" / dimensionless
        theta: Einstein temperature / K
        beta: correction for the (assumed linear) temperature dependence 
            of (acoustic and optical) modes / K**(-1)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Heat capacity value(s) if T is a number or a numpy array.
    """
    return (1 / (1 - beta*T)) * cp_einstein(alpha, theta, T)

_cv_debye_symbolic = R * sympy.sympify('9 * (T / theta)**3 * integrate((x**4)*exp(-x) / (1 - exp(-x))**2, (x, 0, theta/T))')
_cv_debye_numeric = sympy.lambdify(sympy.symbols('theta, T'), _cv_debye_symbolic)

def cv_debye(theta, T = symbolic_T):
    """Heat capacity; Debye function; J/mol/K

    The Debye function. Contains an integral which cannot be evaluated
    symbolically. This function cannot be integrated symbolically.

    Relies on the lambdified numeric implementation, which uses
    scipy.integrate.quad, which doesn't accept numpy arrays as arguments.

    Args:
        theta: Debye temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Heat capacity value if T is a number.
    """
    if type(T) == sympy.Symbol:
        return _cv_debye_symbolic.subs('theta', theta)
    else:
        return _cv_debye_numeric(theta, T)

# These are thermodynamically incorrect, but otherwise very useful aliases).
cp_einstein = cv_einstein
h_einstein = u_einstein
cp_debye = cv_debye


def phi_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, T = symbolic_T):
    """Φ(T); Gurvich textbook and database; J/mol/K

    The "reduced" Gibbs energy function Φ(T) as defined in [1].

    A0 + Aln*log(x) + A_2*x**(-2) + A_1*x**(-1) + A1*x + A2*x**2 + A3*x**3,
    where x = T*1e-4

    References:
        1. Gurvich LV. Thermodynamic properties of individual substances. 
        4th ed. New York: Hemisphere Publishing Corp.; 1989.

    Args:
        A0, Aln, A_2, A_1, A1, A2, A3: coefficients (see the equation above)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Φ value(s) in J/mol/K if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    x = T*1e-4
    phi = A0 + Aln*log(x) + A_2*x**(-2) + A_1*x**(-1) + A1*x + A2*x**2 + A3*x**3
    return phi
        
def g_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, dhf298, h298, T = symbolic_T):
    """Gibbs function; Gurvich textbook and database; J/mol

    Computes Gibbs function via the "reduced" Gibbs energy function Φ(T) 
    as defined in [1].

    Φ(T) = A0 + Aln*log(x) + A_2*x**(-2) + A_1*x**(-1) + A1*x + A2*x**2 + A3*x**3,
    where x = T*1e-4;
    g = -T*Φ(T) + dhf298 - h298

    See also phi_gurvich.

    References:
        1. Gurvich LV. Thermodynamic properties of individual substances. 
        4th ed. New York: Hemisphere Publishing Corp.; 1989.

    Args:
        A0, Aln, A_2, A_1, A1, A2, A3: coefficients (see the equation above)
        dhf298: standard enthalpy of formation at 298.15 K / J/mol
        h298: enthalpy increment H(298.15) - H(0) / J/mol
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Gibbs energy value(s) in J/mol if T is a number or a numpy array.
    """
    phi = phi_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, T)
    return -T*phi + dhf298 - h298


def coeff_to_float(coeff):
    return float(coeff.replace('×10', 'e'))
    
def shomate_function(t, coeff_dict, what):
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] #from string.ascii_uppercase
    coeffs = [coeff_to_float(coeff_dict[letter]) for letter in abc]
    A, B, C, D, E, F, G, H = coeffs
    t /= 1000
    if what == 'cp':
        return A + B*t + C*t**2 + D*t**3 + E/(t**2)
    elif what == 'h':
        return A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - H
    elif what == 's':
        return A*np.log(t) + B*t + C*t**2/2 + D*t**3/3 - E/(2*t**2) + G
    else:
        return None
    
def shomate_cp(*args):
    return shomate_function(*args, what='cp')

def shomate_h(*args):
    return shomate_function(*args, what='h')

def shomate_s(*args):
    return shomate_function(*args, what='s')


def cp_robie(A1, A2, A3, A4, A5, T = symbolic_T):
    """Heat capacity; Robie, Hemingway; J/mol/K

    A1 + A2*T + A3*T**(-2) + A4*T**(-0.5) + A5*T**2

    This heat capacity function used in Robie, Hemingway textbook [1]
    is a kind of a Maier-Kelley polynomial.
    
    References:
        1. Robie RA, Hemingway BS. Thermodynamic properties of minerals 
        and related substances at 298.15 K and 1 bar (10 Pascals) pressure 
        and at higher temperatures. 
        Washington: U.S. goverment printing office; 1995.
    
    Args:
        A1, A2, A3, A4, A5: coefficients (see the equation above)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Heat capacity value(s) if T is a number or a numpy array.
    """
    return A1 + A2*T + A3*T**(-2) + A4*T**(-0.5) + A5*T**2

def cp_barin(A, B, C, D, D_subst = False, T = symbolic_T):
    """Heat capacity; Barin; J/mol/K

    Small polynomials defined in Eq. (29) of [1].

    A + B*1e-3*T + C*1e5*T**(-2) + D**1e-6*T**2

    In some cases the last term in equation is substituted by 
    D*10**8*T**(-3), which should be indicated at the bottom 
    of the respective tablein the textbook.

    References:
        Barin, I., Knacke, O., Kubaschewski, O. (1977). Thermochemical 
        properties of inorganic substances. Springer, Berlin, Heidelberg. 
        doi: 10.1007/978-3-662-02293-1
    
    Args:
        A, B, C, D: coefficients (see the equation above)
        D_subst: if True, substitutes the term multiplied by D
            as explained above
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Heat capacity value(s) if T is a number or a numpy array.
    """
    cp = A + B*1e-3*T + C*1e5*T**(-2)
    if D_subst:
        cp += D*1e8*T**(-3)
    else:
        cp += D**1e-6*T**2
    return cp

def dh298_mks(cp298, b, c, T = symbolic_T):
    """Enthalpy increments; Maier-Kelley-Shomate
    
    Maier-Kelley (polynomial) function modified with Shomate method [1]
    for defining the entalpy increments, H(T) - H(298.15), via the 
    heat capacity value at 298.15 K.

    References:
        1. Shomate CH. A Method for Evaluating and Correlating Thermodynamic 
        Data. The Journal of Physical Chemistry. 1954;58(4):368-72. 
        doi:10.1021/j150514a018
    """
    a = cp298 - 2*298.15*T + c/298.15**2
    d = -298.15*a - 298.15**2*b - c/298.15
    return a*T + b*T**2 + c*T**(-1) + d

#TODO Nasa7
#TODO Nasa9

def poly_from_dict(d, T = symbolic_T):
    return sum([v*T**k for k, v in d.items()])
    
def func_from_dict(d, T = symbolic_T):
    return sum([v*sympy.sympify(k) for k, v in d.items()])


#===========#
# ANOMALIES #
#===========#
# The following functions are used for describing the anomalies
# in various thermodynamic functions

#TODO all from cpfit manual p15
#TODO magnetic contributions

def cp_gaussian(a, b, c, T = symbolic_T):
    """Gaussian function; J/mol/K
    
    Standard Gaussian function which can be used to describe 
    a peak on the heat capacity curve.

    Args:
        a: Gaussian peak height / J/mol/K
        b: Gaussian peak position / K
        c: standard deviation / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Heat capacity value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    return a*exp(-(T - b)**2 / 2 / c**2)

def h_gaussian(a, b, c, T = symbolic_T):
    """Gauss enthalpy; J/mol/K

    Just a symbolic integral of Gaussian peak function.
    
    See also cp_gaussian.

    Args:
        a: Gaussian peak height / J/mol/K
        b: Gaussian peak position / K
        c: standard deviation / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type (or not given explicitly).
        Enthalpy value if T is a number.
    """
    if type(T) == sympy.Symbol:
        erf = sympy.erf
    else:
        erf = scipy.special.erf
    return -(1/2)*a*(2*np.pi)**(0.5)*c*erf((1/2)*2**(0.5)*(-T + b)/c)