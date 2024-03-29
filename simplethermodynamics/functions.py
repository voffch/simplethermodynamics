"""
This file defines some commonly used functions for describing (approximating)
various thermodynamic functions and associated anomalies.
"""

import sympy
import numpy as np
from scipy import special
from .core import R

_symbolic_T = sympy.Symbol('T')

#===========#
# FUNCTIONS #
#===========#
# The following functions are used for approximating
# whole thermodynamic functions or their pieces

# Einstein and Debye functions

# TODO: make Einsteins and Debyes return 0 at (or even very close to) 0 K

def cv_einstein(alpha, theta, T = _symbolic_T):
    """Heat capacity; Einstein function; J/mol/K

    3 * alpha * R * x**2 * exp(x) / (exp(x) - 1)**2,
    where x = theta / T

    Args:
        alpha: "number of oscillators per formula unit" / dimensionless
        theta: Einstein temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    x = theta / T
    return 3 * alpha * R * x**2 * exp(x) / (exp(x) - 1)**2

def s_einstein(alpha, theta, T = _symbolic_T):
    """Entropy; Einstein function; J/mol/K

    3 * alpha * R * ((x / (exp(x) - 1)) - log(1 - exp(-x))),
    where x = theta / T

    Args:
        alpha: "number of oscillators per formula unit" / dimensionless
        theta: Einstein temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
        log = sympy.log
    else:
        exp = np.exp
        log = np.log
    x = theta / T
    return 3 * alpha * R * ((x / (exp(x) - 1)) - log(1 - exp(-x)))

def u_einstein(alpha, theta, T = _symbolic_T):
    """Internal energy; Einstein function; J/mol

    3 * alpha * R * theta / (exp(x) - 1),
    where x = theta / T

    Args:
        alpha: "number of oscillators per formula unit" / dimensionless
        theta: Einstein temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    x = theta / T
    return 3 * alpha * R * theta / (exp(x) - 1)

def cv_einstein_mod(alpha, theta, beta, T = _symbolic_T):
    """Heat capacity; Modified Einstein function; J/mol/K

    The Einstein function, modified in accordance with [1],
    which includes the anharmonic effect correction term 'beta'
    (it was denoted differently in [1]).

    (1 / (1 - beta*T)) * cp_einstein(alpha, theta, T)

    Warning: this function cannot be integrated symbolically.

    References:
        1. Martin CA. Simple treatment of anharmonic effects on the specific 
        heat. Journal of Physics: Condensed Matter. 1991;3(32):5967. 
        https://doi.org/10.1088/0953-8984/3/32/005

    Args:
        alpha: "number of oscillators per formula unit" / dimensionless
        theta: Einstein temperature / K
        beta: correction for the (assumed linear) temperature dependence 
            of (acoustic and optical) modes / K**(-1)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    return (1 / (1 - beta*T)) * cp_einstein(alpha, theta, T)

_cv_debye_symbolic = R * sympy.sympify('alpha * 9 * (T / theta)**3 * integrate((x**4)*exp(-x) / (1 - exp(-x))**2, (x, 0, theta/T))')
# the following will rely on scipy.integrate.quad, which doesn't support numpy arrays...
_cv_debye_numeric = sympy.lambdify(sympy.symbols('alpha, theta, T'), _cv_debye_symbolic)
# ...so we'll make a ufunc from it.
_cv_debye_numeric = np.frompyfunc(_cv_debye_numeric, 3, 1)

def cv_debye(alpha, theta, T = _symbolic_T):
    """Heat capacity; Debye function; J/mol/K

    The Debye function. Contains an integral which cannot be evaluated
    symbolically. This function cannot be integrated symbolically.

    Args:
        alpha: dimensionless multiplier
        theta: Debye temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        return _cv_debye_symbolic.subs({'alpha': alpha, 'theta': theta})
    else:
        return _cv_debye_numeric(alpha, theta, T).astype(float)

# These are thermodynamically incorrect, but otherwise very useful aliases).
cp_einstein = cv_einstein
cp_einstein_mod = cv_einstein_mod
h_einstein = u_einstein
cp_debye = cv_debye

# Gurvich-Glushko functions

def phi_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, T = _symbolic_T):
    """Φ(T); Gurvich textbook and database; J/mol/K

    The "reduced" Gibbs energy function Φ(T) defined in [1].

    A0 + Aln*log(x) + A_2*x**(-2) + A_1*x**(-1) + A1*x + A2*x**2 + A3*x**3,
    where x = T*1e-4

    References:
        1. Gurvich LV. Thermodynamic properties of individual substances. 
        4th ed. New York: Hemisphere Publishing Corp.; 1989.

    Args:
        A0, Aln, A_2, A_1, A1, A2, A3: coefficients (see the equation above)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    x = T*1e-4
    return A0 + Aln*log(x) + A_2*x**(-2) + A_1*x**(-1) + A1*x + A2*x**2 + A3*x**3
        
def cp_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, T = _symbolic_T):
    """Heat capacity; Gurvich textbook and database; J/mol/K

    Heat capacity function derived from Φ(T) defined in [1].

    Aln + 2*A_2*x**(-2) + 2*A1*x + 6*A2*x**2 + 12*A3*x**3,
    where x = T*1e-4

    References:
        1. Gurvich LV. Thermodynamic properties of individual substances. 
        4th ed. New York: Hemisphere Publishing Corp.; 1989.

    Args:
        Aln, A_2, A1, A2, A3: coefficients (see also phi_gurvich)
        A0, A_1: coefficients that are not used for the calculations here,
            but remain among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    x = T*1e-4
    return Aln + 2*A_2*x**(-2) + 2*A1*x + 6*A2*x**2 + 12*A3*x**3

def s_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, T = _symbolic_T):
    """Entropy; Gurvich textbook and database; J/mol/K

    Entropy function derived from Φ(T) defined in [1].

    A0 + Aln*(log(x) + 1) - A_2*x**(-2) + 2*A1*x + 3*A2*x**2 + 4*A3*x**3,
    where x = T*1e-4

    References:
        1. Gurvich LV. Thermodynamic properties of individual substances. 
        4th ed. New York: Hemisphere Publishing Corp.; 1989.

    Args:
        A0, Aln, A_2, A1, A2, A3: coefficients (see also phi_gurvich)
        A_1: coefficient that is not used for the calculations here,
            but remains among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    x = T*1e-4
    return A0 + Aln*(log(x) + 1) - A_2*x**(-2) + 2*A1*x + 3*A2*x**2 + 4*A3*x**3

def dh0_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, T = _symbolic_T):
    """Enthalpy increments; Gurvich textbook and database; J/mol

    Enthalpy increments derived from Φ(T) defined in [1].

    H(T) - H(0) = 
        = 1e4 * (Aln*x - 2*A_2*x**(-1) - A_1 + A1*x**2 + 2*A2*x**3 + 3*A3*x**4)

    References:
        1. Gurvich LV. Thermodynamic properties of individual substances. 
        4th ed. New York: Hemisphere Publishing Corp.; 1989.

    Args:
        Aln, A_2, A_1, A1, A2, A3: coefficients (see also phi_gurvich)
        A0: coefficient that is not used for the calculations here,
            but remains among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    x = T*1e-4
    return 1e4 * (Aln*x - 2*A_2*x**(-1) - A_1 + A1*x**2 + 2*A2*x**3 + 3*A3*x**4)

def h_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, dhf298, dh0_298, T = _symbolic_T):
    """Enthalpy; Gurvich textbook and database; J/mol

    Enthalpy function derived from Φ(T) defined in [1].

    H(T) = ΔfH(298.15) + (H(T) - H(0)) - (H(298.15) - H(0)),
    where the last two terms in brackets are the enthalpy increments.

    The value of (H(298.15) - H(0)) will NOT be calculated correctly with
    the dh0_gurvich function if the coefficients are for the higher-temperature
    Φ(T) function which doesn't evaluate correctly at room temperature.

    References:
        1. Gurvich LV. Thermodynamic properties of individual substances. 
        4th ed. New York: Hemisphere Publishing Corp.; 1989.

    Args:
        A0, Aln, A_2, A_1, A1, A2, A3: coefficients (see also phi_gurvich)
        dhf298: standard enthalpy of formation at 298.15 K / J/mol
        dh0_298: enthalpy increment (H(298.15) - H(0)) / J/mol
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    dh0_T = dh0_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, T)
    return dhf298 + dh0_T - dh0_298

def g_gurvich(A0, Aln, A_2, A_1, A1, A2, A3, dhf298, dh0_298, T = _symbolic_T):
    """Gibbs function; Gurvich textbook and database; J/mol

    Gibbs function derived from Φ(T) defined in [1].

    References:
        1. Gurvich LV. Thermodynamic properties of individual substances. 
        4th ed. New York: Hemisphere Publishing Corp.; 1989.

    Args:
        A0, Aln, A_2, A_1, A1, A2, A3: coefficients (see also phi_gurvich)
        dhf298: standard enthalpy of formation at 298.15 K / J/mol
        dh0_298: enthalpy increment (H(298.15) - H(0)) / J/mol
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    x = T*1e-4
    poly = 1e4*(-A3*x**4 - A2*x**3 - A1*x**2 - Aln*log(x)*x - A0*x - A_1 - A_2/x)
    return poly - dh0_298 + dhf298

# NIST (Webbook) functions

def cp_shomate(A, B, C, D, E, F, G, H, T = _symbolic_T):
    """Heat capacity; NIST Webbook; J/mol/K

    Shomate equation as defined in NIST Webbook ( see, e.g.,
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Type=JANAFL&Table=on ).
    
    A + B*t + C*t**2 + D*t**3 + E/(t**2),
    where t = T / 1000

    Args:
        A, B, C, D, E: coefficients (see the equation above)
        F, G, H: coefficients that are not used for the calculations here,
            but remain among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    t = T / 1000
    return A + B*t + C*t**2 + D*t**3 + E/(t**2)

def dh298_shomate(A, B, C, D, E, F, G, H, T = _symbolic_T):
    """Enthalpy increments; NIST Webbook; J/mol

    Shomate equation as defined in NIST Webbook ( see, e.g.,
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Type=JANAFL&Table=on ).
    
    H(T) - H(298.15) = 
         = (A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - H) kJ/mol,
    where t = T / 1000

    Args:
        A, B, C, D, E, F, H: coefficients (see the equation above)
        G: coefficient that is not used for the calculations here,
            but remains among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    t = T / 1000
    return (A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - H)*1000

def s_shomate(A, B, C, D, E, F, G, H, T = _symbolic_T):
    """Entropy; NIST Webbook; J/mol/K

    Shomate equation as defined in NIST Webbook ( see, e.g.,
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Type=JANAFL&Table=on ).

    This is the absolute entropy, and not the entropy increments.
    
    A*log(t) + B*t + C*t**2/2 + D*t**3/3 - E/(2*t**2) + G,
    where t = T / 1000

    Args:
        A, B, C, D, E, G: coefficients (see the equation above)
        F, H: coefficients that are not used for the calculations here,
            but remain among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    t = T / 1000
    return A*log(t) + B*t + C*t**2/2 + D*t**3/3 - E/(2*t**2) + G

def g_shomate(A, B, C, D, E, F, G, H, dhf298, T = _symbolic_T):
    """Gibbs function; NIST Webbook; J/mol

    Derived from the Shomate equation as defined in NIST Webbook ( see, e.g.,
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Type=JANAFL&Table=on ).

    Uses Standard Element Reference as a reference state.
    In addition to the coefficients in the Shomate equation table,
    requires the enthalpy of formation value (in J/mol).
    
    dhf298 + (-(250/3)*D*t**4-(500/3)*C*t**3-500*B*t**2+(-1000*A*log(t)+
        +1000*A-1000*G)*t+1000*F-1000*H-500*E/t),
    where t = T / 1000

    Args:
        A, B, C, D, E, F, G, H: coefficients (see the equation above)
        dhf298: standard enthalpy of formation at 298.15 K / J/mol
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    t = T / 1000
    return dhf298 + (-(250/3)*D*t**4-(500/3)*C*t**3-500*B*t**2+(-1000*A*log(t)+1000*A-1000*G)*t+1000*F-1000*H-500*E/t)

# Maier-Kelley-like functions

def cp_robie(A1, A2, A3, A4, A5, T = _symbolic_T):
    """Heat capacity; Robie, Hemingway; J/mol/K

    A1 + A2*T + A3*T**(-2) + A4*T**(-0.5) + A5*T**2

    This heat capacity function used in Robie, Hemingway textbook [1]
    is a modified Maier-Kelley polynomial.
    
    References:
        1. Robie RA, Hemingway BS. Thermodynamic properties of minerals 
        and related substances at 298.15 K and 1 bar (10 Pascals) pressure 
        and at higher temperatures. 
        Washington: U.S. goverment printing office; 1995.
    
    Args:
        A1, A2, A3, A4, A5: coefficients (see the equation above)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    return A1 + A2*T + A3*T**(-2) + A4*T**(-0.5) + A5*T**2

def cp_barin(A, B, C, D, T = _symbolic_T, D_subst = False):
    """Heat capacity; Barin; J/mol/K

    Small polynomials defined in Eq. (29) of [1].

    (A + B*1e-3*T + C*1e5*T**(-2) + D*1e-6*T**2) / cal/mol/K

    In some cases the last term in the equation is substituted by 
    D*1e8*T**(-3), which should be indicated at the bottom 
    of the respective table in the textbook.

    References:
        Barin, I., Knacke, O., Kubaschewski, O. (1977). Thermochemical 
        properties of inorganic substances. Springer, Berlin, Heidelberg. 
        https://doi.org/10.1007/978-3-662-02293-1
    
    Args:
        A, B, C, D: coefficients (see the equation above)
        T: temperature / K (if numeric value required)
        D_subst: if True, substitutes the term multiplied by D
            as explained above

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    cp = A + B*1e-3*T + C*1e5*T**(-2)
    if D_subst:
        cp += D*1e8*T**(-3)
    else:
        cp += D*1e-6*T**2
    return 4.184 * cp

def dh298_mks(cp298, b, c, T = _symbolic_T):
    """Enthalpy increments; Maier-Kelley-Shomate; J/mol
    
    Maier-Kelley (polynomial) function modified with Shomate method [1]
    for defining the entalpy increments, H(T) - H(298.15), via the 
    heat capacity value at 298.15 K. The original Maier-Kelley expression is:
        a*T + b*T**2 + c*T**(-1) + d,
    where a can be expressed in terms of the heat capacity at 298.15 K:
        a = cp298 - 2*298.15*b + c/298.15**2,
    and d can be eliminated using the obvious condition that 
        H(298.15) - H(298.15) == 0,
    so
        d = -298.15*a - 298.15**2*b - c/298.15

    References:
        1. Shomate CH. A Method for Evaluating and Correlating Thermodynamic 
        Data. The Journal of Physical Chemistry. 1954;58(4):368-72. 
        https://doi.org/10.1021/j150514a018

    Args:
        cp298: isobaric heat capacity at 298.15 K
        b, c: coefficients (see the equation above)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    a = cp298 - 2*298.15*b + c/298.15**2
    d = -298.15*a - 298.15**2*b - c/298.15
    return a*T + b*T**2 + c*T**(-1) + d

# NASA polynomial functions

def cp_nasa7(a1, a2, a3, a4, a5, a6, a7, T = _symbolic_T):
    """Heat capacity; NASA 7 Polynomials; J/mol/K
    
    These polynomials, initially developed at NASA, are described in [1].

    Cp / R = a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4

    References:
        1. Burcat A., Ruscic B. Third Millennium Ideal Gas and Condensed Phase
        Thermochemical Database for Combustion with Updates from Active 
        Thermochemical Tables. 2005. Argonne National Laboratory publication
        ANL-05/20 TAE 960.
        https://publications.anl.gov/anlpubs/2005/07/53802.pdf

    Args:
        a1 - a5: coefficients (see the equation above)
        a6, a7: coefficients that are not used for the calculations here,
            but remain among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    return R * (a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4)

def h_nasa7(a1, a2, a3, a4, a5, a6, a7, T = _symbolic_T):
    """Enthalpy; NASA 7 Polynomials; J/mol
    
    These polynomials, initially developed at NASA, are described in [1].
    This enthalpy is a sum of the standard enthalpy of formation at 298.15 K
    and the heat capacity integral between 298.15 K and T.

    H / R*T = a1 + (a2/2)*T + (a3/3)*T**2 + (a4/4)*T**3 + (a5/5)*T**4 + a6/T

    References:
        1. Burcat A., Ruscic B. Third Millennium Ideal Gas and Condensed Phase
        Thermochemical Database for Combustion with Updates from Active 
        Thermochemical Tables. 2005. Argonne National Laboratory publication
        ANL-05/20 TAE 960.
        https://publications.anl.gov/anlpubs/2005/07/53802.pdf

    Args:
        a1 - a6: coefficients (see the equation above)
        a7: coefficient that is not used for the calculations here,
            but remains among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    return R * (a1*T + (a2/2)*T**2 + (a3/3)*T**3 + (a4/4)*T**4 + (a5/5)*T**5 + a6)

def s_nasa7(a1, a2, a3, a4, a5, a6, a7, T = _symbolic_T):
    """Entropy; NASA 7 Polynomials; J/mol/K
    
    These polynomials, initially developed at NASA, are described in [1].
    This is the absolute entropy.

    S / R = a1*log(T) + a2*T + (a3/2)*T**2 + (a4/3)*T**3 + (a5/4)*T**4 + a7

    References:
        1. Burcat A., Ruscic B. Third Millennium Ideal Gas and Condensed Phase
        Thermochemical Database for Combustion with Updates from Active 
        Thermochemical Tables. 2005. Argonne National Laboratory publication
        ANL-05/20 TAE 960.
        https://publications.anl.gov/anlpubs/2005/07/53802.pdf

    Args:
        a1 - a5, a7: coefficients (see the equation above)
        a6: coefficient that is not used for the calculations here,
            but remains among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    return R * (a1*log(T) + a2*T + (a3/2)*T**2 + (a4/3)*T**3 + (a5/4)*T**4 + a7)

def g_nasa7(a1, a2, a3, a4, a5, a6, a7, T = _symbolic_T):
    """Gibbs energy; NASA 7 Polynomials; J/mol
    
    These polynomials, initially developed at NASA, are described in [1].

    G / R*T = a1*(1 - log(T)) - (a2/2)*T - (a3/6)*T**2 - (a4/12)*T**3 - 
        - (a5/20)*T**4 + a6/T - a7

    References:
        1. Burcat A., Ruscic B. Third Millennium Ideal Gas and Condensed Phase
        Thermochemical Database for Combustion with Updates from Active 
        Thermochemical Tables. 2005. Argonne National Laboratory publication
        ANL-05/20 TAE 960.
        https://publications.anl.gov/anlpubs/2005/07/53802.pdf

    Args:
        a1 - a7: coefficients (see the equation above)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    return R * ((a1 - a7)*T - a1*T*log(T) - (a2/2)*T**2 - (a3/6)*T**3 - (a4/12)*T**4 - (a5/20)*T**5 + a6)

def cp_nasa9(a1, a2, a3, a4, a5, a6, a7, a8, a9, T = _symbolic_T):
    """Heat capacity; NASA 9 Polynomials; J/mol/K
    
    This is an extended version of NASA 7 polynomials.
    These polynomials, initially developed at NASA, are described in [1].

    Cp / R = a1*T**(-2) + a2*T**(-1) + a3 + a4*T + a5*T**2 + a6*T**3 + a7*T**4

    References:
        1. Burcat A., Ruscic B. Third Millennium Ideal Gas and Condensed Phase
        Thermochemical Database for Combustion with Updates from Active 
        Thermochemical Tables. 2005. Argonne National Laboratory publication
        ANL-05/20 TAE 960.
        https://publications.anl.gov/anlpubs/2005/07/53802.pdf

    Args:
        a1 - a7: coefficients (see the equation above)
        a8, a9: coefficients that are not used for the calculations here,
            but remain among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    return R * (a1*T**(-2) + a2*T**(-1) + a3 + a4*T + a5*T**2 + a6*T**3 + a7*T**4)

def h_nasa9(a1, a2, a3, a4, a5, a6, a7, a8, a9, T = _symbolic_T):
    """Enthalpy; NASA 9 Polynomials; J/mol
    
    This is an extended version of NASA 7 polynomials.
    These polynomials, initially developed at NASA, are described in [1].
    This enthalpy is a sum of the standard enthalpy of formation at 298.15 K
    and the heat capacity integral between 298.15 K and T.

    H / R*T = -a1*T**(-2) + a2*T**(-1)*log(T) + a3 + (a4/2)*T + (a5/3)*T**2 + 
        + (a6/4)*T**3 + (a7/5)*T**4 + a8/T

    References:
        1. Burcat A., Ruscic B. Third Millennium Ideal Gas and Condensed Phase
        Thermochemical Database for Combustion with Updates from Active 
        Thermochemical Tables. 2005. Argonne National Laboratory publication
        ANL-05/20 TAE 960.
        https://publications.anl.gov/anlpubs/2005/07/53802.pdf

    Args:
        a1 - a8: coefficients (see the equation above)
        a9: coefficient that is not used for the calculations here,
            but remains among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    return R * (-a1*T**(-1) + a2*log(T) + a3*T + (a4/2)*T**2 + (a5/3)*T**3 + (a6/4)*T**4 + (a7/5)*T**5 + a8)

def s_nasa9(a1, a2, a3, a4, a5, a6, a7, a8, a9, T = _symbolic_T):
    """Entropy; NASA 7 Polynomials; J/mol/K
    
    This is an extended version of NASA 7 polynomials.
    These polynomials, initially developed at NASA, are described in [1].
    This is the absolute entropy.

    S / R = -a1*T**-2/2 - a2*T**-1 + a3*log(T) + a4*T + (a5/2)*T**2 + 
        + (a6/3)*T**3 + (a7/4)*T**4 + a9

    References:
        1. Burcat A., Ruscic B. Third Millennium Ideal Gas and Condensed Phase
        Thermochemical Database for Combustion with Updates from Active 
        Thermochemical Tables. 2005. Argonne National Laboratory publication
        ANL-05/20 TAE 960.
        https://publications.anl.gov/anlpubs/2005/07/53802.pdf

    Args:
        a1 - a7, a9: coefficients (see the equation above)
        a8: coefficient that is not used for the calculations here,
            but remains among the args to preserve the order
            in which the coefficients are listed
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    return R * (-a1*T**-2/2 - a2*T**-1 + a3*log(T) + a4*T + (a5/2)*T**2 + (a6/3)*T**3 + (a7/4)*T**4 + a9)

def g_nasa9(a1, a2, a3, a4, a5, a6, a7, a8, a9, T = _symbolic_T):
    """Gibbs energy; NASA 7 Polynomials; J/mol
    
    This is an extended version of NASA 7 polynomials.
    These polynomials, initially developed at NASA, are described in [1].

    G / R*T = -(a1/2)*T**(-2) + 2*a2*(1-log(T))/T + a3*(1-log(T)) - 
        - (a4/2)*T - (a5/6)*T**2 - (a6/12)*T**3 - (a7/20)*T**4 + a8/T - a9

    References:
        1. Burcat A., Ruscic B. Third Millennium Ideal Gas and Condensed Phase
        Thermochemical Database for Combustion with Updates from Active 
        Thermochemical Tables. 2005. Argonne National Laboratory publication
        ANL-05/20 TAE 960.
        https://publications.anl.gov/anlpubs/2005/07/53802.pdf

    Args:
        a1 - a9: coefficients (see the equation above)
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        log = sympy.log
    else:
        log = np.log
    return R * (-(a1/2)*T**(-1) + 2*a2*(1-log(T)) + (a3 - a9)*T - a3*T*log(T) - (a4/2)*T**2 - (a5/6)*T**3 - (a6/12)*T**4 - (a7/20)*T**5 + a8)
    
# Miscellaneous functions

def poly_from_dict(d, T = _symbolic_T):
    """Polynomial function from {power : coeff} dict

    Args:
        d: dict with powers as keys and coefficients
            as values
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    return sum([coeff*T**power for power, coeff in d.items()])
    
def expr_from_dict(d):
    """Arbitrary sympy expression from {term : coeff} dict

    Args:
        d: dict with terms (e.g., simpyfiable strings) as keys 
            and coefficients as values

    Returns:
        SymPy expression.
    """
    return sum([coeff*sympy.sympify(term) for term, coeff in d.items()])


#===========#
# ANOMALIES #
#===========#
# The following functions are used for describing the anomalies
# in various thermodynamic functions

# The extra terms such as those implemented in the CpFit software.
# Voskov, A.L. New Possibilities of the CpFit Program for Approximating 
# Heat Contents and Heat Capacities. Russ. J. Phys. Chem. 96, 1895–1900 (2022).
# https://doi.org/10.1134/S0036024422090291
# See also https://td.chem.msu.ru/en/developments/cpfit/

def cp_lambda(b1, b2, b3, Ttr, T = _symbolic_T):
    """Heat capacity lambda transition term; CpFit; J/mol/K
    
    The default (in CpFit) term for describing the lambda transitions.

    Cp / R = b1 * exp(b2*(b3*ΔT - |ΔT|)),
    where ΔT = T - Ttr

    Args:
        b1, b2, b3: coefficients (see the equation above)
        Ttr: transition temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    dT = T - Ttr
    if type(T) == sympy.Symbol:
        exp = sympy.exp
        absdT = sympy.Piecewise((T - Ttr, T >= Ttr), (Ttr - T, True))
    else:
        exp = np.exp
        absdT = np.abs(dT)
    return R * b1 * exp(b2*(b3*dT - absdT))

def cp_leftexp(b1, b2, Tmax, T = _symbolic_T):
    """Heat capacity left exponent; CpFit; J/mol/K
    
    Exponential function for describing the left side of the peak.

    Cp / R = b1 * exp(b2*(T - Tmax)),

    Args:
        b1, b2: coefficients (see the equation above)
        Tmax: peak (position) temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    return R * b1 * exp(b2*(T - Tmax))

def cp_rightexp(b1, b2, Tmin, T = _symbolic_T):
    """Heat capacity right exponent; CpFit; J/mol/K
    
    Exponential function for describing the right side of the peak.

    Cp / R = b1 * exp(-b2*(T - Tmin)),

    Args:
        b1, b2: coefficients (see the equation above)
        Tmin: peak (position) temperature / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    return R * b1 * exp(-b2*(T - Tmin))

def cp_skewed(b1, b2, b3, b4, T = _symbolic_T):
    """Heat capacity asymmetric Gauss term; CpFit; J/mol/K
    
    A (possibly) asymmetric Gauss bell curve.

    Cp / R = b1 * exp(-x**2) / (1 + exp(-b2*x)),
    where x = (T - b3) / b4

    Args:
        b1 - b4: coefficients (see the equation above),
            responsible for the peak height, asymmetry,
            peak position and peak width, respectively
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    x = (T - b3) / b4
    return R * b1 * exp(-x**2) / (1 + exp(-b2*x))

# Miscellaneous anomaly functions

def cp_gauss(a, b, c, T = _symbolic_T):
    """Heat capacity Gauss term; J/mol/K
    
    Standard Gaussian function which can be used to describe 
    a peak on the heat capacity curve.

    Args:
        a: Gaussian peak height / J/mol/K
        b: Gaussian peak position / K
        c: standard deviation / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        exp = sympy.exp
    else:
        exp = np.exp
    return a*exp(-(T - b)**2 / 2 / c**2)

def h_gauss(a, b, c, T = _symbolic_T):
    """Enthalpy Gauss term; J/mol

    Just a symbolic integral of Gaussian peak function.
    
    See also cp_gaussian.

    Args:
        a: Gaussian peak height / J/mol/K
        b: Gaussian peak position / K
        c: standard deviation / K
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value if T is a number.
    """
    if type(T) == sympy.Symbol:
        erf = sympy.erf
    else:
        erf = special.erf
    return -(1/2)*a*(2*np.pi)**(0.5)*c*erf((1/2)*2**(0.5)*(-T + b)/c)

# SGTE descriptions of magnetic contributions (aka lambda-anomalies)

def _g_magnetic_sgte_symbolic(Tc, B0, p):
    T = _symbolic_T
    log = sympy.log
    tau = T / Tc
    D = 518/1125 + 11692/15975*(p**(-1) - 1)
    g_mag_low = 1 - (79*tau**(-1)/140/p + 474/497*(p**(-1) - 1)*(tau**3/6 + tau**9/135 + tau**15/600)) / D
    g_mag_high = -(tau**(-5)/10 + tau**(-15)/315 + tau**(-25)/1500) / D
    return sympy.simplify(R * T * log(B0 + 1) * sympy.Piecewise((g_mag_low, T <= Tc), (g_mag_high, T > Tc)))

_g_magnetic_sgte_numeric = sympy.lambdify(
    sympy.symbols('Tc, B0, p, T'), 
    _g_magnetic_sgte_symbolic(*sympy.symbols('Tc, B0, p')),
    'numpy'
)

def g_magnetic_sgte(Tc, B0, p, T = _symbolic_T):
    """Magnetic Gibbs energy; SGTE; J/mol

    The magnetic contribution to the thermodynamic properties 
    is described in [1].

    References:
        1. Dinsdale AT. SGTE data for pure elements. 
        Calphad. 1991;15(4):317-425. 
        https://doi.org/10.1016/0364-5916(91)90030-N

    Args:
        Tc: the critical temperature 
            (the Curie temperature Tc for ferromagnetic materials 
            or the Neel temperature TN for antiferromagnetic materials)
        B0: the average magnetic moment per atom
        p: can be thought of as the fraction of the magnetic enthalpy 
            absorbed above the critical temperature; depends on the structure
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        return _g_magnetic_sgte_symbolic(Tc, B0, p)
    else:
        return _g_magnetic_sgte_numeric(Tc, B0, p, T)

def _s_magnetic_sgte_symbolic(Tc, B0, p):
    T = _symbolic_T
    log = sympy.log
    tau = T / Tc
    D = 518/1125 + 11692/15975*(p**(-1) - 1)
    s_mag_low = 1 - (474/497*(p**(-1) - 1)*((2/3)*tau**3 + (2/27)*tau**9 + (2/75)*tau**15)) / D
    s_mag_high = ((2/5)*tau**(-5) + (2/45)*tau**(-15) + (2/125)*tau**(-25)) / D
    return sympy.simplify(- R * log(B0 + 1) * sympy.Piecewise((s_mag_low, T <= Tc), (s_mag_high, T > Tc)))

_s_magnetic_sgte_numeric = sympy.lambdify(
    sympy.symbols('Tc, B0, p, T'), 
    _s_magnetic_sgte_symbolic(*sympy.symbols('Tc, B0, p')),
    'numpy'
)

def s_magnetic_sgte(Tc, B0, p, T = _symbolic_T):
    """Magnetic entropy; SGTE; J/mol/K

    The magnetic contribution to the thermodynamic properties 
    is described in [1].

    References:
        1. Dinsdale AT. SGTE data for pure elements. 
        Calphad. 1991;15(4):317-425. 
        https://doi.org/10.1016/0364-5916(91)90030-N

    Args:
        Tc: the critical temperature 
            (the Curie temperature Tc for ferromagnetic materials 
            or the Neel temperature TN for antiferromagnetic materials)
        B0: the average magnetic moment per atom
        p: can be thought of as the fraction of the magnetic enthalpy 
            absorbed above the critical temperature; depends on the structure
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        return _s_magnetic_sgte_symbolic(Tc, B0, p)
    else:
        return _s_magnetic_sgte_numeric(Tc, B0, p, T)

def _h_magnetic_sgte_symbolic(Tc, B0, p):
    T = _symbolic_T
    log = sympy.log
    tau = T / Tc
    D = 518/1125 + 11692/15975*(p**(-1) - 1)
    h_mag_low = (-79*tau**(-1)/140/p + 474/497*(p**(-1) - 1)*(tau**3/2 + tau**9/15 + tau**15/40)) / D
    h_mag_high = -(tau**(-5)/2 + tau**(-15)/21 + tau**(-25)/60) / D
    return sympy.simplify(R * T * log(B0 + 1) * sympy.Piecewise((h_mag_low, T <= Tc), (h_mag_high, T > Tc)))

_h_magnetic_sgte_numeric = sympy.lambdify(
    sympy.symbols('Tc, B0, p, T'), 
    _h_magnetic_sgte_symbolic(*sympy.symbols('Tc, B0, p')),
    'numpy'
)

def h_magnetic_sgte(Tc, B0, p, T = _symbolic_T):
    """Magnetic enthalpy; SGTE; J/mol

    The magnetic contribution to the thermodynamic properties 
    is described in [1].

    References:
        1. Dinsdale AT. SGTE data for pure elements. 
        Calphad. 1991;15(4):317-425. 
        https://doi.org/10.1016/0364-5916(91)90030-N

    Args:
        Tc: the critical temperature 
            (the Curie temperature Tc for ferromagnetic materials 
            or the Neel temperature TN for antiferromagnetic materials)
        B0: the average magnetic moment per atom
        p: can be thought of as the fraction of the magnetic enthalpy 
            absorbed above the critical temperature; depends on the structure
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        return _h_magnetic_sgte_symbolic(Tc, B0, p)
    else:
        return _h_magnetic_sgte_numeric(Tc, B0, p, T)

def _cp_magnetic_sgte_symbolic(Tc, B0, p):
    T = _symbolic_T
    log = sympy.log
    tau = T / Tc
    D = 518/1125 + 11692/15975*(p**(-1) - 1)
    c_mag_low = ((474/497) * (1/p - 1) * (2*tau**3 + (2/3)*tau**9 + (2/5)*tau**15)) / D
    c_mag_high = (2*tau**(-5) + (2/3)*tau**(-15) + (2/5)*tau**(-25)) / D
    return R * log(B0 + 1) * sympy.Piecewise((c_mag_low, T <= Tc), (c_mag_high, T > Tc))

_cp_magnetic_sgte_numeric = sympy.lambdify(
    sympy.symbols('Tc, B0, p, T'), 
    _cp_magnetic_sgte_symbolic(*sympy.symbols('Tc, B0, p')),
    'numpy'
)

def cp_magnetic_sgte(Tc, B0, p, T = _symbolic_T):
    """Magnetic heat capacity; SGTE; J/mol/K

    The magnetic contribution to the thermodynamic properties 
    is described in [1].

    References:
        1. Dinsdale AT. SGTE data for pure elements. 
        Calphad. 1991;15(4):317-425. 
        https://doi.org/10.1016/0364-5916(91)90030-N

    Args:
        Tc: the critical temperature 
            (the Curie temperature Tc for ferromagnetic materials 
            or the Neel temperature TN for antiferromagnetic materials)
        B0: the average magnetic moment per atom
        p: can be thought of as the fraction of the magnetic enthalpy 
            absorbed above the critical temperature; depends on the structure
        T: temperature / K (if numeric value required)

    Returns:
        SymPy expression if T is of Symbol type or is not given explicitly.
        Value(s) if T is a number or a numpy array.
    """
    if type(T) == sympy.Symbol:
        return _cp_magnetic_sgte_symbolic(Tc, B0, p)
    else:
        return _cp_magnetic_sgte_numeric(Tc, B0, p, T)

#?TODO: Inden's models from The role of magnetism in the calculation of phase diagrams Physica 103B (1981) 82-100