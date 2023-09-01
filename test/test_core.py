import unittest
import os, json
import numpy as np
import sympy, matplotlib
from simplethermodynamics import Phase, Compound, ThermodynamicDatabase
import simplethermodynamics.functions as tf

T = sympy.Symbol('T')

class TestPhaseBasic(unittest.TestCase):
    def test_numbers(self):
        """Phase initialization with functions as numeric values"""
        t = 400
        funcs = ['g', 'cp', 'h', 's']
        known_g_values = {
            0 : [0, 0, 0, 0],
        '-1000' : [-1000, 0, -1000, 0]
        }
        known_cphs_values = {
                    (0, 0, 0) : [0, 0, 0, 0],
            (100, -2000, 300) : [
                    (100*(t-298.15)-2000) - t*(300 + 100*np.log(t/298.15)), 
                    100, 
                    100*(t-298.15)-2000, 
                    300 + 100*np.log(t/298.15)
                                ]
        }
        
        for g, known_values in known_g_values.items():
            p = Phase('name', g=g)
            calc_values = [getattr(p, f)(t) for f in funcs]
            for calc, known in zip(calc_values, known_values):
                self.assertAlmostEqual(calc, known, places=8)
        
        for (cp, h298, s298), known_values in known_cphs_values.items():
            p = Phase('name', cp=cp, h298=h298, s298=s298)
            calc_values = [getattr(p, f)(t) for f in funcs]
            for calc, known in zip(calc_values, known_values):
                self.assertAlmostEqual(calc, known, places=8)

    def test_incorrect(self):
        """Phase initialization with incomplete (incorrect) arguments"""
        self.assertRaises(Exception, Phase)
        self.assertRaises(Exception, Phase, 'name_only')
        self.assertRaises(Exception, Phase, 'cp_only', cp=12)


def f_dicts_to_funcs(f_dicts, dhf298, h298, extrapolation_extent):
    """List of Gurvich Φ(T) coefficients into Piecewise thermodynamics

    Args:
        f_dicts: a list of well-structured ordered dicts containing
            the coefficients for the Gurvich polynomials
        dhf298: standard enthalpy of formation at 298.15 K / J/mol
        h298: enthalpy increment (H(298.15) - H(0)) / J/mol
        extrapolation_extent: defines the temperature ranges above max
            and below min temperatures for which the functions will be
            extrapolated

    Returns:
        (g, h, s, cp) tuple containing the respective Piecewise
        thermodynamic functions.
    """
    cond_list = []
    for i, x in enumerate(f_dicts):
        lower = float(x['t'][0])
        upper = float(x['t'][1])
        if i == 0:
            lower -= extrapolation_extent
        if i == len(f_dicts) - 1:
            upper += extrapolation_extent
        cond_list.append(sympy.And(T >= lower, T <= upper))
    g_list = [
        tf.g_gurvich(*[float(coeff) for coeff in x['coefficients'].values()], dhf298, h298) for x in f_dicts
    ]
    h_list = [
        tf.h_gurvich(*[float(coeff) for coeff in x['coefficients'].values()], dhf298, h298) for x in f_dicts
    ]
    s_list = [
        tf.s_gurvich(*[float(coeff) for coeff in x['coefficients'].values()]) for x in f_dicts
    ]
    cp_list = [
        tf.cp_gurvich(*[float(coeff) for coeff in x['coefficients'].values()]) for x in f_dicts
    ]
    g_exprcond = [(expr, cond) for expr, cond in zip(g_list, cond_list)]
    g_exprcond.append((0, True))
    h_exprcond = [(expr, cond) for expr, cond in zip(h_list, cond_list)]
    h_exprcond.append((0, True))
    s_exprcond = [(expr, cond) for expr, cond in zip(s_list, cond_list)]
    s_exprcond.append((0, True))
    cp_exprcond = [(expr, cond) for expr, cond in zip(cp_list, cond_list)]
    cp_exprcond.append((0, True))
    g = sympy.Piecewise(*g_exprcond)
    h = sympy.Piecewise(*h_exprcond)
    s = sympy.Piecewise(*s_exprcond)
    cp = sympy.Piecewise(*cp_exprcond)
    return (g, h, s, cp)

def record_to_phases(record, extrapolation_extent = 20.0):
    """Gurvich record to Phases
    
    Converts a dict record containing the info from one of the Gurvich 
    tables into Phase objects.

    Args:
        record: a list of well-structured dict with the Gurvich data
        extrapolation_extent: to this extent (in K) the thermodynamic
            functions will be extrapolated above and below the respective
            initial limits

    Returns:
        (g, h, s, cp) tuple containing the respective Piecewise
        thermodynamic functions.
    """
    info = f"{record['name']} ({record['state']}) [{record['url']}]"
    dhf298 = float(record['dhf298']) * 1000
    h298 = float(record['h298-h0']) * 1000
    f_dicts = record['f']
    ph = []
    if 'solid,liquid' in record['state']:
        state_parts = record['state'].split(';')
        if len(state_parts) > 1:
            label = state_parts[1]
        else:
            label = ''
        name = state_parts[0]
        #solid
        name = 'solid;' + label if label else 'solid'
        g, h, s, cp = f_dicts_to_funcs(f_dicts[:-1], dhf298, h298, extrapolation_extent)
        ph.append(Phase(name, 'solid', g=g, h=h, s=s, cp=cp))
        #liquid
        name = 'liquid;' + label if label else 'liquid'
        g, h, s, cp = f_dicts_to_funcs([f_dicts[-1]], dhf298, h298, extrapolation_extent)
        ph.append(Phase(name, 'liquid', g=g, h=h, s=s, cp=cp))
    else:
        g, h, s, cp = f_dicts_to_funcs(f_dicts, dhf298, h298, extrapolation_extent)
        ph.append(Phase(record['state'], record['state'], g=g, h=h, s=s, cp=cp))
    return (ph, info)


class TestGurvichCompoundAndDatabase(unittest.TestCase):
    """Testing Compound and ThermodynamicDatabase with real-life data"""
    def setUp(self):
        jsonpath = os.path.join(os.path.dirname(__file__), 'gurvich_Fe2O3.json')
        with open(jsonpath, 'r', encoding="utf-8") as f:
            gurvich = json.load(f)
        self.gurvich = gurvich
        ph, info = record_to_phases(gurvich)
        self.cmpd = Compound(gurvich['formula'], ph, info)
        # from the Gurvich table, let's just remove the duplicate temperatures
        t_table = self.gurvich['table']['t']
        duplicate_t_indices = [i for i in range(1, len(t_table)) if t_table[i] == t_table[i - 1]]
        for i in duplicate_t_indices[::-1]:
            for col in self.gurvich['table'].values():
                del col[i]
        # in this respect, maybe the Gurvich data 
        # doesn't constitute a good test case...

    def test_cp(self):
        """Testing Compound Cp for Fe2O3"""
        X = np.array(self.gurvich['table']['t'][2:], dtype=float)
        Y = np.array(self.gurvich['table']['cp'][2:], dtype=float)
        calc = self.cmpd.cp(X)
        self.assertTrue(np.allclose(Y, calc, atol=5e-04))

    def test_s(self):
        """Testing Compound S for Fe2O3"""
        X = np.array(self.gurvich['table']['t'][2:], dtype=float)
        Y = np.array(self.gurvich['table']['s'][2:], dtype=float)
        calc = self.cmpd.s(X)
        self.assertTrue(np.allclose(Y, calc, atol=5e-04))

    def test_dh(self):
        """Testing Compound H(T) - H(298.15) for Fe2O3"""
        X = np.array(self.gurvich['table']['t'][2:], dtype=float)
        Y = np.array(self.gurvich['table']['ht-h0'][2:], dtype=float)
        Y -= Y[0]
        calc = (self.cmpd.h(X) - self.cmpd.h(298.15)) / 1000
        self.assertTrue(np.allclose(Y, calc, atol=1e-03))

    def test_phi(self):
        """Testing Compound Φ(T) for Fe2O3"""
        X = np.array(self.gurvich['table']['t'][2:], dtype=float)
        Y = np.array(self.gurvich['table']['f'][2:], dtype=float)
        h0 = self.cmpd.h(298.15) - float(self.gurvich['table']['ht-h0'][2]) * 1000
        calc = - (self.cmpd.g(X) - h0) / X
        self.assertTrue(np.allclose(Y, calc, atol=3e-03))

    '''
    def test_table(self):
        """Compound.table returning a nonzero string"""
        table = self.cmpd.table(show=False)
        self.assertIsInstance(table, str)
        self.assertTrue(len(table))

    def test_plot(self):
        """Compound.table returning a matplotlib (fig, axes) tuple"""
        plot = self.cmpd.plot(show=False)
        self.assertIsInstance(plot, tuple)
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot[0], matplotlib.figure.Figure)
        self.assertIsInstance(plot[1], matplotlib.axes.Axes)
    '''