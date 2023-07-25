import unittest
import sympy
import numpy as np
import json
import os
import simplethermodynamics.functions as tf

T = sympy.Symbol('T')

# General guidelines for making the function test cases:
# - returning symbolic representation when no temperature given
# - evaluating symbolic function at one floating-point T
# - evaluating numeric function at one floating-point T
# - evaluating numeric function for ndarray of T

class TestEinsteinDebye(unittest.TestCase):
    def test_c_einstein(self):
        """Einstein heat capacity"""
        self.assertIs(tf.cp_einstein, tf.cv_einstein)
        f = tf.cp_einstein
        args = (5.047, 508.2)
        X = np.array([298.15, 500, 1000])
        Y = np.array([99.37445118, 115.5892442, 123.2144879])
        sym = f(*args).evalf(subs={T: float(X[0])})
        num = f(*args, float(X[0]))
        ndar = f(*args, X)
        self.assertAlmostEqual(sym, Y[0])
        self.assertAlmostEqual(num, Y[0])
        self.assertTrue(np.allclose(ndar, Y))

    def test_s_einstein(self):
        """Einstein entropy"""
        f = tf.s_einstein
        args = (5.047, 508.2)
        X = np.array([298.15, 500, 1000])
        Y = np.array([72.96710312, 129.1240600, 212.4472635])
        sym = f(*args).evalf(subs={T: float(X[0])})
        num = f(*args, float(X[0]))
        ndar = f(*args, X)
        self.assertAlmostEqual(sym, Y[0])
        self.assertAlmostEqual(num, Y[0])
        self.assertTrue(np.allclose(ndar, Y))

    def test_u_einstein(self):
        """Einstein internal energy / enthalpy"""
        self.assertIs(tf.h_einstein, tf.u_einstein)
        f = tf.h_einstein
        args = (5.047, 508.2)
        X = np.array([298.15, 500, 1000])
        Y = np.array([14221.21374, 36283.95612, 96598.64456])
        sym = f(*args).evalf(subs={T: float(X[0])})
        num = f(*args, float(X[0]))
        ndar = f(*args, X)
        self.assertAlmostEqual(sym, Y[0], places = 4)
        self.assertAlmostEqual(num, Y[0], places = 4)
        self.assertTrue(np.allclose(ndar, Y))

    def test_cv_einstein_mod(self):
        """Modified Einstein heat capacity"""
        self.assertIs(tf.cp_einstein_mod, tf.cv_einstein_mod)
        f = tf.cv_einstein_mod
        args = (9.77, 392.406, 0.000228212)
        X = np.array([298.15, 500, 1000])
        Y = np.array([226.8001337, 261.3908057, 311.7355303])
        sym = f(*args).evalf(subs={T: float(X[0])})
        num = f(*args, float(X[0]))
        ndar = f(*args, X)
        self.assertAlmostEqual(sym, Y[0], places = 6)
        self.assertAlmostEqual(num, Y[0], places = 6)
        self.assertTrue(np.allclose(ndar, Y))

    def test_cv_debye(self):
        """Debye heat capacity"""
        self.assertIs(tf.cv_debye, tf.cp_debye)
        f = tf.cp_debye
        args = (3.1, 210.5)
        X = np.array([30, 298.15, 500, 1000, 2000])
        Y = np.array([14.67742324, 75.43110970, 76.64366983, 77.15424844, 77.28276325])
        self.assertIsInstance(f(*args), sympy.Basic)
        num = f(*args, float(X[0]))
        ndar = f(*args, X)
        self.assertAlmostEqual(num, Y[0])
        # allowing to differ in thousandth at high T due to the numeric integration:
        self.assertTrue(np.allclose(ndar, Y, atol=1e-03))

class TestGurvich(unittest.TestCase):
    def setUp(self):
        jsonpath = os.path.join(os.path.dirname(__file__), 'gurvich_Fe2O3.json')
        with open(jsonpath, 'r', encoding="utf-8") as f:
            zn = json.load(f)
        self.table = zn['table']
        self.h298 = 1000 * float(zn['h298-h0'])
        self.dhf298 = 1000 * float(zn['dhf298'])
        t_range = [float(x) for x in zn['f'][0]['t']]
        self.coeffs = [float(x) for x in zn['f'][0]['coefficients'].values()]
        all_t = [float(x) for x in self.table['t']]
        t_indices = [all_t.index(x) for x in t_range]
        self.t_slice = slice(t_indices[0], t_indices[1] + 1)
        self.t_ndar = np.array(all_t[self.t_slice])
        # for testing h and g at the highest temperature
        self.high_t_coeffs = [float(x) for x in zn['f'][-1]['coefficients'].values()]
        self.highest_t = all_t[-1]
        self.highest_h = float(self.table['ht-h0'][-1]) \
                        + (self.dhf298 / 1000) - (self.h298 / 1000)
        self.highest_g = self.highest_h - self.highest_t \
                        * float(self.table['s'][-1]) / 1000

    def test_phi(self):
        """Gurvich Phi(T)"""
        data = np.array(self.table['f'][self.t_slice], dtype=float)
        calc = tf.phi_gurvich(*self.coeffs, self.t_ndar)
        self.assertTrue(np.allclose(data, calc, atol=2e-03))
        calc_num = tf.phi_gurvich(*self.coeffs, float(self.t_ndar[0]))
        self.assertAlmostEqual(calc_num, data[0], places = 2)
        calc_sym = tf.phi_gurvich(*self.coeffs).evalf(subs={T: self.t_ndar[0]})
        self.assertAlmostEqual(calc_sym, data[0], places = 2)

    def test_cp(self):
        """Gurvich heat capacity"""
        data = np.array(self.table['cp'][self.t_slice], dtype=float)
        calc = tf.cp_gurvich(*self.coeffs, self.t_ndar)
        self.assertTrue(np.allclose(data, calc, atol=5e-04))
        calc_num = tf.cp_gurvich(*self.coeffs, float(self.t_ndar[0]))
        self.assertAlmostEqual(calc_num, data[0], places = 2)
        calc_sym = tf.cp_gurvich(*self.coeffs).evalf(subs={T: self.t_ndar[0]})
        self.assertAlmostEqual(calc_sym, data[0], places = 2)

    def test_s(self):
        """Gurvich entropy"""
        data = np.array(self.table['s'][self.t_slice], dtype=float)
        calc = tf.s_gurvich(*self.coeffs, self.t_ndar)
        self.assertTrue(np.allclose(data, calc, atol=5e-04))
        calc_num = tf.s_gurvich(*self.coeffs, float(self.t_ndar[0]))
        self.assertAlmostEqual(calc_num, data[0], places = 2)
        calc_sym = tf.s_gurvich(*self.coeffs).evalf(subs={T: self.t_ndar[0]})
        self.assertAlmostEqual(calc_sym, data[0], places = 2)

    def test_dh0(self):
        """Gurvich enthalpy increments"""
        data = np.array(self.table['ht-h0'][self.t_slice], dtype=float)
        calc = tf.dh0_gurvich(*self.coeffs, self.t_ndar) / 1000 # in kJ/mol
        self.assertTrue(np.allclose(data, calc, atol=1e-03))
        calc_num = tf.dh0_gurvich(*self.coeffs, float(self.t_ndar[0])) / 1000
        self.assertAlmostEqual(calc_num, data[0], places = 2)
        calc_sym = tf.dh0_gurvich(*self.coeffs).evalf(subs={T: self.t_ndar[0]}) / 1000
        self.assertAlmostEqual(calc_sym, data[0], places = 2)

    def test_h(self):
        """Gurvich enthalpy"""
        data = np.array(self.table['ht-h0'][self.t_slice], dtype=float) \
                + (self.dhf298 / 1000) - (self.h298 / 1000)
        calc = tf.h_gurvich(*self.coeffs, 
                            self.dhf298, 
                            self.h298, 
                            self.t_ndar) / 1000 # in kJ/mol
        self.assertTrue(np.allclose(data, calc, atol=5e-04))
        calc_num = tf.h_gurvich(*self.coeffs, 
                                self.dhf298, 
                                self.h298, float(self.t_ndar[0])) / 1000
        self.assertAlmostEqual(calc_num, data[0], places = 2)
        calc_sym = tf.h_gurvich(*self.coeffs,
                                self.dhf298, 
                                self.h298).evalf(subs={T: self.t_ndar[0]}) / 1000
        self.assertAlmostEqual(calc_sym, data[0], places = 2)
        highest = tf.h_gurvich(*self.high_t_coeffs, 
                            self.dhf298, 
                            self.h298, 
                            self.highest_t) / 1000
        self.assertAlmostEqual(highest, self.highest_h, places = 2)

    def test_g(self):
        """Gurvich Gibbs energy"""
        h = np.array(self.table['ht-h0'][self.t_slice], dtype=float) \
                + (self.dhf298 / 1000) - (self.h298 / 1000)
        ts = self.t_ndar * np.array(self.table['s'][self.t_slice], dtype=float)
        data = h - (ts / 1000) # in kJ/mol
        calc = tf.g_gurvich(*self.coeffs, 
                            self.dhf298, 
                            self.h298, 
                            self.t_ndar) / 1000 # in kJ/mol
        self.assertTrue(np.allclose(data, calc, atol=5e-04))
        calc_num = tf.g_gurvich(*self.coeffs, 
                                self.dhf298, 
                                self.h298, float(self.t_ndar[0])) / 1000
        self.assertAlmostEqual(calc_num, data[0], places = 2)
        calc_sym = tf.g_gurvich(*self.coeffs,
                                self.dhf298, 
                                self.h298).evalf(subs={T: self.t_ndar[0]}) / 1000
        self.assertAlmostEqual(calc_sym, data[0], places = 2)
        highest = tf.g_gurvich(*self.high_t_coeffs, 
                            self.dhf298, 
                            self.h298, 
                            self.highest_t) / 1000
        self.assertAlmostEqual(highest, self.highest_g, places = 2)

class TestNistShomate(unittest.TestCase):
    def setUp(self):
        # liquid water, https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=2#Thermo-Condensed
        # table columns: Cp, S, -(G° - H°298.15)/T, H° - H°298.15
        # units: J/mol/K, J/mol/K, J/mol/K, kJ/mol
        nist_table = """298.	75.38	69.92	69.95	-0.01
        300.	75.35	70.42	69.95	0.14
        400.	76.74	92.19	72.91	7.71
        500.	83.66	109.9	78.58	15.66"""
        table = np.fromstring(nist_table, sep='\t', dtype=float).reshape(4, 5).T
        self.t, self.cp, self.s, self.phi, self.dh298 = table
        self.coeffs = (-203.6060, 1523.290, -3196.413, 2474.455, 3.855326, -256.5478, -488.7163, -285.8304)
        self.dhf298 = -285.83 * 1000 # in J/mol

    def test_cp(self):
        """Shomate heat capacity"""
        f = tf.cp_shomate
        X = self.t
        Y = self.cp
        sym = f(*self.coeffs).evalf(subs={T: float(X[0])})
        num = f(*self.coeffs, float(X[0]))
        ndar = f(*self.coeffs, X)
        self.assertAlmostEqual(sym, Y[0], places=2)
        self.assertAlmostEqual(num, Y[0], places=2)
        self.assertTrue(np.allclose(ndar, Y, atol=5e-3))

    def test_dh298(self):
        """Shomate enthalpy increments"""
        f = tf.dh298_shomate
        X = self.t
        Y = self.dh298
        sym = f(*self.coeffs).evalf(subs={T: float(X[0])}) / 1000
        num = f(*self.coeffs, float(X[0])) / 1000
        ndar = f(*self.coeffs, X) / 1000
        self.assertAlmostEqual(sym, Y[0], places=2)
        self.assertAlmostEqual(num, Y[0], places=2)
        self.assertTrue(np.allclose(ndar, Y, atol=5e-3))

    def test_s(self):
        """Shomate entropy"""
        f = tf.s_shomate
        X = self.t
        Y = self.s
        sym = f(*self.coeffs).evalf(subs={T: float(X[0])})
        num = f(*self.coeffs, float(X[0]))
        ndar = f(*self.coeffs, X)
        self.assertAlmostEqual(sym, Y[0], places=2)
        self.assertAlmostEqual(num, Y[0], places=2)
        self.assertTrue(np.allclose(ndar, Y, atol=5e-3))

    def test_g(self):
        """Shomate Gibbs energy"""
        f = tf.g_shomate
        X = self.t
        Y = (- self.phi * self.t + self.dhf298 + self.dh298) / 1000
        sym = f(*self.coeffs, self.dhf298).evalf(subs={T: float(X[0])}) / 1000
        num = f(*self.coeffs, self.dhf298, float(X[0])) / 1000
        ndar = f(*self.coeffs, self.dhf298, X) / 1000
        self.assertAlmostEqual(sym, Y[0], places=2)
        self.assertAlmostEqual(num, Y[0], places=2)
        self.assertTrue(np.allclose(ndar, Y, atol=1.5e-2)) # many rounding errors

class TestMaierKelley(unittest.TestCase):
    def test_cp_robie(self):
        """Robie, Hemingway heat capacity"""
        f = tf.cp_robie
        args = (4.260E+02, -2.508E-01, 4.898E+06, -6.078E+03, 9.244E-05)
        robie_table = """298.15 62.54
300 62.59
400 67.18
500 71.49
600 74.27
700 76.00
800 77.28
900 78.60
1000 80.33
1100 82.76
1200 86.10
1300 90.51
1400 96.12
1500 103.03"""
        table = np.fromstring(robie_table, sep=' ', dtype=float).reshape(len(robie_table.splitlines()), 2).T
        X, Y = table
        sym = f(*args).evalf(subs={T: float(X[0])})
        num = f(*args, float(X[0]))
        ndar = f(*args, X)
        self.assertAlmostEqual(sym, Y[0], places=2)
        self.assertAlmostEqual(num, Y[0], places=2)
        self.assertTrue(np.allclose(ndar, Y, atol=5e-3))

    def test_cp_barin(self):
        """Barin, Knacke, Kubaschewski heat capacity"""
        f = tf.cp_barin
        args = (6.784, -0.478, -0.045, 1.368) # solid Er
        barin_table = """298 6.712
300 6.714
400 6.784
500 6.869
600 6.977
700 7.111
800 7.270
900 7.456
1000 7.669
1100 7.910
1200 8.177
1300 8.472
1400 8.794
1500 9.143
1600 9.520
1700 9.923
1795 10.332"""
        table = np.fromstring(barin_table, sep=' ', dtype=float).reshape(len(barin_table.splitlines()), 2).T
        X, Y = table
        sym = f(*args).evalf(subs={T: float(X[0])}) / 4.184
        num = f(*args, float(X[0])) / 4.184
        ndar = f(*args, X) / 4.184
        self.assertAlmostEqual(sym, Y[0], places=3)
        self.assertAlmostEqual(num, Y[0], places=3)
        self.assertTrue(np.allclose(ndar, Y, atol=5e-4))

    def test_dh298_mks(self):
        """Maier-Kelley-Shomate enthalpy increments"""
        f = tf.dh298_mks
        args = (106.47, 4.23058e-2, 1.1e6) # an artificial example
        X = np.array([298.15, 500, 1000])
        Y = np.array([0, 24223.00086, 101661.0610])
        sym = f(*args).evalf(subs={T: float(X[0])})
        num = f(*args, float(X[0]))
        ndar = f(*args, X)
        self.assertAlmostEqual(sym, Y[0])
        self.assertAlmostEqual(num, Y[0])
        self.assertTrue(np.allclose(ndar, Y))


class TestNasa7(unittest.TestCase):
    pass