import unittest
import sympy
import numpy as np
import json
import os
import simplethermodynamics.functions as tf

T = sympy.Symbol('T')

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
        calc_sym = tf.phi_gurvich(*self.coeffs).evalf(subs={T: self.t_ndar[0]})
        self.assertAlmostEqual(calc_sym, data[0], places = 2)

    def test_cp(self):
        """Gurvich heat capacity"""
        data = np.array(self.table['cp'][self.t_slice], dtype=float)
        calc = tf.cp_gurvich(*self.coeffs, self.t_ndar)
        self.assertTrue(np.allclose(data, calc, atol=5e-04))
        calc_sym = tf.cp_gurvich(*self.coeffs).evalf(subs={T: self.t_ndar[0]})
        self.assertAlmostEqual(calc_sym, data[0], places = 2)

    def test_s(self):
        """Gurvich entropy"""
        data = np.array(self.table['s'][self.t_slice], dtype=float)
        calc = tf.s_gurvich(*self.coeffs, self.t_ndar)
        self.assertTrue(np.allclose(data, calc, atol=5e-04))
        calc_sym = tf.s_gurvich(*self.coeffs).evalf(subs={T: self.t_ndar[0]})
        self.assertAlmostEqual(calc_sym, data[0], places = 2)

    def test_dh0(self):
        """Gurvich enthalpy increments"""
        data = np.array(self.table['ht-h0'][self.t_slice], dtype=float)
        calc = tf.dh0_gurvich(*self.coeffs, self.t_ndar) / 1000 # in kJ/mol
        self.assertTrue(np.allclose(data, calc, atol=1e-03))
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
        highest = tf.h_gurvich(*self.high_t_coeffs, 
                            self.dhf298, 
                            self.h298, 
                            self.highest_t) / 1000
        self.assertAlmostEqual(highest, self.highest_h, places = 2)
        calc_sym = tf.h_gurvich(*self.coeffs,
                                self.dhf298, 
                                self.h298).evalf(subs={T: self.t_ndar[0]}) / 1000
        self.assertAlmostEqual(calc_sym, data[0], places = 2)

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
        calc_sym = tf.g_gurvich(*self.coeffs,
                                self.dhf298, 
                                self.h298).evalf(subs={T: self.t_ndar[0]}) / 1000
        self.assertAlmostEqual(calc_sym, data[0], places = 2)
        highest = tf.g_gurvich(*self.high_t_coeffs, 
                            self.dhf298, 
                            self.h298, 
                            self.highest_t) / 1000
        self.assertAlmostEqual(highest, self.highest_g, places = 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)