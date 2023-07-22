import unittest
import sympy
import numpy as np
import simplethermodynamics.functions as tf

T = sympy.Symbol('T')

class TestFunctions(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main(verbosity=2)