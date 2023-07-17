import unittest
import numpy as np
from simplethermodynamics import Phase, Compound, ThermodynamicDatabase

class TestPhase(unittest.TestCase):
    def test_numbers(self):
        '''Phase initialization with functions as numeric values'''
        t = 400
        funcs = ['g', 'cp', 'h', 's']
        known_g_values = {
            0 : [0, 0, 0, 0],
        -1000 : [-1000, 0, -1000, 0]
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

    
    def test_polynomials(self):
        '''Phase initialization with functions as symbolic strings'''
        #g = "Piecewise()"
        pass
        
    def test_einsteins(self):
        '''Phase initialization with functions as symbolic strings'''
        #g = "Piecewise()"
        pass

    def test_incorrect(self):
        '''Phase initialization with incomplete (incorrect) arguments'''
        self.assertRaises(Exception, Phase)
        self.assertRaises(Exception, Phase, 'name_only')
        self.assertRaises(Exception, Phase, 'cp_only', cp=12)

    def test_transitions(self):
        pass

    def test_compound_iron(self):
        pass

    def test_compound_water(self):
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)