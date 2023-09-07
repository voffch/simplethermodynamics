import numpy as np
import scipy
from scipy import optimize
import numdifftools
import sympy
from sympy import ln, log, Piecewise, Symbol, \
    simplify, lambdify, integrate, diff, sympify
import matplotlib.pyplot as plt
import json
import itertools


T = Symbol('T')
R = 8.314462618 # CODATA 2018 value


class Phase:
    """The basic class for a constant-composition Phase.
    
    This is where the actual thermodynamic functions are stored. This Phase 
    supports only symbolic definitions for the thermodynamic functions, 
    but these symbolic functions are converted (lambdified) to the numpy-
    style functions to allow for the faster numeric computations.
    
    The direct methods - thermodynamic functions of this class (e.g., g, cp) 
    return the lambdified functions. The actual symbolic representations of 
    these thermodynamic functions are stored within the 'symbolic' attribute.

    Attributes:
        name:
            A string representing the name of the phase. Since Phases are 
            supposed to work not by themselves, but from within the respective 
            Compounds, the names of the phases within one Compound should be 
            unique.
        state:
            A string describing the aggregate state of the phase, e.g.,
            solid, liquid or gas. The information regarding the state is
            not used in the computations performed by the Phase methods.
            So, in a way, it is here just for the reference, to assist the 
            further implementations.
        symbolic:
            An instance of the nested SymbolicPhase class. Contains the actual
            symbolic representation of the thermodynamic functions, either 
            calculated and passed to the Phase constructor.
    """
    class SymbolicPhase:
        """The internal class of Phase for the symbolic thermodynamics.
        
        This class is not intended to function on its own and should only be
        used from inside the wrapping Phase class.

        This class is designed to compute the thermodynamic functions that were
        not defined in the constructor on-the-fly when first requested. For
        example, if the Phase was initialized with the Gibbs energy only,
        heat capacity is calculated only upon the first access to the cp 
        attribute.

        For this to work, the actual symbolic functions are stored within the 
        'private' attributes prefixed by _ and should NOT be accessed directly.
        The user access is provided via the properties (those without the 
        underscores).

        For this reason also, the symbolic function definitions should NOT be
        modified in-place, as this will not trigger the recalculation of any
        dependent thermodynamic function and, thus, will result in 'breaking
        the thermodynamics'. Instead, a new phase should be initialized with 
        the modified thermodynamic functions.

        The computations rely on SymPy and are expected to be slow.

        By convention, the measurement units must be: J/mol for the enthalpy 
        and Gibbs energy, K for the temperature, and J/mol/K for the heat 
        capacity and entropy.

        Attributes:
            g:
                Returns the symbolic Gibbs function,
                computing it in case this has not been done yet.
            h:
                Returns the symbolic enthalpy,
                computing it in case this has not been done yet.
            s:
                Returns the symbolic entropy,
                computing it in case this has not been done yet.
            cp:
                Returns the symbolic isobaric heat capacity,
                computing it in case this has not been done yet.
            h298:
                Returns the enthalpy at 298.15 K,
                computing it in case this has not been done yet.
            s298:
                Returns the entropy at 298.15 K,
                computing it in case this has not been done yet.
            _g:
                Symbolic Gibbs function, for internal use only.
            _h:
                Symbolic enthalpy, for internal use only.
            _s:
                Symbolic entropy, for internal use only.
            _cp:
                Symbolic isobaric heat capacity, for internal use only.
            _h298:
                Symbolic enthalpy at 298.15 K, for internal use only.
            _s298:
                Symbolic entropy at 298.15 K, for internal use only.
        """
        def __init__(self, fdict):
            """The constructor for the SymbolicPhase class.
            
            This function initializes the SymbolicPhase with the functions
            given in fdict. For simplicity, this constructor does not check 
            if the thermodynamic functions are self-consistent.

            Args:
                fdict:
                    A dict with the types of thermodynamic functions as keys 
                    and the values as the actual functions.
                    The values must be either numbers, strings that will be 
                    sympified into SymPy expressions, or SymPy expressions 
                    that depend on T.
                    Valid keys:
                    'g': Gibbs function with respect to some reference state,
                        e.g., the Standard Element Reference (SER)
                    'h': enthalpy with respect to some reference state,
                        e.g., the Standard Element Reference (SER)
                    's': absolute entropy
                    'cp': isobaric heat capacity
                    'h298': enthalpy at 298.15 K with respect to some reference 
                        state, e.g., the Standard Element Reference (SER),
                        in which case its value is equal to
                        the standard enthalpy of formation at 298.15 K
                    's298': absolute entropy at 298.15 K

            Returns:
                A new SymbolicPhase instance.

            Raises:
                ValueError:
                    if the information given does not allow deriving all
                    thermodynamic function. In this implementation, 
                    EITHER 'g'
                    OR 'cp', 'h298', 's298'
                    MUST be given in fdict so as not to raise ValueError.
            """
            # The current implementation results in the infinite loop
            # between h and h298 or s and s298 if the phase is initialized with
            # incomplete arguments (e.g. with only cp without h298 and s298).
            # Hence, we're forcing the correct parameter set to be given.
            valid_parameter_sets = [['g'], ['cp', 'h298', 's298']]
            parameters_are_valid = np.any([np.all([p in fdict for p in parameter_set]) for parameter_set in valid_parameter_sets])
            if not parameters_are_valid:
                given_parameters = ', '.join(fdict) if fdict else 'no parameter'
                raise ValueError(f"{given_parameters} given, but EITHER 'g' OR 'cp', 'h298', 's298' must be defined")
            for key in ['g', 'cp', 'h298', 's298', 'h', 's']:
                if key in fdict:
                    value = sympify(fdict[key])
                else:
                    value = None
                setattr(self, '_' + key, value)
                
        @property
        def h(self):
            """Symbolic absolute enthalpy (e.g., with respect to SER)."""
            if self._h is None:
                if self._g is not None:
                    self._h = simplify(-diff(self.g / T, T) * T**2)
                else:
                    self._h = simplify(self.h298 + integrate(self.cp, (T, 298.15, T), heurisch=True))
            return self._h
                
        @property
        def s(self):
            """Symbolic absolute entropy."""
            if self._s is None:
                if self._g is not None:
                    self._s = simplify(-(self.g - self.h) / T)
                else:
                    self._s = simplify(self.s298 + integrate(self.cp / T, (T, 298.15, T), heurisch=True))
            return self._s
            
        @property
        def cp(self):
            """Symbolic isobaric heat capacity."""
            if self._cp is None:
                self._cp = simplify(diff(self.h, T))
            return self._cp
                
        @property
        def g(self):
            """Symbolic Gibbs energy."""
            if self._g is None:
                self._g = simplify(self.h - T*self.s)
            return self._g
        
        @property
        def h298(self):
            """Absolute enthalpy (e.g., with respect to SER) at 298.15 K."""
            if self._h298 is None:
                self._h298 = self.h.evalf(subs={T:298.15})
            return self._h298
        
        @property
        def s298(self):
            """Absolute entropy at 298.15 K."""
            if self._s298 is None:
                self._s298 = self.s.evalf(subs={T:298.15})
            return self._s298
    
    def __init__(self, name, state='undefined', limits=None, **kwargs):
        """The constructor for the Phase class.
        
        Initializes a Phase with the given name and state, and then
        passes the kwargs dict to the SymbolicPhase constructor, which
        returns the symbolic representation of the phase.

        See also the Phase.SymbolicPhase constructor.

        Examples:
            >>> ph = Phase(
                    'Substance', 
                    'solid', 
                    [300, 1000], 
                    cp=100, 
                    h298=-200000, 
                    s298=200
                )

        Args:
            name:
                A string describing the name of the phase.
            state:
                A string describing the aggregate state (e.g., solid or gas).
            limits:
                A 2-element list with [lower, upper] intended 
                temperature limits where the thermodynamic functions for this 
                phase are designed to be meaningful. Currently, this parameter 
                is just for the reference. TODO: display a warning when the 
                user tries to compute a thermodynamic value outside the set 
                limits.
            kwargs:
                Valid keys: g, h, s, cp, s298, h298. These arguments represent
                various thermodynamic functions, defined as either numbers,
                strings (SymPy-fiable ones) or SymPy expressions depending on
                T. This argument is passed directly into the 
                Phase.SymbolicPhase constructor; see its info for more details.

        Returns:
            A new Phase instance.

        Raises:
            AnyError: whatever the SymbolicPhase constructor raises.
        """
        self.name = name
        self.state = state
        self.symbolic = self.SymbolicPhase(kwargs)
        if limits:
            self.limits = limits
        else:
            self.limits = find_phase_limits(self)

    def cp(self, t):
        """Replaces itself with the lambda-function when called."""
        f = lambdify(T, self.symbolic.cp, 'numpy')
        if self.symbolic.cp.is_number:
            self.cp = lambda a : np.full_like(a, f(a))
        else:
            self.cp = f
        return self.cp(t)

    def h(self, t):
        """Replaces itself with the lambda-function when called."""
        f = lambdify(T, self.symbolic.h, 'numpy')
        if self.symbolic.h.is_number:
            self.h = lambda a : np.full_like(a, f(a))
        else:
            self.h = f
        return self.h(t)
    
    def s(self, t):
        """Replaces itself with the lambda-function when called."""
        f = lambdify(T, self.symbolic.s, 'numpy')
        if self.symbolic.s.is_number:
            self.s = lambda a : np.full_like(a, f(a))
        else:
            self.s = f
        return self.s(t)
    
    def g(self, t):
        """Replaces itself with the lambda-function when called."""
        f = lambdify(T, self.symbolic.g, 'numpy')
        if self.symbolic.g.is_number:
            self.g = lambda a : np.full_like(a, f(a))
        else:
            self.g = f
        return self.g(t)
    
    def asdict(self):
        """Returns the dict representation of the Phase.
        
        Args:
            None

        Returns:
            A dict containing the information about the Phase:
            its name, state, and string representations of the
            Gibbs function, enthalpy, entropy and isobaric heat 
            capacity (all functions are symbolic).
        """
        ph_dict = {'name' : self.name, 'state' : self.state, 'limits': self.limits}
        funcs_dict = {prop : str(getattr(self.symbolic, prop)) 
                      for prop in ['g', 'cp', 'h', 's']}
        ph_dict.update(funcs_dict)
        return ph_dict
    
    def table(self, *limits, show=True):
        """Thermodynamic properties of the Phase in tabular form.
        
        Returns or prints (if show == True) the 'spreadsheet string' 
        containing the tabulated standard molar thermodynamic properties, 
        given the specified temperature limits and the interval.

        Standard state: pure solid, liquid and ideal gas at 1 atm.

        The columns are:
        T(K) Cp(J/K) H(J) G(J) S(J/K) H-H298(J).

        Examples:
            >>> phase.table() # use the default limits
            >>> phase.table(300, 520, 20)
            >>> phase.table(300, 520, 20, show = False)
        
        Args:
            limits:
                The temperatures are computed with numpy.arange,
                so the limits are three numbers representing the
                'start', 'stop' and 'step' of numpy.arange.
            show:
                The function prints the resulting table if True,
                and returns it as string - otherwise.

        Returns:
            None if show == True,
            A string containing the output results - otherwise.
        """
        # calculating the values
        if len(limits) < 3:
            limits=(300, 2100, 100)
        t_range = np.arange(*limits)
        values = np.empty((6, t_range.size + 1))
        values[0, 0] = 298.15
        values[0, 1:] = t_range
        for i, func_name in enumerate(['cp', 'h', 'g', 's']):
            func = getattr(self, func_name)
            values[i + 1] = func(values[0])
        values[-1] = values[2] - values[2, 0] # it's h(T) - h298.15(T)
        # converting to string
        # the output table will look very ugly with unrealistic values 
        # (e.g., really huge Cp or S)
        precisions = [2, 3, 1, 1, 3, 1]
        paddings = [5, 4, 9, 9, 4, 9]
        col_sep = '\t'
        mock_factsage_header = f'''
Phase: {self.name}
State: {self.state}
T: {self.limits} K
________{col_sep}________{col_sep}___________{col_sep}___________{col_sep}________{col_sep}___________
  T(K)  {col_sep}Cp(J/K) {col_sep}    H(J)   {col_sep}   G(J)    {col_sep} S(J/K) {col_sep} H-H298(J)
________{col_sep}________{col_sep}___________{col_sep}___________{col_sep}________{col_sep}___________
'''
        values_str = [
            [ np.format_float_positional(x, unique=False, precision=pre, pad_left=pad) for x in col ]
                for col, pre, pad in zip(values, precisions, paddings)
        ]
        values_lines = [col_sep.join(cells) for cells in list(zip(*values_str))]
        table = (mock_factsage_header + '\n'.join(values_lines))
        if show:
            print(table)
        else:
            return table
    
    def plot(self, func='cp', t_min=298.15, t_max=1000, show=True):
        """Plots the specified thermodynamic function of the Phase vs T.
        
        Plots the temperature-dependent values computed with the numeric 
        (lambdified) versions of the thermodynamic functions using the 
        matplotlib.pyplot backend.

        Examples:
            >>> phase.plot('cp', 298.15, 1200)

        Args:
            func:
                A string, which is either 'cp', 'h', 'g', 's', or 'h-h298',
                denoting the function for plotting.
            t_min:
                The lower temperature limit.
            t_max:
                The upper temperature limit.
            show:
                The function displays the plot if True,
                and returns the (fig, axs) tuple otherwise,
                where (fig, axs) are the objects
                returned from plt.subplots().

        Returns:
            None if show == True, the (fig, axs) tuple otherwise.

        Raises:
            ValueError: if trying to plot an invalid function (e.g., 'cv').
        """
        valid_funcs = {
            'cp' : '$C_p$ / $\mathrm{J\cdot mol^{-1}\cdot K^{-1}}$',
            'h' : '$H$ / $\mathrm{J\cdot mol^{-1}}$',
            'g' : '$G$ / $\mathrm{J\cdot mol^{-1}}$',
            's' : '$S$ / $\mathrm{J\cdot mol^{-1}\cdot K^{-1}}$',
            'h-h298' : '$\Delta ^{T}_{298.15}H$ / $\mathrm{J\cdot mol^{-1}}$'
        }
        func = func.lower()
        if func in valid_funcs:
            x = np.linspace(t_min, t_max, 100)
            if func in list(valid_funcs.keys())[:-1]:
                f = getattr(self, func)
            else:
                f = lambda t : self.h(t) - self.h(298.15)
            y = f(x)
            fig, ax = plt.subplots()
            ax.set_xlabel('$T$ / K')
            ax.set_ylabel(valid_funcs[func])
            ax.plot(x, y)
            if show:
                plt.show()
            else:
                return (fig, ax)
        else:
            raise ValueError(f'Cannot plot "{func}"')


def find_phase_limits(phase, default_limits=[298.15, 1e6]):
    """Finds the lower and upper limits of the Gibbs function.

    Outside the returned limits, the Gibbs functions are expected
    to be constant (nonsensical).

    Args:
        phase:
            A Phase instance.
        default_limits:
            The (lower, upper) limits to be returned if no
            respective limit is found inside the Gibbs function
            of the Phase.

    Returns:
        A list containing the [lower, upper] temperature limits.
    """
    limits = list(default_limits)
    expr = phase.symbolic.g
    lambdified = phase.g # lambdify(T, expr, 'numpy')
    delta_t = 0.1
    upper = []
    lower = []
    # traversing the SymPy expression tree, 
    # getting all the Piecewise condition limits
    for arg in sympy.preorder_traversal(expr):
        if isinstance(arg, sympy.core.relational.StrictLessThan) or isinstance(arg, sympy.core.relational.LessThan):
            if arg.lhs == T:
                upper.append(float(arg.rhs))
        elif isinstance(arg, sympy.core.relational.StrictGreaterThan) or isinstance(arg, sympy.core.relational.GreaterThan):
            if arg.lhs == T:
                lower.append(float(arg.rhs))
    # checking whether the limits are "final"
    # i.e. whether the function evaluates to a constant 
    # value outside the limits (i.e., has a zero derivative)
    derivative = numdifftools.Derivative(lambdified, n = 1)
    if lower:
        temp_limit = min(lower)
        if derivative(temp_limit - delta_t) == 0:
            limits[0] = temp_limit
    if upper:
        temp_limit = max(upper)
        if derivative(temp_limit + delta_t) == 0:
            limits[1] = temp_limit
    return limits


def transition_temperatures(phase0, phase1):
    """Finds all possible phase transitions.

    For two Phase instances, tries to obtain all possible transition 
    temperatures (i.e., all temperatures where the Gibbs energies of both
    phases are equal). Uses only the common temperature range where the 
    Gibbs functions of both phases have meaningful values, relying on
    the Phase.limits properties of the respective phases.

    The order in which the phases are listed doesn't matter.
    
    Args:
        phase0:
            A Phase instance.
        phase1:
            A Phase instance.

    Returns:
        A list of dicts {'from': x0, 'to': x1, 'T': x2}, 
        where the 'from' and 'to' values (x0 and x1) are the names 
        of the corresponding phases, and x2 is the transition temperature.

        A single string with the name of the most stable phase (at the minimal 
        common temperature range for both phases) if there's no transition 
        temperature found.

        None if the phases do not share a common temperature range.
    """
    individual_limits = [phase.limits for phase in [phase0, phase1]]
    # finding the intersection of the temperature ranges of both phases
    limits = [max([x[0] for x in individual_limits]), min([x[1] for x in individual_limits])]
    if limits[0] > limits[1]: # the limits ranges do not intersect
        return None
    
    # squaring the Gibbs function difference and finding all the roots between the limits
    fun_squared = lambda t : (phase0.g(t) - phase1.g(t))**2
    prev_len = 0
    while prev_len < len(limits):
        prev_len = len(limits)
        try:
            for i in range(len(limits) - 1):
                res = scipy.optimize.minimize_scalar(fun_squared, bounds=(limits[i], limits[i+1]), method='bounded')
                if res.success and res.fun < 1e-3 and res.x > limits[i] + 0.1 and res.x < limits[i+1] - 0.1:
                    limits.append(res.x)
        except:
            break
        limits.sort()
    
    # finding the stablest phases and defining the transitions
    fun = lambda t : phase0.g(t) - phase1.g(t)
    if len(limits) == 2:
        # no transition detected
        transitions = phase0.name if fun(limits[0]) < 0 else phase1.name
    else:
        transitions = []
        for t in limits[1:-1]:
            diff = numdifftools.Derivative(fun, n=1)(t)
            if diff > 0:
                transitions.append({'from': phase0.name, 'to': phase1.name, 'T': t})
            else:
                transitions.append({'from': phase1.name, 'to': phase0.name, 'T': t})
    
    return transitions


def standard_transitions(phases):
    """Finds the sequence of standard phase transitions upon heating.
    
    This function should determine the sequence of all the standard
    phase transitions occurring between phases upon heating.

    Args:
        phases:
            A dict of Phase instances with the names of the Phases
            as keys (e.g., as stored in Compound.phases).

    Returns:
        A list of dicts [{'t' : temperature_K, 'name' : phase_name}],
        where the phase with phase_name is stable above temperature_K.
        Always returns at least one phase that is stable at the lowest 
        temperature of all the temperature limits.
    """
    all_trans = []
    for phase_pair in itertools.combinations(phases.values(), 2):
        trans = transition_temperatures(*phase_pair)
        if type(trans) != str:
            all_trans.extend(trans)

    names = list(phases.keys())
    t0 = min(phase.limits[0] for phase in phases.values())
    gibbses_t0 = [phase.g(t0) for phase in phases.values()]
    stable_phases = [{'t' : t0, 'name' : names[np.argmin(gibbses_t0)]}]
            
    if len(phases) > 1 and all_trans:
        for i in range(100): # idiotic check, think about changing this
            this_temperature, this_name = stable_phases[-1].values()
            possible_trans = [x for x in all_trans if x['from'] == this_name and x['T'] > this_temperature]
            if possible_trans:
                possible_trans.sort(key = lambda x : x['T'])
                stable_phases.append({'t' : possible_trans[0]['T'], 'name' : possible_trans[0]['to']})
            else:
                break

    return stable_phases


class Compound:
    """The basic class for a constant-composition Compound.
    
    A Compound is a collection of Phases under a common name and chemical
    formula, with an additional stable phase defined for convenience. The
    stable phase is a Phase instance with the Gibbs function defined to 
    take into account all the possible phase transitions occurring in the
    standard conditions (1 atm) between the Phases constituting the Compound.
    Hence, the thermodynamic functions returned by the stable phase instance
    at all T correspond to the phase which is the stablest one (again, in 
    standard conditions only).

    Attributes:
        name:
            An arbitrary name of the Compound. May or may not be the same
            as its chemical formula.
        phases:
            A dict of Phase instances, constituting the Compound,
            with the phase names as keys.
        stable:
            A Phase instance, corresponding to the stable phase.
            This is either a new Phase or just a reference to one of the
            Phases in phases, depending on the compound and comprising
            Phases' thermodynamics.
        transitions:
            A list of dicts [{'t' : temperature_K, 'name' : phase_name}],
            which is the output of the standard_transitions function,
            representing the phase transitions between the phases 
            in the standard conditions.
        transition_thermodynamics:
            A property computing and returning the parameters of the standard
            phase transitions.
        formula:
            A chemical formula of the Compound.
        info:
            Any arbitrary string information accompanying the Compound 
            definition.
    """
    def __init__(self, name, phases, info = "", stable = None, transitions = None, formula = None):
        """Initializes the Compound instance.
        
        Also computes the standard phase transitions and the stable Phase
        (above 298.15 K) if both of these are not passed to the constructor.

        Args:
            name:
                A string describing the Compound. May be anything, even the
                chemical formula.
            phases:
                A list of Phase instances or a dict (with the names of the Phases
                as the keys) of Phase instances constituting the Compound.
            info:
                Any arbitrary string information accompanying the Compound 
                definition.
            stable:
                A Phase object representing the stable phase. If not given,
                it will be constructed from the phases argument.
            transitions:
                A list of dicts [{'t' : temperature_K, 'name' : phase_name}],
                which is the output of the standard_transitions function,
                representing the phase transitions between the phases 
                in the standard conditions. If not given,
                it will be constructed from the phases argument.
            formula:
                A string representing the chemical formula of the Compound.
                If not given, takes the same value as the name.

        Returns:
            A new Compound instance.
        """
        self.name = name
        if not formula:
            self.formula = name
        else:
            self.formula = formula
        if type(phases) == dict:
            self.phases = phases
        else:
            self.phases = {p.name : p for p in phases}
        self.info = info
        if not stable or not transitions:
            # initializing the stable phase
            transitions = standard_transitions(self.phases)
            # TODO: what if the phases do not intersect (i.e., when there's 
            # a temperature range where no phase is defined, but phases exist
            # above and below this range, so there's no standard transition 
            # between the lower- and higher-temperature phases)?
            if len(transitions) == 1:
                # only one stable phase anyway, let's just alias the existing phase
                self.stable = self.phases[transitions[0]['name']]
            else:
                name_cond = []
                for i in range(len(transitions) - 1):
                    name_cond.append((transitions[i]['name'], T <= transitions[i + 1]['t']))
                name_cond.append((transitions[-1]['name'], True))
                expr_cond = [(self.phases[name].symbolic.g, cond) for name, cond in name_cond]
                state_order = ['solid', 'solid,liquid', 'liquid', 'gas']
                key_func = lambda x : state_order.index(x) if x in state_order else 100
                states = sorted(set(self.phases[tr['name']].state for tr in transitions), key=key_func)
                stable_state = ','.join(states)
                used_phases = [self.phases[nc[0]] for nc in name_cond]
                lower_limits = [p.limits[0] for p in used_phases]
                upper_limits = [p.limits[1] for p in used_phases]
                stable_limits = [min(lower_limits), max(upper_limits)]
                self.stable = Phase('stable', stable_state, stable_limits, g=simplify(Piecewise(*expr_cond)))
                # TODO: init other functions, in addition to g, if available (pre-computed) in the self.phases
        else:
            if type(stable) == Phase:
                self.stable = stable
            else:
                # if we're given a name of the existing stable phase, 
                # and not a Phase instance, then we're just aliasing that phase
                try:
                    self.stable = self.phases[stable]
                except:
                    raise ValueError(f'Cannot use "{stable}" as a stable phase (perhaps there\'s no such phase).')
        self.transitions = transitions
        
    def __getitem__(self, key):
        return self.phases[key]
    
    def __len__(self):
        return len(self.phases)
    
    def __str__(self):
        return "Compound '{}' contains {} phases: {}".format( self.name, 
                                                              len(self),
                                                              ', '.join(f'{x.name} ({x.state})' for x in self.phases.values()))
    
    @property
    def symbolic(self):
        """SymbolicPhase of the stable phase."""
        return self.stable.symbolic
    
    def cp(self, t):
        """Isobaric heat capacity of the stable phase."""
        return self.stable.cp(t)
    
    def h(self, t):
        """Absolute enthalpy of the stable phase."""
        return self.stable.h(t)
    
    def s(self, t):
        """Absolute entropy of the stable phase."""
        return self.stable.s(t)
    
    def g(self, t):
        """Gibbs energy of the stable phase."""
        return self.stable.g(t)

    @property
    def transition_thermodynamics(self):
        """Computes the thermodynamic parameters of the phase transitions.

        Args:
            None
        
        Returns:
            A list of dicts containing the thermodynamic parameters of the 
            phase transitions, one dict per transition.
            An empty list if there are no transitions.
        """
        all_transitions = []
        for i in range(len(self.transitions) - 1):
            this_t = self.transitions[i+1]['t']
            phase_from = self.phases[self.transitions[i]['name']]
            phase_to = self.phases[self.transitions[i+1]['name']]
            params = {
                'from' : phase_from.name,
                'to' : phase_to.name,
                't' : this_t,
                't/C' : this_t - 273.15,
                'dh' : phase_to.h(this_t) - phase_from.h(this_t),
                'ds' : phase_to.s(this_t) - phase_from.s(this_t),
                'dcp' : phase_to.cp(this_t) - phase_from.cp(this_t)
            }
            all_transitions.append(params)
        return all_transitions
    
    def table(self, *limits, show=True, transitions_only=False):
        """Thermodynamic properties of the Compound in tabular form.
        
        Returns or prints (if show == True) the 'spreadsheet string' 
        containing the tabulated standard molar thermodynamic properties
        of the stable phase of the Compound.

        In addition, shows the parameters of all standard transitions (if any)
        occurring between the Phases of the Compound.

        Standard state: pure solid, liquid and ideal gas at 1 atm.

        See also Phase.table.

        Examples:
            >>> cmpd.table() # use the default limits
            >>> cmpd.table(300, 520, 20)
            >>> cmpd.table(300, 520, 20, show = False, transitions_only=True)
        
        Args:
            limits:
                The temperatures are computed with numpy.arange,
                so the limits are three numbers representing the
                'start', 'stop' and 'step' of numpy.arange.
            show:
                The function prints the resulting table if True,
                and returns it as string - otherwise.
            transitions_only:
                If True, outputs only the info regarding the phase transitions,
                with no thermodynamic parameters of the stable phase.

        Returns:
            None if show == True,
            A string containing the output results - otherwise.
        """
        header = f'Compound: {self.name}\nFormula: {self.formula}\n{str(self)}\n\n'
        len_transitions = len(self.transitions)
        if len_transitions == 1:
            transition_table = f"The only stable phase is: {self.transitions[0]['name']}"
        else:
            # to compute it only once
            transition_thermodynamics = self.transition_thermodynamics
            descriptions = [f'{x["from"]} -> {x["to"]}' for x in transition_thermodynamics]
            max_description_len = max([len(x) for x in descriptions])
            col_sep = '\t'
            widths = [max_description_len + 1, 8, 8, 12, 8, 8]
            formats = ['', '.2f', '.2f', '.1f', '.3f', '.3f']
            header_formats = ['{:>' + str(w) + '}' for w in widths]
            row_formats = ['{:>' + str(w) + f + '}' for f, w in zip(formats, widths)]
            col_headers = ['Trans', 'T(K)', 'T(C)', 'ΔH(J)', 'ΔS(J/K)', 'ΔCp(J/K)']
            transition_table_rows = [
                'Standard state transitions at 1 atm (per mol):',
                col_sep.join(['_'*x for x in widths]),
                col_sep.join([f.format(x) for f, x in zip(header_formats, col_headers)]),
                col_sep.join(['_'*x for x in widths])
            ]
            for i, trans in enumerate(transition_thermodynamics):
                values = [
                    descriptions[i], 
                    trans['t'],
                    trans['t/C'],
                    trans['dh'],
                    trans['ds'],
                    trans['dcp']
                ]
                row = col_sep.join([f.format(x) for f, x in zip(row_formats, values)])
                transition_table_rows.append(row)
            transition_table = '\n'.join(transition_table_rows)
        table = header + transition_table
        if not transitions_only:
            table += '\n' + self.stable.table(*limits, show=False)
        if show:
            print(table)
        else:
            return table
    
    def plot(self, *args, **kwargs):
        """Plots the specified thermodynamic function of the stable Phase vs T.

        Calls the plot method of the stable phase.
        
        See also the Phase.plot method.
        """
        return self.stable.plot(*args, **kwargs)
    
    def asdict(self):
        """Returns the dict representation of the Compound.
        
        Args:
            None

        Returns:
            A dict containing the information about the Compound
            and all the Phases it contains, including the stable phase.
        """
        if len(self.phases) == 1 or self.stable in self.phases.values():
            stable_phase = self.stable.name
        else:
            stable_phase = self.stable.asdict()
        return {
            'name' : self.name,
            'formula' : self.formula,
            'phases' : [ph.asdict() for ph in self.phases.values()],
            'stable' : stable_phase,
            'transitions' : self.transitions,
            'info' : self.info
        }


class ThermodynamicDatabase:
    """The basic ThermodynamicDatabase class.
    
    This ThermodynamicDatabase is basically a collection (a dict) 
    of the Compound instances, with a few helper methods for 
    storing and loading such collections in or from the json format.

    Getting, setting and deleting the Compounds in the Database can be
    done with the bracket notation, as easy as getting, setting and 
    deleting the dict items.

    Attributes:
        compounds:
            A dict of all the Compounds in the Database, with the
            Compound instances as values and their names as keys.
            The names of the Compounds within one Database must
            be unique.
        info:
            A string containing any additional info to accompany
            the Database.
    """
    def __init__(self, db, info=''):
        """Initializes the ThermodynamicDatabase.
        
        Args:
            db:
                Either of the following:
                - A string - a name of the file containing the json database.
                - A string containing the json database.
                - A list of Compound instances.
                - A dict of Compound instances with their names as keys.
            info:
                An optional info string to accompany the Database.
                This parameter is not used if the database is read 
                from the json file or string.

        Returns:
            A new ThermodynamicDatabase instance.
        """
        self.compounds = {}
        self.info = ''
        if type(db) == str:
            try:
                self.loads(db)
            except:
                # since it's not a valid json string, maybe it's a file name
                self.load(db)
        elif type(db) == dict:
            self.compounds = db
            self.info = info
        elif type(db) == list:
            self.compounds = {cmpd.name : cmpd for cmpd in db}
            self.info = info
                
    def __getitem__(self, key):
        return self.compounds[key]
    
    def __setitem__(self, key, value):
        self.compounds[key] = value
    
    def __delitem__(self, key):
        del self.compounds[key]
    
    def __len__(self):
        return len(self.compounds)
    
    def asdict(self):
        """Returns the dict representation of the ThermodynamicDatabase."""
        return {
            'compounds' : [cmpd.asdict() for cmpd in self.compounds.values()],
            'info' : self.info
        }
    
    def dumps(self):
        """Stores the ThermodynamicDatabase in the json string.
        
        Args:
            None

        Returns:
            A string of the dict representation of this ThermodynamicDatabase
            instance, as returned by json.dumps.
        """
        return json.dumps(self.asdict())
    
    def dump(self, filename):
        """Stores the ThermodynamicDatabase in the json file.
        
        Args:
            filename:
                A valid file name for the write-only access to the json file.

        Returns:
            None
        """
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(self.asdict(), f, ensure_ascii=False)
    
    def loads(self, jsonstr):
        """Initializes the ThermodynamicDatabase from the json string.

        Args:
            jsonstr:
                A valid json string representation of the 
                ThermodynamicDatabase.

        Returns:
            None
        """
        self._json_dict_into_db(json.loads(jsonstr))
    
    def load(self, filename):
        """Initializes the ThermodynamicDatabase from the json file.
        
        Args:
            filename:
                A valid file name for the read-only access to the json file.

        Returns:
            None
        """
        with open(filename, 'r') as f:
            file_dict = json.load(f)
        self._json_dict_into_db(file_dict)
            
    def _json_dict_into_db(self, file_dict):
        """Processes the json-serialized thermodynamic info dict.
        
        Parses the info from the file_dict, stores the results in the 
        current instance of ThermodynamicDatabase.
        
        Args:
            file_dict:
                A compatible dict obtained from the json string or file.

        Returns:
            None
        """
        self.info = file_dict['info']
        for cmpd_dict in file_dict['compounds']:
            cmpd_dict['phases'] = [Phase(**phd) for phd in cmpd_dict['phases']]
            if type(cmpd_dict['stable']) == dict:
                cmpd_dict['stable'] = Phase(**cmpd_dict['stable'])
                # else it's an alias of one of the existing phases
                # which doesn't need to be converted into Phase
                # and should be handled by the Phase constructor
            self[cmpd_dict['name']] = Compound(**cmpd_dict)