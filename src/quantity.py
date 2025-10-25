# quantity.py  — single-file units + quantities with optional NumPy support

# -------- Optional NumPy ------------------------------------------------------
try:
    import numpy as np
except Exception:
    np = None

FLOATING_POINT_EPS = 1.0e-12


# -------- Prefixes ------------------------------------------------------------
class PrefixDefinition:
    def __init__(self, symbol, name, factor):
        self.symbol = symbol
        self.name = name
        self.factor = factor

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return f"PREFIX DEFINITION: Symbol='{self.symbol}', Name='{self.name}', Factor={self.factor:.0e}"


PREFIXES = {
    'y':  PrefixDefinition(symbol='y',  name='yocto', factor=1E-24),
    'z':  PrefixDefinition(symbol='z',  name='zepto', factor=1E-21),
    'a':  PrefixDefinition(symbol='a',  name='atto',  factor=1E-18),
    'f':  PrefixDefinition(symbol='f',  name='femto', factor=1E-15),
    'p':  PrefixDefinition(symbol='p',  name='pico',  factor=1E-12),
    'n':  PrefixDefinition(symbol='n',  name='nano',  factor=1E-09),
    'µ':  PrefixDefinition(symbol='µ',  name='micro', factor=1E-06),
    'u':  PrefixDefinition(symbol='u',  name='micro', factor=1E-06),
    'm':  PrefixDefinition(symbol='m',  name='milli', factor=1E-03),
    'c':  PrefixDefinition(symbol='c',  name='centi', factor=1E-02),
    'd':  PrefixDefinition(symbol='d',  name='deci',  factor=1E-01),
    '':   PrefixDefinition(symbol='',   name='',      factor=1E0),
    'da': PrefixDefinition(symbol='da', name='deca',  factor=1E+01),
    'h':  PrefixDefinition(symbol='h',  name='hecto', factor=1E+02),
    'k':  PrefixDefinition(symbol='k',  name='kilo',  factor=1E+03),
    'M':  PrefixDefinition(symbol='M',  name='mega',  factor=1E+06),
    'G':  PrefixDefinition(symbol='G',  name='giga',  factor=1E+09),
    'T':  PrefixDefinition(symbol='T',  name='tera',  factor=1E+12),
    'P':  PrefixDefinition(symbol='P',  name='peta',  factor=1E+15),
    'E':  PrefixDefinition(symbol='E',  name='exa',   factor=1E+18),
    'Z':  PrefixDefinition(symbol='Z',  name='zetta', factor=1E+21),
    'Y':  PrefixDefinition(symbol='Y',  name='yotta', factor=1E+24)
}


# -------- UnitDefinition ------------------------------------------------------
class UnitDefinition:
    def __init__(self, symbol, name, L=0, M=0, T=0, I=0, THETA=0, N=0, J=0, scale=1.0, bias=0.0):
        self.symbol = symbol
        self.name   = name
        self.L      = L
        self.M      = M
        self.T      = T
        self.I      = I
        self.THETA  = THETA
        self.N      = N
        self.J      = J
        self.scale  = scale
        self.bias   = bias

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return f"UNIT DEFINITION: Symbol='{self.symbol}', Name='{self.name}', Scale={self.scale}, Bias={self.bias}"

    def __eq__(self, other):
        if not isinstance(other, UnitDefinition): return NotImplemented

        def _is_close(a, b): return abs(a - b) <= FLOATING_POINT_EPS

        return (
            _is_close(float(self.L), float(other.L))         and _is_close(float(self.M), float(other.M)) and
            _is_close(float(self.T), float(other.T))         and _is_close(float(self.I), float(other.I)) and
            _is_close(float(self.THETA), float(other.THETA)) and _is_close(float(self.N), float(other.N)) and
            _is_close(float(self.J), float(other.J))         and _is_close(self.scale, other.scale) and
            _is_close(self.bias,  other.bias)
        )

    def __mul__(self, other):
        if not isinstance(other, UnitDefinition): return NotImplemented
        if self.bias != 0 or other.bias != 0:     raise ValueError('Cannot multiply units with non-zero bias (e.g., °C, °F)')
        return UnitDefinition(
            symbol=f'{self.symbol}*{other.symbol}', name=f'{self.name} times {other.name}',
            L = self.L + other.L, M     = self.M     + other.M,     T    = self.T + other.T,
            I = self.I + other.I, THETA = self.THETA + other.THETA, N    = self.N + other.N,
            J = self.J + other.J, scale = self.scale * other.scale, bias = 0.0
        )

    def __truediv__(self, other):
        if not isinstance(other, UnitDefinition): return NotImplemented
        if self.bias != 0 or other.bias != 0:     raise ValueError('Cannot divide units with non-zero bias (e.g., °C, °F)')
        return UnitDefinition(
            symbol=f'{self.symbol}/{other.symbol}', name=f'{self.name} per {other.name}',
            L = self.L - other.L, M     = self.M     - other.M,     T    = self.T - other.T,
            I = self.I - other.I, THETA = self.THETA - other.THETA, N    = self.N - other.N,
            J = self.J - other.J, scale = self.scale / other.scale, bias = 0.0
        )

    def __pow__(self, power):
        try: p = float(power)
        except (TypeError, ValueError): raise ValueError('Power must be a real number.')

        is_one  = abs(p - 1.0) < FLOATING_POINT_EPS
        is_zero = abs(p) < FLOATING_POINT_EPS

        if self.bias != 0 and not (is_one or is_zero):
            raise ValueError('Cannot raise units with non-zero bias (e.g., °C, °F).')

        if is_one:  return self
        if is_zero: return UnitDefinition('1', 'dimensionless', L=0, M=0, T=0, I=0, THETA=0, N=0, J=0, scale=1.0, bias=0.0)

        def fmt_pow(x):
            return str(int(x)) if abs(x - int(x)) < FLOATING_POINT_EPS else str(x)

        base_sym   = self.symbol
        base_name  = self.name
        exp_for_symbol = p

        if '^' in base_sym and ('*' not in base_sym and '/' not in base_sym):
            bsym, e_str = base_sym.split('^', 1)
            try:
                e0 = float(e_str)
                base_sym = bsym
                exp_for_symbol = e0 * p
                if '^' in base_name:
                    bname, _ = base_name.split('^', 1)
                    base_name = bname
            except ValueError:
                pass

        if abs(exp_for_symbol - 1.0) < FLOATING_POINT_EPS:
            new_symbol = base_sym
            new_name   = base_name
        else:
            new_symbol = f'{base_sym}^{fmt_pow(exp_for_symbol)}'
            new_name   = f'{base_name}^{fmt_pow(exp_for_symbol)}'

        return UnitDefinition(
            symbol = new_symbol, name = new_name,
            L = self.L * p, M     = self.M       * p,  T    = self.T * p,
            I = self.I * p, THETA = self.THETA   * p,  N    = self.N * p,
            J = self.J * p, scale = (self.scale ** p), bias = 0.0
        )

    # ---- Public API ----------------------------------------------------------
    def is_same_dimension(self, other):
        if not isinstance(other, UnitDefinition): return NotImplemented
        def _is_close(a, b): return abs(a - b) <= FLOATING_POINT_EPS
        return (_is_close(self.L, other.L) and _is_close(self.M, other.M) and _is_close(self.T, other.T) and
                _is_close(self.I, other.I) and _is_close(self.THETA, other.THETA) and _is_close(self.N, other.N) and
                _is_close(self.J, other.J))

    def is_dimensionless(self):
        return (self.L == 0 and self.M == 0 and self.T == 0 and self.I == 0 and
                self.THETA == 0 and self.N == 0 and self.J == 0)

    def convert_to(self, other, value):
        other = resolve_units(other)
        if not self.is_same_dimension(other): raise ValueError('Units are not compatible (dimension mismatch).')
        # SI_value = scale*(x + bias) — works for scalars and ndarrays
        base_value = self.scale * (value + self.bias)
        return base_value / other.scale - other.bias


# -------- Canonical units & aliases ------------------------------------------
ALIASES = {
    'degC': '°C', 'degF': '°F', 'degR': '°R',
    'inch': 'in', 'inches': 'in',
    'meter': 'm', 'metre': 'm', 'meters': 'm', 'metres': 'm',
    'liter': 'L', 'litre': 'L', 'liters': 'L', 'litres': 'L',
    'ev': 'eV', 'electronvolt': 'eV', 'electron-volt': 'eV',
}

UNITS = {

    # Non-Dimensional 'Units'
    '1':   UnitDefinition('1', 'dimensionless'),
    'deg': UnitDefinition('deg', 'degree', scale=np.pi/180.0),  # dimensionless; converts to radians

    # Base SI units
    'm':   UnitDefinition('m',   'meter',    L=1),
    'kg':  UnitDefinition('kg',  'kilogram', M=1),
    's':   UnitDefinition('s',   'second',   T=1),
    'A':   UnitDefinition('A',   'ampere',   I=1),
    'K':   UnitDefinition('K',   'kelvin',   THETA=1),
    'mol': UnitDefinition('mol', 'mole',     N=1),
    'cd':  UnitDefinition('cd',  'candela',  J=1),

    # Derived SI units
    'Hz':   UnitDefinition('Hz',  'hertz', T=-1),
    'N':    UnitDefinition('N',   'newton', M=1, L=1, T=-2),
    'Pa':   UnitDefinition('Pa',  'pascal', M=1, L=-1, T=-2),
    'J':    UnitDefinition('J',   'joule', M=1, L=2, T=-2),
    'eV':   UnitDefinition('eV', 'electron volt', M=1, L=2, T=-2, scale=1.602176634e-19),
    'W':    UnitDefinition('W',   'watt', M=1, L=2, T=-3),
    'C':    UnitDefinition('C',   'coulomb', T=1, I=1),
    'V':    UnitDefinition('V',   'volt', M=1, L=2, T=-3, I=-1),
    'Ω':    UnitDefinition('Ω',   'ohm', M=1, L=2, T=-3, I=-2),
    'S':    UnitDefinition('S',   'siemens', M=-1, L=-2, T=3, I=2),
    'F':    UnitDefinition('F',   'farad', M=-1, L=-2, T=4, I=2),
    'T':    UnitDefinition('T',   'tesla', M=1, T=-2, I=-1),
    'Wb':   UnitDefinition('Wb',  'weber', M=1, L=2, T=-2, I=-1),
    'H':    UnitDefinition('H',   'henry', M=1, L=2, T=-2, I=-2),
    '°C':   UnitDefinition('°C',  'celsius', THETA=1, bias=273.15),
    'rad':  UnitDefinition('rad', 'radian'),
    'sr':   UnitDefinition('sr',  'steradian'),
    'lm':   UnitDefinition('lm',  'lumen', J=1),
    'lx':   UnitDefinition('lx',  'lux', L=-2, J=1),
    'Bq':   UnitDefinition('Bq',  'becquerel', T=-1),
    'Gy':   UnitDefinition('Gy',  'gray', L=2, T=-2),
    'Sv':   UnitDefinition('Sv',  'sievert', L=2, T=-2),
    'kat':  UnitDefinition('kat', 'katal', T=-1, N=1),
    'L':    UnitDefinition('L', 'liter', L=3, scale=1e-3),
    'P':    UnitDefinition('P', 'poise', M=1, L=-1, T=-1, scale=1e-1),

    # Customary US Unit System
    '°F':   UnitDefinition('°F', 'fahrenheit', THETA=1, bias=459.67, scale=5/9),
    '°R':   UnitDefinition('°R', 'rankine', THETA=1, bias=0, scale=5/9),
    'gal':  UnitDefinition('gal', 'gallon', L=3, bias=0, scale=0.00378541),

    'psi':      UnitDefinition('psi', 'psi', M=1, L=-1, T=-2, scale=6894.76),
    'lb':       UnitDefinition('lb', 'pound', M=1, scale=0.453592),
    'lbf':      UnitDefinition('lbf', 'lbf', M=1, L=1, T=-2, scale=4.44822),
    'thou':     UnitDefinition('th', 'thou', L=1, scale=2.54E-5),
    'in':       UnitDefinition('in', 'inch', L=1, scale=2.54E-2),
    'ft':       UnitDefinition('ft', 'foot', L=1, scale=3.048E-1),
    'yard':     UnitDefinition('yd', 'yard', L=1, scale=9.144E-1),
    'chain':    UnitDefinition('ch', 'chain', L=1, scale=20.1168),
    'furlong':  UnitDefinition('fur', 'furlong', L=1, scale=201.168),
    'mile':     UnitDefinition('mi', 'mile', L=1, scale=1609.344),
    'league':   UnitDefinition('lea', 'league', L=1, scale=4828.032),
    'BTU':      UnitDefinition('BTU', 'btu', M=1, L=2, T=-2, scale=1055.06),

    'ΔK':  UnitDefinition('ΔK',  'kelvin difference',      THETA=1, scale=1.0,     bias=0.0),
    'Δ°C': UnitDefinition('Δ°C', 'celsius difference',     THETA=1, scale=1.0,     bias=0.0),
    'Δ°F': UnitDefinition('Δ°F', 'fahrenheit difference',  THETA=1, scale=5.0/9.0, bias=0.0),
    'Δ°R': UnitDefinition('Δ°R', 'rankine difference',     THETA=1, scale=5.0/9.0, bias=0.0),

    # Miscellaneous units
    'Torr': UnitDefinition('Torr', 'Torr', M=1, L=-1, T=-2, scale=133.322),
    'bar':  UnitDefinition('bar', 'bar', M=1, L=-1, T=-2, scale=1E5),
    'min':  UnitDefinition('min', 'minute', T=1, scale=60),
    'h':    UnitDefinition('h', 'hour', T=1, scale=3600),
}


# -------- Unit parser ---------------------------------------------------------
def normalize_units(symbol):
    if not isinstance(symbol, str): raise TypeError('Normalize expects a string.')
    symbol = ' '.join(symbol.split())  # trim + collapse spaces
    return symbol.replace(' ', '')

class UnitParser:
    def __init__(self, unit_str):
        if not isinstance(unit_str, str): raise TypeError('UnitParser expects a string.')
        self.unit_str = unit_str

    def __str__(self):
        return self.unit_str

    def __repr__(self):
        return f"UNIT PARSER: Unit_str = '{self.unit_str}'."

    def parse(self):
        s = normalize_units(self.unit_str)
        if not s: raise ValueError('Empty Unit String')

        parts = self._split_outside_parens(s, '/')
        numerator, denominators = parts[0], parts[1:]

        parsed_units = self._parse_product(numerator)
        for d in denominators:
            for unit in self._parse_product(d):
                parsed_units.append(unit ** -1)

        if not parsed_units: return resolve_units('1')

        result = parsed_units[0]
        for u in parsed_units[1:]:
            result = result * u

        return result

    def _split_outside_parens(self, s, sep):
        parts, buf, depth = [], [], 0
        for ch in s:
            if ch == '(':
                depth += 1
                buf.append(ch)
            elif ch == ')':
                depth = max(0, depth - 1)
                buf.append(ch)
            elif ch == sep and depth == 0:
                parts.append(''.join(buf))
                buf = []
            else:
                buf.append(ch)
        parts.append(''.join(buf))
        return [p for p in parts if p]

    def _parse_product(self, units_str):
        units = [u.strip() for u in units_str.split('*') if u.strip()]
        return [self._parse_token(u) for u in units]

    def _parse_token(self, token):
        token = token.strip()
        if '^' not in token:
            return self._parse_base(token)

        base, exp = token.split('^', 1)
        base = base.strip()
        exp  = exp.strip()

        # allow '(1/2)'
        if exp.startswith('(') and exp.endswith(')'):
            exp = exp[1:-1].strip()

        if '/' in exp:  # fraction like 3.5/2 or 1/2.0
            num, den = exp.split('/', 1)
            power = float(num.strip()) / float(den.strip())
        else:
            power = float(exp)

        return self._parse_base(base) ** power

    def _parse_base(self, unit_s):
        unit_s = unit_s.strip()

        try:
            return resolve_units(unit_s)
        except ValueError:
            pass

        for prefix in sorted(PREFIXES.keys(), key=len, reverse=True):
            if not prefix:
                continue
            if unit_s.startswith(prefix):
                base_sym = unit_s[len(prefix):]
                try:
                    base = resolve_units(base_sym)
                except ValueError:
                    continue

                if base.bias != 0.0:
                    raise ValueError("Cannot apply SI prefix '{}' to affine unit '{}'.".format(prefix, base.symbol))

                fac = PREFIXES[prefix].factor
                return UnitDefinition(
                    symbol=prefix + base.symbol,
                    name=(PREFIXES[prefix].name + ' ' + base.name) if base.name else prefix + base.symbol,
                    L=base.L, M=base.M, T=base.T, I=base.I, THETA=base.THETA, N=base.N, J=base.J,
                    scale=fac * base.scale,
                    bias=base.bias
                )

        raise ValueError("Unit '{}' not recognized".format(unit_s))






# -------- Quantities (with NumPy support) ------------------------------------
class Quantity:
    """
    Numeric value + UnitDefinition, with safe arithmetic.

    Rules:
      - Addition/Subtraction require dimension match.
      - Affine (bias != 0) is special:
          * affine + affine  -> ERROR (meaningless)
          * affine + delta   -> affine
          * delta  + affine  -> affine
          * affine - affine  -> delta
          * affine - delta   -> affine
          * delta  - absolute-> ERROR
      - Multiplication/Division follow UnitDefinition dunders (already guard affine).
    """
    __slots__ = ('name', 'val', 'unit', 'desc')
    __array_priority__ = 2000  # ensure we win over ndarray in mixed ops

    def __init__(self, name='', i_val=None, i_unit=None, desc=None):
        if i_unit is None: raise TypeError("Quantity requires i_unit.")
        self.name = '' if name is None else name
        self.val  = i_val
        self.unit = resolve_units(i_unit)  # expects a str or UnitDefinition
        self.desc = desc


    # ---- display -------------------------------------------------------------
    def __str__(self):
        return f"{self.val} {self.unit.symbol}" if not self.unit.is_dimensionless() else f"{self.val}"

    def __repr__(self):
        return f"Quantity(val={self.val}, unit={self.unit!s})"

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _np(x):
        return np.asarray(x) if np is not None else x

    def _as_SI(self):
        # SI_value = scale*(value + bias)
        return self.unit.scale * (self._np(self.val) + self.unit.bias)

    @staticmethod
    def _is_affine(u):
        return abs(u.bias) > FLOATING_POINT_EPS

    @staticmethod
    def _temp_kind(u):
        if u.THETA == 0: return 'linear'
        if u.symbol.startswith('Δ') or ('difference' in (u.name or '')): return 'delta'
        return 'absolute'

    @staticmethod
    def _delta_unit_for(u):
        mapping = {'K': 'ΔK', '°C': 'Δ°C', '°F': 'Δ°F', '°R': 'Δ°R'}
        sym = u.symbol
        return UNITS.get(mapping.get(sym, 'ΔK'))

    # ---- conversions ---------------------------------------------------------
    def to(self, target):
        u_to = resolve_units(target)
        if not self.unit.is_same_dimension(u_to):
            raise ValueError('Cannot convert: dimension mismatch.')
        new_val = self.unit.convert_to(u_to, self._np(self.val))
        return Quantity(i_val=new_val, i_unit=u_to, name=self.name, desc=self.desc)

    @property
    def SI(self):
        return self.to_SI_base()

    def as_SI(self):
        q = self.to_SI_base()
        return q.val, q.unit.symbol

    def to_SI_base(self):
        val_si = self._as_SI()
        u = self.unit

        def fmt_exp(x):
            xi = int(round(x))
            return str(xi) if abs(x - xi) < FLOATING_POINT_EPS else str(x)

        parts = []

        def add(sym, exp):
            if abs(exp) < FLOATING_POINT_EPS:
                return
            if abs(exp - 1.0) < FLOATING_POINT_EPS:
                parts.append(sym)
            else:
                parts.append(f"{sym}^{fmt_exp(exp)}")

        temp_kind = self._temp_kind(u)

        add('kg', u.M)
        add('m',  u.L)
        add('s',  u.T)
        add('A',  u.I)

        if abs(u.THETA) >= FLOATING_POINT_EPS:
            sym_T = 'ΔK' if (temp_kind == 'delta' and abs(u.THETA - 1.0) < FLOATING_POINT_EPS) else 'K'
            add(sym_T, u.THETA)

        add('mol', u.N)
        add('cd',  u.J)

        symbol = '*'.join(parts) if parts else '1'

        base_unit = UnitDefinition(
            symbol=symbol, name='SI base',
            L=u.L, M=u.M, T=u.T, I=u.I, THETA=u.THETA, N=u.N, J=u.J,
            scale=1.0, bias=0.0
        )

        return Quantity(i_val=val_si, i_unit=base_unit, name=self.name, desc=self.desc)


    # ---- add/sub -------------------------------------------------------------
    def _addsub(self, other, op):
        if not isinstance(other, Quantity):
            return NotImplemented
        if not self.unit.is_same_dimension(other.unit):
            raise ValueError('Cannot add/subtract: dimension mismatch.')

        ak = self._temp_kind(self.unit)
        bk = self._temp_kind(other.unit)

        if ak == 'linear' and bk == 'linear':
            rhs = other.to(self.unit).val
            v = (self._np(self.val) + self._np(rhs)) if op == '+' else (self._np(self.val) - self._np(rhs))
            return Quantity(i_val=v, i_unit=self.unit)

        if ak == 'delta' and bk == 'delta':
            rhs = other.to(self.unit).val
            v = (self._np(self.val) + self._np(rhs)) if op == '+' else (self._np(self.val) - self._np(rhs))
            return Quantity(i_val=v, i_unit=self.unit)

        if op == '+':
            if ak == 'absolute' and bk == 'absolute':
                raise ValueError('Adding two absolute temperatures is undefined.')
            # absolute + delta  -> absolute
            if ak == 'absolute' and bk == 'delta':
                dunit = self._delta_unit_for(self.unit)
                inc = other.unit.convert_to(dunit, self._np(other.val))
                return Quantity(i_val=self._np(self.val) + inc, i_unit=self.unit)

            if ak == 'delta' and bk == 'absolute':
                dunit = self._delta_unit_for(other.unit)
                inc = self.unit.convert_to(dunit, self._np(self.val))
                return Quantity(i_val=other._np(other.val) + inc, i_unit=other.unit)
        else:
            if ak == 'absolute' and bk == 'absolute':
                si_diff = self._as_SI() - other._as_SI()
                dunit = self._delta_unit_for(self.unit)
                return Quantity(i_val=si_diff / dunit.scale, i_unit=dunit)
            # absolute + delta  -> absolute
            if ak == 'absolute' and bk == 'delta':
                dunit = self._delta_unit_for(self.unit)
                inc = other.unit.convert_to(dunit, self._np(other.val))
                return Quantity(i_val=self._np(self.val) + inc, i_unit=self.unit)

            if ak == 'delta' and bk == 'absolute':
                raise ValueError('delta - absolute temperature is undefined.')

        raise ValueError('Unsupported quantity addition/subtraction case.')

    def __add__(self, other): return self._addsub(other, '+')
    def __sub__(self, other): return self._addsub(other, '-')

    # ---- mul/div -------------------------------------------------------------
    def __mul__(self, other):
        if isinstance(other, (int, float)) or (np is not None and np.isscalar(other)):
            return Quantity(i_val=self._np(self.val) * float(other), i_unit=self.unit)
        if isinstance(other, Quantity):
            return Quantity(i_val=self._np(self.val) * self._np(other.val), i_unit=self.unit * other.unit)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)) or (np is not None and np.isscalar(other)):
            return Quantity(i_val=float(other) * self._np(self.val), i_unit=self.unit)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)) or (np is not None and np.isscalar(other)):
            return Quantity(i_val=self._np(self.val) / float(other), i_unit=self.unit)
        if isinstance(other, Quantity):
            return Quantity(i_val=self._np(self.val) / self._np(other.val), i_unit=self.unit / other.unit)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)) or (np is not None and np.isscalar(other)):
            return Quantity(i_val=float(other) / self._np(self.val), i_unit=self.unit ** -1)
        return NotImplemented

    # ---- power ---------------------------------------------------------------
    def __pow__(self, p):
        try:
            pf = float(p)
        except (TypeError, ValueError):
            raise ValueError('Power must be numeric.')
        return Quantity(i_val=self._np(self.val) ** pf, i_unit=self.unit ** pf)

    # ---- comparisons ---------------------------------------------------------
    def _cmp(self, other, op):
        import operator as _op
        ops = {'==': _op.eq, '!=': _op.ne, '<': _op.lt, '<=': _op.le, '>': _op.gt, '>=': _op.ge}

        if isinstance(other, Quantity):
            # temperature semantics: don't compare absolute vs delta
            ak = self._temp_kind(self.unit)
            bk = self._temp_kind(other.unit)
            if (ak == 'absolute' and bk == 'delta') or (ak == 'delta' and bk == 'absolute'):
                raise ValueError('Cannot compare absolute temperature to a temperature difference.')

            if not self.unit.is_same_dimension(other.unit):
                raise ValueError('Dimension mismatch in comparison.')

            rhs = other.to(self.unit).val
            return ops[op](self._np(self.val), self._np(rhs))

        # allow comparing with bare numbers only if dimensionless
        if isinstance(other, (int, float)) or (np is not None and np.isscalar(other)):
            if not self.unit.is_dimensionless():
                raise ValueError('Cannot compare a dimensioned quantity with a bare number.')
            return ops[op](self._np(self.val), float(other))

        return NotImplemented

    def __eq__(self, other): return self._cmp(other, '==')
    def __ne__(self, other): return self._cmp(other, '!=')
    def __lt__(self, other): return self._cmp(other, '<')
    def __le__(self, other): return self._cmp(other, '<=')
    def __gt__(self, other): return self._cmp(other, '>')
    def __ge__(self, other): return self._cmp(other, '>=')



    # ---- casting / array interop --------------------------------------------
    def __float__(self):
        if not self.unit.is_dimensionless():
            raise TypeError('Only dimensionless quantities can be cast to float.')
        return float(self._np(self.val))

    def __array__(self, dtype=None):
        # prevent silent unit loss
        raise TypeError("Explicitly extract magnitudes before passing to NumPy (e.g., q.val or q.to('unit').val).")

    # ---- NumPy ufunc protocol ------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if np is None:
            return NotImplemented
        if method != '__call__':
            return NotImplemented

        mags = []
        units = []
        for x in inputs:
            if isinstance(x, Quantity):
                mags.append(self._np(x.val))
                units.append(x.unit)
            else:
                mags.append(x)
                units.append(None)

        def all_same_dim(u_list):
            base = None
            for uu in u_list:
                if uu is None: continue
                if base is None: base = uu
                elif not uu.is_same_dimension(base): return False
            return True

        # ----- handle ufuncs -----
        u = None

        if ufunc in (np.add, np.subtract):
            # all Quantity args must share dimension
            if not all_same_dim(units):
                raise ValueError('Dimension mismatch in add/sub.')
            # reference unit = first Quantity's unit
            ref = next((uu for uu in units if uu is not None), UNITS['1'])
            conv = []
            for m, uu in zip(mags, units):
                if uu is None:
                    if not ref.is_dimensionless():
                        raise ValueError('Adding dimensioned and bare numbers is not allowed.')
                    conv.append(m)
                else:
                    conv.append(uu.convert_to(ref, m))
            mags = conv
            u = ref

        elif ufunc in (np.multiply,):
            # combine multiplicatively
            u = None
            for uu in units:
                if uu is None: continue
                u = uu if u is None else (u * uu)

        elif ufunc in (np.true_divide, np.divide, np.floor_divide):
            # assume binary for simplicity
            if len(units) != 2:
                raise ValueError('Only binary division is supported here.')
            a_u, b_u = units
            if a_u is None and b_u is None:
                u = UNITS['1']
            elif a_u is None:
                u = UNITS['1'] / b_u
            elif b_u is None:
                u = a_u
            else:
                u = a_u / b_u

        elif ufunc is np.power:
            base_u, exp_u = units
            if exp_u is not None:
                raise ValueError('Exponent cannot carry units.')
            exp = mags[1]
            if np.ndim(exp) != 0:
                raise ValueError('Non-scalar exponent not supported for units.')
            u = base_u ** float(exp)

        elif ufunc in (np.sqrt,):
            u = units[0] ** 0.5

        elif ufunc in (np.negative, np.positive, np.absolute, np.sign):
            u = units[0]

        elif ufunc in (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal):
            # convert RHS to LHS units if needed, return boolean array
            a_u, b_u = units
            if a_u is not None and b_u is not None:
                if not a_u.is_same_dimension(b_u):
                    raise ValueError('Dimension mismatch in comparison.')
                mags[1] = b_u.convert_to(a_u, mags[1])
            result = getattr(ufunc, method)(*mags, **kwargs)
            return result

        else:
            # trig/exp/log etc. -> require dimensionless inputs
            for uu in units:
                if uu is not None and not uu.is_dimensionless():
                    raise ValueError('This ufunc requires dimensionless input.')
            u = UNITS['1']

        result = getattr(ufunc, method)(*mags, **kwargs)
        return Quantity(i_val=result, i_unit=u)

    # ---- NumPy high-level ops: intentionally not overridden ------------------
    @classmethod
    def __array_function__(cls, *args, **kwargs):
        # Defer high-level functions to user-space helpers / instance methods.
        return NotImplemented

    # ---- reducers (vector-friendly, NumPy optional) --------------------------
    def sum(self, axis=None, dtype=None, keepdims=False):
        if np is None:
            # Pure-Python fallback
            v = self.val
            if isinstance(v, (list, tuple)):
                total = 0.0
                for x in v:
                    total += x
                return Quantity(i_val=total, i_unit=self.unit)       # pure-Python list case
            # Scalar -> just return self
            return Quantity(i_val=self.val, i_unit=self.unit)    # scalar fallback

        arr = np.asarray(self.val)
        return Quantity(i_val=np.sum(arr, axis=axis, dtype=dtype, keepdims=keepdims), i_unit=self.unit)

    def mean(self, axis=None, dtype=None, keepdims=False):
        # Disallow means of affine (absolute) temperatures
        if abs(self.unit.bias) > FLOATING_POINT_EPS:
            raise ValueError('Mean over affine units (e.g., °C, °F) is not allowed.')

        if np is None:
            v = self.val
            if isinstance(v, (list, tuple)):
                return Quantity(i_val=sum(v) / float(len(v)), i_unit=self.unit)   # pure-Python list
            # Scalar -> mean is itself
            return Quantity(i_val=self.val, i_unit=self.unit)                 # scalar

        arr = np.asarray(self.val)
        return Quantity(i_val=np.mean(arr, axis=axis, dtype=dtype, keepdims=keepdims), i_unit=self.unit)




# -------- Module-level helpers -----------------------------------------------
def validate_aliases():
    for alias, target in ALIASES.items():
        if alias in UNITS:      raise RuntimeError(f"Alias '{alias}' collides with a canonical unit.")
        if target not in UNITS: raise RuntimeError(f"Alias '{alias}' points to unknown target '{target}'.")

def resolve_units(obj):
    if isinstance(obj, UnitDefinition): return obj
    if not isinstance(obj, str):        raise TypeError('resolve_units expects a string or UnitDefinition.')
    symbol = normalize_units(obj)
    symbol = ALIASES.get(symbol, symbol)
    if symbol in UNITS: return UNITS[symbol]
    try:
        return UnitParser(symbol).parse()
    except Exception:
        raise ValueError(f'Unknown unit symbol or expression: {obj}.')

def convert(value, from_unit, to_unit):
    u_from = resolve_units(from_unit)
    u_to   = resolve_units(to_unit)
    return u_from.convert_to(u_to, value)


# -------- Self-test -----------------------------------------------------------
if __name__ == '__main__':
    EPS = 1e-12

    # -------------------- Unit parsing & algebra --------------------
    assert UnitParser('N/m^2').parse() == UNITS['Pa']
    assert UnitParser('kg*m/s^2').parse() == UNITS['N']
    assert abs(UnitParser(' m ^ ( 1 / 2 ) ').parse().L - 0.5) < EPS
    try:
        UnitParser('m°C').parse(); raise AssertionError('prefix on affine should fail')
    except ValueError:
        pass
    assert abs(UnitParser('m^(3.5/2)').parse().L - 1.75) < EPS

    # -------------------- Conversions (affine & delta) ---------------
    q = Quantity(i_val=32, i_unit='°F').to('K')
    assert abs(q.val - 273.15) < 1e-12

    # 1/s from scalar division
    assert (2 / Quantity(i_val=4, i_unit='s')).unit == (UNITS['s'] ** -1)

    # Dimension mismatch should fail
    try:
        _ = Quantity(i_val=1, i_unit='m') + Quantity(i_val=1, i_unit='s')
        raise AssertionError('adding different dimensions should raise')
    except ValueError:
        pass

    # Affine + delta -> affine; affine - affine -> delta
    C  = Quantity(i_val=25, i_unit='°C')
    dC = Quantity(i_val=10, i_unit='Δ°C')
    assert (C + dC).unit == UNITS['°C']

    d1 = Quantity(i_val=300, i_unit='K') - Quantity(i_val=25, i_unit='°C')   # -> ΔK
    assert d1.unit == UNITS['ΔK']

    # -------------------- SI base conversion examples ----------------
    q_in = Quantity(i_val=10, i_unit='in').to_SI_base()
    assert q_in.unit.symbol == 'm'
    assert abs(q_in.val - 0.254) < 1e-12

    q_lbf = Quantity(i_val=1, i_unit='lbf').to_SI_base()
    assert 'kg' in q_lbf.unit.symbol and 'm' in q_lbf.unit.symbol and 's^-2' in q_lbf.unit.symbol
    assert abs(q_lbf.val - 4.44822) < 1e-6

    qC = Quantity(i_val=25, i_unit='°C').to_SI_base()
    assert qC.unit.symbol == 'K' and abs(qC.val - 298.15) < 1e-12

    qdF = Quantity(i_val=10, i_unit='Δ°F').to_SI_base()
    assert qdF.unit.symbol == 'ΔK' and abs(qdF.val - (10 * 5.0/9.0)) < 1e-12

    qpsi = Quantity(i_val=1, i_unit='psi').to_SI_base()
    assert 'kg' in qpsi.unit.symbol and 'm^-1' in qpsi.unit.symbol and 's^-2' in qpsi.unit.symbol

    # -------------------- Basic arithmetic --------------------------
    a = Quantity(i_val=2, i_unit='m')
    b = Quantity(i_val=3, i_unit='m')
    c = Quantity(i_val=4, i_unit='s')

    s = a + b
    assert s.unit == UNITS['m'] and s.val == 5

    p = a * b
    assert p.unit == (UNITS['m'] * UNITS['m']) and p.val == 6

    v = a / c
    assert v.unit == (UNITS['m'] / UNITS['s']) and v.val == 0.5

    sq = a ** 2
    assert sq.unit == (UNITS['m'] ** 2) and sq.val == 4

    # -------------------- NumPy interop (if available) ---------------
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        A = Quantity(i_val=[1, 2, 3], i_unit='m')
        B = Quantity(i_val=10, i_unit='cm')
        S = A + B                               # auto-convert B -> m
        assert np.allclose(S.val, np.array([1.1, 2.1, 3.1]))

        F = Quantity(i_val=[1, 2, 3], i_unit='N')
        Area = Quantity(i_val=[0.5, 1.0, 1.5], i_unit='m^2')
        P = F / Area
        assert P.unit == UNITS['Pa']
        assert np.allclose(P.val, np.array([2.0, 2.0, 2.0]))

        # Comparison broadcasting / unit conversion
        cmp = (Quantity(i_val=[100, 110], i_unit='cm') > Quantity(i_val=1, i_unit='m'))
        assert np.all(cmp == np.array([False, True]))

        # Mean should return a Quantity with same unit
        M = Quantity(i_val=[1, 3], i_unit='m').mean()
        assert M.unit == UNITS['m'] and M.val == 2

        # Trig with angles
        x  = Quantity(i_val=[0, np.pi/6, np.pi/2], i_unit='rad')
        y  = np.sin(x)
        assert y.unit == UNITS['1']

        xd = Quantity(i_val=[0, 30, 90], i_unit='deg')
        yd = np.sin(xd)
        # (values ≈ [0, 0.5, 1.0]; unit should be dimensionless)
        assert yd.unit == UNITS['1']

    print("All __main__ tests executed.")
    
    
    
