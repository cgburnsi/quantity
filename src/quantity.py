
FLOATING_POINT_EPS   = 1.0e-12


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
            _is_close(self.bias,  other.bias))
    
    def __mul__(self, other):
        if not isinstance(other, UnitDefinition): return NotImplemented
        if self.bias != 0 or other.bias != 0:     raise ValueError('Cannot multiply units with non-zero bias (e.g., °C, °F)')
        return UnitDefinition(
            symbol=f'{self.symbol}*{other.symbol}', name=f'{self.name} times {other.name}',
            L = self.L + other.L, M     = self.M     + other.M,     T    = self.T + other.T,
            I = self.I + other.I, THETA = self.THETA + other.THETA, N    = self.N + other.N,
            J = self.J + other.J, scale = self.scale * other.scale, bias = 0.0)

    def __truediv__(self, other):
        if not isinstance(other, UnitDefinition): return NotImplemented
        if self.bias != 0 or other.bias != 0:     raise ValueError('Cannot divide units with non-zero bias (e.g., °C, °F)')
        return UnitDefinition(
            symbol=f'{self.symbol}/{other.symbol}', name=f'{self.name} per {other.name}',
            L = self.L - other.L, M     = self.M     - other.M,     T    = self.T - other.T,
            I = self.I - other.I, THETA = self.THETA - other.THETA, N    = self.N - other.N,
            J = self.J - other.J, scale = self.scale / other.scale, bias = 0.0)
    
    def __pow__(self, power):
        # Verify the power exponent is a real number
        try: p = float(power)
        except (TypeError, ValueError): raise ValueError('Power must be a real number.')
        
        # Check if exponent is 1 or 0
        is_one  = abs(p - 1.0) < FLOATING_POINT_EPS
        is_zero = abs(p) < FLOATING_POINT_EPS
        
        # affine units (non-zero bias) cannot be exponentiated, except ^1 (no-op) or ^0 (dimensionless)
        if self.bias != 0 and not (is_one or is_zero): raise ValueError('Cannot raise units with non-zero bias (e.g., °C, °F).')
        
        # u^1 → u (preserve original)
        if is_one: return self
        
        # u^0 → dimensionless
        if is_zero: return UnitDefinition('1', 'dimensionless', L=0, M=0, T=0, I=0, THETA=0, N=0, J=0, scale=1.0, bias=0.0)

        # This just makes the presentation a bit prettier (1 instead of 1.0, etc.)
        def fmt_pow(x):
            return str(int(x)) if abs(x - int(x)) < FLOATING_POINT_EPS else str(x)
        
        # only try to flatten when the current symbol is a simple base with a single ^e
        base_sym = self.symbol
        base_name = self.name
        exp_for_symbol = p
        
        if '^' in base_sym and ('*' not in base_sym and '/' not in base_sym):
            bsym, e_str = base_sym.split('^', 1)
            try:
                e0 = float(e_str)
                base_sym = bsym
                exp_for_symbol = e0 * p  # combine the exponents
                # do the same for the name if it follows the same pattern
                if '^' in base_name:
                    bname, _ = base_name.split('^', 1)
                    base_name = bname
            except ValueError:
                # if we can’t parse the old exponent, just fall back to the default behavior
                pass
    
        # build pretty symbol (avoid ^1)
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
            J = self.J * p, scale = (self.scale ** p), bias = 0.0)

    # ---------- Private Helper Methods ------------------------------------------------------------------------------------------

        
    # ---------- Public API ------------------------------------------------------------------------------------------------------
    def is_same_dimension(self, other):
        if not isinstance(other, UnitDefinition): return NotImplemented
        
        def _is_close(a, b): return abs(a - b) <= FLOATING_POINT_EPS    

        return (
            _is_close(self.L, other.L) and _is_close(self.M,     other.M)     and _is_close(self.T, other.T) and
            _is_close(self.I, other.I) and _is_close(self.THETA, other.THETA) and _is_close(self.N, other.N) and
            _is_close(self.J, other.J))    
    
    def is_dimensionless(self):
        return (self.L     == 0 and self.M == 0 and self.T == 0 and self.I == 0 and
                self.THETA == 0 and self.N == 0 and self.J == 0)

    def convert_to(self, other, value):
        other = resolve_units(other)
        if not self.is_same_dimension(other): raise ValueError('Units are not compatible (dimension mismatch).')
        base_value = self.scale * (value + self.bias)   # SI_value = scale*(x + bias)
        return base_value / other.scale - other.bias
        


ALIASES = {
    'degC': '°C', 'degF': '°F', 'degR': '°R',
    'inch': 'in', 'inches': 'in',
    'meter': 'm', 'metre': 'm', 'meters': 'm', 'metres': 'm',
    'liter': 'L', 'litre': 'L', 'liters': 'L', 'litres': 'L',
}
                
UNITS = {
    
    # Non-Dimensional 'Units'
    '1':   UnitDefinition('1', 'dimensionless'),
    
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
  

class UnitParser:
    def __init__(self, unit_str):
        if not isinstance(unit_str, str): raise TypeError('UnitParser expects a string.')
        self.unit_str = unit_str
        
    def __str__(self):
        return self.unit_str

    def __repr__(self):
        return f"UNIT PARSER: Unit_str = '{self.unit_str}'."   
   
    # ---------- Public API ------------------------------------------------------------------------------------------------------
    def parse(self):
        s       = normalize_units(self.unit_str)
        if not s: raise ValueError('Empty Unit String')
        
        parts   = self._split_outside_parens(s, '/')    # <-- instead of s.split('/')
        numerator, denominators = parts[0], parts[1:]
        
        parsed_units = self._parse_product(numerator)
        for d in denominators:
            for unit in self._parse_product(d):
                parsed_units.append(unit ** -1)
                
        if not parsed_units: return resolve_units('1')
        
        result = parsed_units[0]
        for u in parsed_units[1:]: result = result * u
        
        return result
    
     
    # ---------- Private Helper Methods ------------------------------------------------------------------------------------------
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
        return [p for p in parts if p]  # drop empties

    
    def _parse_product(self, units_str):
        units = [u.strip() for u in units_str.split('*') if u.strip()]
        return [self._parse_token(u) for u in units]
    
    # Token Level 
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
    
        if '/' in exp:  # simple fraction like 1/2
            num, den = exp.split('/', 1)
            power = float(int(num.strip())) / float(int(den.strip()))
        else:
            power = float(exp)
    
        return self._parse_base(base) ** power


    def _parse_base(self, unit_s):
        unit_s = unit_s.strip()
        
        try: return resolve_units(unit_s)
        except ValueError: pass
            
        for prefix in sorted(PREFIXES.keys(), key=len, reverse=True):
            if not prefix:
                continue
            if unit_s.startswith(prefix):
                base_sym = unit_s[len(prefix):]
                # resolve the base (alias-aware)
                try:
                    base = resolve_units(base_sym)
                except ValueError:
                    continue
    
                # forbid prefix on affine units (e.g., °C, °F)
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
          * delta  - affine  -> ERROR
      - Multiplication/Division follow UnitDefinition dunders (already guard affine).
    """
    __slots__ = ("val", "unit", "name", "desc")

    def __init__(self, val, units='1', name='', desc=''):
        self.val  = float(val)
        self.unit = resolve_units(units)  # -> UnitDefinition (aliases/expressions OK)
        self.name = name
        self.desc = desc

    # Display
    def __str__(self):
        return f"{self.val} {self.unit.symbol}" if not self.unit.is_dimensionless() else f"{self.val}"

    def __repr__(self):
        return f"Quantity(val={self.val}, unit={self.unit!s})"

    # Conversions
    def to(self, target):
        u_to = resolve_units(target)
        if not self.unit.is_same_dimension(u_to):
            raise ValueError("Cannot convert: dimension mismatch.")
        new_val = self.unit.convert_to(u_to, self.val)
        return Quantity(new_val, u_to, name=self.name, desc=self.desc)

    # Helpers
    @staticmethod
    def _temp_kind(u):
        # classify unit as 'linear' (no temperature), 'delta' temperature, or 'absolute' temperature
        if u.THETA == 0:
            return 'linear'
        # delta temperature units: symbol starts with 'Δ' or name contains 'difference'
        if u.symbol.startswith('Δ') or ('difference' in (u.name or '')):
            return 'delta'
        # everything else with THETA != 0 is an absolute temperature (K, °C, °F, °R)
        return 'absolute'

    @staticmethod
    def _delta_unit_for(u):
        # map absolute temp units to their delta counterparts (fallback to ΔK)
        mapping = {'K': 'ΔK', '°C': 'Δ°C', '°F': 'Δ°F', '°R': 'Δ°R'}
        sym = u.symbol
        return UNITS.get(mapping.get(sym, 'ΔK'))

    def _as_SI(self):
        # push value to SI numeric, keep dimensions from unit
        return self.unit.scale * (self.val + self.unit.bias)

    @staticmethod
    def _is_affine(u):  # °C, °F, etc.
        return abs(u.bias) > FLOATING_POINT_EPS

    # Arithmetic with another Quantity
    def _addsub(self, other, op):
            if not isinstance(other, Quantity):
                return NotImplemented
            if not self.unit.is_same_dimension(other.unit):
                raise ValueError("Cannot add/subtract: dimension mismatch.")
    
            ak = self._temp_kind(self.unit)
            bk = self._temp_kind(other.unit)
    
            # Non-temperature (linear) math (or both delta temps): convert RHS to LHS unit
            if ak == 'linear' and bk == 'linear':
                v = self.val + other.to(self.unit).val if op == "+" else self.val - other.to(self.unit).val
                return Quantity(v, self.unit)
            if ak == 'delta' and bk == 'delta':
                v = self.val + other.to(self.unit).val if op == "+" else self.val - other.to(self.unit).val
                return Quantity(v, self.unit)
    
            # Temperature cases
            if op == "+":
                # absolute + absolute -> undefined
                if ak == 'absolute' and bk == 'absolute':
                    raise ValueError("Adding two absolute temperatures is undefined.")
                # absolute + delta -> absolute (express delta in LHS's delta unit)
                if ak == 'absolute' and bk == 'delta':
                    dunit = self._delta_unit_for(self.unit)
                    inc = other.unit.convert_to(dunit, other.val)
                    return Quantity(self.val + inc, self.unit)
                # delta + absolute -> absolute (express delta in RHS's delta unit)
                if ak == 'delta' and bk == 'absolute':
                    dunit = self._delta_unit_for(other.unit)
                    inc = self.unit.convert_to(dunit, self.val)
                    return Quantity(other.val + inc, other.unit)
                # linear + (temperature) or (temperature) + linear cannot happen (dimension check)
            else:  # subtraction
                # absolute - absolute -> delta (in LHS's delta unit)
                if ak == 'absolute' and bk == 'absolute':
                    si_diff = self._as_SI() - other._as_SI()
                    dunit = self._delta_unit_for(self.unit)
                    return Quantity(si_diff / dunit.scale, dunit)
                # absolute - delta -> absolute
                if ak == 'absolute' and bk == 'delta':
                    dunit = self._delta_unit_for(self.unit)
                    dec = other.unit.convert_to(dunit, other.val)
                    return Quantity(self.val - dec, self.unit)
                # delta - absolute -> undefined
                if ak == 'delta' and bk == 'absolute':
                    raise ValueError("delta - absolute temperature is undefined.")
    
            # If we reach here, some unexpected combination slipped through
            raise ValueError("Unsupported quantity addition/subtraction case.")

    def __add__(self, other): return self._addsub(other, "+")
    def __sub__(self, other): return self._addsub(other, "-")

    # Multiplication/Division
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(self.val * float(other), self.unit)
        if isinstance(other, Quantity):
            return Quantity(self.val * other.val, self.unit * other.unit)
        return NotImplemented

    def __rmul__(self, other):  # number * Quantity
        if isinstance(other, (int, float)):
            return Quantity(float(other) * self.val, self.unit)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(self.val / float(other), self.unit)
        if isinstance(other, Quantity):
            return Quantity(self.val / other.val, self.unit / other.unit)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            # produces 1/unit when dividing a scalar by a quantity
            return Quantity(float(other) / self.val, self.unit ** -1)
        return NotImplemented

    # Power
    def __pow__(self, p):
        # only numeric exponent
        try:
            pf = float(p)
        except (TypeError, ValueError):
            raise ValueError("Power must be numeric.")
        return Quantity(self.val ** pf, self.unit ** pf)

    # Casting
    def __float__(self):
        if not self.unit.is_dimensionless():
            raise TypeError("Only dimensionless quantities can be cast to float.")
        return float(self.val)



# ---------- Module Level Functions ---------------------------------------------------------------------------------------------
def validate_aliases():
     for alias, target in ALIASES.items():
         if alias in UNITS:      raise RuntimeError(f"Alias '{alias}' collides with a canonical unit.")
         if target not in UNITS: raise RuntimeError(f"Alias '{alias}' points to unknown target '{target}'.")

def normalize_units(symbol):
    if not isinstance(symbol, str): raise TypeError('Normalize expects a string.')
    symbol = ' '.join(symbol.split())  # trim + collapse spaces
    
    return symbol.replace(' ', '')

def resolve_units(obj):
    if isinstance(obj, UnitDefinition): return obj
    if not isinstance(obj, str):        raise TypeError('resolve_units expects a string or UnitDefinition.')
    symbol = normalize_units(obj)
    symbol = ALIASES.get(symbol, symbol)
    
    if symbol in UNITS: return UNITS[symbol]
    
    try: return UnitParser(symbol).parse()
    except Exception: raise ValueError(f'Unknown unit symbol or expression: {obj}.')

def convert(value, from_unit, to_unit):
    u_from = resolve_units(from_unit)
    u_to   = resolve_units(to_unit)
    
    return u_from.convert_to(u_to, value)



if __name__ == '__main__':

    EPS = 1e-12
    
    # parsing & algebra
    assert UnitParser('N/m^2').parse() == UNITS['Pa']
    assert UnitParser('kg*m/s^2').parse() == UNITS['N']
    assert abs(UnitParser(' m ^ ( 1 / 2 ) ').parse().L - 0.5) < EPS
    try:
        UnitParser('m°C').parse(); raise AssertionError("prefix on affine should fail")
    except ValueError:
        pass
    
    # conversion
    from_F_32_to_K = convert(32, '°F', 'K')
    assert abs(from_F_32_to_K - 273.15) < 1e-12
    assert abs(convert(1, 'Δ°F', 'ΔK') - (5/9)) < EPS
    assert abs(convert(10, 'inch', 'm') - 0.254) < 1e-12
    
    # UnitDefinition dunders
    assert UNITS['N'] / (UNITS['m'] ** 2) == UNITS['Pa']
    try:
        _ = UNITS['°C'] * UNITS['m']; raise AssertionError("affine * linear should fail")
    except ValueError:
        pass
    
    # Quantity basics
    q = Quantity(32, '°F').to('K')
    assert abs(q.val - 273.15) < 1e-12
    assert (2 / Quantity(4, 's')).unit == (UNITS['s'] ** -1)
    try:
        _ = Quantity(1,'m') + Quantity(1,'s'); raise AssertionError("dimension mismatch add should fail")
    except ValueError:
        pass
    
    # affine add/sub rules
    C = Quantity(25,'°C'); dC = Quantity(10,'Δ°C')
    assert (C + dC).unit == UNITS['°C']
    d1 = Quantity(300,'K') - Quantity(25,'°C')   # -> Δ
    assert d1.unit == UNITS['ΔK']
