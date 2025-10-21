# Units & Quantities (Python)

A tiny, production‑minded toolkit for **units**, a **unit parser**, and a **Quantity** type (value + unit). It aims to be safe for real work: temperature gotchas are handled, conversions are explicit and testable, and behavior is easy to reason about.

> **Core invariant:** `SI_value = scale × (unit_value + bias)`
>
> * Linear units (e.g., m, s, Pa): `bias = 0`
> * Absolute temperatures (K, °C, °F, °R): **affine** (non‑zero bias except K)
> * Temperature differences (ΔK, Δ°C, Δ°F, Δ°R): **delta** (linear, `bias = 0`)

---

## Features

* **Unit model** with dimension exponents (L, M, T, I, Θ, N, J), `scale`, and `bias`
* **Robust temperature handling** (absolute vs delta)
* **Unit parser** supporting products `*`, quotients `/`, exponents `^n` and `^(p/q)` (e.g., `m^(1/2)`)
* **Aliases** (e.g., `inch → in`, `degC → °C`) without duplicating unit rows
* **SI prefixes** (single prefix, longest‑first), safely **rejected** on affine units (°C/°F)
* **Quantity** arithmetic with unit safety (+ − × ÷, powers, conversions)
* Friendly **helper APIs**: `convert(value, from, to)`, `resolve_units(symbol_or_expr)`

> **What’s deliberately not included yet**: numpy integration, `**` as an exponent operator, full parenthesis grouping in expressions (only inside exponents), pretty‑printing canonical forms.

---

## Install

This project is currently a single‑file module. You can:

1. Drop `quantity.py` into your project, or
2. Put it under `src/quantity.py` and add `src` to your `PYTHONPATH`.

```bash
python3 -c "import quantity; print('ok')"
```

---

## Quick start

```python
from quantity import UNITS, UnitParser, Quantity, convert

# Scalar conversions
print(convert(32, '°F', 'K'))               # 273.15
print(convert(101325, 'N/m^2', 'psi'))      # ≈ 14.6959

# Parse unit expressions
print(UnitParser('N/m^2').parse() == UNITS['Pa'])  # True
print(UnitParser(' m ^ ( 1 / 2 ) ').parse().L)     # 0.5

# Work with quantities
T = Quantity(25, '°C')
dT = Quantity(10, 'Δ°C')
print(T + dT)                                  # 35 °C
print((Quantity(300,'K') - Quantity(25,'°C'))) # 26.85 ΔK

F = Quantity(3, 'N')
A = Quantity(2, 'm^2')
print((F / A).to('Pa'))                        # 1.5 Pa
```

---

## API overview

### UnitDefinition

Represents a physical unit and its mapping to SI.

* Fields: `symbol`, `name`, `L, M, T, I, THETA, N, J`, `scale`, `bias`
* Algebra: `__mul__`, `__truediv__`, `__pow__` (guards forbid affine units in ×/÷; powers allow only ^0/^1 for affine)
* Equality uses an epsilon to handle float noise

**Helpers**

* `is_same_dimension(other)` → compare only exponents (with tolerance)
* `convert_to(other_unit, value)` → numeric conversion via SI

### UnitParser

Parses a string into a `UnitDefinition`.

* Supported syntax: `*`, `/`, exponents `^n`, `^(p/q)`. Parentheses are supported **inside exponents only** (e.g., `m^(1/2)`).
* Respects **aliases** first (e.g., `inch`, `degC`), then tries a single **SI prefix** (longest‑first).
* Rejects prefixes on **affine** units (e.g., `m°C` → error).

```python
UnitParser('kg*m/s^2').parse()   # -> N
UnitParser('mm').parse()          # -> milli·meter (1e-3 m)
```

### Helpers

* `normalize_units(s: str) -> str` — trims/collapses spaces and removes them from unit expressions
* `resolve_units(obj: str|UnitDefinition) -> UnitDefinition` — canonical lookup; falls back to `UnitParser` for expressions (e.g., `'N/m^2'`)
* `convert(value, from_unit, to_unit) -> float`

### Quantity

A number plus a unit, with safe arithmetic and conversions.

```python
q = Quantity(32, '°F').to('K')
q.val  # 273.15
```

**Addition/Subtraction rules**

| Case                | Result    | Notes                                 |
| ------------------- | --------- | ------------------------------------- |
| linear ± linear     | linear    | RHS converted to LHS unit             |
| delta ± delta       | delta     | convert RHS delta to LHS delta family |
| absolute + delta    | absolute  | delta expressed in LHS’s delta family |
| delta + absolute    | absolute  | symmetric                             |
| absolute − absolute | **delta** | returned in LHS’s delta family        |
| absolute + absolute | error     | blocked                               |
| delta − absolute    | error     | blocked                               |

**Multiplication/Division/Powers**

* ×, ÷ combine units via unit algebra; affine units are blocked at the unit layer
* `q ** p` raises value and unit; affine bases allow only ^0/^1

**Conversions**

* `q.to('unit')` converts via SI using the invariant; delta temps convert linearly

---

## Temperature semantics

We explicitly separate **absolute** temperatures (K, °C, °F, °R) from **delta** temperatures (ΔK, Δ°C, Δ°F, Δ°R).

* Absolute temps are **affine**: they include a `bias` (except K) and cannot be multiplied/divided in unit algebra.
* Delta temps are **linear** (`bias = 0`) and can be combined freely in algebra.
* The `Quantity` rules above enforce meaningful operations automatically.

Examples:

```python
Quantity(25,'°C') + Quantity(10,'Δ°C')     # 35 °C
Quantity(300,'K') - Quantity(25,'°C')      # 26.85 ΔK
(Quantity(25,'°C') + Quantity(298.15,'K')) # ValueError (absolute + absolute)
```

---

## Aliases & Prefixes

* **Aliases** avoid duplicates in `UNITS` (e.g., `degC → °C`, `inch → in`, `liter → L`).
* **Prefixes**: single SI prefix, longest‑first (e.g., `da` before `d`).

  * Allowed only when the base unit is **linear** (i.e., not affine). `m°C` → error.
  * Mass prefixes are typically applied to **g** (gram) rather than **kg**.

---

## Known limitations (by design for now)

* Exponent operator is `^` (not `**`).
* Parentheses are supported **only inside exponents** (e.g., `m^(1/2)`), not for grouping full expressions.
* No NumPy overloads yet (`__array_ufunc__`, `__array_function__`). The design allows adding them later.
* Printing of complex composed units is simple (e.g., `N*m^-2`); canonical formatting (e.g., collapsing exponents, ordering) can be added later.

---

## Testing

A few smoke tests you can keep under `if __name__ == '__main__':`

```python
import math
EPS = 1e-12

assert UnitParser('N/m^2').parse() == UNITS['Pa']
assert UnitParser('kg*m/s^2').parse() == UNITS['N']
assert abs(UnitParser(' m ^ ( 1 / 2 ) ').parse().L - 0.5) < EPS

assert abs(convert(32, '°F', 'K') - 273.15) < EPS
assert abs(convert(1, 'Δ°F', 'ΔK') - (5/9)) < EPS
assert abs(convert(10, 'inch', 'm') - 0.254) < EPS

assert UNITS['N'] / (UNITS['m'] ** 2) == UNITS['Pa']

C = Quantity(25,'°C'); dC = Quantity(10,'Δ°C')
assert (C + dC).unit == UNITS['°C']
d = Quantity(300,'K') - Quantity(25,'°C')
assert d.unit == UNITS['ΔK']

F = Quantity(3,'N'); A = Quantity(2,'m^2')
assert (F / A).to('Pa').val == 1.5

# float equality caveat
k = Quantity(32,'°F').to('K').val
assert math.isclose(k, 273.15, rel_tol=1e-12, abs_tol=1e-12)
```

---

## Roadmap (suggested)

* Pretty printer / canonicalizer for composed units (e.g., collapse `m^1 → m`)
* Optional `**` exponent operator (pre‑canonicalize to `^`)
* Numpy interop (`__array_ufunc__`, `__array_function__`) for vectorized math
* `Quantity.is_close` helper and rich comparisons
* Packaging to PyPI with simple wheels

---

## Contributing

Issues and pull requests are welcome. Please keep PRs small and focused, with tests for new behavior. If you propose new units, include source references and round‑trip tests.

---

## License

MIT (or your preferred license). Add a `LICENSE` file and update this section accordingly.

---

## Acknowledgments

Thanks to everyone who helped think through the temperature semantics and unit algebra edge‑cases. The design here aims to keep correctness first and the surface area small so it’s easy to audit and extend.
