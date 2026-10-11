# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A single-module library of structural/mechanical engineering formulas
(`mechFunctions.py`) plus a Jupyter notebook (`scratchPad.ipynb`) used to solve
individual problems with it. There is no package, no build step, no test suite
and no dependency file — it is a personal calculation toolkit, not an
application.

- [mechFunctions.py](mechFunctions.py) — all library code, ~290 lines
- [scratchPad.ipynb](scratchPad.ipynb) — throwaway working notebook; cells are
  overwritten freely and are not a regression suite
- Runtime: Python 3.12 with `numpy`, `sympy`, `matplotlib` (install manually)

## Conventions that matter

**SI units throughout**, unless a docstring says otherwise: length [m],
stress/pressure [Pa], force [N], moment [N·m], angle [rad]. Any deviation must
be called out in the function's comment. `beam_diagrams` is the one exception
inside a function: it converts deflection to mm before plotting and returning.

**`mechFunctions.py` is organised as one flat module split by comment banners**
(`# ----` blocks): centroids & moments of inertia, elastic constants, axial
members, torsion, pressure vessels, beam diagrams. Add new functions under the
matching banner, or create a new banner in the same style — do not split the
file into submodules.

**Every function carries inline unit comments rather than full docstrings.**
Match that style: a one- or two-line comment above the `def` explaining the
formula's assumptions, then trailing `# Quantity [unit]` comments on the
argument coercions and the return. Only the module has a real docstring.

**Naming is by quantity, not by verb:** `I_*` for second moments, `J_*` for
polar moments, `sigma_*` for stresses, then the shape or case
(`I_rectangles`, `J_tube`, `sigma_hoopPV`). Keep that pattern.

## Numeric vs. symbolic — the main trap

Functions deliberately split into two families, and picking the wrong one
breaks callers:

- **Numeric-only:** coerce with `np.asarray(..., dtype=float)` and return
  floats/arrays — `centroid_2Drectangles`, `J_tube`, `J_rectangle`,
  `J_triangle`, `shear_modulus`, `Poissons_ratio`, the `sigma_*PV` family.
  These vectorise over array inputs but reject sympy symbols.
- **Symbolic-tolerant:** coerce with `dtype=object` and return
  `sp.simplify(...)` — `I_rectangles`, `I_triangles`. They accept sympy
  expressions among the section dimensions, so results can come back as
  expressions rather than numbers. Callers doing arithmetic on the result (e.g.
  `M * c / I`) may need `float(...)`.
- **Purely symbolic:** take a sympy expression plus its symbol and integrate —
  `centroid_2Darea`, `I_area`, `axial_displacement`, `twist_angle`.
  `axial_displacement` and `twist_angle` take `numeric=True` to fall back to
  `sp.Integral(...).evalf()` when the closed form is intractable (tapered
  sections with fractional exponents).

Section-based helpers take **varargs of fixed-width lists**, not a single
array: `I_rectangles([xc, yc, b, h], [xc, yc, b, h], ...)`, `I_triangles([d, b,
h], ...)`. `I_*`/`centroid_2Drectangles` take coordinates **about the global
origin** — the parallel-axis shift to the origin is done inside, so pass
centroids already measured from the axis you want results about.

## `beam_diagrams` — sign and support conventions

The largest function, and the one with the most implicit contract:

```python
xs, V, M, v = mf.beam_diagrams(L, P, w, C, pinned, fixed, I, x, E, n=1001)
```

- Euler–Bernoulli beam on `[0, L]`, `x` from the left end.
- **Forces positive upward, couples positive counterclockwise** — downward
  loads are negative (`w = -8000`, `P = [(-15e3, 2.0)]`).
- `P` and `C` are lists of `(magnitude, position)` pairs; `w`, `E`, `I` are
  sympy expressions in the symbol `x` (constants are fine, they get broadcast
  via `* np.ones_like(xs)`).
- `pinned` and `fixed` are lists of support positions. Reactions are solved
  from global equilibrium plus zero deflection at every support and zero slope
  at every fixed joint, so **statically indeterminate beams work without any
  extra input**. Adding supports just adds unknowns and equations to the same
  dense `np.linalg.solve`.
- Everything is computed on a uniform grid by `cumulative_trapz`, so results
  are accurate to the grid spacing `L/(n-1)`; support positions are snapped to
  the nearest grid node with `argmin`. Increase `n` for sharper point-load
  jumps.
- It calls `plt.show()` as a side effect and returns
  `(xs, V [N], M [N·m], v [mm])`.

Reactions are not returned explicitly — read them as the jump in `V` at each
support, as the example in the last notebook cell does.

## Working in the notebook

Cell 0 sets up `%autoreload 2` and imports the library as `mf`, so edits to
`mechFunctions.py` take effect without restarting the kernel. Keep that cell
intact. The final cell is a worked `beam_diagrams` example (propped cantilever)
and is the closest thing to documentation-by-example — prefer updating it over
deleting it when the beam API changes. The sympy-`solve`-a-system-of-`sp.Eq`
pattern in the torsion cell is the house style for statically indeterminate
hand problems: list the equations, list the unknowns, `sp.solve(..., dict=True)[0]`,
then `.subs(consts)`.

Notebook cells are not checked for output cleanliness and mix unit systems
freely (one cell works in inches/psi) — that is intentional scratch work, so do
not "fix" it unasked.

## Known cruft

`mechFunctions.py` opens with a duplicated, unused `from ast import expr`
(lines 9–10). Safe to delete if touching the imports.
