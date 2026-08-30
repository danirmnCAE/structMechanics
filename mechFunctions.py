"""Mechanical / structural engineering utility functions.

A growing collection of mechanics-based helpers for use across the scripts
in this folder. Functions are grouped by topic

SI units throughout unless explicitly noted in a function's docstring:
    length [m], pressure/stress [Pa], force [N], moment [N*m], angle [rad].
"""
from ast import expr
from ast import expr
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Center of Gravity, Centroid and Moments of Inertia
# ---------------------------------------------------------------------------

# Centroid of a cross-section built up from rectangular sub-areas
def centroid_2Drectangles(*sections):
    # ensure input is a 2D array of shape (n, 4) where each row is [xc, yc, w, h]
    sec = np.asarray(sections, dtype=float) 
    if sec.ndim != 2 or sec.shape[1] != 4:
        raise ValueError("each section must be a 4-vector [xc, yc, base, height]")
    # define arrays for the each rectangle's centroid coordinates (xc, yc) and dimensions (b, h)
    xc, yc, b, h = sec[:, 0], sec[:, 1], sec[:, 2], sec[:, 3]
    # area of each rectangle
    Ai = b * h                                 
    A_total = Ai.sum()
    if A_total == 0.0:
        raise ValueError("total area is zero; centroid is undefined")
    # compute the area-weighted centroid coordinates
    x_bar = (Ai * xc).sum() / A_total
    y_bar = (Ai * yc).sum() / A_total
    return x_bar, y_bar

# Centroid of a 2D area bounded above by y = f(x) and below by y = 0, over [a, b]
def centroid_2Darea(function, x, a, b):
    # integrate to find the area
    area = sp.integrate(function, (x, a, b))
    if area == 0:
        raise ValueError("total area is zero; centroid is undefined")
    # compute the area-weighted centroid coordinates
    x_bar = sp.integrate(x * function, (x, a, b)) / area
    y_bar = sp.integrate(function**2 / 2, (x, a, b)) / area
    return sp.simplify(x_bar), sp.simplify(y_bar)

# Second moments of inertia of a cross-section built up from rectangular sub-areas
def I_rectangles(*sections):
    # ensure input is a 2D array of shape (n, 4) where each row is [xc, yc, b, h]
    sec = np.asarray(sections, dtype=object)
    if sec.ndim != 2 or sec.shape[1] != 4:
        raise ValueError("each section must be a 4-vector [xc, yc, base, height]")
    # define arrays for the each rectangle's centroid coordinates (xc, yc) and dimensions (b, h)
    xc, yc, b, h = sec[:, 0], sec[:, 1], sec[:, 2], sec[:, 3]
    # area of each rectangle
    Ai = b * h
    # second moments about each rectangle's own centroidal axes
    Ixi = b * h**3 / 12.0
    Iyi = h * b**3 / 12.0
    # parallel-axis theorem: shift to the global x and y axes through the origin
    Ixx = (Ixi + Ai * yc**2).sum()    # about the x axis [m^4]
    Iyy = (Iyi + Ai * xc**2).sum()    # about the y axis [m^4]
    return sp.simplify(Ixx), sp.simplify(Iyy)

# Second moment of inertia about the x axis of triangular sub-areas,
# each with its base parallel to and offset d above the x axis
def I_triangles(*sections):
    # ensure input is a 2D array of shape (n, 3) where each row is [d, b, h]
    sec = np.asarray(sections, dtype=object)
    if sec.ndim != 2 or sec.shape[1] != 3:
        raise ValueError("each section must be a 3-vector [offset, base, height]")
    # base offset above the x axis, base and height of each triangle
    d, b, h = sec[:, 0], sec[:, 1], sec[:, 2]
    # area and centroid height (the centroid sits h/3 above the base)
    Ai = b * h / 2.0
    yc = d + h / 3.0
    # second moment about each triangle's own centroidal x axis
    Ixi = b * h**3 / 36.0
    # parallel-axis theorem: shift to the global x axis
    Ixx = (Ixi + Ai * yc**2).sum()    # about the x axis [m^4]
    return sp.simplify(Ixx)

# Second moments of inertia of a 2D area bounded above by y = f(x) and below by y = 0, over [a, b]
def I_area(function, x, a, b):
    # calculate Ix
    Ixx = sp.integrate(function**3 / 3, (x, a, b))
    # calculate Iy
    Iyy = sp.integrate(x**2 * function, (x, a, b))
    return sp.simplify(Ixx), sp.simplify(Iyy)

# Polar moment of inertia of a tubular (hollow circular) cross-section
def J_tube(outer_radius, inner_radius=0.0):
    ro = np.asarray(outer_radius, dtype=float)  # Outer radius [m]
    ri = np.asarray(inner_radius, dtype=float)  # Inner radius [m] (0 for a solid shaft)
    # subtract J of the inner solid shaft from that of the outer one
    return np.pi * (ro**4 - ri**4) / 2.0        # Polar moment of inertia [m^4]

# Polar moment of inertia of a rectangular cross-section about its centroid
def J_rectangle(base, height):
    b = np.asarray(base, dtype=float)           # Base [m]
    h = np.asarray(height, dtype=float)         # Height [m]
    # perpendicular-axis theorem: J = Ixx + Iyy = b*h^3/12 + h*b^3/12
    return b * h * (b**2 + h**2) / 12.0         # Polar moment of inertia [m^4]

# Polar moment of inertia of any triangular cross-section about its centroid.
# Vertices at (0, 0), (b, 0) and (a, h), with a the apex offset measured from
# the base's left corner (a = 0 or a = b for a right triangle, a = b/2 if
# isosceles); a may also fall outside [0, b] for an obtuse triangle
def J_triangle(base, height, apex_offset):
    b = np.asarray(base, dtype=float)           # Base [m]
    h = np.asarray(height, dtype=float)         # Height [m]
    a = np.asarray(apex_offset, dtype=float)    # Apex offset along the base [m]
    # second moments about the centroidal axes (Ixx is independent of a)
    Ixx = b * h**3 / 36.0                       # about the x axis [m^4]
    Iyy = h * b * (b**2 - a * b + a**2) / 36.0  # about the y axis [m^4]
    # perpendicular-axis theorem
    return Ixx + Iyy                            # Polar moment of inertia [m^4]

# ---------------------------------------------------------------------------
# Elastic material constants
# ---------------------------------------------------------------------------
# For a linear-elastic isotropic material

# Shear modulus from Young's modulus and Poisson's ratio
def shear_modulus(E, nu):
    E = np.asarray(E, dtype=float)          # Young's modulus [Pa]
    nu = np.asarray(nu, dtype=float)        # Poisson's ratio [-]
    return E / (2.0 * (1.0 + nu))           # Shear modulus [Pa]

# Poisson's ratio from Young's and shear moduli
def Poissons_ratio(E, G):
    E = np.asarray(E, dtype=float)          # Young's modulus [Pa]
    G = np.asarray(G, dtype=float)          # Shear modulus [Pa]
    return E / (2.0 * G) - 1.0              # Poisson's ratio [-]

# ---------------------------------------------------------------------------
# Axially loaded members
# ---------------------------------------------------------------------------
# For a linear-elastic isotropic material

# Relative axial displacement over [a, b] of a member with P(x), A(x) and E(x).
# Set numeric=True when the exact integral is unwieldy (e.g. a tapered area
# with a fractional exponent) to integrate numerically instead.
def axial_displacement(P, A, E, x, a, b, numeric=False):
    P, A, E = sp.sympify(P), sp.sympify(A), sp.sympify(E) # functions of x
    integrand = P / (A * E)
    if numeric:
        return float(sp.Integral(integrand, (x, a, b)).evalf())
    delta = sp.integrate(integrand, (x, a, b))            # relative displacement [m]
    return sp.simplify(delta)

# ---------------------------------------------------------------------------
# Torsion
# ---------------------------------------------------------------------------
# For shafts of linear-elastic isotropic material

# Relative angle of twist over [a, b] of a shaft with T(x), J(x) and G(x).
# Set numeric=True when the exact integral is unwieldy (e.g. a tapered radius
# with a fractional exponent) to integrate numerically instead.
def twist_angle(T, J, G, x, a, b, numeric=False):
    T, J, G = sp.sympify(T), sp.sympify(J), sp.sympify(G) # functions of x
    integrand = T / (J * G)
    if numeric:
        return float(sp.Integral(integrand, (x, a, b)).evalf())
    phi = sp.integrate(integrand, (x, a, b))              # relative angle of twist [rad]
    return sp.simplify(phi)

# ---------------------------------------------------------------------------
# Thin-walled pressure vessels
# ---------------------------------------------------------------------------
# Cylindrical vessel with closed ends under internal pressure.
# Thin-walled assumption valid when r/t >= 10.

# Hoop (circumferential) stress in a thin-walled cylindrical vessel
def sigma_hoopPV(pressure, radius, thickness):
    p = np.asarray(pressure, dtype=float)   # Internal gauge pressure [Pa]
    r = np.asarray(radius, dtype=float)     # Internal radius [m]
    t = np.asarray(thickness, dtype=float)  # Wall thickness [m]
    return p * r / t                        # Hoop (circumferential) stress [Pa]

# Axial (longitudinal) stress in a thin-walled cylindrical vessel with closed ends
def sigma_longPV(pressure, radius, thickness):
    p = np.asarray(pressure, dtype=float)   # Internal gauge pressure [Pa]
    r = np.asarray(radius, dtype=float)     # Internal radius [m]
    t = np.asarray(thickness, dtype=float)  # Wall thickness [m]
    return p * r / (2.0 * t)                # Axial (longitudinal) stress [Pa]

# Biaxial stress in a thin-walled spherical vessel
def sigma_spherePV(pressure, radius, thickness):
    p = np.asarray(pressure, dtype=float)   # Internal gauge pressure [Pa]
    r = np.asarray(radius, dtype=float)     # Internal radius [m]
    t = np.asarray(thickness, dtype=float)  # Wall thickness [m]
    return p * r / (2.0 * t)                # Biaxial stress [Pa]

# ---------------------------------------------------------------------------
# Beams: load, shear, moment and deflection diagrams
# ---------------------------------------------------------------------------
# Euler-Bernoulli beams spanning [0, L], with x measured from the left end and
# forces/deflections positive upward. Internal shear V and bending moment M
# follow the usual convention (sagging moment positive). Reactions come from
# equilibrium plus zero deflection at every support and zero slope at every
# fixed joint, so statically indeterminate beams are handled automatically.

# Cumulative trapezoidal integral of samples y on a uniform grid of spacing dx
def cumulative_trapz(y, dx):
    out = np.zeros_like(y)
    out[1:] = np.cumsum((y[:-1] + y[1:]) / 2.0) * dx
    return out

# Plot the distributed load, shear force, bending moment and deflection of a
# beam of length L. P is a list of point forces as (magnitude [N], position [m])
# pairs and C a list of point moments (couples) as (magnitude [N*m], position
# [m]) pairs, counterclockwise positive like the forces are upward positive.
# w(x) [N/m], E(x) [Pa] and I(x) [m^4] are expressions in the symbol x (E may
# also just be a constant), pinned and fixed are lists of support positions [m]
def beam_diagrams(L, P, w, C, pinned, fixed, I, x, E, n=1001):

    # discretization of beam elements along x-axis
    xs = np.linspace(0.0, float(L), n) 
    dx = xs[1] - xs[0]      # beam element length dx

    # discretize distributed load and bending stiffness using numerical functions
    # coming from original symbolic input functions
    w_vals = sp.lambdify(x, sp.sympify(w), "numpy")(xs) * np.ones_like(xs)
    E_vals = sp.lambdify(x, sp.sympify(E), "numpy")(xs) * np.ones_like(xs)
    I_vals = sp.lambdify(x, sp.sympify(I), "numpy")(xs) * np.ones_like(xs)
    EI = E_vals * I_vals    # bending stiffness of each beam element

    # shear and moment produced by the loads
    V0 = cumulative_trapz(w_vals, dx)            # w(x) = dV/dx
    M0 = cumulative_trapz(V0, dx)                # V(x) = dM/dx   
    for Pi, a in P:                              # loop through all point loads
        V0 = V0 + Pi * (xs >= a)                 # jump in shear force diagram from point load
        M0 = M0 + Pi * np.maximum(xs - a, 0.0)   # ramp in moment diagram from point load
    for Mi, a in C:                              # loop through all point moments
        M0 = M0 - Mi * (xs >= a)                 # jump in moment diagram from point moment

    # shear and moment produced by the pinned and fixed supports
    supports = list(pinned) + list(fixed)
    Vu = [1.0 * (xs >= a) for a in supports] + [0.0 * xs for _ in fixed]
    Mu = [np.maximum(xs - a, 0.0) for a in supports] + [1.0 * (xs >= a) for a in fixed]

    # slope and deflection of every contribution, from EI * v'' = M
    th0 = cumulative_trapz(M0 / EI, dx)
    v0 = cumulative_trapz(th0, dx)
    thu = [cumulative_trapz(Mi / EI, dx) for Mi in Mu]
    vu = [cumulative_trapz(ti, dx) for ti in thu]

    # linear system for the unknowns [reactions..., C1, C2]: shear and moment
    # vanish past the right end (equilibrium), deflection is zero at every
    # support and slope is zero at every fixed joint (compatibility)
    eqs, rhs = [], []
    eqs.append([Vi[-1] for Vi in Vu] + [0.0, 0.0])
    rhs.append(-V0[-1])
    eqs.append([Mi[-1] for Mi in Mu] + [0.0, 0.0])
    rhs.append(-M0[-1])
    for a in supports:
        i = np.argmin(np.abs(xs - a))
        eqs.append([vi[i] for vi in vu] + [xs[i], 1.0])
        rhs.append(-v0[i])
    for a in fixed:
        i = np.argmin(np.abs(xs - a))
        eqs.append([ti[i] for ti in thu] + [1.0, 0.0])
        rhs.append(-th0[i])
    u = np.linalg.solve(np.array(eqs), np.array(rhs))

    # superpose the solved reactions and integration constants C1, C2
    R, C1, C2 = u[:-2], u[-2], u[-1]
    V = V0 + sum(Ri * Vi for Ri, Vi in zip(R, Vu))
    M = M0 + sum(Ri * Mi for Ri, Mi in zip(R, Mu))
    v = v0 + sum(Ri * vi for Ri, vi in zip(R, vu)) + C1 * xs + C2
    v = v * 1000 # convert to mm

    # three stacked diagrams sharing the x axis
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    labels = ["Shear Force [N]", "Moment [Nm]", "Deflection [mm]"]
    for ax, y, label in zip(axs, [V, M, v], labels):
        ax.fill_between(xs, y, color="#2a78d6", alpha=0.2)
        ax.plot(xs, y, color="#2a78d6", lw=2)
        ax.axhline(0.0, color="#c3c2b7", lw=1)
        ax.set_ylabel(label)
        ax.grid(color="#e1e0d9", lw=0.8)
    axs[-1].set_xlabel("x [m]")
    fig.tight_layout()
    plt.show()
    return xs, V, M, v

