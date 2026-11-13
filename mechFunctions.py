"""Mechanics of materials helper functions.

A toolkit for structural / mechanical engineering hand calculations, meant
to be imported into a notebook and combined with numpy and sympy.

    Statics and Mechanics of materials 4th ed (R.C. Hibbeler):

    Centroids and moments of inertia ...... ch. 6 and App. A
    Internal loadings and strain .......... ch. 7
    Mechanical properties of materials .... ch. 8
    Axially loaded members ................ ch. 9
    Torsion ............................... ch. 10
    Bending ............................... ch. 11 (plus curved beams)
    Transverse shear ...................... ch. 12
    Combined loadings ..................... ch. 13

Units
    SI in the comments: length [m], force [N], moment [N*m], stress,
    pressure and moduli [Pa], angle [rad]. No function hard-codes a unit,
    so any consistent system (in, lb, psi) works just as well. The one
    exception is beam_diagrams, which reports deflection in mm.

Function families
    Numeric            Coerce inputs with np.asarray(..., dtype=float), so
                       they broadcast over arrays but reject sympy symbols.

    Symbolic-tolerant  I_rectangles, Ixy_rectangles and I_triangles accept
                       sympy expressions among the section dimensions and
                       return sp.simplify(...). Wrap the result in float()
                       before mixing it with numpy arrays.

    Symbolic           Take a sympy expression and its symbol and integrate:
                       centroid_2Darea, I_area, axial_displacement and
                       twist_angle.
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Centroids and moments of inertia                          (ch. 6, App. A)
# ---------------------------------------------------------------------------
# Built-up sections are passed as varargs of fixed-width lists, one per
# sub-area: I_rectangles([xc, yc, b, h], [xc, yc, b, h], ...). Coordinates are
# measured from the global origin and the parallel-axis shift to the global
# axes is done inside, so pass centroids already measured from the axes the
# results are wanted about. A negative base subtracts a hole.

# Centroid of a cross-section built up from rectangular sub-areas
def centroid_2Drectangles(*sections):
    # one row [xc, yc, base, height] per rectangle
    sec = np.asarray(sections, dtype=float)
    if sec.ndim != 2 or sec.shape[1] != 4:
        raise ValueError("each section must be a 4-vector [xc, yc, base, height]")
    xc, yc, b, h = sec.T                        # centroid [m] and size [m] of each rectangle
    Ai = b * h                                  # area of each rectangle [m^2]
    A_total = Ai.sum()
    if A_total == 0.0:
        raise ValueError("total area is zero; centroid is undefined")
    # area-weighted mean of the sub-area centroids
    x_bar = (Ai * xc).sum() / A_total
    y_bar = (Ai * yc).sum() / A_total
    return x_bar, y_bar                         # Centroid coordinates [m]

# Centroid of a triangle from its three vertices in the xy plane; the centroid
# is the mean of the vertices, whatever their order
def centroid_2Dtriangle(p1, p2, p3):
    pts = np.asarray([p1, p2, p3], dtype=float)     # Vertex coordinates [m]
    if pts.shape != (3, 2):
        raise ValueError("each vertex must be a 2-vector [x, y]")
    (x1, y1), (x2, y2), (x3, y3) = pts
    # collinear vertices enclose no area
    if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0.0:
        raise ValueError("vertices are collinear; centroid is undefined")
    x_bar = (x1 + x2 + x3) / 3.0
    y_bar = (y1 + y2 + y3) / 3.0
    return x_bar, y_bar                             # Centroid coordinates [m]

# Centroid of the area under y = f(x) (and above y = 0) over [a, b]. function
# is a sympy expression in the symbol x; a and b may be symbols too
def centroid_2Darea(function, x, a, b):
    area = sp.integrate(function, (x, a, b))
    if area == 0:
        raise ValueError("total area is zero; centroid is undefined")
    # first moments of the area divided by the area; a vertical strip of
    # height f has its centroid at y = f/2
    x_bar = sp.integrate(x * function, (x, a, b)) / area
    y_bar = sp.integrate(function**2 / 2, (x, a, b)) / area
    return sp.simplify(x_bar), sp.simplify(y_bar)   # Centroid coordinates [m]

# Second moments of inertia of a cross-section built up from rectangular
# sub-areas, about the global x and y axes
def I_rectangles(*sections):
    # one row [xc, yc, base, height] per rectangle; object dtype keeps sympy entries
    sec = np.asarray(sections, dtype=object)
    if sec.ndim != 2 or sec.shape[1] != 4:
        raise ValueError("each section must be a 4-vector [xc, yc, base, height]")
    xc, yc, b, h = sec.T
    Ai = b * h                          # area of each rectangle [m^2]
    # second moments about each rectangle's own centroidal axes
    Ixi = b * h**3 / 12.0
    Iyi = h * b**3 / 12.0
    # parallel-axis theorem: shift to the global axes through the origin
    Ixx = (Ixi + Ai * yc**2).sum()      # About the x axis [m^4]
    Iyy = (Iyi + Ai * xc**2).sum()      # About the y axis [m^4]
    return sp.simplify(Ixx), sp.simplify(Iyy)

# Product of inertia of a cross-section built up from rectangular sub-areas,
# about the global x and y axes. A rectangle's own centroidal product of
# inertia is zero (it is symmetric), so only the parallel-axis term remains
def Ixy_rectangles(*sections):
    # one row [xc, yc, base, height] per rectangle; object dtype keeps sympy entries
    sec = np.asarray(sections, dtype=object)
    if sec.ndim != 2 or sec.shape[1] != 4:
        raise ValueError("each section must be a 4-vector [xc, yc, base, height]")
    xc, yc, b, h = sec.T
    Ixy = (b * h * xc * yc).sum()       # Product of inertia [m^4]
    return sp.simplify(Ixy)

# Second moment of inertia about the x axis of triangular sub-areas, each
# with its base parallel to the x axis at height d and its apex above the base
def I_triangles(*sections):
    # one row [d, base, height] per triangle; object dtype keeps sympy entries
    sec = np.asarray(sections, dtype=object)
    if sec.ndim != 2 or sec.shape[1] != 3:
        raise ValueError("each section must be a 3-vector [offset, base, height]")
    d, b, h = sec.T
    # area and centroid height (the centroid sits h/3 above the base)
    Ai = b * h / 2.0
    yc = d + h / 3.0
    # second moment about each triangle's own centroidal x axis
    Ixi = b * h**3 / 36.0
    # parallel-axis theorem: shift to the global x axis
    Ixx = (Ixi + Ai * yc**2).sum()      # About the x axis [m^4]
    return sp.simplify(Ixx)

# Second moments of inertia of the area under y = f(x) (and above y = 0) over
# [a, b], about the x and y axes. function is a sympy expression in x
def I_area(function, x, a, b):
    # a vertical strip of height f contributes f^3/3 dx about the x axis
    # and x^2 * f dx about the y axis
    Ixx = sp.integrate(function**3 / 3, (x, a, b))
    Iyy = sp.integrate(x**2 * function, (x, a, b))
    return sp.simplify(Ixx), sp.simplify(Iyy)       # About the x and y axes [m^4]

# Principal moments of inertia of an area, from Ix, Iy and Ixy about any pair
# of perpendicular axes through one point (the centroid, for bending): Mohr's
# circle for inertia. theta_p is the counterclockwise angle from the x axis to
# the axis of Imax; the axis of Imin lies at theta_p + pi/2
def I_principal(Ix, Iy, Ixy):
    Ix = np.asarray(Ix, dtype=float)                    # Moment of inertia about x [m^4]
    Iy = np.asarray(Iy, dtype=float)                    # Moment of inertia about y [m^4]
    Ixy = np.asarray(Ixy, dtype=float)                  # Product of inertia [m^4]
    I_avg = (Ix + Iy) / 2.0                             # centre of Mohr's circle
    R = np.hypot((Ix - Iy) / 2.0, Ixy)                  # radius of Mohr's circle
    theta_p = 0.5 * np.arctan2(-2.0 * Ixy, Ix - Iy)     # tan(2*theta_p) = -2*Ixy / (Ix - Iy)
    return I_avg + R, I_avg - R, theta_p                # Imax [m^4], Imin [m^4], theta_p [rad]

# Polar moment of inertia of a solid or hollow circular cross-section
def J_tube(outer_radius, inner_radius=0.0):
    ro = np.asarray(outer_radius, dtype=float)  # Outer radius [m]
    ri = np.asarray(inner_radius, dtype=float)  # Inner radius [m] (0 for a solid shaft)
    # subtract J of the bore from that of the solid outer circle
    return np.pi * (ro**4 - ri**4) / 2.0        # Polar moment of inertia [m^4]

# Polar moment of inertia of a rectangular cross-section about its centroid.
# This is a property of the area only: it is NOT the torsion constant of a
# rectangular shaft (use torsion_rectangle for that)
def J_rectangle(base, height):
    b = np.asarray(base, dtype=float)           # Base [m]
    h = np.asarray(height, dtype=float)         # Height [m]
    # perpendicular-axis theorem: J = Ixx + Iyy = b*h^3/12 + h*b^3/12
    return b * h * (b**2 + h**2) / 12.0         # Polar moment of inertia [m^4]

# Polar moment of inertia of any triangle about its centroid, with vertices at
# (0, 0), (b, 0) and (a, h): a is the apex offset from the left end of the base
# (a = 0 or b for a right triangle, b/2 if isosceles, outside [0, b] if obtuse).
# Like J_rectangle, NOT the torsion constant of a triangular shaft
def J_triangle(base, height, apex_offset):
    b = np.asarray(base, dtype=float)           # Base [m]
    h = np.asarray(height, dtype=float)         # Height [m]
    a = np.asarray(apex_offset, dtype=float)    # Apex offset along the base [m]
    # second moments about the centroidal axes (Ixx does not depend on a)
    Ixx = b * h**3 / 36.0
    Iyy = h * b * (b**2 - a * b + a**2) / 36.0
    # perpendicular-axis theorem
    return Ixx + Iyy                            # Polar moment of inertia [m^4]


# ---------------------------------------------------------------------------
# Internal loadings and strain                                      (ch. 7)
# ---------------------------------------------------------------------------

# Internal resultant loadings on the section through point O of a member in
# 3D, from equilibrium of the segment on one side of the cut. n runs along the
# member axis and points out of that segment through the cut face (its
# outward normal). forces [N] act at points [m] on the segment and couples
# [N*m] are any external couple moments on it, all 3D vectors in global axes.
# Returns the normal force N (tension positive) and torque T (right-hand rule
# about n) as scalars, and the shear force V and bending moment M as vectors
# lying in the plane of the section; stress_circularSection takes them as is
def internal_loads(O, n, forces, points, couples=()):
    O = np.asarray(O, dtype=float)                          # Centroid of the section [m]
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)      # Outward unit normal of the cut face [-]
    F = np.asarray(forces, dtype=float).reshape(-1, 3)      # External forces on the segment [N]
    r = np.asarray(points, dtype=float).reshape(-1, 3)      # Their points of application [m]
    C = np.asarray(couples, dtype=float).reshape(-1, 3)     # External couple moments [N*m]
    if F.shape != r.shape:
        raise ValueError("give one point of application per force")
    # the internal loads balance the external loads acting on the segment
    F_int = -F.sum(axis=0)
    M_int = -np.cross(r - O, F).sum(axis=0) - C.sum(axis=0)
    # resolve them along the member axis and in the plane of the section
    N = F_int @ n               # Normal force [N]
    T = M_int @ n               # Torque [N*m]
    V = F_int - N * n           # Shear force vector [N]
    M = M_int - T * n           # Bending moment vector [N*m]
    return N, V, T, M

# Average normal strains of two line segments AB and AC, and the shear strain
# between them at A, from the positions of points A, B and C before and after
# deformation (2D or 3D). The shear strain is the decrease in the angle BAC,
# gamma = theta - theta', so it is positive when the angle closes up
def strains_fromPoints(A, B, C, A_def, B_def, C_def):
    P0 = np.asarray([A, B, C], dtype=float)                 # Undeformed positions [m]
    P1 = np.asarray([A_def, B_def, C_def], dtype=float)     # Deformed positions [m]
    if P0.shape != P1.shape or P0.shape[1] not in (2, 3):
        raise ValueError("give six points with the same number (2 or 3) of coordinates")
    # segments AB and AC before (0) and after (1) the deformation
    AB0, AC0 = P0[1] - P0[0], P0[2] - P0[0]
    AB1, AC1 = P1[1] - P1[0], P1[2] - P1[0]
    length = np.linalg.norm
    def angle(u, v):
        return np.arccos(np.clip(u @ v / (length(u) * length(v)), -1.0, 1.0))
    eps_AB = length(AB1) / length(AB0) - 1.0        # Normal strain along AB [-]
    eps_AC = length(AC1) / length(AC0) - 1.0        # Normal strain along AC [-]
    gamma = angle(AB0, AC0) - angle(AB1, AC1)       # Shear strain at A [rad]
    return eps_AB, eps_AC, gamma


# ---------------------------------------------------------------------------
# Mechanical properties of materials                                (ch. 8)
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

# Material properties from tension-test data: the points of an engineering
# stress-strain diagram, in order of increasing strain up to fracture. The
# first n_linear points are taken as the linear-elastic region: E is their
# least-squares slope through the origin and the last of them sets the
# proportional limit. The yield strength comes from the offset method, where
# a line of slope E drawn from strain = offset (0.2 % by default) cuts the
# curve. The modulus of resilience is the strain energy density at the
# proportional limit, and the modulus of toughness the area under the whole
# curve. Both moduli come out in [Pa] = [J/m^3]
def tension_test(strain, stress, n_linear, offset=0.002):
    eps = np.asarray(strain, dtype=float)       # Engineering strain [-]
    sig = np.asarray(stress, dtype=float)       # Engineering stress [Pa]
    if eps.ndim != 1 or eps.shape != sig.shape or not 1 <= n_linear <= eps.size:
        raise ValueError("strain and stress must be equal-length 1D arrays with n_linear points or more")
    # modulus of elasticity and proportional limit from the linear region
    e_lin, s_lin = eps[:n_linear], sig[:n_linear]
    if e_lin @ e_lin == 0.0:
        raise ValueError("the linear region needs at least one point with nonzero strain")
    E = (e_lin @ s_lin) / (e_lin @ e_lin)
    sigma_pl = s_lin[-1]
    # offset method: first point where the curve drops below the offset line,
    # then linear interpolation back to the crossing
    gap = sig - E * (eps - offset)
    cross = np.flatnonzero((gap[:-1] > 0.0) & (gap[1:] <= 0.0))
    if cross.size == 0:
        raise ValueError("the offset line never meets the curve; check offset and n_linear")
    i = cross[0]
    f = gap[i] / (gap[i] - gap[i + 1])
    sigma_Y = sig[i] + f * (sig[i + 1] - sig[i])
    sigma_u = sig.max()
    # strain energy densities: triangle under the elastic line, and trapezoidal
    # rule for the area under the whole curve
    u_r = sigma_pl**2 / (2.0 * E)
    u_t = np.sum((sig[1:] + sig[:-1]) / 2.0 * np.diff(eps))
    # E [Pa], proportional limit [Pa], yield strength [Pa], ultimate stress [Pa],
    # modulus of resilience [Pa], modulus of toughness [Pa]
    return E, sigma_pl, sigma_Y, sigma_u, u_r, u_t


# ---------------------------------------------------------------------------
# Axially loaded members                                            (ch. 9)
# ---------------------------------------------------------------------------
# For a linear-elastic isotropic material

# Relative axial displacement over [a, b] of a member with P(x), A(x) and
# E(x), given as sympy expressions in x (or constants). Set numeric=True when
# the exact integral is unwieldy (e.g. a tapered area with a fractional
# exponent) to integrate numerically instead
def axial_displacement(P, A, E, x, a, b, numeric=False):
    P, A, E = sp.sympify(P), sp.sympify(A), sp.sympify(E)  # functions of x
    integrand = P / (A * E)
    if numeric:
        return float(sp.Integral(integrand, (x, a, b)).evalf())
    delta = sp.integrate(integrand, (x, a, b))             # Relative displacement [m]
    return sp.simplify(delta)

# Shared solver for a member made of prismatic segments in series along x and
# fixed at x = 0, used by axial_segmentedBar and torsion_segmentedShaft. For
# each segment, flex is its flexibility (L/(A*E) or L/(J*G)), load the external
# load at its right end and free its deformation under no load (thermal).
# gap is None for a free right end, 0 for a fixed one, or > 0 for a clearance
# that must close before the right support reacts, which then can only push
def _segments_in_series(flex, load, free, gap):
    N0 = np.cumsum(load[::-1])[::-1]        # internal load with the right end free
    d0 = flex * N0 + free                   # deformation of each segment, right end free
    R_B = 0.0
    if gap is not None:
        closure = d0.sum() - gap            # overlap with the right support if it did not react
        if gap == 0.0 or closure > 0.0:
            R_B = -closure / flex.sum()     # compatibility: total deformation equals the gap
    N = N0 + R_B                            # internal load in each segment
    u = np.concatenate(([0.0], np.cumsum(flex * N + free)))     # displacement of each node
    R_A = -(load.sum() + R_B)               # equilibrium of the whole member
    return N, u, R_A, R_B

# Internal forces, displacements and support reactions of a straight bar made
# of prismatic segments in series, fixed at its left end (x = 0). P[i] is the
# external axial force at the right end of segment i, positive towards +x;
# alpha and dT add a free thermal strain to each segment. The right end is
# free when gap is None (statically determinate), fixed when gap = 0, or meets
# a rigid wall after closing a clearance gap > 0. The last two cases are
# statically indeterminate and are solved from compatibility of deformations
def axial_segmentedBar(L, A, E, P, alpha=0.0, dT=0.0, gap=None):
    L = np.atleast_1d(np.asarray(L, dtype=float))                       # Segment lengths [m]
    A = np.broadcast_to(np.asarray(A, dtype=float), L.shape)            # Cross-sectional areas [m^2]
    E = np.broadcast_to(np.asarray(E, dtype=float), L.shape)            # Young's moduli [Pa]
    P = np.atleast_1d(np.asarray(P, dtype=float))                       # Forces at the segment ends [N]
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), L.shape)    # Thermal expansion coefficients [1/K]
    dT = np.broadcast_to(np.asarray(dT, dtype=float), L.shape)          # Temperature changes [K]
    if P.shape != L.shape:
        raise ValueError("give one force P per segment (0 where there is none)")
    N, u, R_A, R_B = _segments_in_series(L / (A * E), P, alpha * dT * L, gap)
    # Segment forces [N] (tension positive), displacements of the n + 1 segment
    # ends [m], reactions at x = 0 and at the right end [N] (positive towards +x)
    return N, u, R_A, R_B


# ---------------------------------------------------------------------------
# Torsion                                                          (ch. 10)
# ---------------------------------------------------------------------------
# For shafts of linear-elastic isotropic material. Torques are vectors along
# the shaft axis, positive by the right-hand rule

# Relative angle of twist over [a, b] of a shaft with T(x), J(x) and G(x),
# given as sympy expressions in x (or constants). Set numeric=True when the
# exact integral is unwieldy (e.g. a tapered radius with a fractional
# exponent) to integrate numerically instead
def twist_angle(T, J, G, x, a, b, numeric=False):
    T, J, G = sp.sympify(T), sp.sympify(J), sp.sympify(G)  # functions of x
    integrand = T / (J * G)
    if numeric:
        return float(sp.Integral(integrand, (x, a, b)).evalf())
    phi = sp.integrate(integrand, (x, a, b))               # Relative angle of twist [rad]
    return sp.simplify(phi)

# Internal torques, rotations and reaction torques of a circular shaft made of
# prismatic segments in series, fixed at its left end (x = 0). T[i] is the
# external torque at the right end of segment i, positive along +x. The right
# end is free, or also fixed when fixed_right=True, which makes the shaft
# statically indeterminate (solved from zero total twist)
def torsion_segmentedShaft(L, J, G, T, fixed_right=False):
    L = np.atleast_1d(np.asarray(L, dtype=float))               # Segment lengths [m]
    J = np.broadcast_to(np.asarray(J, dtype=float), L.shape)    # Polar moments of inertia [m^4]
    G = np.broadcast_to(np.asarray(G, dtype=float), L.shape)    # Shear moduli [Pa]
    T = np.atleast_1d(np.asarray(T, dtype=float))               # Torques at the segment ends [N*m]
    if T.shape != L.shape:
        raise ValueError("give one torque T per segment (0 where there is none)")
    gap = 0.0 if fixed_right else None
    T_int, phi, R_A, R_B = _segments_in_series(L / (J * G), T, np.zeros_like(L), gap)
    # Internal torque in each segment [N*m], rotations of the n + 1 segment
    # ends [rad], reaction torques at x = 0 and at the right end [N*m]
    return T_int, phi, R_A, R_B

# Maximum shear stress and angle of twist of a solid rectangular shaft with
# sides a and b (either order), from the exact Saint-Venant series solution.
# The book's square in Table 10-1 (4.81 T/a^3 and 7.10 T*L/(a^4*G)) is the
# case a = b. tau_max acts at the midpoint of the long sides
def torsion_rectangle(T, a, b, L, G):
    T = np.asarray(T, dtype=float)                  # Torque [N*m]
    a = np.asarray(a, dtype=float)                  # One side [m]
    b = np.asarray(b, dtype=float)                  # Other side [m]
    L = np.asarray(L, dtype=float)                  # Shaft length [m]
    G = np.asarray(G, dtype=float)                  # Shear modulus [Pa]
    w, t = np.maximum(a, b), np.minimum(a, b)       # long and short sides [m]
    # sum the first 50 odd terms along a new leading axis, so that array
    # inputs still broadcast
    n = np.arange(1, 100, 2).reshape((-1,) + (1,) * np.ndim(w / t))
    x = n * np.pi * w / (2.0 * t)
    sech = 2.0 * np.exp(-x) / (1.0 + np.exp(-2.0 * x))     # 1/cosh(x) without overflow
    # torsion constant J_t = k1 * w * t^3 and tau_max = T / (k2 * w * t^2)
    k1 = (1.0 - 192.0 * t / (np.pi**5 * w) * (np.tanh(x) / n**5).sum(axis=0)) / 3.0
    k2 = k1 / (1.0 - 8.0 / np.pi**2 * (sech / n**2).sum(axis=0))
    tau_max = T / (k2 * w * t**2)
    phi = T * L / (k1 * w * t**3 * G)
    return tau_max, phi                             # Max shear stress [Pa], angle of twist [rad]

# Maximum shear stress and angle of twist of a solid shaft whose section is
# an equilateral triangle of side a (Table 10-1). tau_max acts at the
# midpoint of each side. The exact twist coefficient is 80/sqrt(3) = 46.19;
# the book rounds it to 46
def torsion_triangle(T, a, L, G):
    T = np.asarray(T, dtype=float)                  # Torque [N*m]
    a = np.asarray(a, dtype=float)                  # Side of the triangle [m]
    L = np.asarray(L, dtype=float)                  # Shaft length [m]
    G = np.asarray(G, dtype=float)                  # Shear modulus [Pa]
    tau_max = 20.0 * T / a**3
    phi = 80.0 / np.sqrt(3.0) * T * L / (a**4 * G)
    return tau_max, phi                             # Max shear stress [Pa], angle of twist [rad]

# Maximum shear stress and angle of twist of a solid shaft with an elliptical
# section of semi-axes a and b (either order) (Table 10-1). tau_max acts at
# the ends of the minor axis
def torsion_ellipse(T, a, b, L, G):
    T = np.asarray(T, dtype=float)                  # Torque [N*m]
    a = np.asarray(a, dtype=float)                  # One semi-axis [m]
    b = np.asarray(b, dtype=float)                  # Other semi-axis [m]
    L = np.asarray(L, dtype=float)                  # Shaft length [m]
    G = np.asarray(G, dtype=float)                  # Shear modulus [Pa]
    a, b = np.maximum(a, b), np.minimum(a, b)       # semi-major and semi-minor axes [m]
    tau_max = 2.0 * T / (np.pi * a * b**2)
    phi = (a**2 + b**2) * T * L / (np.pi * a**3 * b**3 * G)
    return tau_max, phi                             # Max shear stress [Pa], angle of twist [rad]


# ---------------------------------------------------------------------------
# Bending                                                          (ch. 11)
# ---------------------------------------------------------------------------
# Section axes as in the book: x normal to the section and pointing out of
# it, y and z in its plane with the origin at the centroid. Moment components
# My and Mz act along +y and +z, so a positive Mz compresses the +y side.
# I_rectangles and Ixy_rectangles return Iz, Iy and Iyz if each rectangle is
# entered with its z coordinate as xc and its y coordinate as yc.

# Bending stress at point (y, z) of a cross-section of any shape, bent about
# both centroidal axes. With Iyz = 0 (y and z are principal axes) this is
# sigma = -Mz*y/Iz + My*z/Iy (Eq. 11-17); a nonzero Iyz gives the general
# form derived in Prob. 11-73, so M need not be resolved on principal axes
def sigma_unsymBending(My, Mz, Iy, Iz, y, z, Iyz=0.0):
    My = np.asarray(My, dtype=float)        # Moment component along +y [N*m]
    Mz = np.asarray(Mz, dtype=float)        # Moment component along +z [N*m]
    Iy = np.asarray(Iy, dtype=float)        # Moment of inertia about the y axis [m^4]
    Iz = np.asarray(Iz, dtype=float)        # Moment of inertia about the z axis [m^4]
    y = np.asarray(y, dtype=float)          # y coordinate of the point [m]
    z = np.asarray(z, dtype=float)          # z coordinate of the point [m]
    Iyz = np.asarray(Iyz, dtype=float)      # Product of inertia [m^4]
    D = Iy * Iz - Iyz**2
    return (-(Mz * Iy + My * Iyz) * y + (My * Iz + Mz * Iyz) * z) / D   # Bending stress [Pa], tension positive

# Orientation of the neutral axis under biaxial bending: the angle alpha from
# the +z axis towards the +y axis (Eq. 11-19, tan(alpha) = Iz/Iy * tan(theta),
# when Iyz = 0). Same axes and moments as sigma_unsymBending. An axial force
# shifts the neutral axis off the centroid but does not change its angle
def alpha_neutralAxis(My, Mz, Iy, Iz, Iyz=0.0):
    My = np.asarray(My, dtype=float)        # Moment component along +y [N*m]
    Mz = np.asarray(Mz, dtype=float)        # Moment component along +z [N*m]
    Iy = np.asarray(Iy, dtype=float)        # Moment of inertia about the y axis [m^4]
    Iz = np.asarray(Iz, dtype=float)        # Moment of inertia about the z axis [m^4]
    Iyz = np.asarray(Iyz, dtype=float)      # Product of inertia [m^4]
    # sigma = 0 along the line y/z = tan(alpha)
    alpha = np.arctan2(My * Iz + Mz * Iyz, Mz * Iy + My * Iyz)
    return (alpha + np.pi / 2.0) % np.pi - np.pi / 2.0      # Neutral axis angle in [-pi/2, pi/2) [rad]


# ---------------------------------------------------------------------------
# Curved beams             (not in Statics and MoM; Hibbeler, MoM sec. 6.9)
# ---------------------------------------------------------------------------
# Cross-sections symmetric about the plane of bending, with every radius
# measured from the centre of curvature. The neutral axis lies at
# Rna = A / integral(dA / r), inside the centroid: e = rc - Rna > 0. For the
# single shapes below the section width cancels out, so only the radial
# dimensions are needed.

# Radius to the neutral axis of a rectangular cross-section
def Rna_rectangle(inner_radius, outer_radius):
    ri = np.asarray(inner_radius, dtype=float)  # Inner radius [m]
    ro = np.asarray(outer_radius, dtype=float)  # Outer radius [m]
    return (ro - ri) / np.log(ro / ri)          # Radius to the neutral axis [m]

# Radius to the neutral axis of a cross-section built up from rectangles (T,
# I or box sections), each given as [inner_radius, outer_radius, width]:
# Rna = sum(A_i) / sum(width_i * ln(ro_i / ri_i)). A negative width subtracts a hole
def Rna_rectangles(*sections):
    # one row [inner radius, outer radius, width] per rectangle
    sec = np.asarray(sections, dtype=float)
    if sec.ndim != 2 or sec.shape[1] != 3:
        raise ValueError("each section must be a 3-vector [inner_radius, outer_radius, width]")
    ri, ro, w = sec.T
    return (w * (ro - ri)).sum() / (w * np.log(ro / ri)).sum()  # Radius to the neutral axis [m]

# Radius to the neutral axis of a triangular cross-section with its base at
# radius base_radius and its apex at apex_radius (the apex may point either
# towards or away from the centre of curvature)
def Rna_triangle(base_radius, apex_radius):
    rb = np.asarray(base_radius, dtype=float)   # Radius to the base [m]
    ra = np.asarray(apex_radius, dtype=float)   # Radius to the apex [m]
    return (ra - rb)**2 / (2.0 * (ra * np.log(ra / rb) - (ra - rb)))   # Radius to the neutral axis [m]

# Radius to the neutral axis of a solid circular cross-section. Also valid for
# an ellipse, with radius its semi-axis along the radial direction
def Rna_circle(centroid_radius, radius):
    rc = np.asarray(centroid_radius, dtype=float)       # Radius to the centroid [m]
    c = np.asarray(radius, dtype=float)                 # Radius of the cross-section [m]
    return c**2 / (2.0 * (rc - np.sqrt(rc**2 - c**2)))  # Radius to the neutral axis [m]

# Bending stress in a curved beam, sigma = M*y / (A*e*(Rna - y)). M is positive
# when it tends to straighten the beam (increase its radius of curvature) and
# y is measured from the neutral axis towards the centre of curvature, so the
# point sits at radius r = Rna - y. Tension positive; axial N/A not included
def sigma_bendCurvedBeam(M, y, A, e, Rna):
    M = np.asarray(M, dtype=float)          # Bending moment about the centroid [N*m]
    y = np.asarray(y, dtype=float)          # Distance from the neutral axis, towards the centre [m]
    A = np.asarray(A, dtype=float)          # Cross-sectional area [m^2]
    e = np.asarray(e, dtype=float)          # Centroid radius minus neutral axis radius [m]
    Rna = np.asarray(Rna, dtype=float)      # Radius to the neutral axis [m]
    return M * y / (A * e * (Rna - y))      # Bending stress [Pa]


# ---------------------------------------------------------------------------
# Beams: load, shear, moment and deflection diagrams        (ch. 11, ch. 16)
# ---------------------------------------------------------------------------
# Euler-Bernoulli beams spanning [0, L], with x measured from the left end and
# forces/deflections positive upward. Internal shear V and bending moment M
# follow the book's convention (sagging moment positive). Reactions come from
# equilibrium plus zero deflection at every support and zero slope at every
# fixed support, so statically indeterminate beams are handled automatically.

# Cumulative trapezoidal integral of samples y on a uniform grid of spacing dx,
# starting from zero at the first sample
def cumulative_trapz(y, dx):
    out = np.zeros_like(y)
    out[1:] = np.cumsum((y[:-1] + y[1:]) / 2.0) * dx
    return out

# Plot the shear force, bending moment and deflection of a beam of length L.
# P is a list of point forces as (magnitude [N], position [m]) pairs and C a
# list of point moments (couples) as (magnitude [N*m], position [m]) pairs,
# couples counterclockwise positive just as forces are upward positive.
# w(x) [N/m], E(x) [Pa] and I(x) [m^4] are sympy expressions in the symbol x
# (constants are fine), and pinned and fixed list the support positions [m].
# Everything is sampled on n grid points, so results are accurate to the grid
# spacing L/(n-1). Returns the grid and V [N], M [N*m] and v [mm] on it
def beam_diagrams(L, P, w, C, pinned, fixed, I, x, E, n=1001):

    # uniform grid along the beam
    xs = np.linspace(0.0, float(L), n)
    dx = xs[1] - xs[0]

    # sample the load and bending stiffness; multiplying by ones broadcasts
    # constant expressions to the whole grid
    w_vals = sp.lambdify(x, sp.sympify(w), "numpy")(xs) * np.ones_like(xs)
    E_vals = sp.lambdify(x, sp.sympify(E), "numpy")(xs) * np.ones_like(xs)
    I_vals = sp.lambdify(x, sp.sympify(I), "numpy")(xs) * np.ones_like(xs)
    EI = E_vals * I_vals                         # bending stiffness [N*m^2]

    # shear and moment produced by the applied loads alone
    V0 = cumulative_trapz(w_vals, dx)            # dV/dx = w
    M0 = cumulative_trapz(V0, dx)                # dM/dx = V
    for Pi, a in P:
        V0 = V0 + Pi * (xs >= a)                 # a point force is a jump in V...
        M0 = M0 + Pi * np.maximum(xs - a, 0.0)   # ...and a kink in M
    for Mi, a in C:
        M0 = M0 - Mi * (xs >= a)                 # a counterclockwise couple drops M

    # shear and moment produced by a unit reaction at each support: an upward
    # force at every support, then a moment at every fixed support
    supports = list(pinned) + list(fixed)
    Vu = [1.0 * (xs >= a) for a in supports] + [0.0 * xs for _ in fixed]
    Mu = [np.maximum(xs - a, 0.0) for a in supports] + [1.0 * (xs >= a) for a in fixed]

    # slope and deflection of every contribution, from EI * v'' = M (the
    # integration constants C1, C2 are added at the end)
    th0 = cumulative_trapz(M0 / EI, dx)
    v0 = cumulative_trapz(th0, dx)
    thu = [cumulative_trapz(Mi / EI, dx) for Mi in Mu]
    vu = [cumulative_trapz(ti, dx) for ti in thu]

    # linear system for the unknowns [reactions..., C1, C2]: shear and moment
    # vanish past the right end (equilibrium), deflection is zero at every
    # support and slope is zero at every fixed support (compatibility).
    # Supports are snapped to the nearest grid point
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

    # superpose the solved reactions and integration constants
    R, C1, C2 = u[:-2], u[-2], u[-1]
    V = V0 + sum(Ri * Vi for Ri, Vi in zip(R, Vu))
    M = M0 + sum(Ri * Mi for Ri, Mi in zip(R, Mu))
    v = v0 + sum(Ri * vi for Ri, vi in zip(R, vu)) + C1 * xs + C2
    v = v * 1000.0                               # deflection in mm

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
    return xs, V, M, v                           # Grid [m], shear [N], moment [N*m], deflection [mm]


# ---------------------------------------------------------------------------
# Transverse shear                                                 (ch. 12)
# ---------------------------------------------------------------------------

# First moment Q, about the neutral axis, of the part of a cross-section above
# a horizontal cut at height y, for a section built up from rectangular
# sub-areas [xc, yc, base, height]. Also returns the width t of the section at
# the cut, so the shear stress there is tau = V*Q/(I*t) and the shear flow is
# q = V*Q/I. The neutral axis is found internally, so y and the sections only
# need to share an origin; y may be an array to sample Q over the depth. A
# negative base subtracts a hole. On a joint between rectangles of different
# width, t is the narrower one, where tau is larger
def Q_rectangles(y, *sections):
    y = np.asarray(y, dtype=float)                  # Height of the cut [m]
    # one row [xc, yc, base, height] per rectangle
    sec = np.asarray(sections, dtype=float)
    if sec.ndim != 2 or sec.shape[1] != 4:
        raise ValueError("each section must be a 4-vector [xc, yc, base, height]")
    xc, yc, b, h = sec.T
    y_na = (b * h * yc).sum() / (b * h).sum()       # height of the neutral axis [m]
    bot, top = yc - h / 2.0, yc + h / 2.0
    # part of each rectangle above the cut; the trailing axis runs over the rectangles
    yy = y[..., None]
    lo = np.clip(yy, bot, top)
    Q = (b * (top - lo) * ((top + lo) / 2.0 - y_na)).sum(axis=-1)
    # widths just above and just below the cut, with a small tolerance so a
    # cut placed exactly on a joint sees both rectangles
    tol = 1e-9 * (top.max() - bot.min())
    t_up = (b * ((bot - tol <= yy) & (yy < top - tol))).sum(axis=-1)
    t_dn = (b * ((bot + tol < yy) & (yy <= top + tol))).sum(axis=-1)
    t_min = np.minimum(t_up, t_dn)
    t = np.where(t_min > 0.0, t_min, np.maximum(t_up, t_dn))   # outer faces only see one side
    return Q, t                                     # First moment [m^3], width at the cut [m]

# Maximum transverse shear stress, at the neutral axis, of a solid or tubular
# circular section: tau = V*Q/(I*t) with Q = 2/3*(ro^3 - ri^3) and
# t = 2*(ro - ri). Reduces to 4V/(3A) for a solid section and tends to 2V/A
# for a thin tube
def tau_shearTube(V, outer_radius, inner_radius=0.0):
    V = np.asarray(V, dtype=float)              # Shear force [N]
    ro = np.asarray(outer_radius, dtype=float)  # Outer radius [m]
    ri = np.asarray(inner_radius, dtype=float)  # Inner radius [m] (0 for a solid section)
    return 4.0 * V * (ro**2 + ro * ri + ri**2) / (3.0 * np.pi * (ro**4 - ri**4))   # Max shear stress [Pa]


# ---------------------------------------------------------------------------
# Combined loadings                                                (ch. 13)
# ---------------------------------------------------------------------------
# Thin-walled pressure vessels under internal gauge pressure, with closed
# ends. The thin-walled assumption holds when r/t >= 10.

# Hoop (circumferential) stress in a thin-walled cylindrical vessel
def sigma_hoopPV(pressure, radius, thickness):
    p = np.asarray(pressure, dtype=float)   # Internal gauge pressure [Pa]
    r = np.asarray(radius, dtype=float)     # Inner radius [m]
    t = np.asarray(thickness, dtype=float)  # Wall thickness [m]
    return p * r / t                        # Hoop (circumferential) stress [Pa]

# Longitudinal (axial) stress in a thin-walled cylindrical vessel with closed ends
def sigma_longPV(pressure, radius, thickness):
    p = np.asarray(pressure, dtype=float)   # Internal gauge pressure [Pa]
    r = np.asarray(radius, dtype=float)     # Inner radius [m]
    t = np.asarray(thickness, dtype=float)  # Wall thickness [m]
    return p * r / (2.0 * t)                # Longitudinal (axial) stress [Pa]

# Membrane stress, equal in every direction, in a thin-walled spherical vessel
def sigma_spherePV(pressure, radius, thickness):
    p = np.asarray(pressure, dtype=float)   # Internal gauge pressure [Pa]
    r = np.asarray(radius, dtype=float)     # Inner radius [m]
    t = np.asarray(thickness, dtype=float)  # Wall thickness [m]
    return p * r / (2.0 * t)                # Membrane stress [Pa]

# State of stress at a point of a solid or tubular circular section that
# carries the internal loads returned by internal_loads(): normal force N,
# shear force vector V, torque T and bending moment vector M, with n the
# section's outward normal and rho the position of the point measured from
# the centre of the section. Superposes N/A, the flexure formula, the torsion
# formula T*rho/J and the shear formula V*Q/(I*t). Q and t belong to the chord
# through the point perpendicular to V, which is exact on the neutral axis of
# V and the book's approximation elsewhere
def stress_circularSection(N, V, T, M, n, rho, outer_radius, inner_radius=0.0):
    N = float(N)                                        # Normal force [N]
    V = np.asarray(V, dtype=float)                      # Shear force vector [N]
    T = float(T)                                        # Torque [N*m]
    M = np.asarray(M, dtype=float)                      # Bending moment vector [N*m]
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)  # Outward unit normal of the section [-]
    rho = np.asarray(rho, dtype=float)                  # Position of the point from the centre [m]
    rho = rho - (rho @ n) * n                           # keep only its in-plane part
    ro = float(outer_radius)                            # Outer radius [m]
    ri = float(inner_radius)                            # Inner radius [m] (0 for a solid section)
    A = np.pi * (ro**2 - ri**2)
    I = np.pi * (ro**4 - ri**4) / 4.0
    J = 2.0 * I
    # axial force plus bending: (M x rho) . n / I gives -Mz*y/I + My*z/I
    sigma = N / A + np.cross(M, rho) @ n / I
    # torsion: the shear stress circulates about n in the sense of T
    tau = T * np.cross(n, rho) / J
    # transverse shear: V*Q/(I*t) along V, over the chord at distance s from
    # the centre, measured along V
    V_mag = np.linalg.norm(V)
    if V_mag > 0.0:
        s = abs(rho @ V) / V_mag
        c_o = np.sqrt(max(ro**2 - s**2, 0.0))           # half-chord in the outer circle
        c_i = np.sqrt(max(ri**2 - s**2, 0.0))           # half-chord in the bore
        Q = 2.0 / 3.0 * (c_o**3 - c_i**3)
        t = 2.0 * (c_o - c_i)
        if t > 0.0:
            tau = tau + V * Q / (I * t)
    return sigma, tau       # Normal stress [Pa] (tension positive), shear stress vector on the section [Pa]
