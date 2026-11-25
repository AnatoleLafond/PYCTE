"""
Coupling matrix of C. Nahed, J. de Lamare, Dynamic characterization of cross-physics coupling strengths, a methodology
to coupling and reordering partitioned solvers for multiphysics applications, Computers and Structures 307 (2025)
107615 : measurement only, no reordering and no partition (the Newton solver is unchanged).

k physics (blocks of unknowns) solved by Newton. After each converged time step n the solver hands over its Jacobian
J = [dR/du]^n evaluated at the converged solution, the increment du = u^n - u^(n-1) and the block of every unknown
(CouplingMonitor.afterStep). The coupling strengths are the signed nodal ratios (J_ij du_j) / (J_ii du_i + eps), eps
absolute in mol/s given by the setting nahedEpsilon (default 1e-9, value of the article ; it must stay well below the
perturbations of the rows, which are often 1e-12 to 1e-8 mol/s for small cells or long steps), averaged over the rows of block i (eq. 7, 9, 13, 38, 39) ; |Phi_ij| (eq. 14) is the strength of the
coupling of block i by block j : below 1 weak, above 1 strong.

Independent of the transport : nativeTransport only gives the data (setting nahed=True), in every transport mode.
J is the Jacobian of the equation of the time step, in mol/s, with the time scheme of the mode (theta of the implicit
ADE, implicit Euler for the implicit Nernst-Planck, and the equivalent implicit Euler residual for the explicit
schemes), the lagged terms frozen ; it is analytic when the solver assembles it, otherwise computed by fdJacobian.
The matrix of every step is written in couplingMatrix.txt (current folder) and kept in CouplingMonitor.Phi / history.
"""
import numpy as np

EPS = 1e-9                      # default eps of eq. 9 (absolute, 10^-9 as in eq. 38-40) ; setting 'nahedEpsilon'
FILE = 'couplingMatrix.txt'     # one row per accepted step : t, h, then Phi_ij (i != j, signed)


def blockPerturbations(rows, cols, vals, du, blockOf, k):
    """Y[r, j] = sum over the columns c of block j of J[r, c].du[c] : partial perturbation of row r by block j (eq. 7).
    J given by its entries (rows, cols, vals), duplicates summed."""
    n = len(du)
    w = np.asarray(vals, dtype=float) * du[cols]
    return np.bincount(rows * k + blockOf[cols], weights=w, minlength=n * k).reshape(n, k)


def phiMatrix(Y, blockOf, k, eps=EPS):
    """Averaged Phi_ij (eq. 9, 13, 38, 39) : mean over the rows r of block i of the signed nodal ratio
    Y[r, j] / (Y[r, i] + eps), the perturbation of the row by block j relative to its perturbation by its own block i,
    eps absolute. Signed : the absolute value is taken afterwards, on the mean (eq. 14). A nodal ratio whose
    denominator is exactly zero is counted as 0."""
    own = Y[np.arange(len(blockOf)), blockOf]
    den = own + eps
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((den != 0)[:, None], Y / den[:, None], 0.0)
    P = np.zeros((k, k))
    np.add.at(P, blockOf, ratio)
    P /= np.maximum(np.bincount(blockOf, minlength=k), 1)[:, None]
    np.fill_diagonal(P, 1.0)
    return P


def fdJacobian(residual, u, k, width=1, rel=1e-7):
    """Jacobian of residual(u) by forward differences, for unknowns ranged node by node (u[l.k + b] : block b at node
    l) and a residual whose row of node l depends only on the unknowns of the nodes l - width .. l + width. Columns of
    the same block whose nodes are 2.width + 1 apart do not share any row : they are perturbed together ((2.width + 1).k
    + 1 residual evaluations). Step : rel.max(|u_c|, 1e-3.max |u| of the block) ; for a block zero everywhere,
    rel.(median of the max |u| of the other blocks), so that the change of the residual stays above its round-off.
    Returns (rows, cols, vals), non-zero entries only."""
    u = np.asarray(u, dtype=float)
    n = len(u)
    if n % k:
        raise ValueError(f"{n} unknowns are not a multiple of the {k} blocks")
    N = n // k
    node, block = np.arange(n) // k, np.arange(n) % k
    scale = np.array([np.max(np.abs(u[b::k]), initial=0.0) for b in range(k)])
    positive = scale[scale > 0]
    typical = float(np.median(positive)) if positive.size else 1.0
    floor = np.where(scale > 0, 1e-3 * scale, typical)
    step = rel * np.maximum(np.abs(u), floor[block])
    R0 = np.asarray(residual(u), dtype=float).ravel()
    period = 2 * width + 1
    rows, cols, vals = [], [], []
    for a in range(min(period, N)):
        for b in range(k):
            grp = np.flatnonzero((node % period == a) & (block == b))
            if len(grp) == 0:
                continue
            up = u.copy()
            up[grp] += step[grp]
            dR = np.asarray(residual(up), dtype=float).ravel() - R0
            for o in range(-width, width + 1):
                tgt = node[grp] + o
                ok = (tgt >= 0) & (tgt < N)
                c = np.repeat(grp[ok], k)
                r = (tgt[ok][:, None] * k + np.arange(k)[None, :]).ravel()
                v = dR[r] / step[c]
                nz = v != 0
                rows.append(r[nz])
                cols.append(c[nz])
                vals.append(v[nz])
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def checkJacobian(residual, u, du, rows, cols, vals, k, width=1):
    """Verification (tests) of the Jacobian J (rows, cols, vals) given for the residual at u (unknowns ranged node by
    node) : jac = largest difference with fdJacobian, row by row, relative to the largest entry of the row ; lin =
    |R(u) - R(u - du) - J.du| / |R(u) - R(u - du)| (max norms, first order : small when du is small) ; phi = largest
    difference between the coupling matrices of J and of fdJacobian."""
    from scipy.sparse import csr_matrix
    u, du = np.asarray(u, dtype=float), np.asarray(du, dtype=float)
    n = len(u)
    Ja = csr_matrix((np.asarray(vals, dtype=float), (rows, cols)), shape=(n, n))
    fr, fc, fv = fdJacobian(residual, u, k, width)
    Jf = csr_matrix((fv, (fr, fc)), shape=(n, n))
    D = abs(Ja - Jf).max(axis=1).toarray().ravel()
    scale = np.maximum(abs(Ja).max(axis=1).toarray().ravel(), abs(Jf).max(axis=1).toarray().ravel())
    with np.errstate(divide='ignore', invalid='ignore'):
        jac = float(np.max(np.where(scale > 0, D / scale, 0.0)))
    R1, R0 = np.asarray(residual(u)).ravel(), np.asarray(residual(u - du)).ravel()
    dR = R1 - R0
    lin = float(np.max(np.abs(dR - Ja @ du)) / max(float(np.max(np.abs(dR))), 1e-300))
    blockOf = np.arange(n) % k
    Pa = couplingMatrix({'rows': rows, 'cols': cols, 'vals': vals, 'du': du, 'blockOf': blockOf}, k)
    Pf = couplingMatrix({'rows': fr, 'cols': fc, 'vals': fv, 'du': du, 'blockOf': blockOf}, k)
    phi = None if Pa is None or Pf is None else float(np.max(np.abs(Pa - Pf)))
    return {'jac': jac, 'lin': lin, 'phi': phi, 'res': float(np.max(np.abs(R1))), 'dres': float(np.max(np.abs(dR)))}


def couplingMatrix(data, k, eps=EPS):
    """Phi of one step from the solver data ('rows', 'cols', 'vals', 'du', 'blockOf'), None without increment."""
    du = np.asarray(data['du'], dtype=float)
    if data['vals'] is None or not np.any(du):
        return None
    return phiMatrix(blockPerturbations(data['rows'], data['cols'], data['vals'], du, data['blockOf'], k),
                     data['blockOf'], k, eps)


class CouplingMonitor:
    """Coupling matrix of a Newton solver, step after step : initial(names) at the start of a transport call,
    afterStep(data) after each accepted step. Phi : last matrix ; history : [(t, h, Phi), ...] of every step.
    eps : eps of the denominators (eq. 9, 38-39), absolute, in the unit of the rows of J (mol/s)."""

    def __init__(self, path=FILE, eps=EPS):
        self.names, self.Phi, self.history, self.steps = None, None, [], 0
        self.path = path
        self.eps = float(eps)
        if path:
            open(path, 'w', encoding='utf-8').close()

    def initial(self, names):
        names = list(names)
        if names != self.names:
            self.names, self.Phi = names, None
            if self.path:
                k = len(names)
                head = ['t_s', 'h_s'] + [f"Phi({names[i]},{names[j]})" for i in range(k) for j in range(k) if i != j]
                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write('\t'.join(head) + '\n')

    def afterStep(self, data, t=None, h=None):
        """data : 'rows', 'cols', 'vals' (Newton matrix entries), 'du' (increment of the step), 'blockOf' (block of
        every unknown). Returns Phi of the step (None if no increment)."""
        self.steps += 1
        P = couplingMatrix(data, len(self.names), self.eps)
        if P is None:
            return None
        self.Phi = P
        self.history.append((t, h, P))
        if self.path:
            k = len(P)
            vals = [P[i, j] for i in range(k) for j in range(k) if i != j]
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write('\t'.join(f"{v:.6g}" for v in [t if t is not None else np.nan, h if h is not None else np.nan]
                                  + vals) + '\n')
        return P

    def strongest(self):
        """(|Phi_ij|, i name, j name) of the strongest coupling of the last matrix, None before the first one."""
        if self.Phi is None:
            return None
        A = np.abs(self.Phi)
        np.fill_diagonal(A, 0.0)
        i, j = np.unravel_index(int(np.argmax(A)), A.shape)
        return float(A[i, j]), self.names[i], self.names[j]

    def describe(self):
        s = self.strongest()
        return 'no coupling matrix yet' if s is None else f"max |Phi| = {s[0]:.3g} ({s[1]} perturbed by {s[2]})"


def readCouplingMatrix(path=FILE):
    """(names, t, h, Phi) from a couplingMatrix.txt : t, h (s) of every step, Phi[step, i, j] signed (diagonal 1).
    A file holds a new header when the blocks change : the last block list is read, with all its steps."""
    sections, names, rows = [], None, []
    with open(path, encoding='utf-8') as f:
        for line in f:
            cells = line.rstrip('\n').split('\t')
            if not cells or cells == ['']:
                continue
            if cells[0] == 't_s':
                if names is not None:
                    sections.append((names, rows))
                pairs = [c[4:-1].split(',') for c in cells[2:]]
                names = list(dict.fromkeys([p[0] for p in pairs] + [p[1] for p in pairs]))
                order = {n: i for i, n in enumerate(names)}
                index = [(order[a], order[b]) for a, b in pairs]
                rows = []
            else:
                rows.append([float(x) for x in cells])
    if names is None:
        raise ValueError(f"{path} : no header (t_s, h_s, Phi(i,j) ...)")
    sections.append((names, rows))
    names = sections[-1][0]
    rows = [r for nm, rs in sections if nm == names for r in rs]
    k = len(names)
    A = np.array(rows, dtype=float).reshape(len(rows), 2 + k * (k - 1))
    Phi = np.zeros((len(rows), k, k))
    for col, (i, j) in enumerate(index):
        Phi[:, i, j] = A[:, 2 + col]
    Phi[:, np.arange(k), np.arange(k)] = 1.0
    return names, A[:, 0], A[:, 1], Phi


TIME_UNITS = {'s': 1.0, 'min': 60.0, 'h': 3600.0, 'd': 86400.0, 'y': 365.25 * 86400.0}
# fixed categorical order (colour follows the perturbing block, the same in every panel)
COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']


def plotCouplingMatrix(path=FILE, out=None, timeUnit='s', blocks=None, logTime=False, ylim=None):
    """|Phi_ij| against time from a couplingMatrix.txt : one panel per perturbed block i, one line per perturbing
    block j (same colour in every panel ; a pair zero at every step is not drawn), log scale, dashed line |Phi| = 1
    (weak below, strong above, eq. 14). timeUnit : 's', 'min', 'h', 'd' or 'y'. ylim : (min, max) of |Phi|, default
    the 1st - 99th percentiles of the non-zero values (spikes of near-zero denominators cut).
    blocks : names of the perturbed blocks to show (default all) ; with more than 8 perturbing blocks, the 8 strongest
    (median of |Phi| over all steps and panels) get a colour, the others one grey line 'others (max)'.
    out : figure file (png, pdf, svg...), else the figure is shown. Returns the matplotlib figure."""
    import matplotlib
    if out is not None:
        matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    names, t, h, Phi = readCouplingMatrix(path)
    k = len(names)
    x = t / TIME_UNITS[timeUnit]
    A = np.abs(Phi)
    shown = [names.index(b) for b in blocks] if blocks else list(range(k))
    off = A.copy()
    off[:, np.arange(k), np.arange(k)] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)         # a pair zero at every step : all-NaN
        strength = np.nanmedian(np.where(off > 0, off, np.nan), axis=(0, 1))
    strength = np.nan_to_num(strength, nan=0.0)
    coloured = sorted(np.argsort(-strength, kind='stable')[:len(COLOURS)]) if k > len(COLOURS) else list(range(k))
    colourOf = {j: COLOURS[n] for n, j in enumerate(coloured)}
    nCol = min(3, len(shown))
    nRow = -(-len(shown) // nCol)
    fig, axes = plt.subplots(nRow, nCol, figsize=(4.2 * nCol, 3.0 * nRow), sharex=True, sharey=True, squeeze=False)
    if ylim is None:
        # the spikes of the steps where J_ii.du_i + eps is close to 0 (eq. 13) would flatten everything else : the
        # range is cut at the 1st and 99th percentiles of the non-zero |Phi_ij|
        positive = off[off > 0]
        lo, hi = (np.quantile(positive, 0.01), np.quantile(positive, 0.99)) if positive.size else (1e-3, 1e1)
        ylim = (min(lo, 0.1) / 3, max(hi, 10.0) * 3)
    for ax, i in zip(axes.flat, shown):
        others = [j for j in range(k) if j != i and j not in colourOf]
        drawn = 0
        for j in range(k):
            if j == i or j not in colourOf or not np.any(A[:, i, j] > 0):
                continue
            y = np.where(A[:, i, j] > 0, A[:, i, j], np.nan)
            ax.plot(x, y, color=colourOf[j], lw=1.5, label=names[j])
            drawn += 1
        if others:
            y = np.max(A[:, i, others], axis=1)
            if np.any(y > 0):
                ax.plot(x, np.where(y > 0, y, np.nan), color='#a3a29b', lw=1.2, label='others (max)')
                drawn += 1
        if not drawn:
            ax.text(0.5, 0.5, 'Φ = 0 at every step', transform=ax.transAxes, ha='center', va='center',
                    color='#52514e', fontsize=9)
        ax.axhline(1.0, color='#52514e', lw=1.0, ls='--')
        ax.set_yscale('log')
        if logTime:
            ax.set_xscale('log')
        ax.set_ylim(*ylim)
        ax.set_title(f"{names[i]} perturbed by", fontsize=10, loc='left', color='#0b0b0b')
        ax.grid(True, which='major', color='#e4e3df', lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors='#52514e', labelsize=8)
        if drawn:
            ax.legend(fontsize=8, frameon=False, loc='best')
    for ax in list(axes.flat)[len(shown):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"time ({timeUnit})", color='#52514e', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('|Φ_ij|', color='#52514e', fontsize=9)
    fig.suptitle("Coupling strengths (Nahed & de Lamare 2025) — strong above the dashed line |Φ| = 1",
                 fontsize=11, x=0.01, ha='left', color='#0b0b0b')
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def monitorFor(centralDict):
    """Monitor kept in centralDict['nahedMonitor'] from one transport call to the next, None if 'nahed' is off. Its eps
    is the setting 'nahedEpsilon' (EPS if not given)."""
    if not centralDict.get('nahed'):
        return None
    eps = centralDict.get('nahedEpsilon')
    eps = EPS if eps is None else float(eps)
    monitor = centralDict.get('nahedMonitor')
    if monitor is None:
        monitor = CouplingMonitor(eps=eps)
        centralDict['nahedMonitor'] = monitor
    monitor.eps = eps
    return monitor
