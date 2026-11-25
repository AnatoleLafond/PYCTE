"""Tests of pycte/Source/nahed.py (coupling matrix of Nahed & de Lamare 2025, no reordering)."""
import numpy as np
import pytest

from pycte.Source import nahed


def coo(A):
    r, c = np.nonzero(A)
    return r, c, A[r, c]


def test_phi_two_blocks_by_hand():
    # 2 blocks of 2 unknowns : rows of block 0 perturbed by block 1 three times more than by block 0, block 1 not
    # perturbed by block 0
    J = np.array([[2.0, 0.0, 6.0, 0.0],
                  [0.0, 2.0, 0.0, 6.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    du = np.ones(4)
    blockOf = np.array([0, 0, 1, 1])
    Y = nahed.blockPerturbations(*coo(J), du, blockOf, 2)
    assert np.allclose(Y, [[2, 6], [2, 6], [0, 1], [0, 1]])
    assert np.allclose(nahed.phiMatrix(Y, blockOf, 2), [[1, 3], [0, 1]])


def test_phi_is_the_mean_of_the_nodal_ratios():
    # eq. 13 : nodal ratios 1 and 3 in the rows of block 0 -> Phi_01 = 2
    J = np.array([[1.0, 0, 1.0, 0], [0, 4.0, 0, 12.0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    blockOf = np.array([0, 0, 1, 1])
    Y = nahed.blockPerturbations(*coo(J), np.ones(4), blockOf, 2)
    assert nahed.phiMatrix(Y, blockOf, 2)[0, 1] == pytest.approx(2.0)


def test_phi_is_signed_with_an_absolute_eps():
    # eq. 13, 14, 38 : nodal ratios +1 and -1 average to 0 (absolute value taken after the mean)
    J = np.array([[1.0, 0, 1.0, 0], [0, 1.0, 0, -1.0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    blockOf = np.array([0, 0, 1, 1])
    Y = nahed.blockPerturbations(*coo(J), np.ones(4), blockOf, 2)
    assert abs(nahed.phiMatrix(Y, blockOf, 2)[0, 1]) < 1e-12
    # eps absolute : a row whose own perturbation is 1e-9 counts its ratio as Y / (1e-9 + 1e-9)
    J = np.array([[1e-9, 1e-9], [0.0, 1.0]])
    P = nahed.phiMatrix(nahed.blockPerturbations(*coo(J), np.ones(2), np.array([0, 1]), 2), np.array([0, 1]), 2)
    assert P[0, 1] == pytest.approx(0.5)
    # a zero denominator counts as 0
    J = np.array([[-1e-9, 1.0], [0.0, 1.0]])
    P = nahed.phiMatrix(nahed.blockPerturbations(*coo(J), np.ones(2), np.array([0, 1]), 2), np.array([0, 1]), 2)
    assert P[0, 1] == 0.0


def test_monitor(tmp_path):
    # a perturbed by b three times more than by itself, b perturbed by a by 0.1
    J = np.array([[1.0, 0, 3.0, 0], [0, 1.0, 0, 3.0], [0.1, 0, 1.0, 0], [0, 0.1, 0, 1.0]])
    r, c, v = coo(J)
    data = {'rows': r, 'cols': c, 'vals': v, 'du': np.ones(4), 'blockOf': np.array([0, 0, 1, 1])}
    path = tmp_path / 'couplingMatrix.txt'
    monitor = nahed.CouplingMonitor(str(path))
    monitor.initial(['a', 'b'])
    assert monitor.describe() == 'no coupling matrix yet'
    P = monitor.afterStep(data, t=10.0, h=2.0)
    assert np.allclose(P, [[1, 3], [0.1, 1]]) and monitor.Phi is P
    assert monitor.strongest() == (pytest.approx(3.0), 'a', 'b')
    # no increment : nothing measured, the last matrix kept
    assert monitor.afterStep(dict(data, du=np.zeros(4))) is None and monitor.Phi is P
    assert len(monitor.history) == 1 and monitor.history[0][:2] == (10.0, 2.0)
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines[0].split('\t') == ['t_s', 'h_s', 'Phi(a,b)', 'Phi(b,a)']
    assert [float(x) for x in lines[1].split('\t')] == pytest.approx([10.0, 2.0, 3.0, 0.1])
    # monitorFor : off, then kept in centralDict
    assert nahed.monitorFor({'nahed': None}) is None
    cd = {'nahed': True, 'nahedMonitor': monitor}
    assert nahed.monitorFor(cd) is monitor


def test_epsilon_given_by_the_user(tmp_path):
    # rows of 1e-12 mol/s : eps = 1e-9 dominates the denominator, eps = 1e-15 does not
    J = 1e-12 * np.array([[1.0, 3.0], [0.0, 1.0]])
    r, c, v = coo(J)
    data = {'rows': r, 'cols': c, 'vals': v, 'du': np.ones(2), 'blockOf': np.array([0, 1])}
    assert nahed.couplingMatrix(data, 2)[0, 1] == pytest.approx(3e-12 / (1e-12 + 1e-9))
    assert nahed.couplingMatrix(data, 2, 1e-15)[0, 1] == pytest.approx(3e-12 / (1e-12 + 1e-15))
    # monitorFor reads the setting nahedEpsilon (default 1e-9), also for a monitor kept from a former call
    cd = {'nahed': True, 'nahedMonitor': nahed.CouplingMonitor(str(tmp_path / 'c.txt'))}
    assert nahed.monitorFor(cd).eps == 1e-9
    cd['nahedEpsilon'] = 0.0
    monitor = nahed.monitorFor(cd)
    monitor.initial(['a', 'b'])
    assert monitor.afterStep(data)[0, 1] == pytest.approx(3.0)
    # the setting is checked
    from pycte.Source import keywordValidation
    keywordValidation.checkSetting('nahedEpsilon', 1e-15)
    with pytest.raises(keywordValidation.KeywordError):
        keywordValidation.checkSetting('nahedEpsilon', -1.0)


def test_read_and_plot_coupling_matrix(tmp_path):
    J = np.array([[1.0, 0, 3.0, 0], [0, 1.0, 0, 3.0], [0.1, 0, 1.0, 0], [0, 0.1, 0, 1.0]])
    r, c, v = coo(J)
    data = {'rows': r, 'cols': c, 'vals': v, 'du': np.ones(4), 'blockOf': np.array([0, 0, 1, 1])}
    path = tmp_path / 'couplingMatrix.txt'
    monitor = nahed.CouplingMonitor(str(path), eps=0.0)
    monitor.initial(['Fe(OH)3', 'psi'])
    monitor.afterStep(data, t=60.0, h=60.0)
    monitor.afterStep(dict(data, vals=2 * v), t=120.0, h=60.0)
    names, t, h, Phi = nahed.readCouplingMatrix(str(path))
    assert names == ['Fe(OH)3', 'psi'] and np.allclose(t, [60, 120]) and np.allclose(h, 60)
    assert np.allclose(Phi[0], [[1, 3], [0.1, 1]]) and np.allclose(Phi[1], Phi[0])
    pytest.importorskip('matplotlib')
    out = tmp_path / 'phi.png'
    nahed.plotCouplingMatrix(str(path), out=str(out), timeUnit='min')
    assert out.stat().st_size > 0


def _stencilResidual(width):
    # nonlinear residual of 3 blocks on 7 nodes, row of node l depending on the nodes l - width .. l + width ;
    # returns the residual and its exact Jacobian
    k, N = 3, 7
    rng = np.random.default_rng(1)
    A = rng.normal(size=(k, k, 2 * width + 1))

    def residual(u):
        U = u.reshape(N, k)
        R = np.zeros((N, k))
        for l in range(N):
            for o in range(-width, width + 1):
                if 0 <= l + o < N:
                    R[l] += A[:, :, o + width] @ np.sin(U[l + o]) + 0.5 * (U[l + o] ** 2).sum()
        return R.ravel()

    def jacobian(u):
        U = u.reshape(N, k)
        J = np.zeros((N * k, N * k))
        for l in range(N):
            for o in range(-width, width + 1):
                if 0 <= l + o < N:
                    blk = A[:, :, o + width] * np.cos(U[l + o])[None, :] + U[l + o][None, :]
                    J[l * k:(l + 1) * k, (l + o) * k:(l + o + 1) * k] += blk
        return J
    return residual, jacobian, k, N


@pytest.mark.parametrize("width", [1, 2])
def test_fd_jacobian_matches_the_exact_jacobian(width):
    residual, jacobian, k, N = _stencilResidual(width)
    u = np.random.default_rng(2).normal(size=N * k)
    u[1::k] = 0.0                                      # a block zero everywhere (step floor)
    r, c, v = nahed.fdJacobian(residual, u, k, width)
    J = np.zeros((N * k, N * k))
    np.add.at(J, (r, c), v)
    Jx = jacobian(u)
    assert np.max(np.abs(J - Jx)) < 1e-5 * np.max(np.abs(Jx))
    # outside the stencil : nothing
    rows, cols = np.nonzero(J)
    assert np.all(np.abs(rows // k - cols // k) <= width)


def test_fd_jacobian_rejects_a_bad_layout():
    with pytest.raises(ValueError):
        nahed.fdJacobian(lambda u: u, np.ones(7), 3)


def test_check_jacobian():
    residual, jacobian, k, N = _stencilResidual(1)
    u = np.random.default_rng(3).normal(size=N * k)
    du = 1e-4 * np.random.default_rng(4).normal(size=N * k)
    J = jacobian(u)
    r, c = np.nonzero(J)
    ok = nahed.checkJacobian(residual, u, du, r, c, J[r, c], k, 1)
    assert ok['jac'] < 1e-5 and ok['lin'] < 1e-2 and ok['phi'] < 1e-3
    bad = nahed.checkJacobian(residual, u, du, r, c, 1.5 * J[r, c], k, 1)
    assert bad['jac'] > 0.3 and bad['lin'] > 0.3
