# -*- coding: utf-8 -*-
# Paraopt is a simple parallel optimization toolbox.
# Copyright (C) 2012-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Paraopt.
#
# Paraopt is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Paraopt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--


import numpy as np
from paraopt import *
from common import *




def test_harmonic1():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 1)
        p, status = fmin_async(harmonic, x0, 1.0, wtol=1e-2, max_iter=10000, verbose=50)
        assert status == 'CONVERGED_SIGMA'
        assert harmonic(p.best) < 1e-3


def test_harmonic2():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        p, status = fmin_async(harmonic, x0, 1.0, max_iter=10000, verbose=50)
        assert status == 'CONVERGED_SIGMA'
        assert harmonic(p.best) < 1e-3


def test_harmonic_noise():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        p, status = fmin_async(harmonic_noise, x0, 1.0, wtol=1e-2, max_iter=10000, verbose=50)
        assert status == 'CONVERGED_SIGMA'
        assert harmonic(p.best) < 1e-2


def test_harmonic_noise_loss():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        p, status = fmin_async(harmonic_noise, x0, 1.0, wtol=1e-2, max_iter=10000, verbose=50, loss_rate=0.5)
        assert status == 'CONVERGED_SIGMA'
        assert harmonic(p.best) < 1e-2


def test_rosenbrock2():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        p, status = fmin_async(rosenbrock, x0, 1.0, max_iter=20000, wtol=1e-10, npop=None, verbose=100)
        assert status == 'CONVERGED_SIGMA' or status == 'FAILED_DEGENERATE'
        if status == 'CONVERGED_SIGMA':
            assert rosenbrock(p.best) < 1e-2


def test_rosenbrock2_loss():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        p, status = fmin_async(rosenbrock, x0, 1.0, max_iter=20000, wtol=1e-10, npop=None, verbose=100, loss_rate=0.1)
        assert status == 'CONVERGED_SIGMA' or status == 'FAILED_DEGENERATE'
        if status == 'CONVERGED_SIGMA':
            assert rosenbrock(p.best) < 1e-2


def test_callback():
    lcb = LogCallback()
    x0 = np.random.uniform(2, 3, 2)
    p, status = fmin_async(harmonic, x0, 1.0, max_iter=10, callback=lcb)
    assert len(lcb.log) == 10
    for item in lcb.log:
        assert item == ((p,),{})


def test_reject_errors():
    x0 = np.random.uniform(2, 3, 2)
    p, status = fmin_async(failing, x0, 1.0, max_iter=10, reject_errors=True)


def test_best_empty():
    x0 = np.random.uniform(2, 3, 2)
    p, status = fmin_async(failing, x0, 1.0, max_iter=10, reject_errors=True)
    p.members = []
    assert (p.best == p.m).all()
