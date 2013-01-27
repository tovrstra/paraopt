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
        cm, status = fmin_cma_async(harmonic, x0, 1.0, wtol=1e-3, max_iter=10000, verbose=True)
        assert status == 'CONVERGED_WIDTH'
        assert harmonic(cm.m) < 1e-3


def test_harmonic2():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma_async(harmonic, x0, 1.0, max_iter=10000, verbose=True)
        assert status == 'CONVERGED_WIDTH'
        assert harmonic(cm.m) < 1e-3


def test_harmonic_noise():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma_async(harmonic_noise, x0, 1.0, wtol=1e-2, max_iter=10000, verbose=True)
        assert status == 'CONVERGED_WIDTH'
        assert harmonic(cm.m) < 1e-2


def test_rosenbrock2():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma_async(rosenbrock, x0, 1.0, npop=50, max_iter=1000, wtol=1e-10, verbose=True)
        assert status == 'CONVERGED_WIDTH' or status == 'FAILED_DEGENERATE'
        if status == 'CONVERGED_WIDTH':
            assert rosenbrock(cm.m) < 1e-2

def test_rosenbrock2_hof():
    for i in xrange(10):
        m0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma_async(rosenbrock, m0, 1.0, npop=50, max_iter=1000, verbose=True, hof_rate=0.0)
        assert status == 'CONVERGED_WIDTH'
        assert rosenbrock(cm.m) < 0.3


def test_callback():
    lcb = LogCallback()
    x0 = np.random.uniform(2, 3, 2)
    cm, status = fmin_cma_async(harmonic, x0, 1.0, max_iter=10, callback=lcb)
    assert len(lcb.log) == 10
    for item in lcb.log:
        assert item == ((cm,),{})


def test_reject_errors():
    x0 = np.random.uniform(2, 3, 2)
    cm, status = fmin_cma_async(failing, x0, 1.0, max_iter=10, reject_errors=True)
