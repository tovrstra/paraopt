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
        m0 = np.random.uniform(-1,3, 1)
        cm, status = fmin_cma(harmonic, m0, 1.0, npop=10, wtol=1e-4, verbose=True)
        assert status == 'CONVERGED_WIDTH'
        assert abs(cm.m - 1).max() < 1e-3
        assert harmonic(cm.m) < 1e-6


def test_rosenbrock():
    for i in xrange(10):
        m0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma(rosenbrock, m0, 1.0, npop=50, max_iter=1000, verbose=True)
        assert status == 'CONVERGED_WIDTH'
        assert abs(cm.m - 1).max() < 1e-5
        assert rosenbrock(cm.m) < 1e-9


def test_rosenbrock_rtol():
    for i in xrange(10):
        m0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma(rosenbrock, m0, 1.0, npop=50, max_iter=1000, rtol=1e-10, verbose=True)
        assert status == 'CONVERGED_RANGE'
        assert rosenbrock(cm.m) < 0.3


def test_rosenbrock_hof():
    for i in xrange(10):
        m0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma(rosenbrock, m0, 1.0, npop=50, max_iter=1000, verbose=True, hof_rate=0.0)
        assert status == 'CONVERGED_WIDTH'
        assert rosenbrock(cm.m) < 0.3


def test_callback():
    lcb = LogCallback()
    m0 = np.random.uniform(2, 3, 2)
    cm, status = fmin_cma(rosenbrock, m0, 1.0, npop=50, max_iter=10, rtol=1e-10, callback=lcb)
    assert len(lcb.log) == 10
    for item in lcb.log:
        assert item == ((cm,),{})


def test_reject_errors():
    m0 = np.random.uniform(2, 3, 2)
    cm, status = fmin_cma(failing, m0, 1.0, npop=50, max_iter=10, rtol=1e-10, reject_errors=True)
