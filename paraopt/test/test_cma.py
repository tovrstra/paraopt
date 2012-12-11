# -*- coding: utf-8 -*-
# Paraopt is a simple parallel optimization toolbox.
# Copyright (C) 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>
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


def harmonic(x):
    return 0.5*x[0]**2


def test_harmonic():
    for i in xrange(10):
        m0 = np.random.uniform(-1,3, 1)
        cm, status = fmin_cma(harmonic, m0, 1.0, 10, 1000)
        assert status == 'CONVERGED_SIGMA'
        assert abs(cm.m - 0.0).max() < 1e-5
        assert harmonic(cm.m) < 1e-9


def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2


def test_rosenbrock1():
    for i in xrange(10):
        m0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma(rosenbrock, m0, 1.0, 100, 1000)
        assert status == 'CONVERGED_SIGMA'
        assert abs(cm.m - 1).max() < 1e-5
        assert rosenbrock(cm.m) < 1e-9


def test_rosenbrock2():
    for i in xrange(10):
        m0 = np.random.uniform(-1,3, 2)
        cm, status = fmin_cma(rosenbrock, m0, 1.0, npop=100, maxiter=1000, rtol=1e-3, verbose=True)
        assert status == 'CONVERGED_RANGE'
        assert rosenbrock(cm.m) < 0.3
