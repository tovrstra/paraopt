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
        p, status = fmin_kriging_async(harmonic, x0, 1.0, max_iter=1000, verbose=10)
        assert status == 'CONVERGED_WIDTH'
        assert harmonic(p.best) < 1e-3


def test_harmonic2():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        p, status = fmin_kriging_async(harmonic, x0, 1.0, max_iter=1000, verbose=10)
        assert status == 'CONVERGED_WIDTH'
        assert harmonic(p.best) < 1e-3


def test_rosenbrock2():
    for i in xrange(10):
        x0 = np.random.uniform(-1,3, 2)
        p, status = fmin_kriging_async(rosenbrock, x0, 1.0, npop=50, max_iter=1000, wtol=1e-6, verbose=1)
        assert status == 'CONVERGED_WIDTH'
        assert rosenbrock(p.best) < 1e-2
