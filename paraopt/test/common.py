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


__all__ = ['LogCallback', 'harmonic', 'harmonic_noise', 'rosenbrock', 'failing']


np.seterr(all='raise')


class LogCallback(object):
    def __init__(self):
        self.log = []

    def __call__(self, *args, **kwargs):
        self.log.append((args, kwargs))


def harmonic(x):
    return 0.5*np.linalg.norm(x - np.ones(x.shape))**2


def harmonic_noise(x):
    return harmonic(x) + np.random.uniform(-1e-3, 1e-3)


def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2


def failing(x):
    if np.random.uniform(0, 1) < 0.05:
        raise ValueError
    else:
        return rosenbrock(x)
