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
import bisect

from paraopt import context


__all__ = [
    'CONVERGED_SIGMA', 'CONVERGED_RANGE', 'FAILED_DEGENERATE', 'FAILED_MAXITER',
    'fmin_cma',
]


CONVERGED_SIGMA = 1
CONVERGED_RANGE = 2
FAILED_DEGENERATE = -1
FAILED_MAXITER = -2


def fmin_cma(fun, m0, sigma0, npop, maxiter=100, cntol=1e6, stol=1e-12, rtol=None, verbose=False):
    '''Minimize a function with a basic CMA algorithm

       **Arguments:**

       fun
           The function to be minimized. It is recommended to use scoop for
           internal parallelization.

       m0
            The initial guess. (numpy vector, shape=n)

       sigma0
            The initial value of the width of the distribution (a single scalar,
            a diagonal of the covariance or a covariance matrix)

       npop
            The size of the sample population


       **Optional arguments:**

       maxiter
            The maximum number of iterations

       cntol
            When the condition number of the covariance goes above this
            threshold, the minimum is considered degenerate and the optimizer
            stops.

       stol
            When the largest sqrt(covariance eigenvalue) drops below this value,
            the solution is sufficiently close to the real optimum and the
            optimization has converged.

       rtol
            When the range of the selected results drops below this threshold,
            the optimization has converged.

       verbose
            When set to True, some convergence info is printed on screen
    '''

    # A) Parse the arguments:
    m = np.array(m0, dtype=float)
    if len(m.shape) != 1:
        raise TypeError('The initial guess must be a vector')
    ndof = m.shape[0]

    if isinstance(sigma0, np.ndarray):
        if len(sigma0.shape) == 0:
            covar = np.identity(ndof, float)*sigma0**2
        elif len(sigma0.shape) == 1:
            if sigma0.shape[0] != ndof:
                raise TypeError('The size of sigma0 does not match the size of the initial guess.')
            covar = np.diag(sigma0**2)
        elif len(sigma0.shape) == 2:
            if sigma0.shape[0] != ndof or sigma0.shape[1] != ndof:
                raise TypeError('The size of sigma0 does not match the size of the initial guess.')
            covar = 0.5*(sigma0 + sigma0.T)
        else:
            raise TypeError('sigma0 must have at most two dimensions.')
    else:
        covar = np.identity(ndof, float)*sigma0**2

    if not isinstance(npop, int) or npop < 2:
        raise ValueError('npop must be an integer not smaller than 2.')
    nselect = npop/2

    # B) The main loop
    if verbose:
        print 'Iteration   max(sigmas)   min(sigmas)       min(fs)     range(fs)'
    for i in xrange(maxiter):
        # diagonalize the covariance matrix
        evals, evecs = np.linalg.eigh(covar)
        sigmas = evals**0.5
        max_sigma = abs(sigmas).max()
        min_sigma = abs(sigmas).min()
        # screen info
        if max_sigma < stol:
            if verbose:
                print '%9i  % 12.5e  % 12.5e' % (i, max_sigma, min_sigma)
            return m, CONVERGED_SIGMA
        elif max_sigma > min_sigma*cntol:
            if verbose:
                print '%9i  % 12.5e  % 12.5e' % (i, max_sigma, min_sigma)
            return m, FAILED_DEGENERATE

        # generate input samples
        xs = np.random.normal(0, 1, (npop, ndof))
        xs *= sigmas
        xs = np.dot(xs, evecs)
        xs += m

        # compute the function values
        fs = np.array(context.map(fun, xs))

        # sort by function value and select
        select = fs.argsort()[:nselect]
        xs = xs[select]
        fs = fs[select]
        frange = fs[-1] - fs[0]

        # screen info
        if verbose:
            print '%9i  % 12.5e  % 12.5e  % 12.5e  % 12.5e' % (i, max_sigma, min_sigma, fs[0], frange)

        # check for range convergence
        if rtol is not None and frange < rtol:
            return m, CONVERGED_RANGE

        # determine the new mean and covariance
        weights = nselect-np.arange(nselect, dtype=float)
        weights /= weights.sum()
        ys = xs - m # must be done with old mean!
        covar = np.dot(weights*ys.T, ys)
        m = np.dot(weights, xs)


    return m, FAILED_MAXITER
