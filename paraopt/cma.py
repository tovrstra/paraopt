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
from scipy.special import gamma
import bisect

from paraopt import context


__all__ = [
    'fmin_cma',
]


class CovarianceModel(object):
    def __init__(self, m0, sigma0, nselect):
        # Initialize state
        self.m = np.array(m0, dtype=float)
        if len(self.m.shape) != 1:
            raise TypeError('The initial guess must be a vector')
        self.ndof = self.m.shape[0]
        self.sigma = float(sigma0)
        self.covar = np.identity(self.ndof, dtype=float)
        self._update_derived()

        # Set algorithm parameters
        self.nselect = nselect
        self.weights = self.nselect-np.arange(self.nselect, dtype=float)
        self.weights /= self.weights.sum()
        self.mu_eff = 1/(self.weights**2).sum()

        # Learning rates for the covariance matrix
        self.c_1 = 2/self.ndof**2
        self.c_mu = min(self.mu_eff/self.ndof**2, 1-self.c_1)
        self.c_old = 1 - self.c_1 - self.c_mu

        # Learning rates for the paths
        self.c_path_c = 3.0/self.ndof
        self.root_c = np.sqrt(self.c_path_c*(2-self.c_path_c)*self.mu_eff)
        self.c_path_sigma = 1.0/self.ndof
        self.root_sigma = np.sqrt(self.c_path_sigma*(2-self.c_path_sigma)*self.mu_eff)
        self.ref_path_sigma_norm = np.sqrt(2.0)*gamma(0.5*(self.ndof+1))/gamma(0.5*self.ndof)

        # Initial path arrays
        self.path_c = np.zeros(self.ndof, float)
        self.path_sigma = np.zeros(self.ndof, float)

    def _update_derived(self):
        # Compute derived properties from the covariance matrix
        self.evals, self.evecs = np.linalg.eigh(self.covar)
        self.sigmas = self.evals**0.5
        self.max_sigma = abs(self.sigmas).max()*self.sigma
        self.min_sigma = abs(self.sigmas).min()*self.sigma
        self.inv_root_covar = np.dot(self.evecs/self.sigma, self.evecs.T)

    def generate(self, npop):
        xs = np.random.normal(0, self.sigma, (npop, self.ndof))
        xs *= self.sigmas
        xs = np.dot(xs, self.evecs)
        xs += self.m
        return xs

    def update_covar(self, xs, fs):
        # transform xs to reduced displacements
        ys = (xs - self.m)/self.sigma # must be done with old mean!

        # compute the new mean
        new_m = np.dot(self.weights, xs)

        # update the evolution path for the cumulation effect
        self.path_c = (1-self.c_path_c)*self.path_c + \
                      self.root_c*(new_m - self.m)/self.sigma

        # update the evolution path for the step size
        self.path_sigma = (1-self.c_path_sigma)*self.path_sigma + \
                          self.root_sigma*np.dot(self.inv_root_covar, (new_m - self.m))/self.sigma

        # update the covariance matrix
        self.covar = sum([
            self.c_old * self.covar,
            self.c_1 *   np.outer(self.path_c, self.path_c),
            self.c_mu *  np.dot(self.weights*ys.T, ys),
        ])

        # assign the new mean
        self.m = new_m

        # update the step size
        path_sigma_norm = np.linalg.norm(self.path_sigma)
        self.sigma = self.sigma*np.exp(self.c_path_sigma*(path_sigma_norm/self.ref_path_sigma_norm - 1))

        self._update_derived()


def fmin_cma(fun, m0, sigma0, npop, maxiter=100, cntol=1e6, stol=1e-12, rtol=None, verbose=False):
    '''Minimize a function with a basic CMA algorithm

       **Arguments:**

       fun
           The function to be minimized. It is recommended to use scoop for
           internal parallelization.

       m0
            The initial guess. (numpy vector, shape=n)

       sigma0
            The initial value of the step size

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
    if not isinstance(npop, int) or npop < 2:
        raise ValueError('npop must be an integer not smaller than 2.')
    nselect = npop/2
    cm = CovarianceModel(m0, sigma0, nselect)

    # B) The main loop
    if verbose:
        print 'Iteration   max(sigmas)   min(sigmas)       min(fs)     range(fs)'
    for i in xrange(maxiter):
        # screen info
        if cm.max_sigma < stol:
            if verbose:
                print '%9i  % 12.5e  % 12.5e' % (i, cm.max_sigma, cm.min_sigma)
            return cm, 'CONVERGED_SIGMA'
        elif cm.max_sigma > cm.min_sigma*cntol:
            if verbose:
                print '%9i  % 12.5e  % 12.5e' % (i, cm.max_sigma, cm.min_sigma)
            return cm, 'FAILED_DEGENERATE'

        # generate input samples
        xs = cm.generate(npop)

        # compute the function values
        fs = np.array(context.map(fun, xs))

        # sort by function value and select
        select = fs.argsort()[:nselect]
        xs = xs[select]
        fs = fs[select]
        frange = fs[-1] - fs[0]

        # screen info
        if verbose:
            print '%9i  % 12.5e  % 12.5e  % 12.5e  % 12.5e' % (i, cm.max_sigma, cm.min_sigma, fs[0], frange)

        # check for range convergence
        if rtol is not None and frange < rtol:
            return cm, 'CONVERGED_RANGE'

        # determine the new mean and covariance
        cm.update_covar(xs, fs)

    return cm, 'FAILED_MAXITER'
