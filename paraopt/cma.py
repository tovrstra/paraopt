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
    def __init__(self, m0, sigma0, npop, do_rank1, do_stepscale):
        # Features
        self.do_rank1 = do_rank1
        self.do_stepscale = do_stepscale
        self.linear_slope = False

        # Initialize state
        self.m = np.array(m0, dtype=float)
        if len(self.m.shape) != 1:
            raise TypeError('The initial guess must be a vector')
        self.ndof = self.m.shape[0]
        if npop is None:
            self.npop = 4 + int(np.floor(3*np.log(self.ndof)))
        else:
            self.npop = npop
        self.nselect = self.npop/2
        self.sigma = float(sigma0)
        self.covar = np.identity(self.ndof, dtype=float)
        self._update_derived()

        # Set algorithm parameters
        self.weights = np.log(0.5*(self.npop+1)) - np.log(np.arange(self.nselect)+1)
        assert (self.weights > 0).all()
        self.weights /= self.weights.sum()
        self.mu_eff = 1/(self.weights**2).sum()

        # Learning rates for the covariance matrix
        self.c_1 = 2.0/((self.ndof + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1-self.c_1, 2*(self.mu_eff + 2.0 + 1.0/self.mu_eff)/((self.ndof+1)**2 + self.mu_eff))

        # Learning rates for the cumulation path
        self.c_path_c = (4*self.ndof + self.mu_eff)/(1 + 4*self.ndof + 2*self.mu_eff)
        self.norm_c = np.sqrt(self.c_path_c*(2-self.c_path_c)*self.mu_eff)

        # Learning rates for the step-size path
        self.c_path_sigma = (self.mu_eff + 2)/(self.ndof + self.mu_eff + 5)
        self.d_path_sigma = 1 + 2*max(0, np.sqrt((self.mu_eff-1)/(self.ndof+1))-1) + self.c_path_sigma
        self.norm_sigma = np.sqrt(self.c_path_sigma*(2-self.c_path_sigma)*self.mu_eff)
        self.ref_path_sigma_norm = np.sqrt(2.0)*gamma(0.5*(self.ndof+1))/gamma(0.5*self.ndof)

        # Initial path arrays
        self.path_c = np.zeros(self.ndof, float)
        self.path_sigma = np.zeros(self.ndof, float)

        # Counter for the number of cm updates
        self.update_counter = 0

    def _update_derived(self):
        # Compute derived properties from the covariance matrix
        self.evals, self.evecs = np.linalg.eigh(self.covar)
        self.sigmas = self.evals**0.5
        self.max_sigma = abs(self.sigmas).max()*self.sigma
        self.min_sigma = abs(self.sigmas).min()*self.sigma
        self.inv_root_covar = np.dot(self.evecs/self.sigma, self.evecs.T)

    def generate(self):
        xs = np.random.normal(0, self.sigma, (self.npop, self.ndof))
        xs *= self.sigmas
        xs = np.dot(xs, self.evecs)
        xs += self.m
        return xs

    def update_covar(self, xs, fs):
        # transform xs to reduced displacements
        ys = (xs - self.m)/self.sigma # must be done with old mean!

        # compute the new mean
        new_m = np.dot(self.weights, xs)

        if self.do_rank1 or self.do_stepscale:
            # update the evolution path for the step size
            self.path_sigma = (1-self.c_path_sigma)*self.path_sigma + \
                              self.norm_sigma*np.dot(self.inv_root_covar, (new_m - self.m))/self.sigma
            path_sigma_norm = np.linalg.norm(self.path_sigma)
            self.linear_slope = (
                path_sigma_norm
                /np.sqrt(1-(1-self.c_path_sigma)**(2*self.update_counter+2)) >
                (1.4+2/(self.ndof+1))*
                self.ref_path_sigma_norm
            )

        if self.linear_slope or not self.do_rank1:
            # update the covariance matrix
            self.covar = sum([
                (1 - self.c_mu)*self.covar,
                self.c_mu*np.dot(self.weights*ys.T, ys),
            ])
        else:
            # update the evolution path for the cumulation effect
            self.path_c = (1-self.c_path_c)*self.path_c + \
                          self.norm_c*(new_m - self.m)/self.sigma

            # update the covariance matrix
            self.covar = sum([
                (1 - self.c_1 - self.c_mu)*self.covar,
                self.c_1*np.outer(self.path_c, self.path_c),
                self.c_mu*np.dot(self.weights*ys.T, ys),
            ])

        # assign the new mean
        self.m = new_m

        # update the step size
        if self.do_stepscale:
            scale = np.exp(self.c_path_sigma/self.d_path_sigma*(path_sigma_norm/self.ref_path_sigma_norm - 1))
            #scale = np.clip(scale, 0.5, 1.1)
            #scale = 1
            self.sigma = self.sigma*scale

        self._update_derived()
        self.update_counter += 1


def fmin_cma(fun, m0, sigma0, npop=None, max_iter=100, cntol=1e6, stol=1e-12, rtol=None, smax=1e12, verbose=False, do_rank1=True, do_stepscale=True):
    '''Minimize a function with a basic CMA algorithm

       **Arguments:**

       fun
           The function to be minimized. It is recommended to use scoop for
           internal parallelization.

       m0
            The initial guess. (numpy vector, shape=n)

       sigma0
            The initial value of the step size


       **Optional arguments:**

       npop
            The size of the sample population. By default, this is
            4 + floor(3*ln(ndof)). This is a minimal choice. In case of
            convergence issues, it is recommended to increase npop to something
            proportional to ndof

       max_iter
            The maximum number of iterations

       cntol
            When the condition number of the covariance goes above this
            threshold, the minimum is considered degenerate and the optimizer
            stops.

       stol
            When the largest sqrt(covariance eigenvalue) drops below this value,
            the solution is sufficiently close to the real optimum and the
            optimization has converged.

       smax
            When the largest sqrt(covariance eigenvalue) exceeds this value,
            the CMA algorithm is terminated due to divergence.

       rtol
            When the range of the selected results drops below this threshold,
            the optimization has converged.

       verbose
            When set to True, some convergence info is printed on screen

       do_rank1
            If True, rank-1 updates are used in the CMA algorith. This increases
            efficiency on well-behaved functions but decreases robustness.

       do_stepscale=True
            If True, step-size updates are used in the CMA algorith. This
            increases efficiency on well-behaved functions but decreases
            robustness.
    '''

    # A) Parse the arguments:
    cm = CovarianceModel(m0, sigma0, npop, do_rank1, do_stepscale)
    if not isinstance(cm.npop, int) or cm.npop < 1:
        raise ValueError('npop must be a strictly positive integer.')

    # B) The main loop
    if verbose:
        print 'Iteration   max(sigmas)   min(sigmas)       min(fs)     range(fs) linear  sigma-path-ratio'
    for i in xrange(max_iter):
        # screen info
        if verbose:
            print '%9i  % 12.5e  % 12.5e' % (i, cm.max_sigma, cm.min_sigma),
        if cm.max_sigma < stol:
            if verbose: print
            return cm, 'CONVERGED_SIGMA'
        elif cm.max_sigma > cm.min_sigma*cntol:
            if verbose: print
            return cm, 'FAILED_DEGENERATE'
        elif cm.max_sigma > smax:
            # This typically happens when the initial step size is too small
            if verbose: print
            return cm, 'FAILED_DIVERGENCE'

        # generate input samples
        xs = cm.generate()

        # compute the function values
        fs = np.array(context.map(fun, xs))

        # sort by function value and select
        select = fs.argsort()[:cm.nselect]
        xs = xs[select]
        fs = fs[select]
        frange = fs[-1] - fs[0]

        # screen info
        if verbose:
            print ' % 12.5e  % 12.5e' % (fs[0], frange),

        # check for range convergence
        if rtol is not None and frange < rtol:
            if verbose: print
            return cm, 'CONVERGED_RANGE'

        # determine the new mean and covariance
        cm.update_covar(xs, fs)
        if verbose:
            if cm.linear_slope:
                print 'L',
            else:
                print ' ',
            if cm.do_rank1 or cm.do_stepscale:
                print '      % 12.5e' % (np.linalg.norm(cm.path_sigma))
            else:
                print

    return cm, 'FAILED_MAX_ITER'
