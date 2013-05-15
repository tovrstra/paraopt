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
from scipy.special import gamma
import time

from paraopt.context import context as global_context
from paraopt.common import WorkerWrapper, TimeoutWrapper


__all__ = [
    'fmin_cma', 'fmin_cma_async',
]


class CovarianceModel(object):
    def __init__(self, m0, sigma0, npop, do_rank1, do_stepscale, hof_rate=1.0):
        # Features
        self.do_rank1 = do_rank1
        self.do_stepscale = do_stepscale
        self.linear_slope = False
        self.hof_rate = hof_rate

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
        self.hof = []

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
        '''Compute derived properties from the covariance matrix'''
        self.evals, self.evecs = np.linalg.eigh(self.covar)
        self.widths = abs(self.evals)**0.5
        self.max_width = abs(self.widths).max()*self.sigma
        self.min_width = abs(self.widths).min()*self.sigma
        if self.min_width <= 0:
            self.cond = 0.0
        else:
            self.cond = self.max_width/self.min_width
        self.inv_root_covar = np.dot(self.evecs/self.widths, self.evecs.T)

    def generate(self, npop=None):
        '''Generate a set of new samples based on the latest distribution'''
        if npop is None:
            npop = self.npop
        xs = np.random.normal(0, self.sigma, (npop, self.ndof))
        xs *= self.widths
        xs = np.dot(xs, self.evecs)
        xs += self.m
        return xs

    def update(self, xs, ys, fs):
        '''Update the mean and the covariance based on a set of accepted samples'''
        # update the hall of fame (hof)
        #  A) clean
        if self.hof_rate == 1.0:
            self.hof = []
        else:
            number_to_clean = int(np.floor(self.hof_rate*self.nselect))
            for i in xrange(number_to_clean):
                del self.hof[np.random.randint(len(self.hof))]
        #  B) add
        for i in xrange(self.nselect):
            self.hof.append((xs[i], fs[i]))
        #  C) sort
        self.hof.sort(key=(lambda item: item[1]))
        #  D) chop
        self.hof = self.hof[:self.nselect]

        # compute the new mean
        new_m = 0.0
        for i in xrange(self.nselect):
            new_m += self.weights[i]*self.hof[i][0]

        if self.do_rank1 or self.do_stepscale:
            # update the evolution path for the step size
            self.path_sigma = (1-self.c_path_sigma)*self.path_sigma + \
                              self.norm_sigma*np.dot(self.inv_root_covar, (new_m - self.m))/self.sigma
            self.path_sigma_ratio = np.linalg.norm(self.path_sigma)/self.ref_path_sigma_norm
            self.linear_slope = (
                self.path_sigma_ratio
                /np.sqrt(1-(1-self.c_path_sigma)**(2*self.update_counter+2)) >
                (1.4+2/(self.ndof+1))
            )

        if self.linear_slope or not self.do_rank1:
            # The rank-one update is avoided if the cost function seems linear.
            # Update the covariance matrix:
            self.covar = sum([
                (1 - self.c_mu)*self.covar,
                self.c_mu*np.dot(self.weights*ys.T, ys),
            ])
        else:
            # Update the evolution path for the cumulation effect:
            self.path_c = (1-self.c_path_c)*self.path_c + \
                          self.norm_c*(new_m - self.m)/self.sigma

            # Update the covariance matrix:
            self.covar = sum([
                (1 - self.c_1 - self.c_mu)*self.covar,
                self.c_1*np.outer(self.path_c, self.path_c),
                self.c_mu*np.dot(self.weights*ys.T, ys),
            ])

        # assign the new mean
        self.m = new_m

        # update the step size
        if self.do_stepscale:
            scale = np.exp(self.c_path_sigma/self.d_path_sigma*(self.path_sigma_ratio - 1))
            #scale = np.clip(scale, 0.5, 1.1)
            #scale = 1
            self.sigma = self.sigma*scale

        self._update_derived()
        self.update_counter += 1



def fmin_cma(fun, m0, sigma0, npop=None, max_iter=100, wtol=1e-6, rtol=None,
             cnmax=1e6, wmax=1e6, verbose=False, do_rank1=True,
             do_stepscale=True, callback=None, reject_errors=False, timeout=None,
             hof_rate=1.0, context=None):
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

       wtol
            When the largest width, sqrt(covariance eigenvalue), drops below
            this value, the solution is sufficiently close to the real optimum
            and the optimization has converged.

       rtol
            When the range of the selected results drops below this threshold,
            the optimization has converged.

       cnmax
            When the condition number of the covariance goes above this
            threshold, the minimum is considered degenerate and the optimizer
            stops.

       wmax
            When the largest width, sqrt(covariance eigenvalue), exceeds this
            value, the CMA algorithm is terminated due to divergence.

       verbose
            When set to True, some convergence info is printed on screen

       do_rank1
            If True, rank-1 updates are used in the CMA algorith. This increases
            efficiency on well-behaved functions but decreases robustness.

       do_stepscale
            If True, step-size updates are used in the CMA algorith. This
            increases efficiency on well-behaved functions but decreases
            robustness.

       callback
            If given, this routine is called after each update of the covariance
            model. One argument is given, i.e. the covariance model.

       reject_errors
            When set to True, exceptions in fun will be caught and the
            corresponding trials will be rejected. If there are too many
            rejected attempts in one iteration, such that the number of
            successful ones is below cm.nselect, the algorithm will still fail.

       timeout
            Maximum time that a function is allowed to take. If it runs longer,
            the corresponding parameters are considered to fail and are
            rejected.

       hof_rate
            The rate with which the hall of fame must be purged. The default
            is 1.0, which corresponds to cleaning the hall of fame on every
            update. This is equivalent to the conventional CMA.

       context
            A custom context. If not given, the global context will be used.
    '''

    # A) Parse the arguments:
    cm = CovarianceModel(m0, sigma0, npop, do_rank1, do_stepscale, hof_rate)
    if not isinstance(cm.npop, int) or cm.npop < 1:
        raise ValueError('npop must be a strictly positive integer.')
    if context is None:
        context = global_context

    if verbose:
        print 'CMA parameters'
        print '  Number of unknowns:    %10i' % cm.ndof
        print '  Population size:       %10i' % cm.npop
        print '  Selection size:        %10i' % cm.nselect
        print '  Effective size:        %10.5f' % cm.mu_eff
        print '  Rank-1 learning rate:  %10.5f' % cm.c_1
        print '  Rank-mu learning rate: %10.5f' % cm.c_mu
        print '  Cumul learning rate:   %10.5f' % cm.c_path_c
        print '  Sigma learning rate:   %10.5f' % cm.c_path_sigma
        print '  Sigma damping:         %10.5f' % cm.d_path_sigma
        print '  Maximum iterations:    %10i' % max_iter
        print '  Width tolerance:       %10.3e' % wtol
        print '  Width maximum:         %10.3e' % wmax
        print '  Condition maximum:     %10.3e' % cnmax
        if rtol is not None:
            print '  Range threshold:       %10.3e' % rtol
        print '  Do rank-1 update:      %10s' % do_rank1
        print '  Do step size update:   %10s' % do_stepscale
        print '  Hall-of-fame rate:     %10.5f' % hof_rate
        if timeout is not None:
            print '  Timeout [s]:            %10.3f' % timeout

    if reject_errors:
        fun = WorkerWrapper(fun)
    if timeout is not None:
        fun = TimeoutWrapper(fun, timeout)

    # B) The main loop
    if verbose:
        print '-------------------------------------+-------------------------------------------------------'
        print 'Iteration       min(fs)    range(fs) |  max(sigmas)   cn(signas) lin   p-s-ratio  walltime[s]'
        print '-------------------------------------+-------------------------------------------------------'
    time0 = time.time()
    for counter in xrange(max_iter):
        # generate input samples
        xs = cm.generate()

        # compute the function values
        fs = context.map(fun, xs)
        fs = np.array([value for value in fs if value != 'FAILED'])
        if len(fs) < cm.nselect:
            raise RuntimeError('Too many evaluations failed or timed out.')

        # sort by function value and select
        select = fs.argsort()[:cm.nselect]
        xs = xs[select]
        fs = fs[select]
        frange = fs[-1] - fs[0]

        # compute rescaled displacemets
        ys = (xs - cm.m)/cm.sigma

        # determine the new mean and covariance
        cm.update(xs, ys, fs)

        # Print screen output
        if verbose:
            if cm.linear_slope:
                linear_str = 'L'
            else:
                linear_str = ' '
            if cm.do_rank1 or cm.do_stepscale:
                ratio_str = '% 12.5e' % cm.path_sigma_ratio
            else:
                ratio_str = '            '
            print '%9i  %12.5e %12.5e | %12.5e %12.5e  %s %s %12.3f' % (
                counter, fs[0], fs[-1]-fs[0],
                cm.max_width, cm.cond, linear_str,
                ratio_str, time.time()-time0
            )

        # If provided, call the callback function.
        if callback is not None:
            callback(cm)

        if cm.max_width < wtol:
            return cm, 'CONVERGED_WIDTH'
        elif cm.max_width > cm.min_width*cnmax:
            return cm, 'FAILED_DEGENERATE'
        elif cm.max_width > wmax:
            return cm, 'FAILED_DIVERGENCE'
        elif rtol is not None and frange < rtol:
            return cm, 'CONVERGED_RANGE'

    return cm, 'FAILED_MAX_ITER'


def fmin_cma_async(fun, m0, sigma0, npop=None, nworker=None, max_iter=100,
                   wtol=1e-12, cnmax=1e6, wmax=1e6, verbose=False,
                   do_rank1=True, do_stepscale=True, callback=None,
                   reject_errors=False, hof_rate=1.0, context=None):
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

       nworker
            The number of worker processes that are evaluating the function. By
            default this is equal to npop

       max_iter
            The maximum number of iterations

       wtol
            When the largest width, sqrt(covariance eigenvalue), drops below
            this value, the solution is sufficiently close to the real optimum
            and the optimization has converged.

       rtol
            When the range of the selected results drops below this threshold,
            the optimization has converged.

       cnmax
            When the condition number of the covariance goes above this
            threshold, the minimum is considered degenerate and the optimizer
            stops.

       wmax
            When the largest width, sqrt(covariance eigenvalue), exceeds this
            value, the CMA algorithm is terminated due to divergence.

       verbose
            When set to True, some convergence info is printed on screen

       do_rank1
            If True, rank-1 updates are used in the CMA algorith. This increases
            efficiency on well-behaved functions but decreases robustness.

       do_stepscale
            If True, step-size updates are used in the CMA algorith. This
            increases efficiency on well-behaved functions but decreases
            robustness.

       callback
            If given, this routine is called after each update of the covariance
            model. One argument is given, i.e. the covariance model.

       reject_errors
            When set to True, exceptions in fun will be caught and the
            corresponding trials will be rejected. If there are too many
            rejected attempts in one iteration, such that the number of
            successful ones is below cm.nselect, the algorithm will still fail.

       hof_rate
            The rate with which the hall of fame must be purged. The default
            is 1.0, which corresponds to cleaning the hall of fame on every
            update. This is equivalent to the conventional CMA.

       context
            A custom context. If not given, the global context will be used.
    '''
    # A) Parse the arguments:
    cm = CovarianceModel(m0, sigma0, npop, do_rank1, do_stepscale, hof_rate)
    if not isinstance(cm.npop, int) or cm.npop < 1:
        raise ValueError('npop must be a strictly positive integer.')
    if nworker is None:
        nworker = cm.npop
    if context is None:
        context = global_context

    if verbose:
        print 'Asynchronous CMA parameters'
        print '  Number of unknowns:    %10i' % cm.ndof
        print '  Number of workers:     %10i' % nworker
        print '  Population size:       %10i' % cm.npop
        print '  Selection size:        %10i' % cm.nselect
        print '  Effective size:        %10.5f' % cm.mu_eff
        print '  Rank-1 learning rate:  %10.5f' % cm.c_1
        print '  Rank-mu learning rate: %10.5f' % cm.c_mu
        print '  Cumul learning rate:   %10.5f' % cm.c_path_c
        print '  Sigma learning rate:   %10.5f' % cm.c_path_sigma
        print '  Sigma damping:         %10.5f' % cm.d_path_sigma
        print '  Maximum iterations:    %10i' % max_iter
        print '  Width tolerance:       %10.3e' % wtol
        print '  Width maximum:         %10.3e' % wmax
        print '  Condition maximum:     %10.3e' % cnmax
        print '  Do rank-1 update:      %10s' % do_rank1
        print '  Do step size update:   %10s' % do_stepscale
        print '  Hall-of-fame rate:     %10.5f' % hof_rate

    # B) The main loop
    if verbose:
        print '-------------------------------------+-------------------------------------------------------'
        print 'Iteration       min(fs)    range(fs) |  max(sigmas)   cn(signas) lin   p-s-ratio  walltime[s]'
        print '-------------------------------------+-------------------------------------------------------'

    time0 = time.time()
    counter = 0
    purgatory = []
    workers = []
    while True:
        # make sure there are enough workers
        while len(workers) < nworker:
            x = cm.generate(1)[0]
            if reject_errors:
                worker = context.submit(WorkerWrapper(fun), x)
            else:
                worker = context.submit(fun, x)
            worker.m = cm.m.copy()
            worker.sigma = cm.sigma
            workers.append(worker)

        # wait until one is ready
        done, todo = context.wait_first(workers)
        for worker in done:
            f = worker.result()
            if f == 'FAILED':
                continue
            x = worker.args[0]
            m = worker.m
            sigma = worker.sigma
            purgatory.append((f, x, m, sigma))
        workers = list(todo)

        if len(purgatory) >= cm.npop:
            # Update covariance
            purgatory.sort(key=(lambda item: item[0]))
            xs = []
            ys = []
            fs = []
            for i in xrange(cm.nselect):
                f, x, m, sigma = purgatory[i]
                xs.append(x)
                ys.append((x-m)/sigma)
                fs.append(f)
            xs = np.array(xs)
            ys = np.array(ys)
            fs = np.array(fs)
            cm.update(xs, ys, fs)

            # Print screen output
            if verbose:
                if cm.linear_slope:
                    linear_str = 'L'
                else:
                    linear_str = ' '
                if cm.do_rank1 or cm.do_stepscale:
                    ratio_str = '% 12.5e' % cm.path_sigma_ratio
                else:
                    ratio_str = '            '
                print '%9i  %12.5e %12.5e | %12.5e %12.5e  %s %s %12.3f' % (
                    counter, fs[0], fs[-1]-fs[0],
                    cm.max_width, cm.cond, linear_str,
                    ratio_str, time.time()-time0
                )

            # Do the callback
            if callback is not None:
                callback(cm)

            # prepare for next batch
            counter += 1
            purgatory = []

            # check convergence
            if cm.max_width < wtol:
                return cm, 'CONVERGED_WIDTH'
            elif cm.max_width > cm.min_width*cnmax:
                return cm, 'FAILED_DEGENERATE'
            elif cm.max_width > wmax:
                # This typically happens when the initial step size is too small
                return cm, 'FAILED_DIVERGENCE'
            elif counter >= max_iter:
                return cm, 'FAILED_MAX_ITER'

    assert False
