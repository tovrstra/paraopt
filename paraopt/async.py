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
import time, bisect

from paraopt.context import context
from paraopt.common import WorkerWrapper


__all__ = [
    'fmin_async',
]


class Population(object):
    def __init__(self, m0, sigma0, npop=None, loss_rate=0.0):
        self.ndof = len(m0)

        # Population size
        if npop is None:
            npop = 8 + 2*self.ndof
        if npop <= self.ndof:
            raise RuntimeError('Population too small.')
        self.npop = npop

        self.m = m0
        self.sigma0 = sigma0
        self.loss_rate = loss_rate
        self.members = []

        self.weights = np.log(self.npop+1) - np.log(np.arange(self.npop)+1)
        #self.weights = np.ones(self.npop)
        assert (self.weights > 0).all()
        self.weights /= self.weights.sum()

    def _get_complete(self):
        return len(self.members) == self.npop

    complete = property(_get_complete)

    def _get_best(self):
        return self.members[0][1]

    best = property(_get_best)

    def sample(self):
        if self.complete:
            xs = np.random.normal(0, 1.0, self.ndof)
            xs *= self.sigmas
            xs = np.dot(self.evecs, xs)
            xs += self.m
            return xs
        else:
            return self.m + np.random.uniform(-self.sigma0, self.sigma0, self.m.shape)

    def add_new(self, f, x, m):
        # Add new member
        bisect.insort(self.members, (f, x, m))

        # Throw one out if needed
        if len(self.members) > self.npop:
            # This is a minimization, i.e. remove largest
            if np.random.uniform(0,1) > self.loss_rate:
                del self.members[-1]
            else:
                del self.members[np.random.randint(self.npop)]

        # Build a new model for the next sample
        if len(self.members) == 1:
            return [0]
        else:
            # determine weights
            ws = self.weights[:len(self.members)]

            # New mean
            if self.complete:
                xs = np.array([x for f, x, m in self.members])
                self.m = np.dot(ws, xs)/ws.sum()

            # New covariance
            ys = np.array([x-m for f, x, m in self.members])
            cm = np.dot(ys.T*ws, ys)/ws.sum()
            if self.complete:
                evals, self.evecs = np.linalg.eigh(cm)
                self.sigmas = np.sqrt(evals)
            else:
                if cm.shape == ():
                    evals = np.array([cm])
                else:
                    evals = np.linalg.eigvalsh(cm)
                self.sigmas = np.sqrt(evals)

            return self.sigmas


def fmin_async(fun, x0, sigma0, npop=None, nworker=None, max_iter=100, stol=1e-6, smax=1e6, cnmax=1e6, verbose=False, callback=None, reject_errors=False, loss_rate=0.0):
    workers = []
    p = Population(x0, sigma0, npop, loss_rate)

    if nworker is None:
        nworker = p.npop

    counter = 0
    time0 = time.time()
    if verbose:
        print 'Iteration       Current          Best         Worst  Pop     min(sigmas)   max(sigmas)        walltime[s]'
        print '---------------------------------------------------------------------------------------------------------'

    while counter < max_iter:
        # make sure there are enough workers
        # TODO: add optional argument for number of workers
        while len(workers) < nworker:
            if reject_errors:
                worker = context.submit(WorkerWrapper(fun), p.sample())
            else:
                worker = context.submit(fun, p.sample())
            worker.m = p.m.copy()
            workers.append(worker)
        # wait until one is ready
        done, todo = context.wait_first(workers)
        for worker in done:
            x = worker.args[0]
            f = worker.result()
            m = worker.m
            counter += 1
            print_now = (verbose > 0) and (counter % verbose == 0)
            if f == 'FAILED' and print_now:
                print '%9i  FAILED' % counter
            else:
                evals = p.add_new(f, x, m)
                status = None
                if p.complete:
                    if evals[-1] > evals[0]*cnmax:
                        return p, 'FAILED_DEGENERATE'
                    elif evals[-1] > smax:
                        return p, 'FAILED_DIVERGENCE'
                    elif evals[-1] < stol:
                        return p, 'CONVERGED_SIGMA'
                if print_now:
                    print '%9i  %12.5e  %12.5e  %12.5e  %3i    %12.5e  %12.5e   %16.3f' % (counter, f, p.members[0][0], p.members[-1][0], len(p.members), evals[0], evals[-1], time.time()-time0)
            if callback is not None:
                callback(p)
        workers = list(todo)

    return p, 'FAILED_MAX_ITER'
