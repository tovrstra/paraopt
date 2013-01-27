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

from paraopt.context import context as global_context, Context
from paraopt.common import WorkerWrapper
from paraopt.cma import fmin_cma


class Population(object):
    def __init__(self, x0, sigma0, npop):
        self.ndof = len(x0)
        self.x0 = x0
        self.sigma0 = sigma0
        if npop is None:
            self.npop = (2+self.ndof)**2
        else:
            self.npop = npop

        self.members = []

    def _get_complete(self):
        return len(self.members) >= self.npop

    complete = property(_get_complete)

    def _get_best(self):
        return self.members[0][1]

    best = property(_get_best)

    def _get_width(self):
        if self.complete:
            xs = np.array([x for f, x in self.members])
            xs -= xs.mean(axis=0)
            return np.sqrt((xs**2).mean())
        else:
            return self.sigma0

    def initial_sample(self):
        return self.x0 + np.random.uniform(-self.sigma0, self.sigma0, self.ndof)

    def add_new(self, f, x):
        self.members.append((f, x))
        self.members.sort(key=(lambda item: item[0]))
        if len(self.members) > self.npop:
            del self.members[self.npop:]
        self.width = self._get_width()
        self.frange = self.members[-1][0] - self.members[0][0]


class Kriging(object):
    def __init__(self, members, tau):
        self.tau = tau

        # turn members into two arrays
        self.fs = []
        self.xs = []
        for f, x in members:
            self.fs.append(f)
            self.xs.append(x)
        self.fs = np.log(1+np.arange(len(members)))
        self.xs = np.array(self.xs)

        # The kriging matrix
        self.npop = len(members)
        mat = np.zeros((self.npop+1, self.npop+1))
        mat[self.npop,:self.npop] = 1
        mat[:self.npop,self.npop] = 1
        for i1 in xrange(self.npop):
            for i2 in xrange(i1+1):
                mat[i1,i2] = self._semivariogram(self.xs[i1], self.xs[i2])
                mat[i2,i1] = mat[i1,i2]
        self.matinv = np.linalg.inv(mat)

    def __call__(self, x):
        rhs = np.zeros(self.npop+1)
        for i in xrange(self.npop):
            rhs[i] = self._semivariogram(x,self.xs[i])
        rhs[self.npop] = 1
        coeffs = np.dot(self.matinv, rhs)
        f = np.dot(self.fs, coeffs[:-1])
        e = np.dot(rhs, coeffs)
        return f

    def _semivariogram(self, x1, x2):
        #return np.log(np.linalg.norm(x1-x2)/self.tau+1)**2
        return (1-np.exp(-np.linalg.norm(x1-x2)/self.tau))**2


def kriging_wrapper(fun, p):
    if p.complete:
        meta_model = Kriging(p.members, p.width)
        x0 = p.members[np.random.randint(p.npop)][1]
        #x0 = p.members[0][1]
        cm, status = fmin_cma(
            meta_model, x0, p.width, max_iter=p.ndof*5, wtol=0,
            cnmax=1e6, wmax=p.width*1e6, verbose=False, context=Context())
        x = cm.m
    else:
        x = p.initial_sample()
    return fun(x), x


def fmin_kriging_async(fun, x0, sigma0, npop=None, nworker=None, wtol=1e-6, wmax=1e6, max_iter=100, verbose=False, callback=None, reject_errors=False, context=None):
    workers = []

    p = Population(x0, sigma0, npop)

    if nworker is None:
        nworker = p.npop
    if context is None:
        context = global_context

    counter = 0
    time0 = time.time()
    if verbose:
        print 'Async Kriging parameters'
        print '  Number of unknowns:    %10i' % p.ndof
        print '  Population size:       %10i' % p.npop

        print 'Iteration       Current          Best         Range  Pop         Width       walltime[s]'
        print '----------------------------------------------------------------------------------------'

    while counter < max_iter:
        # make sure there are enough workers
        while len(workers) < nworker:
            if reject_errors:
                worker = context.submit(kriging_wrapper, WorkerWrapper(fun), p)
            else:
                worker = context.submit(kriging_wrapper, fun, p)
            workers.append(worker)

        # wait until one is ready
        done, todo = context.wait_first(workers)
        for worker in done:
            f, x = worker.result()
            counter += 1
            print_now = (verbose > 0) and (counter % verbose == 0)
            if f == 'FAILED' and print_now:
                print '%9i  FAILED' % counter
            else:
                p.add_new(f, x)

                if print_now:
                    print '%9i  %12.5e  %12.5e  %12.5e  %3i  %12.5e  %16.3f' % (
                        counter, f, p.members[0][0], p.frange, len(p.members),
                        p.width, time.time()-time0
                    )

                if p.width < wtol:
                    return p, 'CONVERGED_WIDTH'
                elif p.width > wmax:
                    return p, 'FAILED_DIVERGENCE'
            if callback is not None:
                callback(p)
        workers = list(todo)

    return p, 'FAILED_MAX_ITER'
