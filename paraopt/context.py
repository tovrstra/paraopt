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


__all__ = ['Context', 'context']



class FakeFuture(object):
    def __init__(self, fun, *args, **kargs):
        self.args = args
        self.kargs = kargs
        self._result = fun(*args, **kargs)

    def result(self):
        return self._result


class Context(object):
    def __init__(self):
        # initialize with serial version of map and submit
        self.use_stub()

    def use_stub(self):
        def my_map(fn, l, **kwargs):
            return [fn(i, **kwargs) for i in l]
        def my_wait_first(fs):
            return fs[:1], fs[1:]
        self.map = my_map
        self.wait_first = my_wait_first
        self.submit = FakeFuture

    def use_scoop(self):
        from scoop import futures
        def my_map(*args, **kwargs):
            return list(futures.map(*args, **kwargs))
        def my_wait_first(fs):
            return futures.wait(fs, return_when=futures.FIRST_COMPLETED)
        self.map = my_map
        self.wait_first = my_wait_first
        self.submit = futures.submit


context = Context()
