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
# --


import sys, traceback, signal


__all__ = ["WorkerWrapper", "TimeoutWrapper"]


class WorkerWrapper(object):
    __name__ = "WorkerWrapper"

    def __init__(self, myfn, reraise=False, verbose=False):
        self.myfn = myfn
        self.reraise = reraise
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        try:
            return self.myfn(*args, **kwargs)
        except Exception:
            type, value, tb = sys.exc_info()
            lines = traceback.format_exception(type, value, tb)
            if self.verbose:
                print("".join(lines), file=sys.stderr)
            if self.reraise:
                raise
            else:
                return "FAILED"


class TimeoutError(Exception):
    pass


class TimeoutWrapper(object):
    __name__ = "TimeoutWrapper"

    def __init__(self, myfn, timeout):
        self.myfn = myfn
        self.timeout = timeout

    def __call__(self, *args, **kwargs):
        def _handle_timeout(signum, frame):
            raise TimeoutError()

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)

        try:
            result = self.myfn(*args, **kwargs)
        except TimeoutError:
            result = "FAILED"
        finally:
            signal.alarm(0)
        return result
