#!/usr/bin/env python
#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""An I/O event loop for non-blocking sockets.

Typical applications will use a single `IOLoop` object, in the
`IOLoop.instance` singleton.  The `IOLoop.start` method should usually
be called at the end of the ``main()`` function.  Atypical applications may
use more than one `IOLoop`, such as one `IOLoop` per thread, or per `unittest`
case.

In addition to I/O events, the `IOLoop` can also schedule time-based events.
`IOLoop.add_timeout` is a non-blocking alternative to `time.sleep`.
"""

from __future__ import absolute_import, division, print_function, with_statement

import datetime
import errno
import functools
import heapq
import itertools
import logging
import numbers
import os
import select
import sys
import threading
import time
import traceback
import math

from tornado.concurrent import TracebackFuture, is_future
from tornado.log import app_log, gen_log
from tornado import stack_context
from tornado.util import Configurable, errno_from_exception, timedelta_to_seconds

try:
    import signal
except ImportError:
    signal = None

try:
    import thread  # py2
except ImportError:
    import _thread as thread  # py3

from tornado.platform.auto import set_close_exec, Waker


_POLL_TIMEOUT = 3600.0


class TimeoutError(Exception):
    pass


class IOLoop(Configurable):
    """A level-triggered I/O loop.

    We use ``epoll`` (Linux) or ``kqueue`` (BSD and Mac OS X) if they
    are available, or else we fall back on select(). If you are
    implementing a system that needs to handle thousands of
    simultaneous connections, you should use a system that supports
    either ``epoll`` or ``kqueue``.

    Example usage for a simple TCP server:

    .. testcode::

        import errno
        import functools
        import tornado.ioloop
        import socket

        def connection_ready(sock, fd, events):
            while True:
                try:
                    connection, address = sock.accept()
                except socket.error as e:
                    if e.args[0] not in (errno.EWOULDBLOCK, errno.EAGAIN):
                        raise
                    return
                connection.setblocking(0)
                handle_connection(connection, address)

        if __name__ == '__main__':
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setblocking(0)
            sock.bind(("", port))
            sock.listen(128)

            io_loop = tornado.ioloop.IOLoop.current()
            callback = functools.partial(connection_ready, sock)
            io_loop.add_handler(sock.fileno(), callback, io_loop.READ)
            io_loop.start()

    .. testoutput::
       :hide:

    By default, a newly-constructed `IOLoop` becomes the thread's current
    `IOLoop`, unless there already is a current `IOLoop`. This behavior
    can be controlled with the ``make_current`` argument to the `IOLoop`
    constructor: if ``make_current=True``, the new `IOLoop` will always
    try to become current and it raises an error if there is already a
    current instance. If ``make_current=False``, the new `IOLoop` will
    not try to become current.

    .. versionchanged:: 4.2
       Added the ``make_current`` keyword argument to the `IOLoop`
       constructor.
    """
    # Constants from the epoll module
    _EPOLLIN = 0x001
    _EPOLLPRI = 0x002
    _EPOLLOUT = 0x004
    _EPOLLERR = 0x008
    _EPOLLHUP = 0x010
    _EPOLLRDHUP = 0x2000
    _EPOLLONESHOT = (1 << 30)
    _EPOLLET = (1 << 31)

    # Our events map exactly to the epoll events
    NONE = 0
    READ = _EPOLLIN
    WRITE = _EPOLLOUT
    ERROR = _EPOLLERR | _EPOLLHUP

    # Global lock for creating global IOLoop instance
    _instance_lock = threading.Lock()

    _current = threading.local()


class PollIOLoop(IOLoop):
    """Base class for IOLoops built around a select-like function.

    For concrete implementations, see `tornado.platform.epoll.EPollIOLoop`
    (Linux), `tornado.platform.kqueue.KQueueIOLoop` (BSD and Mac), or
    `tornado.platform.select.SelectIOLoop` (all platforms).
    """
    def initialize(self, impl, time_func=None, **kwargs):
        super(PollIOLoop, self).initialize(**kwargs)
        self._impl = impl
        if hasattr(self._impl, 'fileno'):
            set_close_exec(self._impl.fileno())
        self.time_func = time_func or time.time
        self._handlers = {}
        self._events = {}
        self._callbacks = []
        self._callback_lock = threading.Lock()
        self._timeouts = []
        self._cancellations = 0
        self._running = False
        self._stopped = False
        self._closing = False
        self._thread_ident = None
        self._blocking_signal_threshold = None
        self._timeout_counter = itertools.count()

        # Create a pipe that we send bogus data to when we want to wake
        # the I/O loop when it is idle
        self._waker = Waker()
        self.add_handler(self._waker.fileno(),
                         lambda fd, events: self._waker.consume(),
                         self.READ)

    def close(self, all_fds=False):
        with self._callback_lock:
            self._closing = True
        self.remove_handler(self._waker.fileno())
        if all_fds:
            for fd, handler in self._handlers.values():
                self.close_fd(fd)
        self._waker.close()
        self._impl.close()
        self._callbacks = None
        self._timeouts = None

    def add_handler(self, fd, handler, events):
        fd, obj = self.split_fd(fd)
        self._handlers[fd] = (obj, stack_context.wrap(handler))
        self._impl.register(fd, events | self.ERROR)

    def update_handler(self, fd, events):
        fd, obj = self.split_fd(fd)
        self._impl.modify(fd, events | self.ERROR)

    def remove_handler(self, fd):
        fd, obj = self.split_fd(fd)
        self._handlers.pop(fd, None)
        self._events.pop(fd, None)
        try:
            self._impl.unregister(fd)
        except Exception:
            gen_log.debug("Error deleting fd from IOLoop", exc_info=True)

    def set_blocking_signal_threshold(self, seconds, action):
        if not hasattr(signal, "setitimer"):
            gen_log.error("set_blocking_signal_threshold requires a signal module "
                          "with the setitimer method")
            return
        self._blocking_signal_threshold = seconds
        if seconds is not None:
            signal.signal(signal.SIGALRM,
                          action if action is not None else signal.SIG_DFL)

    def start(self):
        if self._running:
            raise RuntimeError("IOLoop is already running")
        self._setup_logging()
        if self._stopped:
            self._stopped = False
            return
        old_current = getattr(IOLoop._current, "instance", None)
        IOLoop._current.instance = self
        self._thread_ident = thread.get_ident()
        self._running = True

        # signal.set_wakeup_fd closes a race condition in event loops:
        # a signal may arrive at the beginning of select/poll/etc
        # before it goes into its interruptible sleep, so the signal
        # will be consumed without waking the select.  The solution is
        # for the (C, synchronous) signal handler to write to a pipe,
        # which will then be seen by select.
        #
        # In python's signal handling semantics, this only matters on the
        # main thread (fortunately, set_wakeup_fd only works on the main
        # thread and will raise a ValueError otherwise).
        #
        # If someone has already set a wakeup fd, we don't want to
        # disturb it.  This is an issue for twisted, which does its
        # SIGCHLD processing in response to its own wakeup fd being
        # written to.  As long as the wakeup fd is registered on the IOLoop,
        # the loop will still wake up and everything should work.
        old_wakeup_fd = None
        if hasattr(signal, 'set_wakeup_fd') and os.name == 'posix':
            # requires python 2.6+, unix.  set_wakeup_fd exists but crashes
            # the python process on windows.
            try:
                old_wakeup_fd = signal.set_wakeup_fd(self._waker.write_fileno())
                if old_wakeup_fd != -1:
                    # Already set, restore previous value.  This is a little racy,
                    # but there's no clean get_wakeup_fd and in real use the
                    # IOLoop is just started once at the beginning.
                    signal.set_wakeup_fd(old_wakeup_fd)
                    old_wakeup_fd = None
            except ValueError:
                # Non-main thread, or the previous value of wakeup_fd
                # is no longer valid.
                old_wakeup_fd = None

        try:
            while True:
                # Prevent IO event starvation by delaying new callbacks
                # to the next iteration of the event loop.
                with self._callback_lock:
                    callbacks = self._callbacks
                    self._callbacks = []

                # Add any timeouts that have come due to the callback list.
                # Do not run anything until we have determined which ones
                # are ready, so timeouts that call add_timeout cannot
                # schedule anything in this iteration.
                due_timeouts = []
                if self._timeouts:
                    now = self.time()
                    while self._timeouts:
                        if self._timeouts[0].callback is None:
                            # The timeout was cancelled.  Note that the
                            # cancellation check is repeated below for timeouts
                            # that are cancelled by another timeout or callback.
                            heapq.heappop(self._timeouts)
                            self._cancellations -= 1
                        elif self._timeouts[0].deadline <= now:
                            due_timeouts.append(heapq.heappop(self._timeouts))
                        else:
                            break
                    if (self._cancellations > 512
                            and self._cancellations > (len(self._timeouts) >> 1)):
                        # Clean up the timeout queue when it gets large and it's
                        # more than half cancellations.
                        self._cancellations = 0
                        self._timeouts = [x for x in self._timeouts
                                          if x.callback is not None]
                        heapq.heapify(self._timeouts)

                for callback in callbacks:
                    self._run_callback(callback)
                for timeout in due_timeouts:
                    if timeout.callback is not None:
                        self._run_callback(timeout.callback)
                # Closures may be holding on to a lot of memory, so allow
                # them to be freed before we go into our poll wait.
                callbacks = callback = due_timeouts = timeout = None

                if self._callbacks:
                    # If any callbacks or timeouts called add_callback,
                    # we don't want to wait in poll() before we run them.
                    poll_timeout = 0.0
                elif self._timeouts:
                    # If there are any timeouts, schedule the first one.
                    # Use self.time() instead of 'now' to account for time
                    # spent running callbacks.
                    poll_timeout = self._timeouts[0].deadline - self.time()
                    poll_timeout = max(0, min(poll_timeout, _POLL_TIMEOUT))
                else:
                    # No timeouts and no callbacks, so use the default.
                    poll_timeout = _POLL_TIMEOUT

                if not self._running:
                    break

                if self._blocking_signal_threshold is not None:
                    # clear alarm so it doesn't fire while poll is waiting for
                    # events.
                    signal.setitimer(signal.ITIMER_REAL, 0, 0)

                try:
                    event_pairs = self._impl.poll(poll_timeout)
                except Exception as e:
                    # Depending on python version and IOLoop implementation,
                    # different exception types may be thrown and there are
                    # two ways EINTR might be signaled:
                    # * e.errno == errno.EINTR
                    # * e.args is like (errno.EINTR, 'Interrupted system call')
                    if errno_from_exception(e) == errno.EINTR:
                        continue
                    else:
                        raise

                if self._blocking_signal_threshold is not None:
                    signal.setitimer(signal.ITIMER_REAL,
                                     self._blocking_signal_threshold, 0)

                # Pop one fd at a time from the set of pending fds and run
                # its handler. Since that handler may perform actions on
                # other file descriptors, there may be reentrant calls to
                # this IOLoop that update self._events
                self._events.update(event_pairs)
                while self._events:
                    fd, events = self._events.popitem()
                    try:
                        fd_obj, handler_func = self._handlers[fd]
                        handler_func(fd_obj, events)
                    except (OSError, IOError) as e:
                        if errno_from_exception(e) == errno.EPIPE:
                            # Happens when the client closes the connection
                            pass
                        else:
                            self.handle_callback_exception(self._handlers.get(fd))
                    except Exception:
                        self.handle_callback_exception(self._handlers.get(fd))
                fd_obj = handler_func = None

        finally:
            # reset the stopped flag so another start/stop pair can be issued
            self._stopped = False
            if self._blocking_signal_threshold is not None:
                signal.setitimer(signal.ITIMER_REAL, 0, 0)
            IOLoop._current.instance = old_current
            if old_wakeup_fd is not None:
                signal.set_wakeup_fd(old_wakeup_fd)

    def stop(self):
        self._running = False
        self._stopped = True
        self._waker.wake()

    def time(self):
        return self.time_func()

    def call_at(self, deadline, callback, *args, **kwargs):
        timeout = _Timeout(
            deadline,
            functools.partial(stack_context.wrap(callback), *args, **kwargs),
            self)
        heapq.heappush(self._timeouts, timeout)
        return timeout

    def remove_timeout(self, timeout):
        # Removing from a heap is complicated, so just leave the defunct
        # timeout object in the queue (see discussion in
        # http://docs.python.org/library/heapq.html).
        # If this turns out to be a problem, we could add a garbage
        # collection pass whenever there are too many dead timeouts.
        timeout.callback = None
        self._cancellations += 1

    def add_callback(self, callback, *args, **kwargs):
        if thread.get_ident() != self._thread_ident:
            # If we're not on the IOLoop's thread, we need to synchronize
            # with other threads, or waking logic will induce a race.
            with self._callback_lock:
                if self._closing:
                    return
                list_empty = not self._callbacks
                self._callbacks.append(functools.partial(
                    stack_context.wrap(callback), *args, **kwargs))
                if list_empty:
                    # If we're not in the IOLoop's thread, and we added the
                    # first callback to an empty list, we may need to wake it
                    # up (it may wake up on its own, but an occasional extra
                    # wake is harmless).  Waking up a polling IOLoop is
                    # relatively expensive, so we try to avoid it when we can.
                    self._waker.wake()
        else:
            if self._closing:
                return
            # If we're on the IOLoop's thread, we don't need the lock,
            # since we don't need to wake anyone, just add the
            # callback. Blindly insert into self._callbacks. This is
            # safe even from signal handlers because the GIL makes
            # list.append atomic. One subtlety is that if the signal
            # is interrupting another thread holding the
            # _callback_lock block in IOLoop.start, we may modify
            # either the old or new version of self._callbacks, but
            # either way will work.
            self._callbacks.append(functools.partial(
                stack_context.wrap(callback), *args, **kwargs))

    def add_callback_from_signal(self, callback, *args, **kwargs):
        with stack_context.NullContext():
            self.add_callback(callback, *args, **kwargs)


class _Timeout(object):
    """An IOLoop timeout, a UNIX timestamp and a callback"""

    # Reduce memory overhead when there are lots of pending callbacks
    __slots__ = ['deadline', 'callback', 'tiebreaker']

    def __init__(self, deadline, callback, io_loop):
        if not isinstance(deadline, numbers.Real):
            raise TypeError("Unsupported deadline %r" % deadline)
        self.deadline = deadline
        self.callback = callback
        self.tiebreaker = next(io_loop._timeout_counter)

    # Comparison methods to sort by deadline, with object id as a tiebreaker
    # to guarantee a consistent ordering.  The heapq module uses __le__
    # in python2.5, and __lt__ in 2.6+ (sort() and most other comparisons
    # use __lt__).
    def __lt__(self, other):
        return ((self.deadline, self.tiebreaker) <
                (other.deadline, other.tiebreaker))

    def __le__(self, other):
        return ((self.deadline, self.tiebreaker) <=
                (other.deadline, other.tiebreaker))


class PeriodicCallback(object):
    """Schedules the given callback to be called periodically.

    The callback is called every ``callback_time`` milliseconds.
    Note that the timeout is given in milliseconds, while most other
    time-related functions in Tornado use seconds.

    If the callback runs for longer than ``callback_time`` milliseconds,
    subsequent invocations will be skipped to get back on schedule.

    `start` must be called after the `PeriodicCallback` is created.

    .. versionchanged:: 4.1
       The ``io_loop`` argument is deprecated.
    """
    def __init__(self, callback, callback_time, io_loop=None):
        self.callback = callback
        if callback_time <= 0:
            raise ValueError("Periodic callback must have a positive callback_time")
        self.callback_time = callback_time
        self.io_loop = io_loop or IOLoop.current()
        self._running = False
        self._timeout = None

    def start(self):
        """Starts the timer."""
        self._running = True
        self._next_timeout = self.io_loop.time()
        self._schedule_next()

    def stop(self):
        """Stops the timer."""
        self._running = False
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
            self._timeout = None

    def is_running(self):
        """Return True if this `.PeriodicCallback` has been started.

        .. versionadded:: 4.1
        """
        return self._running

    def _run(self):
        if not self._running:
            return
        try:
            return self.callback()
        except Exception:
            self.io_loop.handle_callback_exception(self.callback)
        finally:
            self._schedule_next()

    def _schedule_next(self):
        if self._running:
            current_time = self.io_loop.time()

            if self._next_timeout <= current_time:
                callback_time_sec = self.callback_time / 1000.0
                self._next_timeout += (math.floor((current_time - self._next_timeout) / callback_time_sec) + 1) * callback_time_sec

            self._timeout = self.io_loop.add_timeout(self._next_timeout, self._run)
