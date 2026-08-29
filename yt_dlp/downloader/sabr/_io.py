from __future__ import annotations
import abc
import asyncio
import collections
import io
import os
import queue
import shutil
import threading
import time
import typing


class FormatIOBackend(abc.ABC):
    def __init__(self, fd, filename, buffer=1024 * 1024):
        self.fd = fd
        self.filename = filename
        self.write_buffer = buffer
        self._fp = None
        self._fp_mode = None

    @abc.abstractmethod
    def __len__(self):
        pass

    @property
    def writer(self):
        if self._fp is None or self._fp_mode != 'write':
            return None
        return self._fp

    @property
    def reader(self):
        if self._fp is None or self._fp_mode != 'read':
            return None
        return self._fp

    def initialize_writer(self, resume=False):
        if self._fp is not None:
            raise ValueError('Backend already initialized')

        self._fp = self._create_writer(resume)
        self._fp_mode = 'write'

    @abc.abstractmethod
    def _create_writer(self, resume=False) -> typing.IO:
        pass

    def initialize_reader(self):
        if self._fp is not None:
            raise ValueError('Backend already initialized')
        self._fp = self._create_reader()
        self._fp_mode = 'read'

    @abc.abstractmethod
    def _create_reader(self) -> typing.IO:
        pass

    def close(self):
        if self._fp and not self._fp.closed:
            self._fp.flush()
            self._fp.close()
        self._fp = None
        self._fp_mode = None

    def validate_length(self, expected_length):
        return len(self) == expected_length

    def remove(self):
        self.close()
        self._remove()

    @abc.abstractmethod
    def _remove(self):
        pass

    @abc.abstractmethod
    def exists(self):
        pass

    @property
    def mode(self):
        if self._fp is None:
            return None
        return self._fp_mode

    def write(self, data: io.BufferedIOBase | bytes):
        if not self.writer:
            raise ValueError('Backend writer not initialized')

        if isinstance(data, bytes):
            bytes_written = self.writer.write(data)
        elif isinstance(data, io.BufferedIOBase):
            bytes_written = self.writer.tell()
            shutil.copyfileobj(data, self.writer, length=self.write_buffer)
            bytes_written = self.writer.tell() - bytes_written
        else:
            raise TypeError('Data must be bytes or a BufferedIOBase object')

        self.writer.flush()

        return bytes_written

    def read_into(self, backend):
        if not backend.writer:
            raise ValueError('Destination backend writer not initialized')
        if not self.reader:
            raise ValueError('Backend reader not initialized')
        shutil.copyfileobj(self.reader, backend.writer, length=self.write_buffer)
        backend.writer.flush()


class DiskFormatIOBackend(FormatIOBackend):
    def __len__(self):
        return 0 if not self.exists() else os.path.getsize(self.filename)

    def _create_writer(self, resume=False) -> typing.IO:
        if resume and self.exists():
            write_fp, self.filename = self.fd.sanitize_open(self.filename, 'ab')
        else:
            write_fp, self.filename = self.fd.sanitize_open(self.filename, 'wb')
        return write_fp

    def _create_reader(self) -> typing.IO:
        read_fp, self.filename = self.fd.sanitize_open(self.filename, 'rb')
        return read_fp

    def _remove(self):
        self.fd.try_remove(self.filename)

    def exists(self):
        return os.path.isfile(self.filename)


class SynchronizedBytesIO(io.BytesIO):
    """
    A fully synchronized BytesIO implementation ensuring thread safety and
    async compatibility across multiple event loops.
    """
    # Class-level lock to ensure thread-safe initialization of instance-level locks
    _init_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = None
        self._async_locks = {}

        # Track which task holds the async lock
        self._holder_task = None

    def _get_async_lock(self):
        loop = asyncio.get_running_loop()
        with self._get_thread_lock():
            if loop not in self._async_locks:
                self._async_locks[loop] = asyncio.Lock()
            return self._async_locks[loop]

    async def _run_async(self, func, *args, **kwargs):
        # Only bypass the lock if the CURRENT task is the one that acquired it
        if self._holder_task is asyncio.current_task():
            return func(*args, **kwargs)

        async with self._get_async_lock():
            return func(*args, **kwargs)

    def _get_thread_lock(self):
        if self._thread_lock is None:
            with self._init_lock:
                if self._thread_lock is None:
                    # RLock allows nested 'with' calls on the same thread
                    self._thread_lock = threading.RLock()
        return self._thread_lock

    # ==== Sync Context Manager ====
    def __enter__(self):
        self._get_thread_lock().acquire()
        return self

    def __exit__(self, *args):
        self._get_thread_lock().release()

    # ==== Async Context Manager ====
    async def __aenter__(self):
        await self._get_async_lock().acquire()
        # Record the current task as the lock holder
        self._holder_task = asyncio.current_task()
        return self

    async def __aexit__(self, *args):
        self._holder_task = None
        self._get_async_lock().release()

    # ==== Length Operations ====

    def __len__(self):
        with self._get_thread_lock():
            # Use super() to avoid calling overridden methods
            # that might attempt to acquire the lock again.
            pos = super().tell()
            try:
                return super().seek(0, io.SEEK_END)
            finally:
                super().seek(pos, io.SEEK_SET)

    async def alen(self):
        return await self._run_async(self.__len__, self)

    # ==== Read Operations ====

    def read(self, size=-1):
        with self._get_thread_lock():
            return super().read(size)

    async def aread(self, size=-1):
        return await self._run_async(self.read, size)

    def readline(self, size=-1):
        with self._get_thread_lock():
            return super().readline(size)

    async def areadline(self, size=-1):
        return await self._run_async(self.readline, size)

    def readlines(self, hint=-1):
        with self._get_thread_lock():
            return super().readlines(hint)

    async def areadlines(self, hint=-1):
        return await self._run_async(self.readlines, hint)

    # ==== Write Operations ====

    def write(self, b):
        with self._get_thread_lock():
            return super().write(b)

    async def awrite(self, b):
        return await self._run_async(self.write, b)

    def writelines(self, lines):
        with self._get_thread_lock():
            return super().writelines(lines)

    async def awritelines(self, lines):
        return await self._run_async(self.writelines, lines)

    # ==== Pointer & Buffer Operations ====

    def seek(self, offset, whence=io.SEEK_SET):
        with self._get_thread_lock():
            return super().seek(offset, whence)

    async def aseek(self, offset, whence=io.SEEK_SET):
        return await self._run_async(self.seek, offset, whence)

    def tell(self):
        with self._get_thread_lock():
            return super().tell()

    async def atell(self):
        return await self._run_async(self.tell)

    def truncate(self, size=None):
        with self._get_thread_lock():
            return super().truncate(size)

    async def atruncate(self, size=None):
        return await self._run_async(self.truncate, size)

    def getvalue(self):
        with self._get_thread_lock():
            return super().getvalue()

    def getbuffer(self):
        with self._get_thread_lock():
            return super().getbuffer()

    # ==== Lifecycle Operations ====

    def flush(self):
        with self._get_thread_lock():
            return super().flush()

    def close(self):
        try:
            if self._thread_lock:
                with self._thread_lock:
                    super().close()
            else:
                super().close()
        finally:
            self._thread_lock = None
            self._async_locks.clear()


class MemoryFormatIOBackend(FormatIOBackend):
    def __init__(self, *args, **kwargs):
        self._remove()
        super().__init__(*args, **kwargs)

    def _remove(self):
        self._memory_store = SynchronizedBytesIO()

    def _reset(self):
        with self._memory_store as ms:
            ms.seek(0)
            ms.truncate(0)

    def __len__(self):
        return len(self._memory_store)

    def _create_writer(self, resume=False) -> typing.IO:
        class NonClosingBufferedWriter(io.BufferedWriter):
            def close(self):
                self.flush()
                # Do not close the underlying buffer

        if resume and self.exists():
            self._memory_store.seek(0, io.SEEK_END)
        else:
            self._reset()

        return NonClosingBufferedWriter(self._memory_store)

    def _create_reader(self) -> typing.IO:
        class NonClosingBufferedReader(io.BufferedReader):
            def close(self):
                self.flush()

        # Seek to the beginning of the buffer
        self._memory_store.seek(0)
        return NonClosingBufferedReader(self._memory_store)

    def exists(self):
        return len(self) > 0


class ProxiedIOBackend(DiskFormatIOBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_mem_be = None
        self._lock = threading.RLock()
        self._pending_bytes_count = 0
        self._worker = None
        self._write_queue = queue.Queue()
        self.exc_info = None

        # Initialize the authoritative Disk destination once
        self.initialize_writer(resume=False)

    def __len__(self):
        with self._lock:
            disk_file = super().__len__()
            queued_bytes = self._pending_bytes_count
            memory = 0 if self._current_mem_be is None else len(self._current_mem_be)
        return memory + queued_bytes + disk_file

    def _begin_queue(self):
        with self._lock:
            self.exc_info = None
            # Start or re-start the worker if it's not active
            if self._worker is None or not self._worker.is_alive():
                base_name = os.path.basename(self.filename)
                self._worker = threading.Thread(
                    target=self._drain_queue,
                    name=f'DiskFormatIOQueue-{base_name}',
                    daemon=True,
                )
                self._worker._logs = collections.deque(maxlen=1_000)
                self._worker._queue = self._write_queue
                self._worker.start()

    def _create_mem_backend(self):
        """Creates a fresh, memory-only backend instance."""
        mem_backend = MemoryFormatIOBackend(
            fd=self.fd,
            filename=self.filename,
        )
        mem_backend.initialize_writer(resume=False)
        return mem_backend

    def _create_writer(self, resume=False) -> typing.IO:
        disk_write_fp = super()._create_writer(resume)

        class NullWriter:
            def __init__(self): self.closed = False
            def write(self, data): pass
            def flush(self): pass
            def close(self): self.closed = True

        with self._lock:
            self._fp_mode = 'write'
            self._fp = disk_write_fp
            if not resume:
                self._fp = NullWriter()
            if self._current_mem_be:
                self.append(self._current_mem_be)
                self._current_mem_be = None
        self._begin_queue()
        if not resume:
            self.flush()
        with self._lock:
            self._fp = disk_write_fp

        return disk_write_fp

    def _drain_queue(self):
        """
        Worker thread that consumes sealed backends and streams to disk.
        """

        logs = list()
        log = lambda msg, m=time.monotonic_ns, t=time.time: logs.append(f'{m()}: {t()}: {msg}')
        log('Worker started.')
        with self._lock:
            log_dest = self._worker._logs

        def record():
            nonlocal logs
            log_dest.append(logs)
            logs = list()

        def finish(msg):
            self._write_queue.task_done()
            log(msg)
            record()

        log('Worker initialized.')
        record()
        backend = self._write_queue.get()
        while backend is not None:
            log('Received a new backend.')
            try:
                log('Closing the backend.')
                backend.close()
                log('Initializing the backend reader.')
                backend.initialize_reader()
                log('Reading from the backend...')
                with self._lock:
                    log('Lock acquired.')
                    length = self.write_buffer
                    suceeded = self.exc_info is None
                    writer = self.writer
                log('Lock released.')
                if suceeded and writer and backend.exists():
                    log(f'Backend exists with: size={len(backend)}')
                    log(f'Copy started with: {length=} {writer=}')
                    shutil.copyfileobj(backend.reader, writer, length=length)
                    log('Copy completed.')
                    writer.flush()
                    log('Flush completed.')
                elif not suceeded:
                    log('Skipped copying after the exception information was recorded.')
                log('Reading is complete.')
            except Exception as e:
                with self._lock:
                    self.exc_info = e
                log(f'Recorded the following exception: {e!r}')
            finally:
                with self._lock:
                    log('Lock acquired.')
                    log(f'Pending byte count: {self._pending_bytes_count}')
                    self._pending_bytes_count -= len(backend)
                    log(f'Pending byte count: {self._pending_bytes_count}')
                log('Lock released.')
                # Immediately purge RAM once serialized to disk
                backend.remove()
                log('Backend removed.')

                # Keep the queue happy and the loop going
                finish('Marked the queue task as done.')
                backend = self._write_queue.get()
        # Poison pill received: Finalize the file handle via parent
        log('The loop has ended.')
        super().close()
        log('The call to super().close() has completed.')
        finish('Marked the final queue task as done.')

    def append(self, backend):
        """Seals the backend and adds it to the queue."""
        backend.close()
        if backend.exists():
            with self._lock:
                if self.exc_info is not None:
                    raise self.exc_info
                self._pending_bytes_count += len(backend)
            self._write_queue.put(backend)

    def close(self):
        """Stops the worker thread and finalizes the backend."""

        worker = None
        with self._lock:
            worker = self._worker
            backend = self._current_mem_be
            self._current_mem_be = None
            if backend and self.exc_info is None:
                self.append(backend)
            elif backend:
                backend.remove()

        if worker is not None and worker.is_alive():
            self._write_queue.put(None)
            self.flush()
            worker.join()

        with self._lock:
            self._worker = None

        super().close()

        with self._lock:
            if self.exc_info is not None:
                raise self.exc_info

    def flush(self):
        """Blocks until the queue is completely drained to disk."""
        # Ensure any data currently in the active RAM buffer is queued first
        worker = None
        with self._lock:
            worker = self._worker
            backend = self._current_mem_be
            self._current_mem_be = None
            if worker and self.writer and self.exc_info is not None:
                backend.remove()
                raise self.exc_info
            if backend:
                self.append(backend)

        # Only join if the worker is actually alive to process it
        if worker is not None and worker.is_alive():
            # Block until the worker finishes processing all items currently in the queue
            self._write_queue.join()

    def write(self, data: io.BufferedIOBase | bytes):
        """
        Writes to the live memory buffer.
        Queues the buffer only AFTER it crosses the limit.
        """

        with self._lock:
            if self._current_mem_be is None:
                self._current_mem_be = self._create_mem_backend()
            written = self._current_mem_be.write(data)

            if len(self._current_mem_be) > self.write_buffer:
                self.append(self._current_mem_be)
                self._current_mem_be = None

            return written
