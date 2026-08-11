import os
import sys
from contextlib import contextmanager


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    A context manager to temporarily redirect stdout to another file. This is useful when you want to suppress the
    output from blender, especially when you are rendering a large number of images.

    Example usage:
    >>> import os
    >>> filename = os.devnull
    >>> with stdout_redirected(to=filename):
    ...     print("from Python")
    ...     os.system("echo non-Python applications are also supported")

    :param to: The file to redirect stdout to. Default is os.devnull.
    :return: The context manager.
    """

    fd = sys.stdout.fileno()

    def _redirect_stdout(_to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(_to.fileno(), fd)  # fd writes to '_to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(_to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(_to=old_stdout)  # restore stdout.
