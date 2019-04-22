from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["utils/metrics.pyx",
                             "utils/nms.pyx"])
)