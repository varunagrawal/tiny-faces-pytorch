from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["utils/metrics.pyx",
                             "utils/dense_overlap.pyx",
                             "utils/nms.pyx",
                             "processing/image.pyx"])
)