# models/__init__.py
import os
import pkgutil
import importlib

# Automatically import all modules in the current package directory.
package_dir = os.path.dirname(__file__)
for (_, module_name, _) in pkgutil.iter_modules([package_dir]):
    importlib.import_module(f"{__name__}.{module_name}")

# Optionally, define __all__ if you want to control what’s exported.
__all__ = [name for (_, name, _) in pkgutil.iter_modules([package_dir])]
