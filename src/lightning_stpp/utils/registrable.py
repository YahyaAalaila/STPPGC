from collections import defaultdict

class Registrable:
    """
    This is inspired by EasyTPP structure. Any class that inherits from Registrable can be registered and instantiated. 
    gains access to a named registry for its subclasses. This usefull if we want to include another paradigm of STPPs 
    that are not based on DL techniques, and we want to be able to instatiate them using the same interface.

    As it stands (10/04/2025), this is used only to create a DL-based runner.
    """
    _registry = defaultdict(dict)
    _default_impl = None
    @classmethod
    def register(cls, name, constructor=None, overwrite=False):
        """Register a class under a particular name.
        Args:
            name (str): The name to register the class under.
            constructor (str): optional (default=None)
                The name of the method to use on the class to construct the object.  If this is given,
                we will use this method (which must be a ``classmethod``) instead of the default
                constructor.
            overwrite (bool) : optional (default=False)
                If True, overwrites any existing models registered under ``name``. Else,
                throws an error if a model is already registered under ``name``.

        # Examples
        To use this class, you would typically have a base class that inherits from ``Registrable``:
        ```python
        class Transform(Registrable):
            ...
        ```
        Then, if you want to register a subclass, you decorate it like this:
        ```python
        @Transform.register("shift-transform")
        class ShiftTransform(Transform):
            def __init__(self, param1: int, param2: str):
                ...
        ```
        """
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass):
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                message = (
                    f"Cannot register {name} as {cls.__name__}; "
                    f"name already in use for {registry[name][0].__name__}"
                )
                raise RuntimeError(message)
            registry[name] = (subclass, constructor)
            return subclass

        return add_subclass_to_registry
    
    @classmethod
    def by_name(cls, name):
        """
        Returns a callable function that constructs an argument of the registered class.  Because
        you can register particular functions as constructors for specific names, this isn't
        necessarily the ``__init__`` method of some class.
        """
        #logger.debug(f"instantiating registered subclass {name} of {cls}")
        subclass, _ = cls.resolve_class_name(name)      
        return subclass
    @classmethod
    def resolve_class_name(cls, name):
        """
        Returns the subclass that corresponds to the given ``name``, along with the name of the
        method that was registered as a constructor for that ``name``, if any.
        This method also allows ``name`` to be a fully-specified module name, instead of a name that
        was already added to the ``Registry``.  In that case, you cannot use a separate function as
        a constructor (as you need to call ``cls.register()`` in order to tell us what separate
        function to use).
        """
        if name in Registrable._registry[cls]:
            subclass, constructor = Registrable._registry[cls].get(name)
            return subclass, constructor
        else:
            for base_cls, v in Registrable._registry.items():
                if name in v:
                    subclass, constructor = Registrable._registry[base_cls].get(name)
                    return subclass, constructor

        if "." in name:
            # This might be a fully qualified class name, so we'll try importing its "module"
            # and finding it there.
            parts = name.split(".")
            submodule = ".".join(parts[:-1])
            class_name = parts[-1]
            import importlib
            try:
                module = importlib.import_module(submodule)
            except ModuleNotFoundError:
                raise RuntimeError(
                    f"tried to interpret {name} as a path to a class "
                    f"but unable to import module {submodule}"
                )

            try:
                subclass = getattr(module, class_name)
                constructor = None
                return subclass, constructor
            except AttributeError:
                raise RuntimeError(
                    f"tried to interpret {name} as a path to a class "
                    f"but unable to find class {class_name} in {submodule}"
                )

        else:
            # is not a qualified class name
            raise RuntimeError(
                f"{name} is not a registered name for {cls.__name__}. "
                "You probably need to use the --include-package flag "
                "to load your custom code. Alternatively, you can specify your choices "
                """using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} """
                "in which case they will be automatically imported correctly."
            )

    @classmethod
    def list_available(cls):
        """List default first if it exists"""
        keys = list(Registrable._registry[cls].keys())
        default = cls._default_impl

        if default is None:
            return keys
        elif default not in keys:
            raise RuntimeError(f"Default implementation {default} is not registered")
        else:
            return [default] + [k for k in keys if k != default]

    @classmethod
    def registry_dict(cls):
        return Registrable._registry[cls]
