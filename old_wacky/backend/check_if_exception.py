from wacky.backend import WackyTypeError, WackyValueError


def raise_type_error(var, allowed_types, var_name=None):
    if var_name is None:
        raise TypeError("Must be types:", allowed_types, "not", type(var))
    else:
        raise TypeError(var_name, "must be types:", allowed_types, "not", type(var))


def check_type(var, allowed_types, var_name=None, allow_none=False):
    if allow_none and var is None:
        pass
    elif not isinstance(var, allowed_types):
        raise WackyTypeError(var, allowed_types, var_name, allow_none)


def check_value(var, allowed_vals, var_name=None, allow_none=False):
    if allow_none and var is None:
        pass
    elif isinstance(allowed_vals, (tuple, list)):
        if var not in allowed_vals:
            raise WackyValueError(var, allowed_vals, var_name, allow_none)


def main():
    check_value('summ', ('mean', 'sum'), 'wacky_reduce', True)


if __name__ == '__main__':
    main()
