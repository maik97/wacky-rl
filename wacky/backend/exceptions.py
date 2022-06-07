

class WackyException(Exception):

    def __init__(self, *args):
        super(WackyException, self).__init__(*args)


class WackyTypeError(WackyException):

    def __init__(self, var, allowed_types, parameter=None, optional=False):
        super(WackyTypeError, self).__init__()
        self.var, self.allowed_types, self.parameter, self.optional = var, allowed_types, parameter, optional

    def __str__(self):
        message = f"\n TypeError: Expected type {self.allowed_types} got instead {type(self.var)}"
        if self.parameter is not None:
            message += f" for parameter '{self.parameter}'."
        if self.optional:
            message += f"\n Note the parameter is optional, 'None' is allowed."
        return message


class WackyValueError(WackyException):

    def __init__(self, var, allowed_values, parameter=None, optional=False):
        super(WackyValueError, self).__init__()
        self.var, self.allowed_values, self.parameter, self.optional = var, allowed_values, parameter, optional

    def __str__(self):
        message = f"\n ValueError: Expected value {self.allowed_values} got instead '{self.var}'"
        if self.parameter is not None:
            message += f" for parameter '{self.parameter}'."
        if self.optional:
            message += f"\n Note the parameter is optional, 'None' is allowed."
        return message
