import numpy as np
from scipy import interpolate
from wacky import functional as funky


def check_list(val):
    if not isinstance(val, list):
        val = [val]
    return val


def assign_ramp_points(val_a: float, val_b: float, point_a: float, point_b: float):
    x_vals = np.array([])
    y_vals = np.array([])
    val_a, val_b, point_a, point_b = check_list(val_a), check_list(val_b), check_list(point_a), check_list(point_b)
    if point_a[0] == 0.0:
        x_vals = np.append(x_vals, point_a)
        y_vals = np.append(y_vals, val_a)
    else:
        x_vals = np.append(x_vals, [0.0, *point_a])
        y_vals = np.append(y_vals, [val_a[0], *val_a])
    if point_b[-1] == 1.0:
        x_vals = np.append(x_vals, point_b)
        y_vals = np.append(y_vals, val_b)
    else:
        x_vals = np.append(x_vals, [*point_b, 1.0])
        y_vals = np.append(y_vals, [*val_b, val_b[-1]])
    return x_vals, y_vals


def get_ramp_interp1d(
        val_a: float,
        val_b: float,
        point_a: float,
        point_b: float,
        kind: str = 'linear',
        *args, **kwargs
) -> interpolate.interp1d:
    """
    Initialize a 1-D linear interpolation class, to interpolate a ramp. Points will be assigned as follows:

        x = [0.0, point_a, point_b, 1.0]
        y = [val_a,val_a, val_b, val_b]

    :param val_a:
        Corresponds as y to given x by point_a
    :param val_b:
        Corresponds as y to given x by point_b
    :param point_a:
        Sets ramp start point, value on x-axis between 0.0 and 1.0
    :param point_b:
        Sets ramp end point, value on x-axis between 0.0 and 1.0
    :param kind:
        Documentation taken from scipy:
            Specifies the kind of interpolation as a string or as an integer specifying the order of the spline
            interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
            ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
            interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or
            next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5)
            in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
    :return:
        interpolate.interp1d object, that interpolates the ramp, when called
    """
    x, y = assign_ramp_points(val_a, val_b, point_a, point_b)
    return interpolate.interp1d(
        x=np.array([0.0, point_a, point_b, 1.0]),
        y=np.array([val_a,val_a, val_b, val_b]),
        kind=kind,
        *args, **kwargs
    )

class RampInterpolator(funky.WackyBase):

    def __init__(
            self,
            val_a: [float, list],
            val_b: [float, list],
            point_a: [float, list],
            point_b: [float, list],
            kind: str = 'linear',
            clip=True,
            *args, **kwargs
    ):
        super(RampInterpolator, self).__init__()

        self.x_vals, self.y_vals= assign_ramp_points(val_a, val_b, point_a, point_b)
        print(self.x_vals)
        print(self.y_vals)

        self.f = interpolate.interp1d(
            x=self.x_vals,
            y=self.y_vals,
            kind=kind,
            *args, **kwargs
        )
        self.clip = clip
        if self.clip:
            self.a_min = np.min(self.y_vals)
            self.a_max = np.max(self.y_vals)

    def call(self, x):
        y = self.f(x)
        if self.clip:
            y = np.clip(y, self.a_min, self.a_max)
        return y

    def plot(self, x_new=None, mode='show'):
        import matplotlib.pyplot as plt
        if x_new is None:
            x_new = np.arange(0, 1.01, 0.01)
        y_new = self(x_new)
        plt.plot(self.x_vals, self.y_vals, 'o', x_new, y_new, '-')
        if mode == 'show':
            plt.show()


def main():

    ramp_interp = RampInterpolator(
        val_a=[1.0],
        val_b=[0.55, 0.5, 0.45,0.01],
        point_a=[0.0],
        point_b=[0.4, 0.5, 0.6,0.9],
        kind='quadratic'
    )

    ramp_interp.plot()


if __name__ == '__main__':
    main()