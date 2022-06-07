from torch import optim
from wacky import functional as funky
from wacky.optimizer import WackyOptimizer


class TorchOptimizer:

    def __init__(self, optimizer: (str, optim.Optimizer), network_parameter, lr:float, *args, **kwargs):
        """
        Creates torch.optimizer class if optimizer is given as string for the name of the optimizer.
        Possible optimizer names are:
            - 'Adadelta'
            - 'Adagrad'
            - 'Adam'
            - 'AdamW'
            - 'SparseAdam'
            - 'Adamax'
            - 'ASGD'
            - 'LBFGS'
            - 'NAdam'
            - 'RAdam'
            - 'RMSprop'
            - 'Rprop'
            - 'SGD'
        Alternatively, optimizer is given as (not initialized) subclass of optim.Optimizer.

        .. warning::
            Parameters need to be specified as collections that have a deterministic
            ordering that is consistent between runs. Examples of objects that don't
            satisfy those properties are sets and iterators over values of dictionaries.

        :param optimizer: (str, torch.optim.Optimizer): Either a the name of an optimizer as str
            or a subclass of torch.optim.Optimizer
        :param network_parameter: (iterable, nn.Module, wacky.WackNetwork): either is a network
            or iterable of parameters to optimize or dicts defining parameter groups
        :param lr: coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        :param args: Additional Parameter passed to optimizer class
        :param kwargs: Additional Parameter passed to optimizer class
        """

        params = funky.maybe_get_network_params(network_parameter)
        self.th_optimizer = funky.get_optim(optimizer, params, lr, *args, **kwargs)

    def apply_loss(self, loss, set_to_none: bool = False, *args, **kwargs):
        self.zero_grad(set_to_none)
        loss.backward()
        loss =  self.step(*args, **kwargs)
        return loss


    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        return self.th_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.th_optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        self.th_optimizer.zero_grad(set_to_none)

    def step(self, *args, **kwargs):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        return self.th_optimizer.step(*args, **kwargs)

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        self.th_optimizer.add_param_group(param_group)
