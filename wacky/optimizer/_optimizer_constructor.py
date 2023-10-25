from typing import Type, Dict, Optional, Union, Any
import torch.optim as optim


class OptimizerConstructor:

    @staticmethod
    def special_kwargs_handling(optimizer_class: Type[optim.Optimizer], optimizer_kwargs: Dict) -> Dict:
        # Add special handling of arguments here if necessary
        return optimizer_kwargs

    @staticmethod
    def from_class(optimizer_class: Type[optim.Optimizer], params,
                   optimizer_kwargs: Optional[Dict] = None) -> optim.Optimizer:
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        #  optimizer_kwargs = OptimizerConstructor.special_kwargs_handling(optimizer_class, optimizer_kwargs)
        return optimizer_class(params, **optimizer_kwargs)

    @staticmethod
    def from_string(optimizer_name: str, params, optimizer_kwargs: Optional[Dict] = None) -> optim.Optimizer:
        optimizer_class = getattr(optim, optimizer_name)
        return OptimizerConstructor.from_class(optimizer_class, params, optimizer_kwargs)

    @staticmethod
    def from_dict(optimizer_dict: Dict, params) -> optim.Optimizer:
        optimizer_class = optimizer_dict.get('type')
        optimizer_kwargs = optimizer_dict.get('kwargs', {})
        if isinstance(optimizer_class, str):
            optimizer_class = getattr(optim, optimizer_class)
        return OptimizerConstructor.from_class(optimizer_class, params, optimizer_kwargs)

    @staticmethod
    def assert_no_kwargs(optimizer_kwargs):
        if optimizer_kwargs:
            raise ValueError(
                "Additional optimizer_kwargs are not allowed when using a dictionary to construct an optimizer."
            )

    @staticmethod
    def construct(optimizer: Optional[Union[str, Dict, Type[optim.Optimizer]]],
                  params,
                  optimizer_kwargs: Any = None) -> Optional[optim.Optimizer]:

        optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        if optimizer is None:
            return None
        elif isinstance(optimizer, str):
            return OptimizerConstructor.from_string(optimizer, params, optimizer_kwargs)
        elif isinstance(optimizer, type) and issubclass(optimizer, optim.Optimizer):
            return OptimizerConstructor.from_class(optimizer, params, optimizer_kwargs)
        elif isinstance(optimizer, dict):
            OptimizerConstructor.assert_no_kwargs(optimizer_kwargs)
            return OptimizerConstructor.from_dict(optimizer, params)
        else:
            raise ValueError("Invalid type for optimizer")
