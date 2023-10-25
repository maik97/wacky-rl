from typing import List, Type, Dict, Union, Optional, Any
import torch.nn as nn


class ModuleConstructor:

    @staticmethod
    def special_kwargs_handling(module_class: Type[nn.Module], module_kwargs: Dict) -> Dict:
        if module_class == nn.Softmax and 'dim' not in module_kwargs:
            module_kwargs['dim'] = -1
        return module_kwargs

    @staticmethod
    def from_class(module_class: Type[nn.Module], module_kwargs: Optional[Dict] = None) -> nn.Module:
        if module_kwargs is None:
            module_kwargs = {}
        module_kwargs = ModuleConstructor.special_kwargs_handling(module_class, module_kwargs)
        return ModuleConstructor.from_module(module_class(**module_kwargs), module_kwargs)

    @staticmethod
    def from_string(module_name: str, module_kwargs: Optional[Dict] = None) -> nn.Module:
        if module_name == 'SimpleMLP':
            return ModuleConstructor.construct([64, 'Tanh', 64, 'Tanh'])
        layer_class = getattr(nn, module_name)
        return ModuleConstructor.from_class(layer_class, module_kwargs)

    @staticmethod
    def special_executions(module_instance: nn.Module, module_kwargs: Dict) -> nn.Module:
        init_method = module_kwargs.get('init_method', None)
        if init_method:
            init_method(module_instance.weight)
        return module_instance

    @staticmethod
    def from_module(module_instance: nn.Module, module_kwargs: Dict) -> nn.Module:
        return ModuleConstructor.special_executions(module_instance, module_kwargs)

    @staticmethod
    def from_dict(module_dict: Dict) -> nn.Module:
        module_class = module_dict.get('type')
        module_kwargs = module_dict.get('kwargs', {})
        if isinstance(module_class, str):
            module_class = getattr(nn, module_class)
        return ModuleConstructor.from_class(module_class, module_kwargs)

    @staticmethod
    def from_list(module_list: List[Union[int, Dict, str]]) -> nn.Module:
        layers = []
        for m in module_list:
            if isinstance(m, int):
                layers.append(nn.LazyLinear(m))
            else:
                layers.append(ModuleConstructor.construct(m))
        return nn.Sequential(*layers)

    @staticmethod
    def assert_no_kwargs(module_kwargs):
        if module_kwargs:
            raise ValueError(
                "Additional module_kwargs are not allowed when using a dictionary or list "
                "to construct a network."
            )

    @staticmethod
    def construct(module: Optional[Union[str, Dict, Type[nn.Module], nn.Module, List[Union[int, Dict]]]],
                  module_kwargs: Any = None) -> Optional[nn.Module]:

        module_kwargs = module_kwargs if module_kwargs is not None else {}

        if module is None:
            return None
        elif isinstance(module, str):
            return ModuleConstructor.from_string(module, module_kwargs)
        elif isinstance(module, type) and issubclass(module, nn.Module):
            return ModuleConstructor.from_class(module, module_kwargs)
        elif isinstance(module, nn.Module):
            return ModuleConstructor.from_module(module, module_kwargs)
        elif isinstance(module, dict):
            ModuleConstructor.assert_no_kwargs(module_kwargs)
            return ModuleConstructor.from_dict(module)
        elif isinstance(module, list):
            ModuleConstructor.assert_no_kwargs(module_kwargs)
            return ModuleConstructor.from_list(module)
        else:
            raise ValueError("Invalid type for module")
