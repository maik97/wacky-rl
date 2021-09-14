from tensorflow import layers

class ContinActionLayer(layers.Layer):

    def __init__(
            self,
            layers,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.layers = layers
        self.is_functional = False

    def __call__(self, *args, **kwargs):

        try:
            return super().__call__(*args, **kwargs)
        except:
            self.last_layer = args[0]
            self.out_list = [l(*args, **kwargs) for l in self.layers]
            self.is_functional = True
            return super().__call__(self.out_list)

    def call(self, inputs, **kwargs):

        if not self.is_functional:
            out_list = [l(inputs) for l in self.layers]
        else:
            out_list = super().call(inputs)

        return out_list