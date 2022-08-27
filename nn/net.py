from losses import Loss
from layers import Function, Layer


class Net:
    __slots__ = ["layers", "loss_fn"]

    def __init__(self, layers, loss):
        assert isinstance(loss, Loss), "loss must be an instance of nn.losses.Loss"
        for layer in layers:
            assert isinstance(layer, Function), (
                "layer should be an instance of " "nn.layers.Function or nn.layers.Layer"
            )

        self.layers = layers
        self.loss_fn = loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, y):
        loss = self.loss_fn(x, y)
        return loss

    def backward(self):
        d = self.loss_fn.backward()
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def update_weights(self, lr):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer._update_weights(lr)