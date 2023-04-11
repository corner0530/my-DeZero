import chainer
import cupy as np  # !! CUPY !!
import pytest

import dezero
from dezero.models import VGG16
from dezero.utils import array_allclose


def test_forward1():
    x = np.random.randn(1, 3, 224, 224).astype("f")
    _model = chainer.links.VGG16Layers(None)
    _model.to_gpu()

    with chainer.using_config("train", False):
        with chainer.using_config("enable_backprop", False):
            out_layer_name = "fc8"
            _y = _model.forward(x, [out_layer_name])[out_layer_name]

    model = VGG16()
    layers = _model.available_layers
    for l in layers:
        if "conv" in l or "fc" in l:
            m1 = getattr(model, l)
            m2 = getattr(_model, l)
            m1.W.data = m2.W.data
            m1.b.data = m2.b.data
            if "fc" in l:
                m1.W.data = m1.W.data.T
    model.to_gpu()

    with dezero.test_mode():
        y = model(x)

    assert array_allclose(y.data, _y.data)


def test_forward2():
    x = np.random.randn(1, 3, 224, 224).astype("f")
    model = VGG16()
    model.to_gpu()
    y = model(x)
    assert y.dtype == np.float32


def test_backward1():
    x = np.random.randn(2, 3, 224, 224).astype("f")
    _model = chainer.links.VGG16Layers(None)
    _model.to_gpu()

    with chainer.using_config("train", False):
        out_layer_name = "fc8"
        _y = _model.forward(x, [out_layer_name])[out_layer_name]
        _y.grad = np.ones_like(_y.data)
        _y.backward()

    model = VGG16()
    layers = _model.available_layers
    for l in layers:
        if "conv" in l or "fc" in l:
            m1 = getattr(model, l)
            m2 = getattr(_model, l)
            m1.W.data = m2.W.data
            m1.b.data = m2.b.data
            if "fc" in l:
                m1.W.data = m1.W.data.T
    model.to_gpu()

    with dezero.test_mode():
        y = model(x)
        y.backward()

    layers = _model.available_layers
    for l in layers:
        if "conv" in l:
            m1 = getattr(model, l)
            m2 = getattr(_model, l)
            assert array_allclose(m1.W.data, m2.W.data)
            assert array_allclose(m1.b.data, m2.b.data)
        elif "fc" in l:
            m1 = getattr(model, l)
            m2 = getattr(_model, l)
            assert array_allclose(m1.W.data, m2.W.data.T)
            assert array_allclose(m1.b.data, m2.b.data)


if __name__ == "__main__":
    pytest.main()
