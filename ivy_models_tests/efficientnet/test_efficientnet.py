import os
import ivy
import pytest
import numpy as np
from ivy_models_tests import helpers
from ivy_models.efficientnet import efficientnet_b0


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_efficientnet_b0_img_classification(device, f, fw, batch_shape, load_weights):
    """Test EfficientNet B0 image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "dog.jpg"), 256, 224
    )

    # Create model
    model = efficientnet_b0(pretrained=load_weights)
    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        # todo: getting the max for now as the sequence
        # often reversed due to close logits (even in native)
        true_indices = np.array([258])
        calc_indices = np.argsort(np_out)[-1:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([9.095613])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-1)