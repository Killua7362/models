import ivy


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = ivy.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = ivy.equal(mask.view(-1), 1)
    index = ivy.astype(ivy.arange(len(mask))[mask], ivy.int64)
    return heads, index


def prune_linear_layer(layer, index, dim=0):
    index = index.to(layer.weight.device)
    W = ivy.gather(layer.weight, index, axis=dim)
    W = ivy.array(W, copy=True).stop_gradient(preserve_type=False)
    if layer.bias is not None:
        if dim == 1:
            b = ivy.array(layer.bias, copy=True).stop_gradient(preserve_type=False)
        else:
            b = ivy.array(layer.bias[index], copy=True).stop_gradient(
                preserve_type=False
            )
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = ivy.Linear(
        new_size[1],
        new_size[0],
        with_bias=layer.bias is not None,
        device=layer.weight.device,
    )
    new_layer.weight = new_layer.weight.stop_gradient(preserve_type=False)
    new_layer.weight = ivy.copy_array(W, to_ivy_array=True)  ##TODO contiguous
    new_layer.weight = new_layer.weight.stop_gradient(preserve_type=True)
    if layer.bias is not None:
        new_layer.bias.stop_gradient(preserve_type=False)
        new_layer.bias = ivy.copy_array(b, to_ivy_array=True)  ##TODO contiguous
        new_layer.bias.stop_gradient(preserve_type=True)
    return new_layer
