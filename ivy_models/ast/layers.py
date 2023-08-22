import ivy
from ivy_models.base import BaseModel, BaseSpec
import math
from utils import find_pruneable_heads_and_indices, prune_linear_layer


class ASTConfig(BaseSpec):
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        patch_size=16,
        qkv_bias=True,
        frequency_stride=10,
        time_stride=10,
        max_length=1024,
        num_mel_bins=128,
        device=None,
    ):
        device = ivy.default(device, ivy.default_device())
        super(ASTConfig, self).__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            frequency_stride=frequency_stride,
            time_stride=time_stride,
            max_length=max_length,
            num_mel_bins=num_mel_bins,
            device=device,
        )


class ASTEmbeddings(ivy.Module):
    def __init__(
        self,
        config: ASTConfig,
        v: ivy.Container = None,
    ):
        self.hidden_size = config.hidden_size
        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        self.num_patches = frequency_out_dimension * time_out_dimension
        self.config = config
        super(ASTEmbeddings, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.patch_embeddings = ASTPatchEmbeddings(self.config)
        self.dropout = ivy.Dropout(self.config.hidden_dropout_prob)

    def _create_variables(self, device, dtype=None):
        cls_token = ivy.zeros(shape=(1, 1, self.hidden_size))
        distillation_token = ivy.zeros(shape=(1, 1, self.hidden_size))
        position_embeddings = ivy.zeros(shape=(1, self.num_patches + 2, self.hidden_size))
        return {
            "cls_token": cls_token,
            "distillation_token": distillation_token,
            "position_embeddings": position_embeddings,
        }

    def get_shape(self, config):
        frequency_out_dimension = (
            config.num_mel_bins - config.patch_size
        ) // config.frequency_stride + 1
        time_out_dimension = (
            config.max_length - config.patch_size
        ) // config.time_stride + 1
        return frequency_out_dimension, time_out_dimension

    def _forward(self, input_values):
        batch_size = input_values.shape[0]
        embeddings = self.patch_embeddings(input_values)

        cls_tokens = ivy.expand(self.v.cls_token,[batch_size, -1, -1])
        distillation_tokens = ivy.expand(self.v.distillation_token,[batch_size, -1, -1])
        embeddings = ivy.concat((cls_tokens, distillation_tokens, embeddings), axis=1)
        embeddings = embeddings + self.v.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class ASTPatchEmbeddings(ivy.Module):
    def __init__(
        self,
        config,
        v: ivy.Container = None,
    ):
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.frequency_stride = config.frequency_stride
        self.time_stride = config.time_stride
        super(ASTPatchEmbeddings, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.projection = ivy.Conv2D(
            1,
            self.hidden_size,
            [self.patch_size, self.patch_size],
            [self.frequency_stride, self.frequency_stride],
            0,
            data_format="NCHW",
        )

    def _forward(self, input_values):
        input_values = ivy.expand_dims(input_values, axis=1)
        input_values = input_values.permute_dims(axes=(0,1,3,2))
        embeddings = self.projection(input_values).flatten(start_dim=2).permute_dims(axes=(0,2,1))
        return embeddings


class ASTSelfAttention(ivy.Module):
    def __init__(self, config: ASTConfig, v: ivy.Container = None):
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.hidden_size = config.hidden_size
        self.qkv_bias = config.qkv_bias
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        super(ASTSelfAttention, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.query = ivy.Linear(
            self.hidden_size, self.all_head_size, with_bias=self.qkv_bias
        )
        self.key = ivy.Linear(
            self.hidden_size, self.all_head_size, with_bias=self.qkv_bias
        )
        self.value = ivy.Linear(
            self.hidden_size, self.all_head_size, with_bias=self.qkv_bias
        )
        self.dropout = ivy.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute_dims(axes=(0, 2, 1, 3))

    def _forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = ivy.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = ivy.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = ivy.matmul(attention_probs, value_layer)
        context_layer = ivy.permute_dims(
            context_layer, (0, 2, 1, 3)
        )  ##todo contiguous array
        new_context_layer_shape = context_layer.shape()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class ASTSelfOutput(ivy.Module):
    def __init__(
        self,
        config: ASTConfig,
        v: ivy.Container = None,
    ):
        self.config = config
        super(ASTSelfOutput, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.dense = ivy.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = ivy.Dropout(self.config.hidden_dropout_prob)

    def _forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ASTAttention(ivy.Module):
    def __init__(
        self,
        config: ASTConfig,
        v: ivy.Container = None,
    ):
        self.config = config
        super(ASTAttention, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.attention = ASTAttention(self.config)
        self.output = ASTSelfOutput(self.config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attention.num_attention_heads,
            self.attention.attention_head_size,
            self.pruned_heads,
        )
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        self.attention.num_attention_heads = self.attention.num_attention_heads - len(
            heads
        )
        self.attention.all_head_size = (
            self.attention.attention_head_size * self.attention.num_attention_heads
        )
        self.pruned_heads = ivy.unique_values(ivy.concat(self.pruned_heads, heads))

    def _forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class ASTIntermediate(ivy.Module):
    def __init__(self,
                 config: ASTConfig,
                v: ivy.Container = None,
                ):
        self.config = config
        super(ASTIntermediate,self).__init__(v=v)
    
    def _build(self, *args, **kwargs):
        self.dense = ivy.Linear(self.config.hidden_size,self.config.intermediate_size)
        if isinstance(self.config.hidden_act,str):
            self.intermediate_act_fn = ivy.Gelu() ##todo adding multiple functions
        else:
            self.intermediate_act_fn = config.hidden_act
    
    def _forward(self,hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class ASTOutput(ivy.Module):
    def __init__(self,
                 config: ASTConfig,
                v: ivy.Container = None,
                ):
        self.config = config
        super(ASTOutput,self).__init__(v=v)
    
    def _build(self,*args,**kwargs):
        self.dense = ivy.Linear(self.config.hidden_states)
        self.dropout = ivy.Dropout(self.config.hidden_dropout_prob)
    
    def _forward(self,hidden_states,input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ASTLayer(ivy.Module):
    def __init__(self,
                 config: ASTConfig,
                v: ivy.Container = None,
                ):
        self.config = config
        self.seq_len_dim = 1
        super(ASTLayer,self).__init__(v=v)
    
    def _build(self,*args,**kwargs):
        self.attention = ASTAttention(self.config)
        self.intermediate = ASTIntermediate(self.config)
        self.output = ASTOutput(self.config)
        self.layernorm_before = ivy.LayerNorm(self.config.hidden_size,eps=self.config.layer_norm_eps)
        self.layernorm_after = ivy.LayerNorm(self.config.hidden_size,eps=self.config.layer_norm_eps)

    def _forward(self,hidden_states,head_mask=None,output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + outputs
        return outputs

class ASTEncoder(ivy.Module):
    def __init__(self,config: ASTConfig,
                v: ivy.Container = None,
):
        self.config = config
        self.gradient_checkpointing = False
        super(ASTEncoder,self).__init(v=v)
    
    def _build(self,*args,**kwargs):
        self.layer =  [ASTLayer(config) for _ in range(config.num_hidden_layers)] ##TODO torch modulelist
    
    def _forward(self,hidden_states,head_mask=None,output_attentions=False,output_hidden_states=False,return_dict=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i,layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            ##TODO adding self training and gradient checkpoint
            layer_outputs = layer_module(hidden_states,layer_head_mask,output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_self_attentions = all_self_attentions + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states,all_hidden_states,all_self_attentions] if v is not None)


config = ASTConfig()
testing = ASTSelfAttention(config)
import numpy as np

loaded_myarray = np.loadtxt("/workspaces/models/ivy_models/ast/myarray.txt")
backtomyarray = loaded_myarray.reshape(1, 1024, 128)
print(testing(ivy.array(backtomyarray)))
