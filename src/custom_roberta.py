import math

import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, RobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
    RobertaSelfOutput,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

# todo: Teile die Attention in Attention Map und Attention Module auf:


class CustomRobertaAttention(nn.Module):
    def __init__(self, model, config: RobertaConfig, index: int):
        super().__init__()
        self.self1 = AttentionMap(model=model, config=config, index=index)
        self.self2 = Attention(model=model, config=config, index=index)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        attention_probs, value_layer = self.self1(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        self_outputs2 = self.self2(attention_probs, value_layer)
        attention_output = self.output(self_outputs2[0], hidden_states)
        outputs = (attention_output,) + self_outputs2[
            1:
        ]  # add attentions if we output them
        return outputs


class AttentionMap(nn.Module):
    def __init__(self, model, config: RobertaConfig, index: int):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = model.roberta.encoder.layer[index].attention.self.query
        self.key = model.roberta.encoder.layer[index].attention.self.key
        self.value = model.roberta.encoder.layer[index].attention.self.value

        self.dropout = model.roberta.encoder.layer[index].attention.self.dropout
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        return attention_probs, value_layer


class Attention(nn.Module):
    def __init__(self, model, config: RobertaConfig, index: int):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = model.roberta.encoder.layer[index].attention.self.query
        self.key = model.roberta.encoder.layer[index].attention.self.key
        self.value = model.roberta.encoder.layer[index].attention.self.value

        self.dropout = model.roberta.encoder.layer[index].attention.self.dropout
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

    def forward(
        self,
        attention_probs,
        value_layer,
        head_mask=None,
        output_attentions=False,
    ):
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


# Todo: Custom roberta Layer in den Encoder einbauen:
# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class CustomRobertaEncoder(nn.Module):
    def __init__(self, model, config: RobertaConfig):

        ### hier müssen alles Custom Roberta Encoder sein

        super().__init__()
        self.config = config

        # todo hier die layer verändern
        self.layer = nn.ModuleList(
            [CustomRobertaLayer(model=model, config=config, index=i) for i in range(12)]
        )

    def forward(
        self,
        attention_probs,
        value_layer,
        layer_index=0,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        i = layer_index
        while i < 12:
            layer_module = self.layer[i]
            # for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                if i == layer_index:
                    layer_outputs = layer_module.first_index_forward(
                        attention_probs,
                        value_layer,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer_module.forward(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        output_attentions,
                    )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            i += 1
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def output_attention(
        self,
        hidden_states,
        layer_index,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        for i in range(layer_index + 1):
            layer_module = self.layer[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module.forward(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        return self.layer[layer_index].output_attention_prob(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )

    def get_attention_probabilities(self, hidden_states):
        return self.layer[0].attention.self1.first_indx_forward(
            hidden_states,
        )[0]


# Todo: Implement Custom Roberta Layer:
# Todo: Implemenet a attention_prob Flag in CustomRobertaEncoder


class CustomRobertaLayer(nn.Module):
    def __init__(self, model, config: RobertaConfig, index: int):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attentionfirst = Attention(model=model, config=config, index=index)
        self.attention = CustomRobertaAttention(model=model, config=config, index=index)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = RobertaAttention(config)
        self.intermediate = model.roberta.encoder.layer[index].intermediate
        self.output = model.roberta.encoder.layer[index].output

    def first_index_forward(
        self,
        attention_probs,
        value_layer,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attentionfirst.forward(
            attention_probs, value_layer, head_mask=None, output_attentions=False
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(self, "crossattention"), (
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                f"cross-attention layers by setting `config.add_cross_attention=True`"
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # todo hier ändern!!!!!
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def output_attention_prob(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        attention_probs, value_layer = self.attention.self1(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        return attention_probs, value_layer

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Todo: Attention Map --> Richtige Gewichte!!
class CustomQuestionAnswering:
    def __init__(self, model_name: str, config: RobertaConfig):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, config=config
        )
        self.custom_encoder = CustomRobertaEncoder(model=self.model, config=config)
        self.qa_outputs = self.model.qa_outputs
        # in attention map noch die richtigen Gewichte verwendne!!!
        self.attention_map = AttentionMap(model=self.model, config=config, index=0)

    def custom_forward(self, attention_probs, value_layer, index):
        last_hidden = self.custom_encoder.forward(attention_probs, value_layer, index)[
            "last_hidden_state"
        ]
        return self.qa_outputs(last_hidden)

    def forward(self, inputs):
        return self.model.forward(inputs)

    def get_embeddings(self, input):
        # inputs is of size [1, _num_tokens]
        return self.model.roberta.embeddings(input)
        # return self.attention_map(embeddings)


if __name__ == "__main__":
    model_name = "deepset/roberta-base-squad2"
    # context = "The option to convert models between FARM and transformers
    # gives freedom to the user and let people easily switch between frameworks."
    config = RobertaConfig.from_pretrained(model_name)

    layer_in = torch.ones((1, 35, 768), requires_grad=True)
    # att.forward(layer_in)

    attention_probs = torch.ones(1, 12, 35, 35, requires_grad=True)
    value_layer = torch.ones(1, 12, 35, 64)

    # layer = CustomRobertaLayer(model_name, config)
    # print(layer.forward(layer_in))
    import time

    t0 = time.time()
    model = CustomQuestionAnswering(model_name, config)
    for i in range(12):
        print(model.custom_encoder.output_attention(layer_in, i))

    model.custom_encoder.output_attention()
    print("doooone")
    t1 = time.time()
    print(t0 - t1)
    out = model.custom_forward(attention_probs, value_layer)

    print(a)
    print(out)
    out[0][1][0].backward()
    print(attention_probs.grad)
    print("##########################")
    t2 = time.time()
    print(t2 - t1)
    print(encoder.output_attention(attention_probs, value_layer, 4))
    print(time.time() - t2)
