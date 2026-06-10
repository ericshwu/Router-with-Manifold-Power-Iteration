import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import List
from functools import partial

import stk

import megablocks.ops as ops

from megablocks import Arguments
from megablocks.layers import sharedexpert_registry
from megablocks.layers.moe import MoE, ParallelMLP
from megablocks.layers.dmoe import common, dMoE, ParallelDroplessMLP
from megablocks.layers.router import LearnedRouter


from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.experiments.llama4 import TransformerModelArgs, AttentionKVCache
from torchtitan.experiments.llama4 import Attention, FeedForward, precompute_freqs_cis


class Router(LearnedRouter):
    def __init__(self, args: Arguments):
        super().__init__(args)
        self.register_buffer("expert_bias", torch.zeros(args.moe_num_experts, dtype=torch.float32), persistent=True)

    def forward(self, x: torch.Tensor):
        logits = self.layer(x.view(-1, x.shape[-1]))

        logits -= logits.max(-1, keepdim=True)[0]
        logits = torch.exp(logits.to(torch.float32))
        
        _, expert_indices = self._top_k(logits + self.expert_bias)
        expert_weights = torch.gather(logits, -1, expert_indices)
                
        return logits, expert_weights, expert_indices


class ParallelMoEDroplessMLP(ParallelDroplessMLP):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.aux_loss_coef = args.aux_loss_coef
    
    def sparse_forward_once(self, x, expert_weights, top_experts):
        # x: [sl, bs, hs]
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, padded_bins, tokens_per_expert = (self.indices_and_padded_bins(top_experts))

        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])

        x = ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            padded_bins,
            self.top_k,
        )

        # Create the sparse matrix topology.
        with torch.no_grad():
            topo = self.topology(x, padded_bins)

        # Perform the expert computation.
        x = self.mlp(x, topo)

        # Un-route the data for the MoE output.
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            self.top_k,
        )
        return x, tokens_per_expert

    
    def forward(self, x, scores, expert_weights, top_experts):
        in_shape = x.shape
        slen, bs, _ = in_shape

        # Compute the experts.
        out, tokens_per_expert = self.sparse_forward_once(x, expert_weights, top_experts)

        bl_loss = self.load_balancing_loss_func(top_experts, scores, bs, slen)
        out = out.view(in_shape)

        return out, bl_loss, tokens_per_expert


    def load_balancing_loss_func(self, top_experts, scores, bs, slen):
        num_experts = self.args.moe_num_experts

        one_hot_topk = F.one_hot(top_experts, num_classes=num_experts).sum(dim=1).float()
        one_hot_topk = one_hot_topk.view(slen, bs, num_experts)
        fi = one_hot_topk.sum(dim=0) * num_experts / (self.top_k * slen)  # (bs, num_experts)

        scores = (scores / scores.sum(dim=-1, keepdim=True))
        pi = scores.view(slen, bs, self.args.moe_num_experts).sum(dim=0) / slen

        return self.aux_loss_coef * (fi * pi).sum(dim=-1).mean()


class MoEMbs(MoE):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.router = Router(args)
        self.top_k = args.moe_top_k
        self.load_balance_coeff = None  # empty parameters for parallel_llama        
        assert self.shared_expert is None

    def _init_experts_mlp(self, args: Arguments):
        return ParallelMoEDroplessMLP(args)
    
    def forward(self, x: torch.Tensor):
        """x: (bs, slen, d)"""
        x = common.cast_if_autocast_enabled(x)

        x = x.transpose(0, 1).contiguous()
        in_shape = x.shape

        scores, expert_weights, top_experts = self.router(x)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(x.dtype)
        
        out, bl_loss, tokens_per_expert = self.experts(x, scores, expert_weights, top_experts)
        out = out.view(in_shape).transpose(0, 1).contiguous()

        return out, bl_loss


class MoEBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.moe_enabled = True
        self.n_heads = model_args.n_heads

        attn_use_rope = True
        fixed_attn_block_size = None
        # init slightly different here, attention is not initialzied in our implementation
        self.attention = Attention(model_args, attn_use_rope, fixed_attn_block_size)
        
        # TODO: Notice that MegaBlocks requires ffn_hidden_size divides 128
        args = Arguments(
            # Model arguments.
            hidden_size=model_args.dim,
            ffn_hidden_size=model_args.ffn_hidden_size,
            bias=False,
            return_bias=False,
            activation_fn=F.silu,   # swiglu
            # MoE arguments.
            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            moe_num_experts=model_args.num_experts,
            moe_top_k=model_args.top_k,
            moe_capacity_factor=0,
            # Parallelism arguments.
            # by default
            # Compute arguments.
            mlp_type='glu',
            mlp_impl='sparse',
            # Initialization arguments. ~ FSDP
            fp16=False,
            bf16=False,
            init_method=partial(nn.init.trunc_normal_, mean=0., std=0.02, a=-0.06, b=0.06),
            shared_expert=False,  # enable using shared expert
        )
        setattr(args, 'aux_loss_coef', model_args.aux_loss_coef)
        self.moe = MoEMbs(args)        
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache,
    ):  
        h = x + self.attention(self.attention_norm(x), freqs_cis, kv_cache)
        out, bl_loss = self.moe(self.ffn_norm(h))
        out = h + out
        return out, bl_loss
            
    
class Transformer(nn.Module, ModelProtocol):
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = MoEBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.freqs_cis.device
        weight_init_std = self.model_args.weight_init_std
        cutoff = 3 * weight_init_std

        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        
        weight_init_fn = self.model_args.weight_init_fn
        assert weight_init_fn in ["adamw", "muonh", "adamwh", "moonlight"]

        if weight_init_fn == "adamw":
            for linear in (self.tok_embeddings, self.output):
                nn.init.trunc_normal_(linear.weight, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
        elif weight_init_fn == "muonh":
            # if embed / output use adamw
            for linear in (self.tok_embeddings, self.output):
                nn.init.trunc_normal_(linear.weight, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
        elif weight_init_fn == "adamwh":
            nn.init.trunc_normal_(self.tok_embeddings.weight, mean=0, std=weight_init_std, a=-cutoff, b=cutoff)
            nn.init.normal_(self.output.weight, mean=0., std=1. / (self.model_args.dim ** 0.5))
        elif weight_init_fn == "moonlight":
            for linear in (self.tok_embeddings, self.output):
                nn.init.trunc_normal_(linear.weight, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
        else:
            raise NotImplementedError

        for layer in self.layers.values():
            # initialize Norm weights
            layer.attention_norm.reset_parameters()
            layer.ffn_norm.reset_parameters()
            # EP Rt weights has been initialized with Megablocks when created
            # we reintialize EP weights here

            if weight_init_fn == "adamw":
                # init attention weights & expert weights as a whole
                _ = layer.attention
                for linear in (_.wq, _.wk, _.wv, _.wo):
                    nn.init.trunc_normal_(linear.weight, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
                _ = layer.moe.experts.mlp
                for linear in (_.w1, _.w2, _.v1):
                    nn.init.trunc_normal_(linear, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
                nn.init.trunc_normal_(layer.moe.router.layer.weight, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
            elif weight_init_fn == "muonh":
                _ = layer.attention
                for linear in (_.wq, _.wk, _.wv, _.wo):
                    nn.init.normal_(linear.weight, mean=0., std=1. / (self.model_args.dim ** 0.5))
                _ = layer.moe.experts.mlp
                for linear in (_.w1, _.w2, _.v1):
                    nn.init.normal_(linear, mean=0., std=1. / (self.model_args.dim ** 0.5))
                nn.init.normal_(layer.moe.router.layer.weight, mean=0., std=1. / (self.model_args.dim ** 0.5))
            elif weight_init_fn == "adamwh":
                _ = layer.attention
                for linear in (_.wq, _.wk, _.wv, _.wo):
                    nn.init.normal_(linear.weight, mean=0., std=1. / (self.model_args.dim ** 0.5))
                _ = layer.moe.experts.mlp
                for linear in (_.w1, _.w2, _.v1):
                    nn.init.normal_(linear, mean=0., std=1. / (self.model_args.dim ** 0.5))
                nn.init.normal_(layer.moe.router.layer.weight, mean=0., std=1. / (self.model_args.dim ** 0.5))
            elif weight_init_fn == "moonlight":
                # init attention weights & expert weights as a whole
                _ = layer.attention
                for linear in (_.wq, _.wk, _.wv, _.wo):
                    nn.init.trunc_normal_(linear.weight, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
                _ = layer.moe.experts.mlp
                for linear in (_.w1, _.w2, _.v1):
                    nn.init.trunc_normal_(linear, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
                nn.init.trunc_normal_(layer.moe.router.layer.weight, mean=0., std=weight_init_std, a=-cutoff, b=cutoff)
            else:
                raise NotImplementedError
           
        self.norm.reset_parameters()

                
    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(self.model_args.dim // self.model_args.n_heads, self.model_args.max_seq_len, self.model_args.rope_theta)

    def forward(self, tokens: torch.Tensor, input_batch: torch.Tensor | None = None, kv_caches: List[AttentionKVCache] | None = None):
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        all_bl_loss = 0
        for layer_idx, layer in enumerate(self.layers.values()):
            kv_cache = kv_caches[layer_idx] if kv_caches is not None else None
            h, bl_loss = layer(h, self.freqs_cis, kv_cache)
            all_bl_loss += bl_loss
        
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        
        return output, all_bl_loss / self.n_layers

    @classmethod
    def from_model_args(cls, model_args: TransformerModelArgs):
        return cls(model_args)




