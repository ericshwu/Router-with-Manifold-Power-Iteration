import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from torch.distributed.tensor import DTensor

from collections import defaultdict

from typing import cast, Tuple, Optional, List

from torch.optim.optimizer import Optimizer, _get_scalar_dtype

from torch import Tensor
from torchtitan.config_manager import JobConfig
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.components.optimizer import OptimizersContainer

from .muon_utils import to_local, momentum_update_pre_orthogonalize, muon_update


class MuonML(Optimizer):
    def __init__(
        self,
        params, 
        distributed_mesh: DeviceMesh,
        lr: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,  # default titan 
        eps: float = 1e-8,
        fused: bool = False,
        foreach: bool = False,
        ep_per_rank: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            step=0,
            eps=eps,
            fused=fused,
            foreach=foreach,
            ep_per_rank=ep_per_rank,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )
        super().__init__(params, defaults)

        self.beta = betas[0]
        
        self.ep_per_rank = ep_per_rank
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        
        self._distributed_mesh = distributed_mesh

        """TODO: Only supports with simple FSDP"""
        if isinstance(self._distributed_mesh, DeviceMesh):
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_adamw()
        self.step_muon()
        return loss
        
    @torch.no_grad()
    def step_adamw(self):
        assert self.param_groups[0]["algorithm"] == "adamw"
        group = self.param_groups[0]
        beta1, beta2 = cast(Tuple[float, float], group["betas"])

        # as we only use adamw 2 optimze rmsnorm
        # notice here, we don't use wd for norm weights
        adamw_update_kwargs = {
            "lr": torch.tensor(group["lr"]),
            "weight_decay": torch.tensor(group["weight_decay"]),
            "eps": torch.tensor(group["eps"]),
            "amsgrad": False,
            "maximize": False,
            "beta1": beta1,
            "beta2": beta2,
        }
        # split into decay / no decay params
        decay_params, no_decay_params = [], []

        for module_name, param in self.param_groups[0]["named_params"].items():
            if 'norm.weight' in module_name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # update decay params
        self.adamw_update(decay_params, adamw_update_kwargs=adamw_update_kwargs)
        # update no decay params like norm.weight
        adamw_update_kwargs["weight_decay"] = torch.tensor(0.0)
        self.adamw_update(no_decay_params, adamw_update_kwargs=adamw_update_kwargs)

    @torch.no_grad()
    def adamw_update(self, params, adamw_update_kwargs):
        if not params:
            return
        
        params_with_grad: List[Tensor] = []
        grads: List[Tensor] = []
        exp_avgs: List[Tensor] = []
        exp_avg_sqs: List[Tensor] = []
        max_exp_avg_sqs: List[Tensor] = []
        state_steps: List[Tensor] = []

        for p in params:
            assert p.grad is not None
            params_with_grad.append(p)
            grads.append(p.grad)            

            state = self.state[p]

            if "exp_avg" not in state:
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
        
        torch.optim._functional.adamw(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            **adamw_update_kwargs,
        )

    @torch.no_grad()
    def step_muon(self):
        assert self.param_groups[1]["algorithm"] == "muon"
        lr = torch.tensor(self.param_groups[1]["lr"])
        wd = torch.tensor(self.param_groups[1]["weight_decay"])

        for module_name, param in self.param_groups[1]["named_params"].items():
            assert isinstance(param, DTensor) and len(param.placements) == 1 and param.placements[0].dim == 0
            assert param.grad is not None
            if 'attention' in module_name:
                if "momentum" not in self.state[param]:
                    self.state[param]["momentum"] = torch.zeros_like(param)
                momentum = self.state[param]["momentum"]
                W_local, G_local, M_local = to_local(param), to_local(param.grad), to_local(momentum)
                # weight decay in place
                W_local.mul_(1 - lr * wd)         
                U_local = momentum_update_pre_orthogonalize(G_local, M_local, beta=self.beta)
                U_full = [torch.empty_like(U_local) for _ in range(self._world_size)]
                dist.all_gather(U_full, U_local, group=self._process_group)
                U_full = torch.cat(U_full, dim=0) 
                U_full = muon_update(U_full, lr, scaler="moonlight") 
                U_shards = U_full.chunk(self._world_size, dim=0)
                W_local.sub_(U_shards[self._device_rank])
                dist.barrier(group=self._process_group)
            elif 'mlp' in module_name:
                ep_per_rank = param.ep_per_rank
                if "momentum" not in self.state[param]:
                    self.state[param]["momentum"] = torch.zeros_like(param)
                momentum = self.state[param]["momentum"]
                W_local, G_local, M_local = to_local(param), to_local(param.grad), to_local(momentum)
                # weight decay in place
                W_local.mul_(1 - lr * wd) 
                U_local = momentum_update_pre_orthogonalize(G_local, M_local, beta=self.beta)
                U_local = muon_update(U_local.view(ep_per_rank, -1, U_local.shape[-1]), lr, scaler="moonlight")
                U_local = U_local.view(-1, U_local.shape[-1])
                W_local.sub_(U_local)
                dist.barrier(group=self._process_group)
            elif 'router' in module_name:   
                # use Muon update router
                if "momentum" not in self.state[param]:
                    self.state[param]["momentum"] = torch.zeros_like(param)
                momentum = self.state[param]["momentum"]
                W_local, G_local, M_local = to_local(param), to_local(param.grad), to_local(momentum)
                # weight decay in place
                W_local.mul_(1 - lr * wd) 
                U_local = momentum_update_pre_orthogonalize(G_local, M_local, beta=self.beta)
                U_full = [torch.empty_like(U_local) for _ in range(self._world_size)]
                dist.all_gather(U_full, U_local, group=self._process_group)
                U_full = torch.cat(U_full, dim=0)
                U_full = muon_update(U_full, lr, scaler="moonlight")
                U_full = U_full.view(-1, U_full.shape[-1])
                U_shards = U_full.chunk(self._world_size, dim=0)
                W_local.sub_(U_shards[self._device_rank])
                dist.barrier(group=self._process_group)
            else:
                raise NotImplementedError
 

class MuonMLOptimizers(OptimizersContainer):
    def __init__(self, model_parts, world_mesh, optimizer_kwargs,):
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        
        name = optimizer_kwargs.pop("name")
        ep_per_rank = optimizer_kwargs.pop("ep_per_rank")
        n_heads = optimizer_kwargs.pop("n_heads")
        n_kv_heads = optimizer_kwargs.pop("n_kv_heads")
        
        assert name in ["MuonML"]

        for model in self.model_parts:
            params = []
            muon_params, adamw_params = [], []
            muon_named_params, adamw_named_params = defaultdict(DTensor), defaultdict(DTensor)

            for n, p in model.named_parameters():
                assert p.requires_grad 
                if 'output.weight' in n or 'tok_embeddings.weight' in n or 'norm.weight' in n:  
                    # use AdamW for embed, unembed and 1D parameters
                    adamw_params.append(p)
                    adamw_named_params[n] = p
                elif 'mlp' in n:
                    # split expert weights
                    p.ep_per_rank = ep_per_rank
                    muon_params.append(p)
                    muon_named_params[n] = p
                else:
                    # split attention weights into heads
                    if 'attention.wq' in n or 'attention.wo' in n:
                        p.n_heads = n_heads
                    elif 'attention.wk' in n or 'attention.wv' in n:
                        p.n_heads = n_kv_heads
                    muon_params.append(p)
                    muon_named_params[n] = p

            params.append({"params": adamw_params, "named_params": adamw_named_params,  "algorithm": "adamw"})        
            params.append({"params": muon_params, "named_params": muon_named_params, "algorithm": "muon"})

            self.optimizers.append(MuonML(params, world_mesh, **optimizer_kwargs))
            all_params.extend(params)

        # temporarily commment this line
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

        if isinstance(world_mesh, DeviceMesh):
            self._device_rank = world_mesh.get_local_rank()
            self._world_size = world_mesh.size()
            self._process_group = world_mesh.get_group()
        else:
            raise NotImplementedError


def build_muonml_optimizers(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    world_mesh: DeviceMesh,
) -> OptimizersContainer:
    """We don't consider ft_mangaer (FaultTolerance) and optim_in_bwd"""
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    eps = job_config.optimizer.eps
    ep_per_rank = job_config.muon.ep_per_rank 
    n_heads = job_config.muon.n_heads
    n_kv_heads = job_config.muon.n_kv_heads

    assert name in ["MuonML"]
    assert ep_per_rank != -1 and n_heads != -1 and n_kv_heads != -1

    # fused as default
    optim_implementation = job_config.optimizer.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_kwargs = {
        "name": name,
        "lr": lr,
        "eps": eps,
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": foreach,
        "ep_per_rank": ep_per_rank,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
    }

    return MuonMLOptimizers(model_parts, world_mesh, optimizer_kwargs)





