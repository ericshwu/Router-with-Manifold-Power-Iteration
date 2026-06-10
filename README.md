# Router-with-Manifold-Power-Iteration

This repository contains the core TorchTitan implementation of our ongoing project: 

**"Redesign Mixture-of-Expert Routers with Manifold Power Iteration"**.

## Key Features
This repository primarily includes:
- **Customized Advanced Optimizers:** Our tailored implementation of advanced optimizers. *(Note: Current support is limited to FSDP and carries some limitations regarding the number of experts.)*
- **Manifold Power Iteration Routers:** The complete modeling and implementation of routers using Manifold Power Iteration.

For researchers interested in our work, you can easily integrate these modules into your own TorchTitan codebase to test the methods.

## Future Roadmap & Collaboration
We are actively working on adapting our method for industrial MoE pretraining, and we will share any future updates through this repository.

Since the official TorchTitan repository does not currently include these advanced optimizers, we warmly welcome community contributions. 
If you are a researcher interested in extending our codebase to expand optimizer compatibility for TorchTitan, please reach out. We are eager to share our code and provide support!
