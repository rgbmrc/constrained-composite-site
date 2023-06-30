# constrained-composite-site

Python module for building the site of quantum lattice models with local constraints.

Deals with composite computational Hilbert spaces constructed as a subspace of the tensor product of some "elementary" degrees of freedom (dofs). 
The model exposes functions that:
- represent states and operators of the composite computational basis as linear combinations of tensor products of states and operators of the "elementary" dofs
- apply composite operators to composite states and compute matrix elements
