from collections import defaultdict
from functools import reduce, wraps
from itertools import chain, product
from math import *
from operator import matmul
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

__version__ = "1.0.0"

__all__ = [
    "linsum",
    "extend_by_linearity",
    "load_dofs_and_basis",
    "dump_ops",
    "hc",
    "apply",
    "compose",
    "inner",
    "matrix_elems",
]


def linsum(*args):
    return list(chain.from_iterable(args))


def extend_by_linearity(operation):
    @wraps(operation)
    def wrapper(*args):
        return linsum(*(operation(*arg) for arg in product(*args)))

    return wrapper


def kron(basis, v=None):
    if v is None:
        v = basis[0]
    return (np.array(basis) == v).astype(int)


# region parse input & write output


def build_op(op_inds, basis):
    ii, jj, vv = zip(*op_inds)
    iijj = [[basis.index(k) for k in kk] for kk in (ii, jj)]
    mat = sp.coo_matrix((vv, iijj), shape=(len(basis),) * 2, dtype=int)
    return mat.toarray()


def build_dofs(config):
    groups = defaultdict(list)
    dofs = {}
    ops = {}
    JW = {}
    for group, gr_data in config.items():
        gr_data = gr_data.copy()
        gr_basis = gr_data["basis"]
        gr_ops = {}
        for op, op_data in gr_data.pop("operators").items():
            gr_ops[op] = build_op(op_data, gr_basis)
        gr_JW = gr_ops.pop("JW", None)
        for dof in product(*gr_data.pop("indices").values()):
            dof = "".join(dof)
            groups[group].append(dof)
            dofs[dof] = gr_data
            for op, op_data in gr_ops.items():
                op_data = {dof: op_data}
                if gr_JW is not None:
                    op_data |= JW
                ops[f"{dof}{op}"] = [op_data]
            if gr_JW is not None:
                JW[dof] = gr_JW
    return groups, dofs, ops


def build_state(labels, coeffs, dofs, ops):
    @extend_by_linearity
    def _build_state(label):
        coeff_key, label = label.split(maxsplit=1)
        psi = [{"coeff": coeffs[coeff_key]}]
        for k, dof in dofs.items():
            psi[0][k] = kron(dof["basis"])
        for l in label.split():
            if ":" in l:
                k, v = l.split(":")
                psi[0][k] = kron(dofs[k]["basis"], v)
            else:
                psi = apply(hc(ops[l]), psi)
        return psi

    return _build_state(labels)


def lines_block(file):
    while line := file.readline().strip():
        yield line


def load_dofs_and_basis(in_dir="res"):
    with open(f"{in_dir}/dofs.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(f"{in_dir}/basis.txt") as f:
        coeffs = {k: eval(v) for k, v in (l.split(maxsplit=1) for l in lines_block(f))}
        labels = [[lbl.strip() for lbl in l.split(",")] for l in lines_block(f)]
    groups, dofs, ops = build_dofs(config)
    basis = [build_state(lbl, coeffs, dofs, ops) for lbl in labels]
    return config, groups, dofs, ops, labels, coeffs, basis


def dump_ops(mels, out_dir="out"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for k, op in mels.items():
        sp.save_npz(f"{out_dir}/{k}", op)


# endregion


# region operators & states manipulation


@extend_by_linearity
def hc(op):
    """Hermitian conjugate of operator `op`"""
    return [{k: np.conjugate(np.transpose(o)) for k, o in op.items()}]


@extend_by_linearity
def apply(op, ket):
    """Apply operator `op` to the state `ket`"""
    return [{k: np.dot(op[k], v) if k in op else v for k, v in ket.items()}]


@extend_by_linearity
def compose(*ops):
    """Compose operators"""
    ops = ops[::-1]  # first applied (rightmost) operator is first (leftmost) argument
    return [
        {k: reduce(matmul, (op[k] for op in ops if k in op)) for k in set(chain(*ops))}
    ]


@extend_by_linearity
def inner(bra, ket):
    """Dot product between states `bra` and `ket`"""
    return [prod(np.vdot(bra[k], ket[k]) for k in bra)]


def matrix_elems(op, basis):
    """Matrix elements of `op` on `basis`"""
    basis_image = (apply(op, ket) for ket in basis)
    mat = [[sum(inner(bra, ket)) for bra in basis] for ket in basis_image]
    mat = sp.csr_matrix(np.array(mat))
    mat.eliminate_zeros()
    mat.sort_indices()
    return mat.transpose()


# endregion
