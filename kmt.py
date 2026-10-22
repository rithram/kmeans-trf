import math

import torch
import torch.nn.functional as F
from entmax import sparsemax
from torch import nn

ATTDICT = {
    "softmax": F.softmax,
    "sparsemax": sparsemax,
}


def scaled_dot_prod(
    queries: torch.FloatTensor,
    keys: torch.FloatTensor,
    values: torch.FloatTensor,
    inv_temp: float = 1.0,
    dropout_p: float = 0,
    act: str = "softmax",
):
    assert act in ATTDICT.keys()
    attn = ATTDICT[act](
        -inv_temp * torch.square(torch.cdist(queries, keys)) / math.sqrt(keys.size(-1)), dim=-1
    )
    if dropout_p > 0:
        # FIXME: F.dropout defaults to training=True, so this dropout is still active under
        # model.eval() (the functional form ignores the module's mode). It needs to be
        # deactivated during model.eval(), e.g., by passing the module's `self.training` here.
        attn = F.dropout(attn, p=dropout_p)
    attn_out = attn @ values
    return attn_out


class KMeansTransformer(nn.Module):
    def __init__(
        self,
        d_emb: int,
        d_qkv: int,
        inv_temp: float = 1.0,
        dropout_p: float = 0.0,
        act: str = "softmax",
    ):
        super().__init__()
        self.dropout_p = dropout_p
        self.d_emb = d_emb
        self.d_qkv = d_qkv
        self.inv_temp = inv_temp
        self.act = act

        self.W_q = nn.Linear(self.d_emb, self.d_qkv)
        self.W_k = nn.Linear(self.d_emb, self.d_qkv)
        self.W_v = nn.Linear(self.d_emb, self.d_qkv)

    def forward(
        self,
        queries_in: torch.FloatTensor,
        keys_values_in: torch.FloatTensor,
    ):
        d_batch, n_queries, d_emb = queries_in.shape
        assert d_emb == self.d_emb and keys_values_in.shape[2] == self.d_emb
        assert d_batch == keys_values_in.shape[0]

        Q = self.W_q(queries_in)
        K = self.W_k(keys_values_in)
        V = self.W_v(keys_values_in)

        attn = scaled_dot_prod(Q, K, V, self.inv_temp, self.dropout_p, self.act)
        # FIXME: this assertion only holds when d_qkv == d_emb. For d_qkv != d_emb (e.g.,
        # --dqkv 2 in train.py, which sets d_qkv = dqkv * d_emb) attn has width d_qkv, so the
        # assertion fails, and the residual connections in KModel.forward would also need an
        # output projection from d_qkv back to d_emb. Only --dqkv 1 currently works.
        assert attn.shape == queries_in.shape
        return attn


class KModel(nn.Module):
    def __init__(
        self,
        d_emb: int,
        d_qkv: int,
        inv_temp: float = 1.0,
        dropout_p: float = 0.0,
        act: str = "softmax",
    ):
        super().__init__()
        trfs = [
            KMeansTransformer(d_emb, d_qkv, inv_temp=inv_temp, dropout_p=dropout_p, act=act)
            for _ in range(4)
        ]
        self.CAX = trfs[0]
        self.SAX = trfs[1]
        self.CAC = trfs[2]
        self.SAC = trfs[3]

    def forward(
        self,
        XX: torch.FloatTensor,
        CC: torch.FloatTensor,
    ):
        XXX = XX + (self.CAX(XX, CC) + self.SAX(XX, XX))
        CCC = CC + (self.CAC(CC, XXX) + self.SAC(CC, CC))
        return XXX, CCC


if __name__ == "__main__":
    print("test ....")
    demb = 5
    CA = KMeansTransformer(demb, demb, inv_temp=5)
    SA = KMeansTransformer(demb, demb)
    nq, nk, nb = 25, 3, 2
    XX = torch.rand((nb, nq, demb))
    CC = torch.rand((nb, nk, demb))
    print(XX.shape, CC.shape)
    up1 = CA(XX, CC)
    up2 = SA(CC, CC)
    print(up1.shape, up2.shape)
    CA = KMeansTransformer(demb, demb, inv_temp=5, act="sparsemax")
    up3 = CA(XX, CC)
    print(up3.shape)
    model = KModel(demb, demb, inv_temp=5, act="sparsemax")
    XX1, CC1 = model(XX.clone().detach(), CC.clone().detach())
    print(XX.shape, XX1.shape)
    print(CC.shape, CC1.shape)
