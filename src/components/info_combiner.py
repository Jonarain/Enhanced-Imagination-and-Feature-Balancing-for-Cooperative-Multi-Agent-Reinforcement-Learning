import torch as th
import torch.nn as nn

class Dynamic_balance_Gate_v3(nn.Module):
    def __init__(self, args):
        super(Dynamic_balance_Gate_v3, self).__init__()
        self.args = args
        self.tgt_shape = args.state_shape
        self.src_shape = args.agent_latent_dim * args.n_agents
        self.s_w = None
        self.img_w = None

        self.gru = nn.GRU(
            input_size=self.src_shape,
            num_layers=1,
            hidden_size=args.state_shape,
            batch_first=True,)

        self.fcn1 = nn.Sequential(
            nn.Linear(args.state_shape, 32),
            nn.ReLU(),
        )
        self.fcn2 = nn.Sequential(
            nn.Linear(args.state_shape, 32),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, tgt, src):
        bs, sl = tgt.size()[:2]
        tgt = tgt.flatten(0, 1)
        src = src.flatten(0, 1)
        h_0 = th.zeros([1, bs * sl, self.args.state_shape], device=self.args.device)

        _, last_h = self.gru(src.flip(-2), h_0)
        last_h = last_h.squeeze()

        s_img_embed = th.cat([self.fcn1(tgt), self.fcn2(last_h.detach())], dim=-1)
        weight = self.gate(s_img_embed)
        self.s_w = weight[:, :1]
        self.img_w = weight[:, 1:]
        res = self.s_w*tgt + self.img_w*last_h

        return res.reshape(bs, sl, -1)



