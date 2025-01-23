import torch
import torch.nn as nn
import torch.nn.functional as F

class oracle_ssm(nn.Module):
    def __init__(self, input_shape, args, traj_dim=None):
        super(oracle_ssm, self).__init__()
        self.agent_latent_dim = args.agent_latent_dim #default 16
        self.obs_dim = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_latent_dim = self.agent_latent_dim * self.n_agents
        self.action_embedding_dim = args.action_embedding_dim # default 4
        self.hidden_dim = args.hidden_dim # default 128
        self.state_dim = args.state_shape if traj_dim==None else traj_dim

        # * prior
        self.prior_state_encoder = nn.Sequential(
            nn.Linear(self.state_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim),
        )

        self.prior_action_encoder = nn.Sequential(
            nn.Linear(self.n_actions, self.action_embedding_dim)
        )

        self.prior_cat_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_embedding_dim * self.n_agents, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.prior_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.state_latent_dim)
        )

        self.prior_logvar = nn.Sequential(
            nn.Linear(self.hidden_dim, self.state_latent_dim)
        )

        # * posterior
        self.posterior_encoder = nn.Linear(self.obs_dim, self.hidden_dim)
        self.posterior_logvar = nn.Linear(self.hidden_dim, self.agent_latent_dim)
        self.posterior_mu = nn.Linear(self.hidden_dim, self.agent_latent_dim)

        self.shared_decoder = nn.Sequential(
            nn.Linear(self.agent_latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.obs_dim)
        )

        # * supervise
        self.shared_ava_decoder = nn.Sequential(
            nn.Linear(self.agent_latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions)
        )

        self.rew_decoder = nn.Sequential(
            nn.Linear(self.state_latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # * trajectories encoder
        self.traj_encoder = nn.GRUCell(self.state_latent_dim,
                                   self.state_dim,)


    def prior_encode(self, s, a):
        state_encoder = self.prior_state_encoder(s)
        action_encoder = self.prior_action_encoder(a)
        encoder = torch.cat([state_encoder, action_encoder.view(action_encoder.size(0), action_encoder.size(1), -1)], dim=-1)
        encoder = self.prior_cat_encoder(encoder)
        mu, logvar = self.prior_mu(encoder), self.prior_logvar(encoder)
        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)

    def prior_forward(self, s, a):
        mu, logvar = self.prior_encode(s, a)
        z = self.sample_z(mu, logvar)
        return mu, logvar, z

    def posterior_encode(self, obs):
        bs = obs.size(0)
        sl = obs.size(1)
        encoder = F.relu(self.posterior_encoder(obs))
        mu = self.posterior_mu(encoder).view(bs, sl, self.state_latent_dim)
        logvar = self.posterior_logvar(encoder).view(bs, sl, self.state_latent_dim)
        return mu, logvar

    def decode(self, z):
        z = z.view(z.size(0), z.size(1), self.n_agents, self.agent_latent_dim)
        recon_obs = F.relu(self.shared_decoder(z))
        return recon_obs

    def posterior_forward(self, obs):
        mu, logvar = self.posterior_encode(obs)
        z = self.sample_z(mu, logvar)
        recon_obs = self.decode(z)
        return recon_obs, mu, logvar, z

    def get_rew(self, mu):
        return self.rew_decoder(mu)

    def get_avail_action(self, mu):
        return self.shared_ava_decoder(mu)

    def get_traj_encode(self, traj):
        bs = traj.size(0)
        sl = traj.size(1)
        traj = traj.flatten(start_dim=0, end_dim=1)
        h = torch.zeros([bs*sl, self.state_dim]).to(traj.device)
        for i in reversed(range(traj.size(-2))):
            h = self.traj_encoder(traj[:, i], h)
        return h.reshape(bs, sl, -1)

