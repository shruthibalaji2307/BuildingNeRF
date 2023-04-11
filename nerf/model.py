import torch
import torch.nn.functional as F

from ray_utils import RayBundle

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.mlp = MLPWithInputSkips(
            n_layers=cfg.n_layers_xyz,
            input_dim=embedding_dim_xyz,
            output_dim=3,
            skip_dim=embedding_dim_xyz,
            hidden_dim=cfg.n_hidden_neurons_xyz,
            input_skips=cfg.append_xyz,
        )

        self.mlp.apply(init_weights)

        self.rgb_layer = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 3)
        self.rgb_layer.apply(init_weights)
        self.density_layer = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1)
        self.density_layer.apply(init_weights)

    def forward(self, ray_bundle):
        xyz = ray_bundle.sample_points
        dir = ray_bundle.directions

        xyz_embed = self.harmonic_embedding_xyz(xyz)
        dir_embed = self.harmonic_embedding_dir(dir)

        embedding = self.mlp(xyz_embed, xyz_embed)
        rgb = torch.nn.functional.sigmoid(self.rgb_layer(embedding))
        density = torch.nn.functional.relu(self.density_layer(embedding))

        out = {
            'density': density,
            'feature': rgb
        }

        return out

# TODO (4.1): Implement NeRF MLP with View Dependence
class NeuralRadianceFieldViewDependent(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.mlp = MLPWithInputSkips(
            n_layers=cfg.n_layers_xyz,
            input_dim=embedding_dim_xyz,
            output_dim=3,
            skip_dim=embedding_dim_xyz,
            hidden_dim=cfg.n_hidden_neurons_xyz,
            input_skips=cfg.append_xyz,
        )

        # self.mlp.apply(init_weights)
        self.rgb_layer = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz + embedding_dim_dir, 3),
        )
        # self.rgb_layer.apply(init_weights)
        self.density_layer = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1)
        # self.density_layer.apply(init_weights)

    def forward(self, ray_bundle):
        xyz = ray_bundle.sample_points
        dir = ray_bundle.directions

        xyz_embed = self.harmonic_embedding_xyz(xyz)
        dir_embed = self.harmonic_embedding_dir(dir)

        embedding = self.mlp(xyz_embed, xyz_embed)
        dir_embed = dir_embed.unsqueeze(1).repeat(1, 128, 1)
        view_embedding = torch.cat([embedding, dir_embed], -1)
        rgb = torch.sigmoid(self.rgb_layer(view_embedding))
        density = torch.nn.functional.relu(self.density_layer(embedding))

        out = {
            'density': density.view(-1, 128, 1),
            'feature': rgb.view(-1, 128, 3)
        }

        return out

volume_dict = {
    'nerf': NeuralRadianceField,
    'nerf_view': NeuralRadianceFieldViewDependent,
}