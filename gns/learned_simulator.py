import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict
from gns.localise import FrameMLP, FrameTransformer, SpatioTemporalFrame
from gns.locs_utils import rotate, rotation_matrices_to_quaternions, cart_to_n_spherical
from gns.geometry import construct_3d_basis_from_2_vectors
import pdb


class LearnedSimulator_locs(nn.Module):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(
        self,
        particle_dimensions: int,
        nnode_in: int,
        nedge_in: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        connectivity_radius: float,
        boundaries: np.ndarray,
        normalization_stats: Dict,
        nparticle_types: int,
        particle_type_embedding_size: int,
        localizer_type: str,
        device="cpu",
    ):
        """Initializes the model.

        Args:
          particle_dimensions: Dimensionality of the problem.
          nnode_in: Number of node inputs.
          nedge_in: Number of edge inputs.
          latent_dim: Size of latent dimension (128)
          nmessage_passing_steps: Number of message passing steps.
          nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
          connectivity_radius: Scalar with the radius of connectivity.
          boundaries: Array of 2-tuples, containing the lower and upper boundaries
                of the cuboid containing the particles along each dimensions, matching
                the dimensionality of the problem.
          normalization_stats: Dictionary with statistics with keys "acceleration"
                and "velocity", containing a named tuple for each with mean and std
                fields, matching the dimensionality of the problem.
          nparticle_types: Number of different particle types.
          particle_type_embedding_size: Embedding size for the particle type.
          localizer_type: Type of localizer to use. Can be "mlp" or "spatio_temporal_transformer".
          device: Runtime device (cuda or cpu).

        """
        super(LearnedSimulator_locs, self).__init__()
        self._boundaries = boundaries
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._nparticle_types = nparticle_types

        # Particle type embedding has shape (9, 16)
        self._particle_type_embedding = nn.Embedding(
            nparticle_types, particle_type_embedding_size
        )
        self.localizer_type = localizer_type
        if localizer_type == "locs_velacc":
            self._localizer = get_matrix_from_vel_accel
        elif localizer_type == "locs_mlp":
            self._localizer = FrameMLP(
                embed_dim=128,
                hidden_dim=256,
                num_features=6,
                trajectory_size=5,
                dropout=0.0,
                n_layers=4,
                n_out_dims=6,
                particle_type_embedding_size=particle_type_embedding_size,
            )
        elif localizer_type == "locs_st_transformer":
            self._localizer = SpatioTemporalFrame(
                embed_dim=128,
                hidden_dim=256,
                num_features=6,
                trajectory_size=5,
                dropout=0.0,
                n_layers=4,
                n_out_dims=6,
                particle_type_embedding_size=particle_type_embedding_size,
                device=device,
            )
        elif localizer_type == "locs_temporal_transformer":
            self._localizer = FrameTransformer(
                embed_dim=128,
                hidden_dim=256,
                num_features=6,
                trajectory_size=5,
                dropout=0.0,
                n_layers=4,
                num_heads=8,
                n_out_dims=6,
                particle_type_embedding_size=particle_type_embedding_size,
            )
        # Initialize the EncodeProcessDecode
        self._encode_process_decode = graph_network.EncodeProcessDecode(
            nnode_in_features=nnode_in,
            nnode_out_features=particle_dimensions,
            nedge_in_features=nedge_in,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self._device = device

    def forward(self):
        """Forward hook runs on class instantiation"""
        pass

    def _compute_graph_connectivity(
        self,
        node_features: torch.tensor,
        nparticles_per_example: torch.tensor,
        radius: float,
        add_self_edges: bool = True,
    ):
        """Generate graph edges to all particles within a threshold radius

        Args:
          node_features: Node features with shape (nparticles, dim).
          nparticles_per_example: Number of particles per example. Default is 2
                examples per batch.
          radius: Threshold to construct edges to all particles within the radius.
          add_self_edges: Boolean flag to include self edge (default: True)
        """
        # Specify examples id for particles
        batch_ids = torch.cat(
            [
                torch.LongTensor([i for _ in range(n)])
                for i, n in enumerate(nparticles_per_example)
            ]
        ).to(self._device)

        # radius_graph accepts r < radius not r <= radius
        # A torch tensor list of source and target nodes with shape (2, nedges)
        edge_index = radius_graph(
            node_features, r=radius, batch=batch_ids, loop=add_self_edges
        )

        # The flow direction when using in combination with message passing is
        # "source_to_target"
        receivers = edge_index[0, :]
        senders = edge_index[1, :]

        return receivers, senders

    def _encoder_preprocessor(
        self,
        position_sequence: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
    ):
        """Extracts important features from the position sequence. Returns a tuple
        of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
        edge_features (nparticles, 3).

        Args:
          position_sequence: A sequence of particle positions. Shape is
                (nparticles, 6, dim). Includes current + last 5 positions
          nparticles_per_example: Number of particles per example. Default is 2
                examples per batch.
          particle_types: Particle types with shape (nparticles).
        """
        nparticles = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1]  # (n_nodes, 2) or (n_nodes, 3)
        velocity_sequence = time_diff(position_sequence)
        most_recent_velocity = velocity_sequence[:, -1]  # (n_nodes, 2) or (n_nodes, 3)
        # Get connectivity of the graph with shape of (nparticles, 2)
        senders, receivers = self._compute_graph_connectivity(
            most_recent_position, nparticles_per_example, self._connectivity_radius
        )
        node_features = []

        # Relative displacement and distances normalized to radius
        # with shape (nedges, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) -
        #     torch.gather(most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius
        normalized_relative_displacements_norms = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True
        )
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"]
        ) / velocity_stats["std"]
        flat_velocity_sequence = normalized_velocity_sequence.view(nparticles, -1)
        # There are 5 previous steps, with dim 2
        # node_features shape (nparticles, 5 * 2 = 10)
        # In the 3D case, node_features shape (nparticles, 5 * 3 = 15)
        node_features.append(flat_velocity_sequence)

        # Past positions wrt to the most recent position (nparticles, 5, dim)
        rel_position_sequence = (
            position_sequence[:, :-1] - most_recent_position[:, None]
        )
        node_features.append(rel_position_sequence.view(nparticles, -1))
        # Normalized clipped distances to lower and upper boundaries.
        # boundaries are an array of shape [num_dimensions, 2], where the second
        # axis, provides the lower/upper boundaries.
        # In the 2D case, boundaries shape (2, 2)
        # In the 3D case, boundaries shape (3, 2)
        boundaries = (
            torch.tensor(self._boundaries, requires_grad=False).float().to(self._device)
        )
        distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
        distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position
        distance_to_boundaries = torch.cat(
            [distance_to_lower_boundary, distance_to_upper_boundary], dim=1
        )
        normalized_clipped_distance_to_boundaries = torch.clamp(
            distance_to_boundaries / self._connectivity_radius, -1.0, 1.0
        )
        # The distance to 4 boundaries (top/bottom/left/right)
        # node_features shape (nparticles, 10+4)
        # In the 3D case, the distance to 6 boundaries (top/bottom/left/right/front/back)
        # node_features shape (nparticles, 15+6)
        node_features.append(normalized_clipped_distance_to_boundaries)

        # Node state is a concatenation of the velocity sequence, the positions (nparticles, 5, 2*dim)
        # and the distance to the boundaries (nparticles, 2*dim)
        node_state = torch.cat(node_features, dim=1)

        # Particle type
        if self._nparticle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)

        if self.localizer_type == "locs_st_transformer":
            edge_index = torch.stack([senders, receivers], dim=0).to(self._device)
            edge_features = normalized_relative_displacements_norms
            R = self._localizer(
                node_state, particle_type_embeddings, edge_index, edge_features
            )
        elif self.localizer_type == "locs_velacc":
            R = self._localizer(normalized_velocity_sequence)
        else:
            R = self._localizer(node_state, particle_type_embeddings)
        # nparticles, 3, 3
        Rinv = R.transpose(-1, -2)

        gforce = torch.zeros(nparticles, 3).to(self._device)
        gforce[:, 2] = -1.0
        # nparticles,(2T+3), 3,1
        node_state = torch.cat([node_state, gforce], dim=1).reshape(-1, 3).unsqueeze(-1)

        # Repeat the rotation matrix for each particle such that we can apply it to the node state
        Rinv = Rinv.unsqueeze(1).repeat(1, 13, 1, 1).reshape(-1, 3, 3)

        canonicalized_node_features = (
            torch.bmm(Rinv, node_state).squeeze(-1).reshape(nparticles, -1)
        )
        # rotate(
        #     node_state.reshape(nparticles, -1, 3), Rinv.unsqueeze(1)
        # ).reshape(nparticles, -1)
        final_node_features = [canonicalized_node_features, particle_type_embeddings]
        # Final node_features shape (nparticles, 30) for 2D
        # 30 = 10 (5 velocity sequences*dim) + 4 boundaries + 16 particle embedding
        # Canonicalized node_features shape (nparticles, 30) for 2D
        # 42 = 10 (5 velocity sequences*dim) + 10 (5 position sequences) + 4 boundaries + 16 particle embedding + 2 gforce
        # Final node_features shape (nparticles, 55) for 3D
        # 37 = 15 (5 velocity sequences*dim) + 6 boundaries + 16 particle embedding
        # Canonicalized node_features shape (nparticles, 55) for 3D
        # 55 = 15 (5 velocity sequences*dim) + 15 (5 position sequences*dim)+ 6 boundaries + 16 particle embedding + 3 gforce
        # Collect edge features.

        send_R = R[senders, :]
        recv_Rinv = Rinv[receivers, :]

        edge_features = []

        rotated_relative_displacements = torch.bmm(
            recv_Rinv, normalized_relative_displacements.unsqueeze(-1)
        ).squeeze(-1)

        # rotate(
        #     normalized_relative_displacements, recv_Rinv
        # )
        # Add relative displacement between two particles as an edge feature
        # with shape (nparticles, ndim)
        edge_features.append(rotated_relative_displacements)

        rotated_relative_velocities = torch.bmm(
            recv_Rinv, most_recent_velocity[senders, :].unsqueeze(-1)
        ).squeeze(-1)

        edge_features.append(rotated_relative_velocities)

        rotated_orientations = recv_Rinv @ send_R
        rotated_quaternions = rotation_matrices_to_quaternions(rotated_orientations)

        # Add relative distance between 2 particles with shape (nparticles, 1)
        # Edge features has a final shape of (nparticles, ndim + 1)
        # normalized_relative_distances = torch.norm(
        #     normalized_relative_displacements, dim=-1, keepdim=True
        # )
        edge_features.append(normalized_relative_displacements_norms)
        # Add relative orientation between two particles as an edge feature
        # with shape (nparticles, 4)
        edge_features.append(rotated_quaternions)
        # Edge features has a final shape of (nparticles, 2*ndim  + 1 + 4)

        return (
            torch.cat(final_node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1),
            R,
        )

    def _decoder_postprocessor(
        self, normalized_acceleration: torch.tensor, position_sequence: torch.tensor
    ) -> torch.tensor:
        """Compute new position based on acceleration and current position.
        The model produces the output in normalized space so we apply inverse
        normalization.

        Args:
          normalized_acceleration: Normalized acceleration (nparticles, dim).
          position_sequence: Position sequence of shape (nparticles, dim).

        Returns:
          torch.tensor: New position of the particles.

        """
        # Extract real acceleration values from normalized values
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats["std"]
        ) + acceleration_stats["mean"]

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        # TODO: Fix dt
        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1
        return new_position

    def predict_positions(
        self,
        current_positions: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
    ) -> torch.tensor:
        """Predict position based on acceleration.

        Args:
          current_positions: Current particle positions (nparticles, dim).
          nparticles_per_example: Number of particles per example. Default is 2
                examples per batch.
          particle_types: Particle types with shape (nparticles).

        Returns:
          next_positions (torch.tensor): Next position of particles.
        """
        node_features, edge_index, edge_features, R = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types
        )
        # Node features shape (nparticles, 30) for 2D and (nparticles, 37) for 3D
        # Edge index shape (2, nedges)
        # Edge features shape (nparticles, ndim + 1)
        # Add localizer module here to canonicalize the inputs

        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, edge_features
        )

        # Add globalizer module here to denormalize the outputs
        # Rotate the acceleration back to the original frame
        global_normalized_acceleration = torch.bmm(
            R, predicted_normalized_acceleration.unsqueeze(-1)
        ).squeeze(-1)

        next_positions = self._decoder_postprocessor(
            global_normalized_acceleration, current_positions
        )

        return next_positions

    def predict_accelerations(
        self,
        next_positions: torch.tensor,
        position_sequence_noise: torch.tensor,
        position_sequence: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
    ):
        """Produces normalized and predicted acceleration targets.

        Args:
          next_positions: Tensor of shape (nparticles_in_batch, dim) with the
                positions the model should output given the inputs.
          position_sequence_noise: Tensor of the same shape as `position_sequence`
                with the noise to apply to each particle.
          position_sequence: A sequence of particle positions. Shape is
                (nparticles, 6, dim). Includes current + last 5 positions.
          nparticles_per_example: Number of particles per example. Default is 2
                examples per batch.
          particle_types: Particle types with shape (nparticles).

        Returns:
          Tensors of shape (nparticles_in_batch, dim) with the predicted and target
                normalized accelerations.

        """

        # Add noise to the input position sequence.
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Perform the forward pass with the noisy position sequence.
        node_features, edge_index, edge_features, R = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types
        )
        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, edge_features
        )

        # Add globalizer module here to denormalize the outputs
        # Rotate the acceleration back to the original frame
        global_normalized_acceleration = torch.bmm(
            R, predicted_normalized_acceleration.unsqueeze(-1)
        ).squeeze(-1)
        # Calculate the target acceleration, using an `adjusted_next_position `that
        # is shifted by the noise in the last input position.
        next_position_adjusted = next_positions + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence
        )
        # As a result the inverted Euler update in the `_inverse_decoder` produces:
        # * A target acceleration that does not explicitly correct for the noise in
        #   the input positions, as the `next_position_adjusted` is different
        #   from the true `next_position`.
        # * A target acceleration that exactly corrects noise in the input velocity
        #   since the target next velocity calculated by the inverse Euler update
        #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
        #   matches the ground truth next velocity (noise cancels out).

        return global_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(
        self, next_position: torch.tensor, position_sequence: torch.tensor
    ):
        """Inverse of `_decoder_postprocessor`.

        Args:
          next_position: Tensor of shape (nparticles_in_batch, dim) with the
                positions the model should output given the inputs.
          position_sequence: A sequence of particle positions. Shape is
                (nparticles, 6, dim). Includes current + last 5 positions.

        Returns:
          normalized_acceleration (torch.tensor): Normalized acceleration.

        """
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats["mean"]
        ) / acceleration_stats["std"]
        return normalized_acceleration

    def save(self, path: str = "model.pt"):
        """Save model state

        Args:
          path: Model path
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load model state from file

        Args:
          path: Model path
        """
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))


class LearnedSimulator(nn.Module):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(
        self,
        particle_dimensions: int,
        nnode_in: int,
        nedge_in: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        connectivity_radius: float,
        boundaries: np.ndarray,
        normalization_stats: Dict,
        nparticle_types: int,
        particle_type_embedding_size,
        device="cpu",
    ):
        """Initializes the model.

        Args:
          particle_dimensions: Dimensionality of the problem.
          nnode_in: Number of node inputs.
          nedge_in: Number of edge inputs.
          latent_dim: Size of latent dimension (128)
          nmessage_passing_steps: Number of message passing steps.
          nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
          connectivity_radius: Scalar with the radius of connectivity.
          boundaries: Array of 2-tuples, containing the lower and upper boundaries
            of the cuboid containing the particles along each dimensions, matching
            the dimensionality of the problem.
          normalization_stats: Dictionary with statistics with keys "acceleration"
            and "velocity", containing a named tuple for each with mean and std
            fields, matching the dimensionality of the problem.
          nparticle_types: Number of different particle types.
          particle_type_embedding_size: Embedding size for the particle type.
          device: Runtime device (cuda or cpu).

        """
        super(LearnedSimulator, self).__init__()
        self._boundaries = boundaries
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._nparticle_types = nparticle_types

        # Particle type embedding has shape (9, 16)
        self._particle_type_embedding = nn.Embedding(
            nparticle_types, particle_type_embedding_size
        )

        # Initialize the EncodeProcessDecode
        self._encode_process_decode = graph_network.EncodeProcessDecode(
            nnode_in_features=nnode_in,
            nnode_out_features=particle_dimensions,
            nedge_in_features=nedge_in,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self._device = device

    def forward(self):
        """Forward hook runs on class instantiation"""
        pass

    def _compute_graph_connectivity(
        self,
        node_features: torch.tensor,
        nparticles_per_example: torch.tensor,
        radius: float,
        add_self_edges: bool = True,
    ):
        """Generate graph edges to all particles within a threshold radius

        Args:
          node_features: Node features with shape (nparticles, dim).
          nparticles_per_example: Number of particles per example. Default is 2
            examples per batch.
          radius: Threshold to construct edges to all particles within the radius.
          add_self_edges: Boolean flag to include self edge (default: True)
        """
        # Specify examples id for particles
        batch_ids = torch.cat(
            [
                torch.LongTensor([i for _ in range(n)])
                for i, n in enumerate(nparticles_per_example)
            ]
        ).to(self._device)

        # radius_graph accepts r < radius not r <= radius
        # A torch tensor list of source and target nodes with shape (2, nedges)
        edge_index = radius_graph(
            node_features, r=radius, batch=batch_ids, loop=add_self_edges
        )

        # The flow direction when using in combination with message passing is
        # "source_to_target"
        receivers = edge_index[0, :]
        senders = edge_index[1, :]

        return receivers, senders

    def _encoder_preprocessor(
        self,
        position_sequence: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
    ):
        """Extracts important features from the position sequence. Returns a tuple
        of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
        edge_features (nparticles, 3).

        Args:
          position_sequence: A sequence of particle positions. Shape is
            (nparticles, 6, dim). Includes current + last 5 positions
          nparticles_per_example: Number of particles per example. Default is 2
            examples per batch.
          particle_types: Particle types with shape (nparticles).
        """
        nparticles = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1]  # (n_nodes, 2)
        velocity_sequence = time_diff(position_sequence)

        # Get connectivity of the graph with shape of (nparticles, 2)
        senders, receivers = self._compute_graph_connectivity(
            most_recent_position, nparticles_per_example, self._connectivity_radius
        )
        node_features = []

        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"]
        ) / velocity_stats["std"]
        flat_velocity_sequence = normalized_velocity_sequence.view(nparticles, -1)
        # There are 5 previous steps, with dim 2
        # node_features shape (nparticles, 5 * 2 = 10)
        node_features.append(flat_velocity_sequence)

        # Normalized clipped distances to lower and upper boundaries.
        # boundaries are an array of shape [num_dimensions, 2], where the second
        # axis, provides the lower/upper boundaries.
        boundaries = (
            torch.tensor(self._boundaries, requires_grad=False).float().to(self._device)
        )
        distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
        distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position
        distance_to_boundaries = torch.cat(
            [distance_to_lower_boundary, distance_to_upper_boundary], dim=1
        )
        normalized_clipped_distance_to_boundaries = torch.clamp(
            distance_to_boundaries / self._connectivity_radius, -1.0, 1.0
        )
        # The distance to 4 boundaries (top/bottom/left/right)
        # node_features shape (nparticles, 10+4)
        node_features.append(normalized_clipped_distance_to_boundaries)

        # Particle type
        if self._nparticle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)
        # Final node_features shape (nparticles, 30) for 2D
        # 30 = 10 (5 velocity sequences*dim) + 4 boundaries + 16 particle embedding

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius
        # with shape (nedges, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) -
        #     torch.gather(most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius

        # Add relative displacement between two particles as an edge feature
        # with shape (nparticles, ndim)
        edge_features.append(normalized_relative_displacements)

        # Add relative distance between 2 particles with shape (nparticles, 1)
        # Edge features has a final shape of (nparticles, ndim + 1)
        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True
        )
        edge_features.append(normalized_relative_distances)

        return (
            torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1),
        )

    def _decoder_postprocessor(
        self, normalized_acceleration: torch.tensor, position_sequence: torch.tensor
    ) -> torch.tensor:
        """Compute new position based on acceleration and current position.
        The model produces the output in normalized space so we apply inverse
        normalization.

        Args:
          normalized_acceleration: Normalized acceleration (nparticles, dim).
          position_sequence: Position sequence of shape (nparticles, dim).

        Returns:
          torch.tensor: New position of the particles.

        """
        # Extract real acceleration values from normalized values
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats["std"]
        ) + acceleration_stats["mean"]

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        # TODO: Fix dt
        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1
        return new_position

    def predict_positions(
        self,
        current_positions: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
    ) -> torch.tensor:
        """Predict position based on acceleration.

        Args:
          current_positions: Current particle positions (nparticles, dim).
          nparticles_per_example: Number of particles per example. Default is 2
            examples per batch.
          particle_types: Particle types with shape (nparticles).

        Returns:
          next_positions (torch.tensor): Next position of particles.
        """
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types
        )
        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, edge_features
        )
        next_positions = self._decoder_postprocessor(
            predicted_normalized_acceleration, current_positions
        )
        return next_positions

    def predict_accelerations(
        self,
        next_positions: torch.tensor,
        position_sequence_noise: torch.tensor,
        position_sequence: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
    ):
        """Produces normalized and predicted acceleration targets.

        Args:
          next_positions: Tensor of shape (nparticles_in_batch, dim) with the
            positions the model should output given the inputs.
          position_sequence_noise: Tensor of the same shape as `position_sequence`
            with the noise to apply to each particle.
          position_sequence: A sequence of particle positions. Shape is
            (nparticles, 6, dim). Includes current + last 5 positions.
          nparticles_per_example: Number of particles per example. Default is 2
            examples per batch.
          particle_types: Particle types with shape (nparticles).

        Returns:
          Tensors of shape (nparticles_in_batch, dim) with the predicted and target
            normalized accelerations.

        """

        # Add noise to the input position sequence.
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Perform the forward pass with the noisy position sequence.
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types
        )
        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, edge_features
        )

        # Calculate the target acceleration, using an `adjusted_next_position `that
        # is shifted by the noise in the last input position.
        next_position_adjusted = next_positions + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence
        )
        # As a result the inverted Euler update in the `_inverse_decoder` produces:
        # * A target acceleration that does not explicitly correct for the noise in
        #   the input positions, as the `next_position_adjusted` is different
        #   from the true `next_position`.
        # * A target acceleration that exactly corrects noise in the input velocity
        #   since the target next velocity calculated by the inverse Euler update
        #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
        #   matches the ground truth next velocity (noise cancels out).

        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(
        self, next_position: torch.tensor, position_sequence: torch.tensor
    ):
        """Inverse of `_decoder_postprocessor`.

        Args:
          next_position: Tensor of shape (nparticles_in_batch, dim) with the
            positions the model should output given the inputs.
          position_sequence: A sequence of particle positions. Shape is
            (nparticles, 6, dim). Includes current + last 5 positions.

        Returns:
          normalized_acceleration (torch.tensor): Normalized acceleration.

        """
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats["mean"]
        ) / acceleration_stats["std"]
        return normalized_acceleration

    def save(self, path: str = "model.pt"):
        """Save model state

        Args:
          path: Model path
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load model state from file

        Args:
          path: Model path
        """
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))


def get_matrix_from_vel_accel(vel: torch.tensor) -> torch.tensor:
    """Compute acceleration and then get matrix from velocity and
       acceleration using Gram-Schmidt process
       (construct_3d_basis_from_2_vectors)

    Args:
      vel: Velocity tensor & shape(nparticles, 6 steps, dim)

    Returns:
      torch.tensor: Matrix & shape(nparticles, dim, dim)
    """
    acc = time_diff(vel)
    return construct_3d_basis_from_2_vectors(vel[:, -1], acc[:, -1])


def time_diff(position_sequence: torch.tensor) -> torch.tensor:
    """Finite difference between two input position sequence

    Args:
      position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

    Returns:
      torch.tensor: Velocity sequence
    """
    return position_sequence[:, 1:] - position_sequence[:, :-1]
