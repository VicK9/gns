import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from pyevtk.hl import pointsToVTK

flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout `.pkl` file")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
flags.DEFINE_enum("output_mode", "gif", ["gif", "vtk"], help="Type of render output")

FLAGS = flags.FLAGS

TYPE_TO_COLOR = {
    1: "red",  # for droplet
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
}


class Render:
    def __init__(self, input_dir, input_name):
        # Texts to describe rollout cases for data and render
        rollout_cases = [
            ["ground_truth_rollout", "Reality"],
            ["predicted_rollout", "GNS"],
        ]
        self.rollout_cases = rollout_cases
        self.input_dir = input_dir
        self.input_name = input_name
        self.output_dir = input_dir
        self.output_name = input_name

        # Get trajectory
        with open(f"{self.input_dir}{self.input_name}.pkl", "rb") as file:
            rollout_data = pickle.load(file)
        self.rollout_data = rollout_data
        trajectory = {}
        for rollout_case in rollout_cases:
            trajectory[rollout_case[0]] = np.concatenate(
                [rollout_data["initial_positions"], rollout_data[rollout_case[0]]],
                axis=0,
            )
        self.trajectory = trajectory

        # Trajectory information
        self.dims = trajectory[rollout_cases[0][0]].shape[2]
        self.num_particles = trajectory[rollout_cases[0][0]].shape[1]
        self.num_steps = trajectory[rollout_cases[0][0]].shape[0]
        self.boundaries = rollout_data["metadata"]["bounds"]
        self.particle_type = rollout_data["particle_types"]

    def color_map(self):
        """
        Get color map array for each particle type for visualization
        """
        # color mask for visualization for different material types
        color_map = np.empty(self.num_particles, dtype="object")
        for material_id, color in TYPE_TO_COLOR.items():
            print(material_id, color)
            color_index = np.where(np.array(self.particle_type) == material_id)
            print(color_index)
            color_map[color_index] = color
        color_map = list(color_map)
        return color_map

    def color_mask(self):
        """
        Get color mask and corresponding colors for visualization
        """
        color_mask = []
        for material_id, color in TYPE_TO_COLOR.items():
            mask = np.array(self.particle_type) == material_id
            if mask.any() == True:
                color_mask.append([mask, color])
        return color_mask

    def render_gif_animation(
        self,
        point_size=1,
        timestep_stride=3,
        vertical_camera_angle=20,
        viewpoint_rotation=0.5,
    ):
        """
        Render `.gif` animation from `,pkl` trajectory.
        :param point_size: particle size for visualization
        :param timestep_stride: numer of timesteps to stride for visualization (i.e., sampling rate)
        :param vertical_camera_angle: camera angle in airplane view in 3d render
        :param viewpoint_rotation: speed of viewpoint rotation in 3d render
        :return: gif format animation
        """
        # Init figures
        fig = plt.figure()
        if self.dims == 2:
            ax1 = fig.add_subplot(1, 2, 1, projection="rectilinear")
            ax2 = fig.add_subplot(1, 2, 2, projection="rectilinear")
            axes = [ax1, ax2]
        elif self.dims == 3:
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            axes = [ax1, ax2]

        # Define datacase name
        trajectory_datacases = [self.rollout_cases[0][0], self.rollout_cases[1][0]]
        render_datacases = [self.rollout_cases[0][1], self.rollout_cases[1][1]]

        # Get boundary of simulation
        xboundary = self.boundaries[0]
        yboundary = self.boundaries[1]
        if self.dims == 3:
            zboundary = self.boundaries[2]

        # Get color mask for visualization
        color_mask = self.color_mask()

        # Fig creating function for 2d
        if self.dims == 2:

            def animate(i):
                print(f"Render step {i}/{self.num_steps}")

                fig.clear()
                for j, datacase in enumerate(trajectory_datacases):
                    # select ax to plot at set boundary
                    axes[j] = fig.add_subplot(1, 2, j + 1, autoscale_on=False)
                    axes[j].set_aspect(1.0)
                    axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                    axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                    for mask, color in color_mask:
                        axes[j].scatter(
                            self.trajectory[datacase][i][mask, 0],
                            self.trajectory[datacase][i][mask, 1],
                            s=point_size,
                            color=color,
                        )
                    axes[j].grid(True, which="both")
                    axes[j].set_title(render_datacases[j])

        # Fig creating function for 3d
        elif self.dims == 3:

            def animate(i):
                print(f"Render step {i}/{self.num_steps} for {self.output_name}")

                fig.clear()
                for j, datacase in enumerate(trajectory_datacases):
                    # select ax to plot at set boundary
                    axes[j] = fig.add_subplot(
                        1, 2, j + 1, projection="3d", autoscale_on=False
                    )
                    axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                    axes[j].set_zlim([float(zboundary[0]), float(zboundary[1])])
                    axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                    axes[j].set_xlabel("x")
                    axes[j].set_zlabel("z")
                    axes[j].set_ylabel("y")
                    for mask, color in color_mask:
                        axes[j].scatter(
                            self.trajectory[datacase][i][mask, 0],
                            self.trajectory[datacase][i][mask, 2],
                            self.trajectory[datacase][i][mask, 1],
                            s=point_size,
                            color=color,
                        )
                    # rotate viewpoints angle little by little for each timestep
                    axes[j].view_init(
                        elev=vertical_camera_angle, azim=i * viewpoint_rotation
                    )
                    axes[j].grid(True, which="both")
                    axes[j].set_title(render_datacases[j])

        # Creat animation
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=np.arange(0, self.num_steps, timestep_stride),
            interval=10,
        )

        ani.save(
            f"{self.output_dir}{self.output_name}.gif",
            dpi=100,
            fps=30,
            writer="imagemagick",
        )
        print(f"Animation saved to: {self.output_dir}{self.output_name}.gif")

    def write_vtk(self):
        for rollout_case, label in self.rollout_cases:
            path = f"{self.output_dir}{self.output_name}_vtk-{label}"
            if not os.path.exists(path):
                os.makedirs(path)
            initial_position = self.trajectory[rollout_case][0]
            for i, coord in enumerate(self.trajectory[rollout_case]):
                disp = np.linalg.norm(coord - initial_position, axis=1)
                pointsToVTK(
                    f"{path}/points{i}",
                    np.array(coord[:, 0]),
                    np.array(coord[:, 1]),
                    np.zeros_like(coord[:, 1])
                    if self.dims == 2
                    else np.array(coord[:, 2]),
                    data={"displacement": disp},
                )
        print(f"vtk saved to: {self.output_dir}{self.output_name}...")


def main(_):
    if not FLAGS.rollout_dir:
        raise ValueError("A `rollout_dir` must be passed.")
    if not FLAGS.rollout_name:
        raise ValueError("A `rollout_name`must be passed.")

    render = Render(input_dir=FLAGS.rollout_dir, input_name=FLAGS.rollout_name)

    if FLAGS.output_mode == "gif":
        render.render_gif_animation(
            point_size=1,
            timestep_stride=FLAGS.step_stride,
            vertical_camera_angle=20,
            viewpoint_rotation=0.3,
        )
    elif FLAGS.output_mode == "vtk":
        render.write_vtk()


if __name__ == "__main__":
    app.run(main)
