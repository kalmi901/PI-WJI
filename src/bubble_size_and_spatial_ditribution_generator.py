"""
Random Bubble Position Generator
--------------------------------
This script generates random bubble positions with randomized sizes.
"""

import tyro
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance_matrix as get_distance_matrix
from typing import List
from utils import CLUSTER_DATA
from utils import ClusterConfig
from utils import make_file_name, generate_bubble_position, generate_bubble_size, compute_coupling_matrix
from utils.plot_tools import plot_bubble_positions



def generate_scene(args):
    file_name = make_file_name(args,
        exclude_keys=["plot_scene_to_html", "save_config_to_file", "save_as_mat"])

    # -- Generate bubble sizes --
    bubble_sizes = generate_bubble_size(
        args.bubble_size_range,
        args.number_of_bubbles,
        args.bubble_size_type,
        args.seed
    )

    # -- Generate bubble positions --
    bubble_positions = generate_bubble_position(
        args.distance_variance,
        args.minimum_distance,
        args.number_of_bubbles,
        args.distance_type,
        args.seed
    )

    # -- Check minimum distance --
    distance_matrix = get_distance_matrix(bubble_positions, bubble_positions)
    #np.fill_diagonal(distances, 1.0e9)
    min_distance = np.min(distance_matrix + np.eye(args.number_of_bubbles) * 1.0e9)
    assert min_distance >= args.minimum_distance, "Minimum distance error!"

    # -- Calculate the coupling matrix 
    coupling_matrix = compute_coupling_matrix(distance_matrix, bubble_sizes)

    if args.plot_scene_to_html:
        plot_bubble_positions(bubble_positions, bubble_sizes, 
                    file_name=file_name)


    if args.save_config_to_file:
        np.savez_compressed(
            CLUSTER_DATA / f"{file_name}.npz",
            bubble_sizes = bubble_sizes,
            bubble_positions = bubble_positions,
            distance_matrix = distance_matrix,
            coupling_matrix = coupling_matrix,
            min_distance = min_distance)
        
        if args.save_as_mat:
            from scipy.io import savemat
            savemat(
            CLUSTER_DATA / f"{file_name}.mat",
            {
                'bubble_sizes': bubble_sizes,
                'bubble_positions': bubble_positions,
                'distance_matrix': distance_matrix,
                'coupling_matrix': coupling_matrix,
                'min_distance': min_distance
            }
        )


if __name__ == "__main__":
    args = tyro.cli(ClusterConfig)
    generate_scene(args)
    