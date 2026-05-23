"""
Visualisation utilities.

Corresponds to Figures 2–15 in the paper:
    - Fig. 2-3: Decoded fields and latent projections
    - Fig. 4: Optimization results
    - Fig. 10-11: Probe recovery and concept-vector arithmetic
    - Fig. 6-9: Probe accuracy and parity plots
    - Fig. 14: Optimization trajectories
    - Fig. 15: Disentanglement matrix
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_latent_pca(
    proj: np.ndarray,
    color_values: np.ndarray,
    title: str = "Latent PCA Projection",
    save_path: str = None,
    cmap: str = "viridis",
    colorbar_label: str = None,
):
    """PCA projection of latent vectors with color encoding  (Fig. 3)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_values, cmap=cmap, s=10, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label=colorbar_label)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_concept_matrix(
    matrix: np.ndarray,
    param_names: list,
    title: str = "Concept-Vector Disentanglement (§F.4)",
    save_path: str = None,
):
    """
    Concept-vector disentanglement matrix  (Fig. 15).

    Rows = latent direction walked, Columns = design parameter read out.
    Diagonal = intended response, Off-diagonal = cross-talk.
    """
    K = len(param_names)
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="equal")

    # Annotate cells
    for i in range(K):
        for j in range(K):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(val) < 0.8 else "white")

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_xlabel("Design Parameter Read Out")
    ax.set_ylabel("Latent Direction Walked")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Sensitivity (σ/γ)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_probe_recovery(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names: list = None,
    title: str = "Linear Probe Recovery",
    save_path: str = None,
):
    """Parity plot for probe recovery  (Fig. 6, 10)."""
    K = y_true.shape[1]
    n_cols = min(4, K)
    n_rows = int(np.ceil(K / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if K > 1 else [axes]

    for k in range(K):
        ax = axes[k]
        ax.scatter(y_true[:, k], y_pred[:, k], s=5, alpha=0.5)
        lims = [min(y_true[:, k].min(), y_pred[:, k].min()),
                max(y_true[:, k].max(), y_pred[:, k].max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        r2 = 1 - np.sum((y_true[:, k] - y_pred[:, k]) ** 2) / np.sum((y_true[:, k] - y_true[:, k].mean()) ** 2)
        label = param_names[k] if param_names else f"Param {k}"
        ax.set_title(f"{label} (R² = {r2:.3f})")

    for idx in range(K, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_decoded_field(
    query_pts: np.ndarray,
    predicted_field: np.ndarray,
    true_field: np.ndarray = None,
    field_name: str = "Field",
    title: str = "Decoded Field Comparison  (Fig. 2, 12)",
    save_path: str = None,
):
    """
    Visualise decoded field  (Fig. 2).

    Scatter plot over surface points coloured by field value.
    """
    if true_field is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        titles = ["Predicted", "Ground Truth", "Error"]
        fields = [predicted_field, true_field, np.abs(predicted_field - true_field)]
        for ax, fld, tl in zip(axes, fields, titles):
            sc = ax.scatter(query_pts[:, 0], query_pts[:, 2], c=fld,
                            s=1, cmap="RdBu_r", alpha=0.6)
            ax.set_title(f"{tl} — {field_name}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            plt.colorbar(sc, ax=ax, shrink=0.8)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sc = ax.scatter(query_pts[:, 0], query_pts[:, 2], c=predicted_field,
                        s=1, cmap="RdBu_r", alpha=0.6)
        ax.set_title(f"{title}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_optimization_trajectory(
    proj_train: np.ndarray,
    clcd_train: np.ndarray,
    traj_proj: np.ndarray,
    optimum: np.ndarray,
    nearest_neighbour: np.ndarray,
    trust_region: tuple = None,
    save_path: str = None,
):
    """
    Latent-space optimization trajectory  (Fig. 14).

    Reference (§F.3, Fig. 14):
        "All restarts ascend the CL/CD field and converge inside the trust region."
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Background training points
    sc = ax.scatter(proj_train[:, 0], proj_train[:, 1], c=clcd_train,
                    cmap="viridis", s=5, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="CL/CD")

    # Trust region (projected)
    if trust_region is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(trust_region[0] + trust_region[2] * np.cos(theta),
                trust_region[1] + trust_region[3] * np.sin(theta),
                "k--", lw=1, alpha=0.6, label="Trust region (95% Mahalanobis)")

    # Trajectory
    ax.plot(traj_proj[:, 0], traj_proj[:, 1], "w-", lw=2, alpha=0.8)
    ax.scatter(traj_proj[0, 0], traj_proj[0, 1], c="green", s=80, marker="^",
               label="Start", edgecolors="black")
    ax.scatter(optimum[0], optimum[1], c="red", s=100, marker="*",
               label="Optimum", edgecolors="black")
    ax.scatter(nearest_neighbour[0], nearest_neighbour[1], c="blue", s=60,
               marker="o", label="Nearest neighbour", edgecolors="black",
               facecolors="none")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent-Space Optimization Trajectory  (§F.3, Fig. 14)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
