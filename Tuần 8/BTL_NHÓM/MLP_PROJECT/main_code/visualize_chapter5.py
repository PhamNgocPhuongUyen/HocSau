"""Create Chapter 5 visualizations for the MLP 2D Gaussian approximation project.

Run from the project root (the folder containing main_code/ and checkpoints/):
    python -m main_code.visualize_chapter5

The script loads the saved Experiment A checkpoint; it does NOT retrain the model.
It saves two figures in the project's figures/ folder.
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from main_code import config as cfg
from main_code.model import MLPRegressor
from main_code.data import gaussian_2d


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model_A.pt"
FIGURE_DIR = ROOT / "figures"


def load_trained_model(checkpoint_path: Path):
    """Load Experiment A model architecture and weights from its checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}\n"
            "Hãy kiểm tra thư mục checkpoints/ và tên file best_model_A.pt."
        )

    # weights_only=False is needed because this checkpoint also stores config/metadata.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint không đúng định dạng mong đợi: cần có khóa 'model_state_dict'."
        )

    model_cfg = checkpoint.get("config", {})
    required = ("input_dim", "hidden_dim", "num_layers", "output_dim")
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise KeyError(f"Checkpoint thiếu thông tin kiến trúc: {missing}")

    model = MLPRegressor(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        output_dim=int(model_cfg["output_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def create_grid(n_points: int = 180):
    """Create a dense 2D grid over the ranges defined in config.py."""
    x1 = np.linspace(cfg.X1_RANGE[0], cfg.X1_RANGE[1], n_points)
    x2 = np.linspace(cfg.X2_RANGE[0], cfg.X2_RANGE[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    points = np.column_stack((X1.ravel(), X2.ravel()))
    return X1, X2, points


def predict_on_grid(model, points: np.ndarray, grid_shape):
    """Predict model outputs for every grid point."""
    x_tensor = torch.as_tensor(points, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(x_tensor).cpu().numpy().reshape(grid_shape)
    return predictions


def plot_true_vs_prediction(X1, X2, Z_true, Z_pred, output_path: Path):
    """Figure 5.1: compare the true function and MLP approximation surfaces."""
    z_min = float(min(np.min(Z_true), np.min(Z_pred)))
    z_max = float(max(np.max(Z_true), np.max(Z_pred)))

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surface1 = ax1.plot_surface(
        X1, X2, Z_true, cmap="viridis", linewidth=0, antialiased=True
    )
    ax1.set_title("Hàm Gaussian thực tế")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("Giá trị y")
    ax1.set_zlim(z_min, z_max)
    fig.colorbar(surface1, ax=ax1, shrink=0.62, pad=0.08)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surface2 = ax2.plot_surface(
        X1, X2, Z_pred, cmap="viridis", linewidth=0, antialiased=True,
        vmin=z_min, vmax=z_max
    )
    ax2.set_title("Kết quả xấp xỉ của MLP (Experiment A)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("Giá trị dự đoán")
    ax2.set_zlim(z_min, z_max)
    fig.colorbar(surface2, ax=ax2, shrink=0.62, pad=0.08)

    fig.suptitle("So sánh hàm mục tiêu và kết quả dự đoán của MLP", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_error(X1, X2, Z_true, Z_pred, output_path: Path):
    """Figure 5.2: spatial distribution of absolute approximation error."""
    abs_error = np.abs(Z_true - Z_pred)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    contour = ax.contourf(X1, X2, abs_error, levels=30, cmap="magma")
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Sai số tuyệt đối |y thực tế - y dự đoán|")
    ax.set_title("Phân bố sai số tuyệt đối trên lưới 2D")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return abs_error


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_trained_model(CHECKPOINT_PATH)
    X1, X2, points = create_grid(n_points=180)

    # Ground-truth values are generated using the same Gaussian function
    # implemented in main_code/data.py.
    Z_true = gaussian_2d(points).reshape(X1.shape)
    Z_pred = predict_on_grid(model, points, X1.shape)

    fig1_path = FIGURE_DIR / "chapter5_fig_5_1_true_vs_pred.png"
    fig2_path = FIGURE_DIR / "chapter5_fig_5_2_absolute_error.png"

    plot_true_vs_prediction(X1, X2, Z_true, Z_pred, fig1_path)
    abs_error = plot_absolute_error(X1, X2, Z_true, Z_pred, fig2_path)

    # These metrics are calculated on the dense visualization grid, NOT the
    # held-out test set. Do not report them as the project's test metrics.
    grid_mse = float(np.mean((Z_true - Z_pred) ** 2))
    grid_rmse = float(np.sqrt(grid_mse))
    grid_mae = float(np.mean(abs_error))

    print("\nĐÃ TẠO XONG HAI HÌNH CHO CHƯƠNG 5")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Thư mục lưu hình: {FIGURE_DIR}")
    print(f"- {fig1_path.name}")
    print(f"- {fig2_path.name}")
    print("\nSai số trên lưới trực quan hóa (không phải test metrics):")
    print(f"Grid MSE  : {grid_mse:.8f}")
    print(f"Grid RMSE : {grid_rmse:.8f}")
    print(f"Grid MAE  : {grid_mae:.8f}")

    best_run = checkpoint.get("best_run_info", {})
    if best_run:
        print("\nThông tin run tốt nhất được lưu trong checkpoint:")
        for key in ("A", "seed", "hidden_dim", "num_layers", "best_epoch",
                    "test_MSE", "test_RMSE", "test_MAE"):
            if key in best_run:
                print(f"{key}: {best_run[key]}")


if __name__ == "__main__":
    main()
