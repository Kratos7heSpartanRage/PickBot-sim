import torch
from torch.utils.data import DataLoader
import os
import time

from models.ggcnn import GGCNN, load_trained_model
from .dataset import SyntheticGraspDataset

def train_model(epochs=5, batch_size=8, lr=1e-3, input_channels=1,
                init_checkpoint="checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt",
                save_path="checkpoints/ggcnn_finetuned.pt"):
    """
    Fine-tunes GG-CNN on auto-labeled simulation grasps to close the sim-to-real domain gap.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Fine-tuning GG-CNN on Device: {device} ===")

    # Initialize model
    model = load_trained_model(
        checkpoint_path=init_checkpoint if os.path.exists(init_checkpoint) else None,
        input_channels=input_channels,
        device=device
    )
    model.train()

    dataset = SyntheticGraspDataset(num_samples=120, input_channels=input_channels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        p_loss_total = 0.0
        ang_loss_total = 0.0

        for x, yc in loader:
            x = x.to(device)
            yc = [t.to(device) for t in yc]

            optimizer.zero_grad()
            out = model.compute_loss(x, yc)
            loss = out['loss']
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            p_loss_total += out['losses']['p_loss']
            ang_loss_total += out['losses']['cos_loss'] + out['losses']['sin_loss']

        avg_loss = running_loss / len(loader)
        dt = time.time() - t0
        print(f"Epoch [{epoch}/{epochs}] ({dt:.1f}s) - Loss: {avg_loss:.4f} | Pos Loss: {p_loss_total/len(loader):.4f} | Angle Loss: {ang_loss_total/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuning complete! Model saved to: {save_path}")
    return model
