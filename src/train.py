"""Train on synthetic cases and compare with gridding on held-out anatomy."""
import argparse
import json
from pathlib import Path
import tomllib

import torch
from torch.nn.functional import l1_loss

from src.dl_data import case
from src.unet import UNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true", help="Use 12 train, 4 validation, and 4 test cases")
    mode.add_argument("--large", action="store_true", help="Use 160 train, 20 validation, and 20 test cases")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("runs/unet"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    with (Path(__file__).resolve().parents[1] / "configs/default.toml").open("rb") as stream:
        config = tomllib.load(stream)
    if args.smoke:
        config.update(size=32, spokes=600, tr=.02)
    width = 8 if args.smoke else 16
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = UNet(2 * config["coils"] * config["bins"], config["bins"], width)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = (2, 1, 1) if args.smoke else (400, 50, 50)
    if args.pilot:
        counts = (12, 4, 4)
    if args.large:
        counts = (160, 20, 20)
    train_seeds = range(counts[0])
    validation_seeds = range(10_000, 10_000 + counts[1])
    test_seeds = range(20_000, 20_000 + counts[2])

    splits = tuple(map(set, (train_seeds, validation_seeds, test_seeds)))
    if len(set.union(*splits)) != sum(counts):
        raise ValueError("Phantom seed splits overlap")

    def dataset(seeds):
        return [tuple(torch.from_numpy(x).unsqueeze(0) for x in case(config, seed))
                for seed in seeds]

    train, validation = dataset(train_seeds), dataset(validation_seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "model.pt"
    best = float("inf")
    history = []
    for epoch in range(1 if args.smoke else args.epochs):
        model.train()
        losses = []
        for index in torch.randperm(len(train)).tolist():
            image, baseline, target = train[index]
            optimizer.zero_grad()
            loss = l1_loss(model(image, baseline), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            val_loss = sum(l1_loss(model(x, b), y).item() for x, b, y in validation) / len(validation)
        record = dict(epoch=epoch + 1, train_l1=sum(losses) / len(losses), val_l1=val_loss)
        print(record, flush=True)
        history.append(record)
        if val_loss < best:
            best = val_loss
            torch.save(dict(weights=model.state_dict(), config=config, width=width), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["weights"])
    test = dataset(test_seeds)
    with torch.inference_mode():
        scores = dict(gridding_l1=sum(l1_loss(b, y).item() for _, b, y in test) / len(test),
                      unet_l1=sum(l1_loss(model(x, b), y).item() for x, b, y in test) / len(test))
    (args.output / "metrics.json").write_text(json.dumps(dict(history=history, test=scores), indent=2))
    print(scores)
    save_comparison(model, test[0], args.output / "comparison.png")


def save_comparison(model, sample, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image, baseline, target = sample
    with torch.inference_mode():
        prediction = model(image, baseline)
    states = target.shape[1]
    figure, axes = plt.subplots(states, 3, figsize=(7, 2 * states), squeeze=False,
                                layout="constrained")
    for state, row in enumerate(axes):
        for axis, data, label in zip(row, (target, baseline, prediction),
                                     ("Truth", "Gridding", "U-Net")):
            axis.imshow(data[0, state], cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"State {state + 1} | {label}")
            axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
