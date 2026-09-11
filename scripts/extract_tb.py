import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logdir = 'results/ood/logs'
outdir = 'results/ood'

ea = EventAccumulator(logdir)
ea.Reload()

tags = ea.Tags()['scalars']
print(f"Found {len(tags)} scalar tags")

loss_tags = [t for t in tags if 'loss' in t.lower()]
auroc_tags = [t for t in tags if 'AUROC' in t]
fpr_tags = [t for t in tags if '/FPR' in t and 'Threshold' not in t]
lr_tags = [t for t in tags if 'lr' in t.lower()]

print(f"Loss tags: {loss_tags}")
print(f"AUROC tags: {auroc_tags}")
print(f"FPR tags: {fpr_tags}")
print(f"LR tags: {lr_tags}")

# Plot loss curve
if loss_tags:
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag in loss_tags[:3]:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        label = tag.split('/')[-1] if '/' in tag else tag
        ax.plot(steps, values, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (OOD)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'loss_curve.png'), dpi=150)
    print("Saved loss_curve.png")
    plt.close()

# Plot AUROC per epoch
if auroc_tags:
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag in auroc_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        label = tag.replace('Assigner/OOD metric/', '').replace('/AUROC', '')
        ax.plot(steps, values, marker='o', label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC')
    ax.set_title('AUROC by Method across Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'auroc_curve.png'), dpi=150)
    print("Saved auroc_curve.png")
    plt.close()

# Plot FPR per epoch
if fpr_tags:
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag in fpr_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        label = tag.replace('Assigner/OOD metric/', '').replace('/FPR', '')
        ax.plot(steps, values, marker='o', label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('FPR (%)')
    ax.set_title('FPR by Method across Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fpr_curve.png'), dpi=150)
    print("Saved fpr_curve.png")
    plt.close()

# Plot LR schedule
if lr_tags:
    fig, ax = plt.subplots(figsize=(10, 4))
    for tag in lr_tags[:1]:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'lr_schedule.png'), dpi=150)
    print("Saved lr_schedule.png")
    plt.close()

print("All plots saved.")
