#!/usr/bin/env python3
"""
BƯỚC 7: ULTRA-FAST TRAINING (4-5 HOURS) - FIXED
Key fixes:
1. Removed invalid parameters from config
2. Added proper error handling
3. Cleaned up config before passing to trainer
"""
import os
import sys
import torch
import gc
from pathlib import Path
import time
import json

# Aggressive optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')

# Memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

print("🚀 ULTRA-FAST TRAINING STARTING")
print("="*70)
print(f"💻 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("="*70)

# Imports
sys.path.insert(0, '/kaggle/working')
from src.training.mask2former_trainer import Mask2FormerTrainer
from src.data.dataset import UbirisDataset
from torch.utils.data import DataLoader

# FIXED: Load and clean config
print("\n📋 Loading configuration...")
with open('/kaggle/working/configs/mask2former_config_kaggle.json', 'r') as f:
    config = json.load(f)

# FIXED: Remove invalid parameters that EnhancedMask2Former doesn't accept
if 'model' in config:
    invalid_params = ['hidden_dim', 'use_checkpoint']
    for param in invalid_params:
        if param in config['model']:
            print(f"⚠️  Removing invalid parameter: model.{param}")
            del config['model'][param]

# Update dataset paths for Kaggle
config['data']['dataset_root'] = '/kaggle/input/iris-segmentation-ubiris-v2/dataset'
config['data']['dataset_dir'] = '/kaggle/input/iris-segmentation-ubiris-v2/dataset'
config['data']['images_dir'] = '/kaggle/input/iris-segmentation-ubiris-v2/dataset/images'
config['data']['masks_dir'] = '/kaggle/input/iris-segmentation-ubiris-v2/dataset/masks'

print("✅ Config loaded and cleaned")

# Collate function
def collate_fn(batch):
    """Memory-efficient collate function"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    result = {'pixel_values': pixel_values, 'labels': labels}
    
    if 'boundary' in batch[0]:
        result['boundary'] = torch.stack([item['boundary'] for item in batch])
    
    del batch
    return result

# Create datasets
print("\n📂 Creating datasets...")
try:
    train_dataset = UbirisDataset(
        dataset_root=config['data']['dataset_root'],
        split='train',
        use_subject_split=True,
        preserve_aspect=False,
        image_size=config['data']['image_size']
    )

    val_dataset = UbirisDataset(
        dataset_root=config['data']['dataset_root'],
        split='val',
        use_subject_split=True,
        preserve_aspect=False,
        image_size=config['data']['image_size']
    )
    
    print(f"✅ Datasets created successfully")
    
except Exception as e:
    print(f"❌ Failed to create datasets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create dataloaders
print("\n📊 Creating dataloaders...")
try:
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=False
    )

    print(f"✅ Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"✅ Val: {len(val_loader)} batches, {len(val_dataset)} samples")
    
except Exception as e:
    print(f"❌ Failed to create dataloaders: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Memory cleanup
gc.collect()
torch.cuda.empty_cache()

# Create trainer
print("\n🔧 Initializing trainer...")
device = torch.device('cuda')

try:
    trainer = Mask2FormerTrainer(
        config=config,
        device=device,
        use_wandb=False,
        resume_from=None
    )
    
    print("✅ Trainer initialized successfully")
    
except Exception as e:
    print(f"❌ Failed to initialize trainer: {e}")
    print("\n🔍 Debug info:")
    print(f"   Model config: {config.get('model', {})}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Memory stats
if torch.cuda.is_available():
    print(f"\n💾 GPU Memory before training:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Start training
print("\n" + "="*70)
print("🚀 STARTING ULTRA-FAST TRAINING")
print("="*70)
print("📊 Settings:")
print(f"   Image: {config['data']['image_size']}x{config['data']['image_size']}")
print(f"   Batch: {config['data']['batch_size']} x {config['training']['accumulation_steps']} accumulation")
print(f"   Queries: {config['model']['num_queries']}")
print(f"   Epochs: {config['training']['num_epochs']}")
print(f"   Eval: Every {config['training']['eval_freq']} epochs")
print(f"   Expected time: 4-5 hours")
print(f"   Target mIoU: 0.85-0.87")
print("="*70 + "\n")

start_time = time.time()

try:
    # Train
    trainer.train(train_loader, val_loader)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅✅✅ TRAINING COMPLETED! ✅✅✅")
    print("="*70)
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"🏆 Best validation mIoU: {trainer.best_metric:.4f}")
    print(f"📁 Checkpoint: /kaggle/working/outputs/mask2former_iris/checkpoints/best.pt")
    print("="*70)
    
    # Save final info
    final_info = {
        'total_time_hours': total_time/3600,
        'best_miou': float(trainer.best_metric),
        'total_epochs': trainer.current_epoch + 1,
        'config': config
    }
    
    with open('/kaggle/working/training_info.json', 'w') as f:
        json.dump(final_info, f, indent=2)
    
    print("\n💾 Training info saved to training_info.json")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n❌ OUT OF MEMORY!")
        print("\n🔥 Try these fixes:")
        print("1. Restart notebook and clear all outputs")
        print("2. Reduce batch_size to 2")
        print("3. Reduce image_size to 256")
        
        if torch.cuda.is_available():
            print(f"\n💾 GPU Memory at OOM:")
            print(f"   Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        print(f"\n❌ Runtime error: {e}")
        import traceback
        traceback.print_exc()
        
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    total_time = time.time() - start_time
    print(f"⏱️  Ran for: {total_time/3600:.2f} hours")
    print(f"💾 Progress saved to checkpoint")
    
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    torch.cuda.empty_cache()
    gc.collect()
    print("\n🧹 Cleanup done")