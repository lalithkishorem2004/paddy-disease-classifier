"""
Generalization Gap Analysis Script.
Compares the performance of models across different domains (clean vs field)
to measure the generalization drop and the impact of domain-specific fine-tuning.
"""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add parent directory of this script to system path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import config
from dataset import PaddyDataset, get_transforms
from evaluate import load_model, evaluate_model, print_metrics
from sklearn.metrics import accuracy_score

def check_eval_possibility(model_path, domain, split="test"):
    """Helper to check if model path and dataset split exist."""
    model_exists = Path(model_path).exists()
    data_dir = config.PROCESSED_DIR / domain / split
    data_exists = data_dir.exists()
    return model_exists, data_exists

def run_evaluation(model_path, domain, split="test", device="cpu"):
    """Loads a model and evaluates it on a specific domain/split, returning the accuracy."""
    data_dir = config.PROCESSED_DIR / domain / split
    dataset = PaddyDataset(data_dir, transform=get_transforms(split))
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    model, _ = load_model(model_path, device)
    y_true, y_pred = evaluate_model(model, dataloader, device)
    
    acc = accuracy_score(y_true, y_pred)
    return acc

def main():
    parser = argparse.ArgumentParser(description="Analyze Generalization Gap between Clean and Field domains.")
    parser.add_argument("--clean-model", type=str, default="models/best_model_clean_light.pth",
                        help="Path to the model checkpoint trained on clean data.")
    parser.add_argument("--field-model", type=str, default="models/best_model_field_light.pth",
                        help="Path to the model checkpoint trained on field data.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Split to use for evaluation (val or test).")
    args = parser.parse_args()
    
    device = config.DEVICE
    
    print("="*70)
    print("                Generalization Gap Analysis Study")
    print("="*70)
    
    # Check what is available
    clean_model_ok, clean_data_ok = check_eval_possibility(args.clean_model, "clean", args.split)
    field_model_ok, field_data_ok = check_eval_possibility(args.field_model, "field", args.split)
    
    results = {}
    
    # 1. Clean model on Clean data
    if clean_model_ok and clean_data_ok:
        print(f"[Run] Evaluating Clean Model on Clean {args.split} split...")
        acc = run_evaluation(args.clean_model, "clean", args.split, device)
        results["clean_on_clean"] = acc
    else:
        results["clean_on_clean"] = None
        
    # 2. Clean model on Field data
    if clean_model_ok and field_data_ok:
        print(f"[Run] Evaluating Clean Model on Field {args.split} split...")
        acc = run_evaluation(args.clean_model, "field", args.split, device)
        results["clean_on_field"] = acc
    else:
        results["clean_on_field"] = None

    # 3. Field model on Field data
    if field_model_ok and field_data_ok:
        print(f"[Run] Evaluating Field Model on Field {args.split} split...")
        acc = run_evaluation(args.field_model, "field", args.split, device)
        results["field_on_field"] = acc
    else:
        results["field_on_field"] = None
        
    # Present the findings
    print("\n" + "="*70)
    print("                        Metrics Summary")
    print("="*70)
    
    # Case A: Both clean and field datasets/models exist
    if (results["clean_on_clean"] is not None and 
        results["clean_on_field"] is not None and 
        results["field_on_field"] is not None):
        
        gen_gap = results["clean_on_clean"] - results["clean_on_field"]
        adaptation_gain = results["field_on_field"] - results["clean_on_field"]
        
        print(f"1. Clean-domain Baseline (Clean Model on Clean Test): {results['clean_on_clean']*100:.2f}%")
        print(f"2. Cross-domain Generalization (Clean Model on Field Test): {results['clean_on_field']*100:.2f}%")
        print(f"3. Field-domain Optimized Model (Field Model on Field Test):   {results['field_on_field']*100:.2f}%")
        print("-" * 70)
        print(f"--> Generalization Gap (Clean-on-Clean - Clean-on-Field):   {gen_gap*100:.2f}%")
        print(f"--> Domain Adaptation Improvement (Field-on-Field - Clean-on-Field): +{adaptation_gain*100:.2f}%")
        
    # Case B: Only field dataset/model exists (Current initial state)
    elif results["field_on_field"] is not None:
        print(f"1. Field-domain Optimized Model (Field Model on Field Test): {results['field_on_field']*100:.2f}%")
        print("-" * 70)
        print("[Notice] Clean model or clean dataset splits are not yet available.")
        print("To compute the Generalization Gap:")
        print("  1. Place the clean-background dataset under 'data/processed/clean/'.")
        print("  2. Train the baseline model: python src/train.py --domain clean")
        print("  3. Run this script again: python src/generalization_gap.py")
        
    else:
        print("[Error] No evaluation could be performed. Check if models are trained and data splits exist.")
        print(f"Field Model Status: {'Found' if field_model_ok else 'Not Found'} | Field Data Split Status: {'Found' if field_data_ok else 'Not Found'}")
        print(f"Clean Model Status: {'Found' if clean_model_ok else 'Not Found'} | Clean Data Split Status: {'Found' if clean_data_ok else 'Not Found'}")
        
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
