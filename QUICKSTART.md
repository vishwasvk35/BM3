# BM3 Quick Start Guide

## 🚀 To Reuse an Existing Model

### Option 1: Download Trained Model (5 minutes)

```bash
# 1. Install gdown for Google Drive downloads
pip install gdown

# 2. Download trained Baby model (385MB)
cd /home/vishwas/Desktop/RDS/BM3_implementation
./download_trained.sh
# Select: 1 (Baby)

# 3. Test the model
cd src
source /home/vishwas/Desktop/RDS/.venv/bin/activate
python load_and_test.py -c ../models/trained/baby/*.pth -d baby
```

### Option 2: Use Your Just-Trained Model

Your model just finished training with these results:
- **Recall@20: 0.0856**
- **NDCG@20: 0.0367**

The model isn't saved to disk yet. To save and reuse it:

```bash
# See your training results
cd /home/vishwas/Desktop/RDS/BM3_implementation/src
grep "best valid result:" training_output.log

# Re-train with model saving (see full guide for instructions)
# Or download trained model instead (Option 1)
```

---

## 📊 What You Can Do

### 1. Evaluate Model on Test Set

```bash
cd src
python load_and_test.py -c <path_to_model.pth> -d baby
```

### 2. Get Recommendations for Users

```bash
python load_and_test.py -c <path_to_model.pth> -d baby \
    --mode recommend \
    --user_ids 0 1 2 3 4 \
    --top_k 20
```

### 3. Check Your Training Results

```bash
# View all results
cat training_output.log

# View best results
grep "+++++Finished training" -A 10 training_output.log
```

---

## 📁 File Locations

- **Training script**: `src/main.py`
- **Test script**: `src/load_and_test.py` ✨ (newly created)
- **Download script**: `download_trained.sh` ✨ (newly created)
- **Training logs**: `src/log/`
- **Training output**: `src/training_output.log`
- **Models (when saved)**: `src/saved/` or `models/trained/`

---

## 🔍 Trained Models Available

| Dataset | Size | Download Link |
|---------|------|---------------|
| Baby | 385 MB | [Google Drive](https://drive.google.com/drive/folders/1OmRY9Z9fC01HjzT1qDuUZEwUBq03oZ5s) |
| Sports | 984 MB | [Google Drive](https://drive.google.com/drive/folders/12DngCZbMafipy2cpcWEyHz9kmKejOGcu) |
| Elec | 3.34 GB | [Google Drive](https://drive.google.com/drive/folders/1QSojeVMGQvGsgNE5GD4T7ZA7ncllWU5N) |

---

## ✅ Your Current Results (Config 1)

**Configuration:**
- n_layers: 1
- dropout: 0.3
- reg_weight: 0.1

**Performance:**
- Recall@5: 0.0322
- Recall@10: 0.0548
- Recall@20: 0.0856 ⭐
- Recall@50: 0.1470
- NDCG@20: 0.0367

---

## 💡 Next Steps

1. **Quick option**: Download trained model and test it (5 min)
2. **Full option**: Re-train with model saving enabled (2 hours)
3. **Analysis**: Compare your results with trained model

For detailed instructions, see the full guide: "How to Reuse Existing BM3 Model"
