# IndicTrans2 Finetuning - Quick Start Guide

## 🎉 What's Been Implemented

A complete, production-ready finetuning pipeline for IndicTrans2 with:

✅ **LoRA & Full Finetuning** - Choose your approach  
✅ **Flexible Data Selection** - Original or corrected translations  
✅ **Multilingual Training** - All 4 languages (Hindi, Bengali, Malayalam, Odia) in one model  
✅ **Automatic Evaluation** - BLEU scores on dev set  
✅ **Easy Inference** - Simple Python API and CLI  
✅ **Mac M2 Optimized** - Works great on Apple Silicon  

---

## 📁 Project Structure

```
src/wat_mmt/finetuning/
├── __init__.py              # Package exports
├── config.py                # Configuration classes
├── data_processor.py        # Data loading & preprocessing
├── finetune.py             # Main training script
├── inference.py            # Translation inference
└── README.md               # Detailed documentation

examples/
├── finetune_example.sh     # Training example script
└── inference_example.py    # Inference example script
```

---

## 🚀 Quick Start (3 Commands)

### 1. Train the Model

```bash
# LoRA with corrected data (recommended)
python -m src.wat_mmt.finetuning.finetune \
    --method lora \
    --use-corrected \
    --output-dir models/my-model
```

### 2. Translate Text

```bash
# Single translation
python -m src.wat_mmt.finetuning.inference \
    --model-path models/my-model \
    --text "Hello world" \
    --target-lang hindi
```

### 3. Interactive Mode

```bash
# Chat-like interface
python -m src.wat_mmt.finetuning.inference \
    --model-path models/my-model \
    --interactive \
    --target-lang bengali
```

---

## 📊 Training Options Comparison

| Feature | LoRA | Full Finetuning |
|---------|------|----------------|
| **Memory** | ~8-10GB | ~16GB+ |
| **Speed** | 2-4 hours | 8-12 hours |
| **Checkpoint Size** | ~10-50MB | ~800MB |
| **Overfitting Risk** | Lower | Higher |
| **Quality** | Excellent | Slightly better |
| **Recommended For** | Most cases | Max performance |

**💡 Recommendation: Start with LoRA!**

---

## 🎯 Common Use Cases

### Compare Original vs Corrected

```bash
# Train on original
python -m src.wat_mmt.finetuning.finetune \
    --use-original \
    --output-dir models/model-original

# Train on corrected
python -m src.wat_mmt.finetuning.finetune \
    --use-corrected \
    --output-dir models/model-corrected

# Compare both models' outputs
python -m src.wat_mmt.finetuning.inference \
    --model-path models/model-original \
    --text "test text" --target-lang hindi

python -m src.wat_mmt.finetuning.inference \
    --model-path models/model-corrected \
    --text "test text" --target-lang hindi
```

### Batch Translation from CSV

```bash
python -m src.wat_mmt.finetuning.inference \
    --model-path models/my-model \
    --input-file my_data.csv \
    --output-file translations.csv \
    --target-lang malayalam \
    --text-column english_caption \
    --batch-size 32
```

### Python API

```python
from src.wat_mmt.finetuning import IndicTrans2Translator

translator = IndicTrans2Translator("models/my-model")

# Single translation
result = translator.translate("Hello", target_lang="hindi")

# Batch translation
results = translator.translate(
    ["Hello", "World"], 
    target_lang="bengali"
)
```

---

## ⚙️ Configuration Flags

### Training

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--method` | `lora`, `full` | `lora` | Training method |
| `--use-corrected` | flag | `True` | Use corrected data |
| `--use-original` | flag | `False` | Use original data |
| `--num-epochs` | int | `3` | Training epochs |
| `--batch-size` | int | `8` | Batch size |
| `--learning-rate` | float | `3e-5` | Learning rate |
| `--lora-r` | int | `16` | LoRA rank |

### Inference

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--target-lang` | `hindi`, `bengali`, `malayalam`, `odia` | `hindi` | Target language |
| `--text` | string | - | Text to translate |
| `--interactive` | flag | - | Interactive mode |
| `--input-file` | path | - | Input CSV/JSON |
| `--num-beams` | int | `5` | Beam search size |

---

## 🔍 Supported Languages

| Language | Code | FLORES-200 |
|----------|------|------------|
| English | `english` | `eng_Latn` |
| Hindi | `hindi` | `hin_Deva` |
| Bengali | `bengali` | `ben_Beng` |
| Malayalam | `malayalam` | `mal_Mlym` |
| Odia | `odia` | `ory_Orya` |

---

## 🐛 Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
--batch-size 4

# Or use gradient accumulation
--gradient-accumulation-steps 4
```

### Training Too Slow?
```bash
# Increase batch size (if memory allows)
--batch-size 16

# Reduce evaluation frequency
--eval-steps 1000
```

### Low Quality Translations?
```bash
# Use corrected data
--use-corrected

# Increase LoRA rank
--lora-r 32 --lora-alpha 64

# Train longer
--num-epochs 5
```

---

## 📈 Expected Performance

**Training Dataset:**
- ~28,898 training examples (×4 languages = ~115,592 examples)
- ~999 dev examples (×4 languages = ~3,996 examples)

**Expected BLEU Scores (after 3 epochs):**
- Baseline (pretrained): ~15-25 BLEU
- After finetuning: ~25-35 BLEU
- With corrected data: +2-5 BLEU improvement

---

## 📚 Next Steps

1. **Train your first model:**
   ```bash
   bash examples/finetune_example.sh
   ```

2. **Test translations:**
   ```bash
   python examples/inference_example.py
   ```

3. **Read full docs:**
   - See `src/wat_mmt/finetuning/README.md` for detailed documentation
   - Check example scripts in `examples/`

4. **Experiment:**
   - Try different LoRA ranks
   - Compare original vs corrected data
   - Test on your own data

---

## 🆘 Need Help?

- **Documentation**: `src/wat_mmt/finetuning/README.md`
- **Examples**: `examples/` directory
- **Code**: Well-commented source in `src/wat_mmt/finetuning/`

---

## 📝 Notes for Mac M2 Users

✅ **MPS Support**: Automatic GPU acceleration via Metal  
✅ **Memory Efficient**: LoRA works great with unified memory  
⚠️ **No FP16**: Uses FP32 (more compatible, slightly slower)  
💡 **Tip**: Start with batch size 8, adjust based on memory

---

**Happy Finetuning! 🎉**

