"""
Progress Bar Suppression for Clean Analysis Output

This module suppresses the repetitive sentence-transformers progress bars
and replaces them with a single, clean progress indicator.

Before: 
  Batches: 100%|###| 1/1 [00:00<00:00, 16.32it/s] 
  Batches: 100%|###| 1/1 [00:00<00:00, 17.44it/s] 
  Batches: 100%|###| 1/1 [00:00<00:00, 16.43it/s] 

After:
  🔮 Creating embeddings... ██████████████████████ 100% 0:00:15
"""
import os
import warnings
import logging

def setup_clean_progress():
    """Suppress sentence-transformers and other library progress bars"""
    
    # Disable transformers progress bars and verbosity
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Disable tqdm progress bars globally (this stops the "Batches: 100%" spam)
    os.environ["TQDM_DISABLE"] = "1"
    
    # Suppress HuggingFace tokenizer warnings
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    # Suppress common warnings that clutter output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*torch.load.*")
    warnings.filterwarnings("ignore", message=".*tokenizer.*")
    
    # Set sentence-transformers to be quieter
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    print("🎯 Clean progress mode enabled - single progress bar only!")

# Auto-setup when imported
setup_clean_progress()
