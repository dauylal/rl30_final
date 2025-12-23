import os
import json
import torch
import numpy as np
import spacy
import argparse
from tqdm import tqdm
from os.path import join as pjoin

# Import VimoRAG modules
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper

# -----------------------------------------------------------------------------
# Configuration & Helper Classes
# -----------------------------------------------------------------------------

class Options:
    """Mock options object to initialize EvaluatorModelWrapper"""
    def __init__(self, device):
        self.dataset_name = 't2m' # Assuming HumanML3D/T2M format based on 263 dims
        self.checkpoints_dir = './checkpoints'
        self.device = device
        
        # Text Encoder Params
        self.dim_word = 300
        self.max_motion_length = 196
        self.dim_pos_ohot = 15 # len(POS_enumerator)
        self.max_text_len = 20
        self.dim_text_hidden = 512
        
        # Co-embedding Params
        self.dim_coemb_hidden = 512
        
        # Motion/Movement Encoder Params (ADDED THESE MISSING PARAMS)
        self.dim_motion_hidden = 1024
        self.dim_movement_enc_hidden = 512 
        self.dim_movement_latent = 512
        
        self.unit_length = 4 # Standard for VQVAE downsampling

def load_text_processor():
    """Load Spacy for POS tagging"""
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        exit()
    return nlp

def process_text(text, nlp, word_vectorizer, max_text_len=20):
    """
    Convert raw text string into word embeddings and POS one-hots.
    Returns: tensor(1, seq, 300), tensor(1, seq, 15), int(seq_len)
    """
    doc = nlp(text)
    
    word_embs = []
    pos_ohots = []
    
    # Process tokens
    for token in doc:
        word = token.text.lower()
        pos = token.pos_
        # Format required by WordVectorizer: "word/POS"
        token_key = f"{word}/{pos}"
        vec, pos_vec = word_vectorizer[token_key]
        word_embs.append(vec)
        pos_ohots.append(pos_vec)
        
    # Truncate or Pad
    valid_len = len(word_embs)
    if valid_len > max_text_len:
        word_embs = word_embs[:max_text_len]
        pos_ohots = pos_ohots[:max_text_len]
        valid_len = max_text_len
    elif valid_len < max_text_len:
        # Pad with zeros
        padding_len = max_text_len - valid_len
        for _ in range(padding_len):
            word_embs.append(np.zeros(300))
            pos_ohots.append(np.zeros(15))
            
    # Convert to Tensor (Batch size 1)
    word_embs_tensor = torch.tensor(np.array(word_embs), dtype=torch.float32).unsqueeze(0)
    pos_ohots_tensor = torch.tensor(np.array(pos_ohots), dtype=torch.float32).unsqueeze(0)
    
    return word_embs_tensor, pos_ohots_tensor, valid_len

def calculate_score(eval_wrapper, word_embs, pos_ohots, cap_len, motion_feats, device):
    """
    Calculate Euclidean distance between text and motion embeddings.
    LOWER score means BETTER alignment.
    """
    # Prepare motion tensor (Batch size 1)
    # motion_feats shape is (Seq, Dims), typically (Seq, 263)
    motion_tensor = torch.tensor(motion_feats, dtype=torch.float32).unsqueeze(0).to(device)
    m_len = torch.tensor([motion_tensor.shape[1]], dtype=torch.long).to(device)
    
    word_embs = word_embs.to(device)
    pos_ohots = pos_ohots.to(device)
    cap_len_tensor = torch.tensor([cap_len], dtype=torch.long).to(device)

    # Get Co-Embeddings
    text_emb, motion_emb = eval_wrapper.get_co_embeddings(
        word_embs, pos_ohots, cap_len_tensor, motion_tensor, m_len
    )
    
    # Calculate Euclidean Distance (L2 Norm)
    # dist shape: (1)
    dist = torch.norm(text_emb - motion_emb, p=2, dim=1)
    return dist.item()

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_a', type=str, default='motionA', help='First motion folder')
    parser.add_argument('--folder_b', type=str, default='motionB', help='Second motion folder')
    parser.add_argument('--output', type=str, default='custom_preference_dataset.json', help='Output JSON file')
    parser.add_argument('--seed_a', type=int, default=77, help='Seed for folder A')
    parser.add_argument('--seed_b', type=int, default=60, help='Seed for folder B')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading models...")
    opt = Options(device)
    eval_wrapper = EvaluatorModelWrapper(opt)
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    nlp = load_text_processor()

    # 2. Scan Directory
    print(f"Scanning {args.folder_a}...")
    files_a = os.listdir(args.folder_a)
    
    # Filter for input text files to identify IDs
    ids = [f.split('_')[0] for f in files_a if f.endswith('_in.txt')]
    ids = sorted(list(set(ids)), key=lambda x: int(x) if x.isdigit() else x)
    
    data_pairs = []

    # 3. Process Pairs
    print(f"Processing {len(ids)} pairs...")
    for idx in tqdm(ids):
        try:
            # File paths
            in_txt_path = pjoin(args.folder_a, f"{idx}_in.txt")
            
            feat_a_path = pjoin(args.folder_a, f"{idx}_out_feats.npy")
            gif_a_path = pjoin("gifs", f"a_{idx}_out.gif")
            npy_a_path = pjoin("npys", f"a_{idx}_out.npy")
            
            feat_b_path = pjoin(args.folder_b, f"{idx}_out_feats.npy")
            gif_b_path = pjoin("gifs", f"b_{idx}_out.gif")
            npy_b_path = pjoin("npys", f"b_{idx}_out.npy")

            # Check existence
            if not (os.path.exists(feat_a_path) and os.path.exists(feat_b_path)):
                print(f"Skipping ID {idx}: Missing feature files.")
                continue

            # Read Text
            with open(in_txt_path, 'r') as f:
                prompt_text = f.read().strip()
                # If text is in a list format "['text']", clean it
                if prompt_text.startswith("['") and prompt_text.endswith("']"):
                    prompt_text = eval(prompt_text)[0]

            # Vectorize Text
            w_embs, p_ohots, c_len = process_text(prompt_text, nlp, w_vectorizer)

            # Read Motions
            feat_a = np.load(feat_a_path)
            feat_b = np.load(feat_b_path)

            # Calculate Scores (Lower is Better for Euclidean Distance)
            score_a = calculate_score(eval_wrapper, w_embs, p_ohots, c_len, feat_a, device)
            score_b = calculate_score(eval_wrapper, w_embs, p_ohots, c_len, feat_b, device)

            # print(feat_a_path)
            # print(feat_b_path)
            # print(score_a)
            # print(score_b)

            # Calculate Difference (Positive diff means A is larger/worse than B -> B wins)
            diff = score_a - score_b 
            abs_diff = abs(diff)
            
            # Determine technical winner (Lowest Distance wins)
            if score_a < score_b:
                winner = "sample_1" # A wins
            else:
                winner = "sample_2" # B wins

            # Store Data
            pair_info = {
                "id": idx,
                "prompt": prompt_text,
                "abs_diff": abs_diff,
                "winner": winner,
                "sample_1": {
                    "gif": gif_a_path,
                    "npy": npy_a_path,
                    "features": f"features/a_{idx}_out_feats.npy",
                    "from": "motionGPT",
                    "seed": args.seed_a
                },
                "sample_2": {
                    "gif": gif_b_path,
                    "npy": npy_b_path,
                    "features": f"features/b_{idx}_out_feats.npy",
                    "from": "motionGPT",
                    "seed": args.seed_b
                }
            }
            data_pairs.append(pair_info)

        except Exception as e:
            print(f"Error processing ID {idx}: {e}")

    # 4. Distribution Logic
    if not data_pairs:
        print("No valid pairs found to process.")
        return

    print("Assigning labels based on distribution...")
    # Sort by magnitude of difference (Largest gap first)
    sorted_data = sorted(data_pairs, key=lambda x: x['abs_diff'], reverse=True)
    
    total = len(sorted_data)
    idx_much = int(total * 0.28)
    idx_better = int(total * 0.45)   # 28 + 17
    idx_slightly = int(total * 0.59) # 45 + 14
    idx_neg = int(total * 0.62)      # 59 + 3
    # Remaining 20% is Unsure

    final_output = []

    for i, item in enumerate(sorted_data):
        if i < idx_much:
            label = "Much better"
        elif i < idx_better:
            label = "Better"
        elif i < idx_slightly:
            label = "Slightly better"
        elif i < idx_neg:
            label = "Negligibly better"
        else:
            label = "Unsure"
        
        # Build JSON Entry
        entry = {
            "id": item["id"],
            "prompt": item["prompt"],
            "sample_1": item["sample_1"],
            "sample_2": item["sample_2"],
            "chosen": [{
                "choice": item["winner"],
                "degree of preference": label,
                "user": "VimoRAG_Reward_Algo"
            }],
            "comment": f"Auto-labeled. Score Diff: {item['abs_diff']:.4f}"
        }
        final_output.append(entry)

    # 5. Save JSON
    # Re-sort by ID for cleanliness (optional)
    final_output.sort(key=lambda x: int(x['id']) if x['id'].isdigit() else x['id'])
    
    with open(args.output, 'w') as f:
        for entry in final_output:
            json.dump(entry, f) 
            f.write('\n')       
            
    print(f"Successfully saved {len(final_output)} labeled samples to {args.output}")

if __name__ == "__main__":
    main()