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


class Options:
    """Mock options object to initialize EvaluatorModelWrapper"""
    def __init__(self, device):
        self.dataset_name = 't2m'
        self.checkpoints_dir = './checkpoints'
        self.device = device
        
        # Text Encoder Params
        self.dim_word = 300
        self.max_motion_length = 196
        self.dim_pos_ohot = 15 
        self.max_text_len = 20
        self.dim_text_hidden = 512
        
        # Co-embedding Params
        self.dim_coemb_hidden = 512
        
        # Motion/Movement Encoder Params
        self.dim_motion_hidden = 1024
        self.dim_movement_enc_hidden = 512 
        self.dim_movement_latent = 512
        
        self.unit_length = 4 

def load_text_processor():
    """Load Spacy for POS tagging"""
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Spacy model 'en_core_web_sm' not found.")
        exit()
    return nlp

def process_text(text, nlp, word_vectorizer, max_text_len=20):
    """Convert raw text string into word embeddings and POS one-hots."""
    doc = nlp(text)
    word_embs = []
    pos_ohots = []
    
    for token in doc:
        word = token.text.lower().replace('/', '') 
        
        pos = token.pos_
        token_key = f"{word}/{pos}"
        
        try:
            vector_data = word_vectorizer[token_key]
            
            if len(vector_data) == 2:
                vec, pos_vec = vector_data
            elif len(vector_data) > 2:
                vec, pos_vec = vector_data[0], vector_data[1]
            else:
                # Fallback
                vec = np.zeros(300)
                pos_vec = np.zeros(15)
                
            word_embs.append(vec)
            pos_ohots.append(pos_vec)
            
        except Exception:
            word_embs.append(np.zeros(300))
            pos_ohots.append(np.zeros(15))
        
    valid_len = len(word_embs)
    if valid_len > max_text_len:
        word_embs = word_embs[:max_text_len]
        pos_ohots = pos_ohots[:max_text_len]
        valid_len = max_text_len
    elif valid_len < max_text_len:
        padding_len = max_text_len - valid_len
        for _ in range(padding_len):
            word_embs.append(np.zeros(300))
            pos_ohots.append(np.zeros(15))
            
    word_embs_tensor = torch.tensor(np.array(word_embs), dtype=torch.float32).unsqueeze(0)
    pos_ohots_tensor = torch.tensor(np.array(pos_ohots), dtype=torch.float32).unsqueeze(0)
    
    return word_embs_tensor, pos_ohots_tensor, valid_len

def calculate_emb_and_score(eval_wrapper, word_embs, pos_ohots, cap_len, motion_feats, device):
    """
    Calculate Euclidean distance between text and motion embeddings.
    RETURNS: (distance, motion_embedding)
    We return motion_embedding to reuse it for GT comparison.
    """
    # Ensure motion is float32 and on device
    motion_tensor = torch.tensor(motion_feats, dtype=torch.float32).to(device)
    
    # If input is (Seq, Dim), unsqueeze to (1, Seq, Dim)
    if motion_tensor.ndim == 2:
        motion_tensor = motion_tensor.unsqueeze(0)
        
    m_len = torch.tensor([motion_tensor.shape[1]], dtype=torch.long).to(device)
    
    word_embs = word_embs.to(device)
    pos_ohots = pos_ohots.to(device)
    cap_len_tensor = torch.tensor([cap_len], dtype=torch.long).to(device)

    # Get embeddings from the model
    text_emb, motion_emb = eval_wrapper.get_co_embeddings(
        word_embs, pos_ohots, cap_len_tensor, motion_tensor, m_len
    )
    
    # Calculate Text-to-Motion Distance
    dist = torch.norm(text_emb - motion_emb, p=2, dim=1)
    
    return dist.item(), motion_emb

# -----------------------------------------------------------------------------
# Main Logic (Updated)
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument('--input_text_dir', type=str, default='/home/michael/code/VimoRAG/McDPO/preference_data/in_text', help='Directory containing *_in.txt files')
    parser.add_argument('--dataset_mapping', type=str, required=True, help='Path to the JSON file containing source_id mapping (e.g., test.json)')
    parser.add_argument('--gt_motion_dir', type=str, required=True, help='Directory containing GT .npy files (suggest: new_joint_vecs)')
    parser.add_argument('--output', type=str, default='custom_preference_dataset.jsonl', help='Output JSON file')
    
    # Config Arguments
    parser.add_argument('--seed_a', type=int, default=77, help='Seed for motion A')
    parser.add_argument('--seed_b', type=int, default=60, help='Seed for motion B')
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

    # 2. Load Dataset Mapping (JSON)
    print(f"Loading mapping from {args.dataset_mapping}...")
    with open(args.dataset_mapping, 'r') as f:
        mapping_data = json.load(f)
    print(f"Loaded {len(mapping_data)} mapping entries.")

    # 3. Scan Directory for IDs
    print(f"Scanning texts in {args.input_text_dir}...")
    files_scan = os.listdir(args.input_text_dir)
    
    ids = [f.split('_')[0] for f in files_scan if f.endswith('_in.txt')]
    # Sort numerically
    ids = sorted(list(set(ids)), key=lambda x: int(x) if x.isdigit() else x)
    
    data_pairs = []

    DIR_GIFS = "/home/michael/code/VimoRAG/McDPO/preference_data/gifs"
    DIR_FEATURES = "/home/michael/code/VimoRAG/McDPO/preference_data/features"
    DIR_NPYS = "/home/michael/code/VimoRAG/McDPO/preference_data/npys"

    count = 0
    # 4. Process Pairs
    print(f"Processing {len(ids)} pairs...")

 
    data_root = os.path.dirname(args.input_text_dir.rstrip(os.sep))
    
    for idx in tqdm(ids):
        idx_int = int(idx)
        
        # --- A. Get Text & Source ID ---
        in_txt_path = pjoin(args.input_text_dir, f"{idx}_in.txt")
        
        # Read Prompt Text
        with open(in_txt_path, 'r') as f:
            prompt_text = f.read().strip()
            if prompt_text.startswith("['") and prompt_text.endswith("']"):
                prompt_text = eval(prompt_text)[0]

        # --- B. Determine GT Path & Existence ---
        gt_path = None
        source_id = None
        has_gt = False

        if idx_int < len(mapping_data):
            source_id = mapping_data[idx_int]['source_id']
            temp_gt_path = pjoin(args.gt_motion_dir, f"{source_id}.npy")
            if os.path.exists(temp_gt_path):
                gt_path = temp_gt_path
                has_gt = True
        
        # --- C. Define Paths (Absolute for loading, Relative for JSON) ---
        
        fname_feat_a = f"a_{idx}_out_feats.npy"
        fname_feat_b = f"b_{idx}_out_feats.npy"
        fname_gif_a  = f"a_{idx}_out.gif"
        fname_gif_b  = f"b_{idx}_out.gif"
        fname_npy_a  = f"a_{idx}_out.npy"
        fname_npy_b  = f"b_{idx}_out.npy"

        dir_feat = "features"
        dir_gif = "gifs"
        dir_npy = "npys"

        full_feat_a = pjoin(data_root, dir_feat, fname_feat_a)
        full_feat_b = pjoin(data_root, dir_feat, fname_feat_b)
        
        rel_feat_a = pjoin(dir_feat, fname_feat_a)
        rel_feat_b = pjoin(dir_feat, fname_feat_b)
        
        rel_gif_a = pjoin(dir_gif, fname_gif_a)
        rel_gif_b = pjoin(dir_gif, fname_gif_b)
        
        rel_npy_a = pjoin(dir_npy, fname_npy_a)
        rel_npy_b = pjoin(dir_npy, fname_npy_b)

        if not (os.path.exists(full_feat_a) and os.path.exists(full_feat_b)):
            continue

        w_embs, p_ohots, c_len = process_text(prompt_text, nlp, w_vectorizer)

        feat_a = np.load(full_feat_a)
        feat_b = np.load(full_feat_b)
        
        # --- F. Calculate Scores ---
        t2m_score_a, emb_a = calculate_emb_and_score(eval_wrapper, w_embs, p_ohots, c_len, feat_a, device)
        t2m_score_b, emb_b = calculate_emb_and_score(eval_wrapper, w_embs, p_ohots, c_len, feat_b, device)
        
        m2m_score_a = 0.0
        m2m_score_b = 0.0
        
        if has_gt:
            try:
                feat_gt = np.load(gt_path)
                _, emb_gt = calculate_emb_and_score(eval_wrapper, w_embs, p_ohots, c_len, feat_gt, device)
                
                m2m_score_a = torch.norm(emb_a - emb_gt, p=2, dim=1).item()
                m2m_score_b = torch.norm(emb_b - emb_gt, p=2, dim=1).item()
            except Exception as e:
                print(f"Warning: GT error for {idx}: {e}")
                has_gt = False
                m2m_score_a = 0.0
                m2m_score_b = 0.0
        
        # 3. Combine Scores
        total_score_a = t2m_score_a + m2m_score_a
        total_score_b = t2m_score_b + m2m_score_b

        # --- G. Determine Winner ---
        diff = total_score_a - total_score_b 
        abs_diff = abs(diff)
        
        winner = "sample_1" if total_score_a < total_score_b else "sample_2"

        pair_info = {
            "id": idx,
            "prompt": prompt_text,
            "gt_source_id": source_id if source_id else "N/A",
            "used_gt": has_gt,
            "abs_diff": abs_diff,
            "scores": {
                "a": {"t2m": t2m_score_a, "m2m": m2m_score_a, "total": total_score_a},
                "b": {"t2m": t2m_score_b, "m2m": m2m_score_b, "total": total_score_b}
            },
            "winner": winner,
            "sample_1": {
                "gif": rel_gif_a,   
                "npy": rel_npy_a,  
                "features": rel_feat_a,
                "from": "motionGPT",
                "seed": args.seed_a
            },
            "sample_2": {
                "gif": rel_gif_b,     
                "npy": rel_npy_b,      
                "features": rel_feat_b,
                "from": "motionGPT",
                "seed": args.seed_b
            }
        }
        data_pairs.append(pair_info)

        # --- REMOVED EXCEPT BLOCK ---

    print(f'count: {count}')
    if not data_pairs:
        print("No valid pairs found to process.")
        return

    print("Assigning labels based on distribution...")
    sorted_data = sorted(data_pairs, key=lambda x: x['abs_diff'], reverse=True)
    
    total = len(sorted_data)
    idx_much = int(total * 0.28)
    idx_better = int(total * 0.45)    
    idx_slightly = int(total * 0.59) 
    idx_neg = int(total * 0.62)       

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
        
        entry = {
            "id": item["id"],
            "prompt": item["prompt"],
            "gt_id": item["gt_source_id"],
            "sample_1": item["sample_1"],
            "sample_2": item["sample_2"],
            "chosen": [{
                "choice": item["winner"],
                "degree of preference": label,
                "user": "VimoRAG_Reward_Algo",
                "rationale": f"Total Score (T2M+M2M) Diff: {item['abs_diff']:.4f}"
            }],
            # Optionally save detailed scores in comment or extra field
            "debug_scores": item["scores"]
        }
        final_output.append(entry)

    # 6. Save JSON
    final_output.sort(key=lambda x: int(x['id']) if x['id'].isdigit() else x['id'])
    
    with open(args.output, 'w') as f:
        for entry in final_output:
            json.dump(entry, f)
            f.write('\n')

    print(f"Successfully saved {len(final_output)} labeled samples to {args.output}")

if __name__ == "__main__":
    main()