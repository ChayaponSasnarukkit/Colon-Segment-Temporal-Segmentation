import torch
import numpy as np
import random
import os
from model.ASFormer import Trainer

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file):
        # Reads the "split file" which is just a list of video IDs
        with open(vid_list_file, 'r') as f:
            self.list_of_examples = [line.strip() for line in f if line.strip()]
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            if features.shape[0] > features.shape[1]: features = features.T
            features = features[:, ::self.sample_rate]
            
            with open(self.gt_path + vid.split('.')[0] + '.txt', 'r') as f:
                content = f.read().split('\n')
                if content[-1] == '': content = content[:-1]
            
            labels = [self.actions_dict[c] for c in content][::self.sample_rate]
            min_len = min(features.shape[1], len(labels))
            batch_input.append(features[:, :min_len])
            batch_target.append(labels[:min_len])

        length_of_sequences = [len(l) for l in batch_target]
        max_len = max(length_of_sequences)
        
        np_batch_input = np.zeros((batch_size, batch_input[0].shape[0], max_len), dtype='float32')
        np_batch_target = np.ones((batch_size, max_len), dtype='int64') * -100
        mask = np.zeros((batch_size, 1, max_len), dtype='float32')

        for i in range(batch_size):
            l = length_of_sequences[i]
            np_batch_input[i, :, :l] = batch_input[i]
            np_batch_target[i, :l] = batch_target[i]
            mask[i, :, :l] = 1

        return torch.tensor(np_batch_input), torch.tensor(np_batch_target), torch.tensor(mask)
    
import torch
import os
import argparse
import random
import numpy as np
from model import Trainer
from dataset import BatchGenerator

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
SEED = 19980125
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='train', choices=['train', 'predict'])
    parser.add_argument('--split', default='1', help='Split number (e.g., 1)')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Setup Paths
    data_root = os.path.join("/scratch/lt200353-pcllm/location/cas_colon/")
    features_path = os.path.join("/scratch/lt200353-pcllm/location/cas_colon/", "features/") 
    gt_path = os.path.join("/scratch/lt200353-pcllm/location/cas_colon/", "ground_truth/")   
    
    splits_dir = os.path.join(data_root, "splits")
    mapping_file = os.path.join(data_root, "mapping.txt")
    
    train_split_file = os.path.join(splits_dir, f"train.split{args.split}.bundle")
    test_split_file = os.path.join(splits_dir, f"test.split{args.split}.bundle")
    
    save_dir = os.path.join(data_root, f"split_{args.split}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Load Mapping (Class list)
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found at {mapping_file}. Run generate_labels.py first!")

    with open(mapping_file, 'r') as f:
        actions = f.read().splitlines()
        
    actions_dict = {}
    for a in actions:
        parts = a.split()
        # Format: "0 Terminal_Ileum" -> ID: 0, Name: Terminal_Ileum
        actions_dict[parts[1]] = int(parts[0])
        
    num_classes = len(actions_dict)
    print(f"Loaded {num_classes} classes from mapping file.")

    # 3. Initialize Trainer
    # Parameters matched to ASFormer paper defaults
    trainer = Trainer(
        num_layers=10, 
        r1=2, r2=2, 
        num_f_maps=64, 
        features_dim=2048, 
        num_classes=num_classes, 
        channel_mask_rate=0.3
    )

    # 4. Initialize Data Generators
    # Train Generator
    batch_gen_train = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate=1)
    if os.path.exists(train_split_file):
        batch_gen_train.read_data(train_split_file)
    else:
        raise FileNotFoundError(f"Train split not found: {train_split_file}")

    # Test Generator
    batch_gen_test = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate=1)
    if os.path.exists(test_split_file):
        batch_gen_test.read_data(test_split_file)
    else:
        print(f"Warning: Test split not found at {test_split_file}. Validation will be skipped.")
        batch_gen_test = None

    # 5. Run Training
    if args.action == 'train':
        print(f"Starting training for {args.num_epochs} epochs...")
        trainer.train(
            save_dir=save_dir,
            batch_gen=batch_gen_train,
            num_epochs=args.num_epochs,
            batch_size=1, # ASFormer uses batch_size=1 for full video context
            learning_rate=args.lr,
            batch_gen_tst=batch_gen_test
        )
        
    elif args.action == 'predict':
        print("Starting prediction...")
        trainer.predict(
            model_dir=save_dir,
            results_dir=os.path.join(save_dir, "results"),
            features_path=features_path,
            batch_gen=batch_gen_test,
            epoch=args.num_epochs, # Loads the last epoch by default
            actions_dict=actions_dict,
            sample_rate=1
        )

if __name__ == "__main__":
    main()