import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from torch.utils.data import DataLoader

class GRUFusion(nn.Module):
    """
    Computes a transition: h_new = (1-z) * h_old + z * h_candidate
    Used for the Motion Expert to evolve the state.
    """
    def __init__(self, state_dim, input_dim):
        super().__init__()
        # Gate generator: Decides how much to update
        self.z_gate = nn.Sequential(
            nn.Linear(state_dim + input_dim, state_dim),
            nn.Sigmoid()
        )
        # Candidate generator: Proposed new state based on input
        self.h_candidate = nn.Sequential(
            nn.Linear(state_dim + input_dim, state_dim),
            nn.Tanh()
        )

    def forward(self, h_old, x_input):
        combined = torch.cat([h_old, x_input], dim=1)
        
        z = self.z_gate(combined)        # Update gate [0, 1]
        h_tilde = self.h_candidate(combined) # Candidate state [-1, 1]
        
        # Convex combination (GRU-style)
        h_new = (1 - z) * h_old + z * h_tilde
        return h_new

class ResidualFusion(nn.Module):
    """
    Computes an enrichment: y = x + z * context
    Used for the Vision Expert to condition the image on history.
    """
    def __init__(self, feat_dim, context_dim):
        super().__init__()
        # Project context to match feature dimension
        self.context_proj = nn.Linear(context_dim, feat_dim)
        
        # Gate generator: Decides how relevant the context is
        self.z_gate = nn.Sequential(
            nn.Linear(feat_dim + context_dim, feat_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x_feat, context):
        # x_feat: Vision Features [B, D]
        # context: State Embedding [B, S]
        
        # 1. Project context to match x
        ctx_proj = self.context_proj(context)
        
        # 2. Calculate relevance gate
        combined = torch.cat([x_feat, context], dim=1)
        z = self.z_gate(combined)
        
        # 3. Residual addition
        out = x_feat + (z * ctx_proj)
        return self.norm(out)
    
# ==========================================
# The Bayesian Neural Filter
# ==========================================

class GatedFusionBayesianNeuralFilter_Implicit(nn.Module):
    def __init__(self, backbone, num_classes=8, embed_dim=768, state_dim=768):
        """
        Args:
            backbone: Instance of JointSurgformer
            num_classes: Number of anatomical segments (K)
            embed_dim: Output dimension of the backbone (e.g. 768)
            state_dim: Dimension for the internal state embedding
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # 1. State Embedding (Soft State Injection)
        # Embeds the previous label (or belief) into a vector
        self.state_embedding = nn.Linear(num_classes, state_dim, bias=False)
        
        # self.motion_adapter = nn.Linear(embed_dim, embed_dim)
        self.motion_fusion = GRUFusion(state_dim=state_dim, input_dim=embed_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_classes)
        )

        # 3. Vision Expert (Uses Residual-style Fusion)
        # It takes 'vision_feat' (768) and adds context from 'state' (128)
        # self.vision_adapter = nn.Linear(embed_dim, embed_dim)
        self.vision_fusion = ResidualFusion(feat_dim=embed_dim, context_dim=state_dim)
        
        # Vision head now takes the enriched 768-dim vector
        self.vision_head = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_classes)
        )

    def forward(self, x_clip, prev_state):
        """
        Args:
            x_clip: Video clip tensor [B, C, T, H, W]
            prev_state: Ground Truth label of previous step [B] (Teacher Forcing)
        """
        # 1. Extract Features
        # vision_feat: [B, 768], motion_feat: [B, 768]
        if prev_state.dim() == 1 and prev_state.dtype == torch.long:
            prev_dist = F.one_hot(prev_state, num_classes=self.num_classes).float()
        else:
            prev_dist = prev_state

        # Get embeddings
        prev_emb = self.state_embedding(prev_dist) # [B, 768]
        vision_feat, motion_feat = self.backbone(x_clip) # [B, 768] each

        # motion_feat = F.gelu(self.motion_adapter(motion_feat))
        # vision_feat = F.gelu(self.vision_adapter(vision_feat))
        
        # --- Motion Expert (Prior) ---
        # "Evolve the previous state using motion info"
        prior_latent = self.motion_fusion(h_old=prev_emb, x_input=motion_feat)
        prior_logits = self.prior_head(prior_latent)
        
        # --- Vision Expert (Likelihood) ---
        # "Look at the image, but use the previous state to resolve ambiguity"
        vision_latent = self.vision_fusion(x_feat=vision_feat, context=prev_emb)
        likelihood_logits = self.vision_head(vision_latent)
        
        # --- Bayesian Update ---
        posterior_logits = prior_logits + likelihood_logits
        
        return {
            "posterior": posterior_logits,
            "prior": prior_logits,
            "likelihood": likelihood_logits,
            "belief": F.softmax(posterior_logits, dim=1)
        }

class ExplicitMatrixTransition(nn.Module):
    """
    The "Classical" Approach.
    Predicts a KxK Transition Matrix and performs explicit matrix multiplication.
    """
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # MLP to predict the flattened matrix (K*K elements)
        # Input: Motion Features (768) -> Output: K*K (64)
        self.matrix_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes * num_classes)
        )

    def forward(self, motion_feat, prev_belief):
        """
        Args:
            motion_feat: [B, Embed_Dim] (From Surgformer)
            prev_belief: [B, K] (Probability distribution from previous step)
        """
        B, K = prev_belief.shape
        
        # 1. Predict raw scores for the matrix
        raw_matrix = self.matrix_head(motion_feat) # [B, K*K]
        
        # 2. Reshape to [B, K, K]
        # T[i, j] = Probability of moving from state i to state j
        T_logits = raw_matrix.view(B, K, K)
        
        # 3. Softmax over the last dimension to ensure rows sum to 1
        # This makes it a valid stochastic matrix.
        T_matrix = F.softmax(T_logits, dim=2) 
        
        # 4. Perform the Belief Update: p_t = b_{t-1} * T_t
        # Shapes: [B, 1, K] @ [B, K, K] -> [B, 1, K]
        prev_belief_unsqueezed = prev_belief.unsqueeze(1)
        prior_prob = torch.matmul(prev_belief_unsqueezed, T_matrix)
        
        # Remove the extra dimension -> [B, K]
        prior_prob = prior_prob.squeeze(1)
        
        return prior_prob, T_matrix
    
class GatedFusionBayesianNeuralFilter_Explicit(nn.Module):
    def __init__(self, backbone, num_classes=8, embed_dim=768, state_dim=768):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # --- 1. State Embedding (Still needed for Vision Expert) ---
        self.state_embedding = nn.Linear(num_classes, state_dim, bias=False)
        
        # --- 2. Motion Expert (EXPLICIT MATRIX VERSION) ---
        self.motion_module = ExplicitMatrixTransition(embed_dim, num_classes)

        # --- 3. Vision Expert (Likelihood) ---
        # Keeping the sophisticated fusion to ensure fair comparison
        # (Only changing the Motion part for ablation)
        self.vision_fusion = ResidualFusion(feat_dim=embed_dim, context_dim=state_dim)
        self.vision_head = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_classes)
        )

    def forward(self, x_clip, prev_state_input):
        # 1. Handle Input (Indices vs Probs)
        if prev_state_input.dim() == 1 and prev_state_input.dtype == torch.long:
            # Training: One-hot encode the hard label
            prev_dist = F.one_hot(prev_state_input, num_classes=self.num_classes).float()
        else:
            # Inference: Use the soft belief directly
            prev_dist = prev_state_input

        # 2. Get Features
        # prev_emb is still needed for the Vision Expert's context
        prev_emb = self.state_embedding(prev_dist) 
        vision_feat, motion_feat = self.backbone(x_clip)
        
        # 3. Motion Expert (Explicit Matrix Update)
        # Returns PROBABILITIES, not logits
        prior_prob, T_matrix = self.motion_module(motion_feat, prev_dist)
        
        # CRITICAL STEP: Convert Probability to Logits for Fusion
        # We add epsilon to prevent log(0)
        epsilon = 1e-9
        prior_logits = torch.log(prior_prob + epsilon)
        
        # 4. Vision Expert (Likelihood)
        vision_latent = self.vision_fusion(x_feat=vision_feat, context=prev_emb)
        likelihood_logits = self.vision_head(vision_latent)
        
        # 5. Bayesian Fusion (Log Space)
        posterior_logits = prior_logits + likelihood_logits
        
        return {
            "posterior": posterior_logits,
            "prior": prior_logits,      # These are now Log-Probs
            "likelihood": likelihood_logits,
            "belief": F.softmax(posterior_logits, dim=1),
            "transition_matrix": T_matrix # Return for visualization/analysis
        }

class BayesianNeuralFilter_Explicit(nn.Module):
    # no fusion version => This should help with the exposure bias problem
    def __init__(self, backbone, num_classes=8, embed_dim=768, state_dim=768):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Motion Expert (EXPLICIT MATRIX VERSION) ---
        self.motion_module = ExplicitMatrixTransition(embed_dim, num_classes)

        # --- 3. Vision Expert (Likelihood) ---
        self.vision_head = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_classes)
        )

    def forward(self, x_clip, prev_state_input):
        # 1. Handle Input (Indices vs Probs)
        if prev_state_input.dim() == 1 and prev_state_input.dtype == torch.long:
            # Training: One-hot encode the hard label
            prev_dist = F.one_hot(prev_state_input, num_classes=self.num_classes).float()
        else:
            # Inference: Use the soft belief directly
            prev_dist = prev_state_input

        # 2. Get Features
        # prev_emb is still needed for the Vision Expert's context
        vision_feat, motion_feat = self.backbone(x_clip)
        
        # 3. Motion Expert (Explicit Matrix Update)
        # Returns PROBABILITIES, not logits
        prior_prob, T_matrix = self.motion_module(motion_feat, prev_dist)
        
        # CRITICAL STEP: Convert Probability to Logits for Fusion
        # We add epsilon to prevent log(0)
        epsilon = 1e-9
        prior_logits = torch.log(prior_prob + epsilon)
        
        # 4. Vision Expert (Likelihood)
        likelihood_logits = self.vision_head(vision_feat)
        
        # 5. Bayesian Fusion (Log Space)
        posterior_logits = prior_logits + likelihood_logits
        
        return {
            "posterior": posterior_logits,
            "prior": prior_logits,      # These are now Log-Probs
            "likelihood": likelihood_logits,
            "belief": F.softmax(posterior_logits, dim=1),
            "transition_matrix": T_matrix # Return for visualization/analysis
        }
    
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)

# Optimizer & Scheduler
from transformers import get_cosine_schedule_with_warmup
import math

class PLWrapper(L.LightningModule):
    def __init__(self, model, config, all_samples):
        super().__init__()
        self.all_samples = all_samples
        self.save_hyperparameters(ignore=['model'])
        self.config = config
        self.model = model
        self.num_classes = self.config["num_classes"]
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro')

        # --- STATEFUL VALIDATION VARIABLES ---
        # We store these to carry memory across batches
        self.last_video_id = None
        self.current_belief = None

    def forward(self, x, prev_state):
        return self.model(x, prev_state)

    def training_step(self, batch, batch_idx):
        # Training is STANDARD (Teacher Forcing, Shuffled)
        # We ignore video_id here
        video_clip, current_label, prev_label_gt, _ = batch
        
        one_hot_gt = F.one_hot(
            prev_label_gt, num_classes=self.num_classes
        ).float()

        # --- 2. Add Random Noise & Softmax ---
        # use_noise = getattr(self, "use_input_noise", True)
        
        if self.config.get("use_noise", True):
            # A. Create Random Noise (Gaussian/Normal distribution)
            # This creates values like [-0.5, 0.2, 1.1, -0.1...]
            noise = torch.randn_like(one_hot_gt, device=self.device)
            
            # B. Define "Confidence" vs "Noise" strength
            # High 'confidence_scale' (e.g., 10.0) ensures the GT remains the Argmax.
            # 'noise_level' controls how messy the other classes look.
            confidence_scale = self.config.get("confidence_scale", 3.5) 
            noise_level = self.config.get("noise_level", 1.0)  

            # C. Create "Noisy Logits"
            # We multiply the One-Hot by a large number so the correct class 
            # has a much higher logit than the others (e.g., 10.0 vs 0.0).
            # Then we add the random noise.
            noisy_logits = (one_hot_gt * confidence_scale) + (noise * noise_level)
            
            # D. Apply Softmax (User Request)
            # This squashes logits into a valid probability distribution (Sum = 1.0).
            # The GT will likely be ~0.90, and others will be ~0.01, ~0.03, etc.
            prev_belief = F.softmax(noisy_logits, dim=1)
            
        else:
            # Clean Teacher Forcing (Strict One-Hot)
            prev_belief = one_hot_gt

        outputs = self(video_clip, prev_belief)
        
        # Losses
        loss_post = self.criterion(outputs["posterior"], current_label)
        loss_prior = self.criterion(outputs["prior"], current_label)
        loss_like = self.criterion(outputs["likelihood"], current_label)
        
        total_loss = loss_post + 0.5 * loss_prior + 0.5 * loss_like
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self):
        # Reset state at start of epoch
        self.last_video_id = None
        self.current_belief = None

    def validation_step(self, batch, batch_idx):
        """
        True Self-Forcing Validation.
        Batch Size MUST be 1. Shuffle MUST be False.
        """
        video_clip, current_label, _, video_id = batch 
        
        # Unpack list (because batch_size=1 wraps strings in a tuple)
        # video_id comes out as ('Video_01',)
        current_vid_id = video_id[0] 
        
        # --- 1. State Management ---
        # If this is a new video, or the very first batch
        if current_vid_id != self.last_video_id:
            # RESET BELIEF: Start with Uniform Distribution (We know nothing)
            self.current_belief = torch.zeros(
                (1, self.num_classes), 
                device=self.device
            )
            
            # Set Index 0 (Terminal Ileum) to 1.0 (100% Probability)
            self.current_belief[:, 0] = 1.0
            
        # --- 2. Forward Pass using SELF-GENERATED Belief ---
        # Note: We use self.current_belief, NOT the ground truth from the batch
        outputs = self(video_clip, self.current_belief)
        
        # --- 3. Update Belief for NEXT Step ---
        # We take the posterior (Logits) -> Softmax -> Store as next input
        # Detach is crucial to stop gradients (though in val it doesn't matter much)
        self.current_belief = F.softmax(outputs["posterior"], dim=1).detach()
        if self.config.get("use_noise", True):
            pred_idx = torch.argmax(self.current_belief, dim=1) 
            
            # 2. Convert to One-Hot (Same format as Ground Truth in training)
            self.current_belief = F.one_hot(pred_idx, num_classes=self.num_classes).float()
        # --- 4. Metrics ---
        val_loss = self.criterion(outputs["posterior"], current_label)
        self.val_acc(outputs["posterior"], current_label)
        self.val_f1(outputs["posterior"], current_label)
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)
        
        return val_loss

    def configure_optimizers(self):
        """
        Sets up AdamW with parameter grouping (weight decay handling AND differential LR) 
        and Cosine Scheduler with Warmup.
        """
        # 1. Define Learning Rates
        head_lr = self.config["lr"]
        backbone_lr = head_lr * 0.5  # <--- Half the LR for backbone
        weight_decay = self.config.get("weight_decay", 0.01)

        # 2. Identify Parameters
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']

        # Helper to check if a parameter belongs to the backbone
        # Assumes your model has 'self.backbone' so names start with "backbone."
        def is_backbone(n):
            return n.startswith("backbone.")

        optimizer_grouped_parameters = [
            # --- Group A: Backbone (Low LR) ---
            {
                'params': [p for n, p in param_optimizer if is_backbone(n) and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': backbone_lr,
            },
            {
                'params': [p for n, p in param_optimizer if is_backbone(n) and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': backbone_lr,
            },
            
            # --- Group B: Head / Rest (High LR) ---
            {
                'params': [p for n, p in param_optimizer if not is_backbone(n) and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': head_lr,
            },
            {
                'params': [p for n, p in param_optimizer if not is_backbone(n) and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': head_lr,
            }
        ]

        # Initialize Optimizer (Base LR is set to head_lr, but groups override it anyway)
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=head_lr 
        )

        # Scheduler Calculation (Unchanged)
        TOTAL_SAMPLES = int(self.all_samples)
        BATCH_SIZE = self.config["batch_size"]
        GRAD_ACCUM_STEPS = self.config.get("grad_accum_steps", 1)
        EPOCHS = self.config["epochs"]
        
        total_batches = TOTAL_SAMPLES // BATCH_SIZE
        num_training_steps = math.ceil(total_batches / GRAD_ACCUM_STEPS) * EPOCHS
        warmup_percent = self.config.get("warmup", 0.01)
        num_warmup_steps = int(num_training_steps * warmup_percent) 

        print(f"📉 Scheduler: {num_warmup_steps} warmup steps, {num_training_steps} total steps.")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
