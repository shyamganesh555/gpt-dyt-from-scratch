
Character-Level GPT with Dynamic Tanh (DyT)

A complete PyTorch implementation of a decoder-only Transformer language model trained from scratch at the character level on William Shakespeare's A Midsummer Night's Dream. Instead of relying on traditional Layer Normalization (LayerNorm), this project replaces every normalization layer with Dynamic Tanh (Dynamic_Tanh), demonstrating that learnable bounded non-linearities can stabilize Transformer training without computing batch or layer statistics.

1. Overview and Research Foundation

• Core Objective: Builds a Generative Pre-trained Transformer (GPT) from scratch using pure PyTorch tensor operations, avoiding high-level Transformer wrappers.

• Andrej Karpathy (nanoGPT): Adopts the character-level tokenization pipeline, batch sampling mechanism (get_batch), residual block layout, and autoregressive multinomial generation loop from Karpathy's "Let's build GPT: from scratch, in code, spelled out".
• Attention Is All You Need (Vaswani et al., 2017): Implements scaled dot-product multi-head self-attention, causal lower-triangular masking, and position-wise feed-forward networks from the original Transformer architecture.
• Transformers without Normalization (Zhu et al., 2025): Replaces standard LayerNorm modules with Dynamic Tanh (Dynamic_Tanh), defined as \(y = \gamma \cdot \tanh(\alpha x) + \beta\), where α is a learnable scalar parameter that dynamically scales input activations before applying the smooth hyperbolic tangent curve.
• Backpropagation (Rumelhart, Hinton, & Williams, 1986): Trains all embedding tables, attention projections, feed-forward weights, and dynamic scaling parameters end-to-end via automatic differentiation and gradient descent.

2. Detailed Model Architecture

• Character-Level Tokenization:
	• Scans the raw text corpus to extract all unique characters, forming a sorted vocabulary of 75 tokens (vocab_size = 75).
	• Maps characters to integer indices (stoi) and reverses indices back to characters (itos) using lookup dictionaries.
• Input Embedding Layer:
	• Token Embeddings: Maps input token indices of shape (B, T) into dense vectors of size n_embd = 256 using nn.Embedding(75, 256).
	• Positional Embeddings: Encodes token positions 0 to T - 1 using a learned embedding table nn.Embedding(256, 256) and adds them element-wise to the token embeddings.
• Dynamic Tanh Module (Dynamic_Tanh):
	• Replaces LayerNorm before the self-attention sub-layer (dyt1), before the feed-forward sub-layer (dyt2), and at the final output projection (final_norm).
	• Initializes a learnable scalar alpha to 0.5, an element-wise learnable scale vector weight to ones, and an optional shift vector bias to zeros.
• Causal Multi-Head Self-Attention (MultiheadAttention):
	• Projects normalized inputs into combined Query, Key, and Value tensors using a single bias-free linear layer (nn.Linear(256, 768, bias=False)).
	• Splits and reshapes the projections across 4 parallel attention heads, each operating on a head_size of 64 (256 // 4).
	• Computes scaled attention scores \((Q K^T) / \sqrt{64}\) and applies a lower-triangular causal mask (torch.tril) filled with -inf to prevent tokens from attending to future positions.
	• Applies softmax, 0.5 dropout regularization, value aggregation, head concatenation, and a final output projection (nn.Linear(256, 256)).
• Position-Wise Feed-Forward Network (FeedForward):
	• Expands representations from 256 to 1,024 dimensions (4 * n_embd) via a linear layer.
	• Applies a non-linear GELU activation function and projects back down to 256 dimensions.
• Transformer Decoder Stack & Language Modeling Head:
	• Stacks 4 sequential Block modules connected via residual skip connections (x = x + Sublayer(x)).
	• Passes the final block output through final_norm (Dynamic_Tanh) and projects logits across all 75 vocabulary characters via lm_head (nn.Linear(256, 75)).

3. Hyperparameters & Training Configuration

• Context Window (block_size): 256 characters per input sequence.
• Batch Size (batch_size): 64 independent sequences processed per iteration (16,384 tokens per step).
• Embedding Dimension (n_embd): 256
• Attention Heads (n_heads): 4 (64 dimensions per head)
• Transformer Layers (n_layers): 4 decoder blocks
• Regularization (dropout): 0.5 applied to attention probability weights
• Optimizer: AdamW with a learning rate of 3e-4 (0.0003)
• Training Duration (max_iter): 3,000 iterations
• Hardware Acceleration: Automatically detects and runs on CUDA-enabled GPUs (cuda), falling back to cpu when unavailable.

4. Dataset & Preprocessing

• Source Text: Downloads the Folger Shakespeare Library edition of A Midsummer Night's Dream (a-midsummer-nights-dream_TXT_FolgerShakespeare.txt, 100,554 bytes) directly from GitHub.
• Train / Validation Split: Encodes the entire text into a 1D torch.long tensor and splits it sequentially into 90% training data and 10% validation data.
• Batch Sampling (get_batch): Randomly samples 64 starting offsets per batch, slicing input windows x of length 256 and target windows y shifted right by one character (i+1 to i+block_size+1).

5. Training Results & Convergence

• Loss Evaluation Protocol (estimate_loss): Every 100 steps, the model switches to .eval() mode with @torch.no_grad() and averages cross-entropy loss across 50 random batches from both splits.
• Progression Across Training Steps:
	• Step 0 (Initialization): Train Loss 4.3738 | Validation Loss 4.3769 (matches theoretical uniform random guess \(-\ln(1/75) \approx 4.3175\)).
	• Step 100: Train Loss 2.5920 | Validation Loss 2.6822
	• Step 500: Train Loss 2.3939 | Validation Loss 2.4793
	• Step 1,000: Train Loss 2.0405 | Validation Loss 2.0997
	• Step 1,500: Train Loss 1.7527 | Validation Loss 1.8409
	• Step 2,000: Train Loss 1.5733 | Validation Loss 1.7066
	• Step 2,500: Train Loss 1.4360 | Validation Loss 1.6459
	• Step 2,999 (Final): Train Loss 1.3068 | Validation Loss 1.6353
• Visualization: Logs evaluation metrics into lists and generates a high-resolution training vs. validation loss curve using matplotlib.

6. Autoregressive Text Generation

• Generation Algorithm (generate):
	• Crops the conditioning context to the last block_size (256) tokens on every step.
	• Extracts logits from the final time step (logits[:, -1, :]), converts them into probabilities via F.softmax, and samples the next character index using torch.multinomial.
	• Appends the newly predicted token to the running sequence and repeats for 500 iterations.
• Qualitative Output Sample:
	• Even without normalization layers or subword tokenization, the model learns theatrical script layout, uppercase speaker cues, bracketed stage directions, punctuation, and Early Modern English rhythm:
[Quin a not, oncry.]
Dever and in the Duch.
[He exit.]
HELENA
Ay, and, the you moonly. My are what I veer you.
Hit go! Our nother! That yourself, speak Helena!
O, thou rects you part. The could tentle nothing,
To he sew'ritg night,
When sould the may devision did thought never the was
Lysander's flower?
THESEUS  Let noble And you night, starding That us your I confeight
Tomiantial. Now no greamss, chay an deater to steeth,
Dance when Preasseblosson you
you truest a Thisbe's fure day man blood

7. Setup and Usage

• Prerequisites: Python 3.8 or newer with PyTorch and Matplotlib installed:bash
  pip install torch matplotlib
• Running in Google Colab: Open the notebook (model_gpt.ipynb), select a GPU runtime (T4 or better), and run all cells sequentially.
• Running Locally: Export the notebook cells to a .py script, download the dataset file into the working directory, and execute python model_gpt.py.

8. References

1. Karpathy, A. (2023). Let's build GPT: from scratch, in code, spelled out (nanoGPT).
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS).
3. Zhu, J., Chen, X., He, K., LeCun, Y., & Liu, Z. (2025). Transformers without Normalization.
4. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533–536.
