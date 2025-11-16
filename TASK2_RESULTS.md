# Task 2: KGramMLPSeqModel Implementation Results

## Implementation Summary

### Architecture
Implemented `KGramMLPSeqModel` with the following structure:
- **Input**: k concatenated one-hot vectors (size: `k × vocab_size`)
- **Input Layer**: Linear(k×vocab_size, embed_size) + SiLU activation
- **Hidden Layers**: Configurable number of Linear(embed_size, embed_size) + SiLU blocks
- **Output Layer**: Linear(embed_size, vocab_size)
- **Network**: Built using `nn.Sequential` for `self.net`

### Design Decision: One-Hot Encoding vs Embeddings

**Why NOT use `torch.nn.Embedding`?**

The k-gram MLP model uses one-hot encoding instead of embeddings for the following reasons:

1. **Preserves Discrete Nature**: K-gram MLPs directly learn mappings from specific token combinations to outputs. One-hot encoding preserves the discrete identity of each token position.

2. **No Semantic Assumptions**: One-hot vectors don't assume any relationship between tokens, allowing the MLP to learn arbitrary pattern mappings purely from data.

3. **Traditional Approach**: This is the classical k-gram MLP architecture. Using embeddings would fundamentally change it into a hybrid model.

4. **Memory Trade-off**: While embeddings would reduce input dimension (k×50257 → k×embed_size), it would alter the learning dynamics of the pure k-gram approach.

## Sanity Check Results

### Test Configuration
```bash
python pico-llm.py --block_size 32 --tinystories_weight 0.0 --input_files 3seqs.txt --prompt "0 1 2 3 4" --device_id cpu --num_inner_mlp_layers 2 --kgram_k 5 --max_steps_per_epoch 50
```

**Parameters:**
- Block size: 32
- K-gram window: 5
- Number of inner MLP layers: 2
- Embed size: 1024
- Dataset: 3333 sequences from 3seqs.txt (simple numeric patterns)
- Training: 3 epochs, 50 steps per epoch

### Training Progress

#### Epoch 1 - Initial Learning
Early in training (step 1-10), the model produced confused/repetitive outputs:
- Step 1: `0 1 2 3 4 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8`
- Step 3: `0 1 2 3 4 8 8 8 8 16 16 16 16 8 8 8 8 16 16 16 16 8 8 8 8`
- Step 7: `0 1 2 3 4 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16`

The model started learning patterns but got stuck on specific tokens.

#### Mid-Epoch 1 - Pattern Recognition Emerging
Around step 38-44:
- Step 38: `0 1 2 3 4 2 6 6 2 6 6 2 6 6 2 6 6 2 6 6 2 6 6 2 6`
- Step 43: `0 1 2 3 4 16 16 128 16 128 5 128 4096 5192 5 128 128 5192 2048 4096 5 5 5`

Model exploring various patterns, showing it's learning but not yet converged.

#### Epoch 2 - Significant Improvement
By step 10:
- Output: `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24`

**The model correctly learned the sequential pattern!**

#### Epoch 3 - Stable Performance
From step 1 onwards, model consistently generated correct sequences:
- All outputs: `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24`

### Final Results

**K-gram MLP Model:**
- Final Average Loss: 0.3238
- Generation Result: `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24` ✓
- All sampling strategies (greedy, top-p=0.95, top-p=1.0) produced identical correct outputs

**LSTM Model (for comparison):**
- Final Average Loss: 0.0174
- Generation Result: `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24` ✓
- Lower loss than k-gram MLP, showing LSTM's advantage in sequence modeling

## Comparison: K-gram MLP vs LSTM

| Metric | K-gram MLP | LSTM |
|--------|------------|------|
| Final Loss | 0.3238 | 0.0174 |
| Generation Quality | Perfect | Perfect |
| Training Speed | Slower (more forward passes) | Faster |
| Memory Efficiency | High memory (one-hot) | Low memory (embeddings) |
| Context Window | Fixed (k=5) | Unlimited |

## Conclusion

✅ **Implementation Successful**: The KGramMLPSeqModel correctly implements a k-gram MLP with:
- Proper one-hot encoding of input tokens
- Configurable MLP architecture
- Sliding window sequence-to-sequence conversion via the forward method

✅ **Sanity Check Passed**: Model successfully learned to continue simple numeric sequences, demonstrating:
- Correct forward pass implementation
- Proper gradient flow through the network
- Ability to learn sequential patterns

✅ **Design Choice Validated**: Using one-hot encoding (rather than embeddings) is appropriate for the k-gram MLP architecture, maintaining the classical approach while successfully learning patterns.

## Code Reference

Implementation location: `pico-llm.py`, lines 147-217
- `__init__`: Lines 155-179 (MLP construction)
- `forward`: Lines 181-217 (sequence processing with sliding window)

