# Notes

## Q: How does the model converge its probabilities into 4 labels?

### Architecture Overview

```
Input: "[CLS] qualification text [SEP] candidate text [SEP]"
         “
   DistilBERT Encoder (6 transformer layers)
         “
   [CLS] token embedding (768-dim vector)
         “
   Classification Head (Linear: 768 ’ 4)
         “
   Raw logits: [score_0, score_1, score_2, score_3]
         “
   Softmax
         “
   Probabilities: [p_NOT_MATCH, p_PARTIAL_MATCH, p_MATCH, p_FORBIDDEN]
```

### Key Components

1. **The Classification Head** - When creating the model with `num_labels=4`, Hugging Face adds a linear layer that projects the 768-dim `[CLS]` embedding to 4 values (one per label).

2. **Training with Cross-Entropy Loss** - The model outputs 4 raw logits, compares them to the true label, and backpropagates to make the correct label's logit highest.

3. **Softmax at Inference** - Converts logits to probabilities that sum to 1.

### Why It Converges

The model learns to:
- **Encode semantic meaning** - DistilBERT understands text (pretrained on massive data)
- **Compare Q and C** - The `[CLS]` token aggregates information from both segments
- **Map patterns to labels** - The classification head learns which representations correspond to which labels

The loss function penalizes wrong predictions, so over epochs, the model adjusts weights until the correct label consistently gets the highest probability.
