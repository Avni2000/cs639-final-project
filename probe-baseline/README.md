# Training the baseline probe
- chose layer 28 (the last hidden layer) as the one to probe,
   - note that the hidden states the probe was trained on doesn't have to be the one we steer with. In fact, this approach is more common -- train on last hidden layer, and steer with middle-late layers. 
- paragraph breaks (\\n\\n) as the positions to probe, and
   - if there are more than MAX_HIDDEN_PER_PROBLEM (default=100) such positions, we sample a subset uniformly at random.
- It's entirely possible that we don't have enough compute to train the probe for all 1000 problems + run COT + extract hidden states for 100 paragraphs per problem/COT. 
    - If that's the case, we can train on a subset of the problems, and/or reduce the number of paragraphs we probe per problem. Shouldn't be too bad though, with a small model and a good GPU.