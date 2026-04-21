These are all claude's notes, but it's a great writeup.

> Note $h^{+}$ is the set of all hidden states where the model ended up getting the right answer (1), and symmetrically $h^{-}$ 

## Sidenote:

Assume our good hidden states are just the bad ones shifted by some n-dim tensor.

So, $h^+ = h^- + c$ (i.e. the good hidden states are just the bad ones shifted by some constant vector $c$), then:

$$\vec{v} = \text{mean}(h_+) - \text{mean}(h_-)$$

$$= \text{mean}(h_- + c) - \text{mean}(h_-)$$

$$= (\text{mean}(h_-) + c) - \text{mean}(h_-)$$

$$= c$$

So **v literally recovers c**, which is the exact translation between the two distributions.

The intuition this unlocks is nice: if you believe the successful and unsuccessful hidden states are roughly the same "shape" in activation space but just offset from each other, then adding $\vec{v} \times \alpha$ to a failing hidden state is geometrically moving it toward where the successful states live. You're not doing anything magical — you're just translating it by (a scaled fraction of) that offset.

The assumption $h_+ = h_- + c$ is obviously a simplification — real distributions won't be perfect translates of each other — but it's a clean justification for why the mean-difference vector is a principled thing to steer with, rather than an arbitrary choice.

---


The document describes a two-phase use of **v**:


**Training time** — it's computed as:

$$\vec{v} = \text{mean}(h^+) - \text{mean}(h^-)$$



$$\vec{v} = \text{mean}(h_+) - \text{mean}(h_-)$$

$$= \text{mean}(h_- + c) - \text{mean}(h_-)$$

$$= (\text{mean}(h_-) + c) - \text{mean}(h_-)$$

$$= c$$

So **v literally recovers c** — the exact translation between the two distributions.

The intuition this unlocks is nice: if you believe the successful and unsuccessful hidden states are roughly the same "shape" in activation space but just offset from each other, then adding $\vec{v} \times \alpha$ to a failing hidden state is geometrically moving it toward where the successful states live. You're not doing anything magical — you're just translating it by (a scaled fraction of) that offset.

The assumption $h_+ = h_- + c$ is obviously a simplification — real distributions won't be perfect translates of each other — but it's a clean justification for why the mean-difference vector is a principled thing to steer with, rather than an arbitrary choice.






where $h^+$ are hidden states from successful traces and $h^-$ from failed ones. So **v** represents the direction in activation space that distinguishes good reasoning strategies from bad ones.

**Inference time** — when the probe fires and returns a low confidence score (e.g. P(correct) = 0.35), **v** is added directly to the model's hidden state at a particular layer:

$$w \mathrel{+}= \vec{v} \times \alpha$$

The idea is that by adding **v** to the hidden state **w**, you're nudging the model's internal representation toward the "direction of a good strategy" — essentially pushing the model's activations closer to where they tend to be when reasoning is going well, before it fully commits to and executes a poor approach.

The document specifies this should happen in the **middle-to-late layers**, explicitly to influence higher-level reasoning and strategy rather than low-level token/word representations (which live in early layers).

One thing worth noting as a gap: the document doesn't address whether **v** is computed per-problem-type, per-strategy-type, or globally across all traces. A single global **v** is the simplest reading, but it may be too coarse if different problem types have meaningfully different "good strategy" directions.