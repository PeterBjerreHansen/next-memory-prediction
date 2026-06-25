## Testing Priority List

### Round 1: establish architectural baselines
1. Transformer NTP baseline.
    h_t → lm_head → x_{t+1}
2. Memory transformer NTP baseline.
    h_t → lm_head → x_{t+1}
    h_t → mem_head → m_t
    memory is used internally across passes, but no NMP loss
3. Memory transformer + teacher-forced NextLat-style memory transition.
    U(m_t, e(x_{t+1})) → m̂_{t+1}
    SmoothL1(m̂_{t+1}, stopgrad(m_{t+1}))
4. “Last hidden state as memory” NextLat/MPTT-style baseline.
    U(h_t, e(x_{t+1})) → ĥ_{t+1}
    SmoothL1(ĥ_{t+1}, stopgrad(h_{t+1}))

If 3 helps but 5 helps just as much, then the effect is probably not about explicit memories. It may just be “latent transition regularization helps.”

Goal:
Separate gains from architecture, teacher-forced transition regularization, and explicit memory states.


### Round 2: test non-teacher-forced memory prediction
Round 2: Non-teacher-forced memory prediction

1. Short-horizon vector prediction baseline
    h_t → lm_head(h_t) for NTP
    h_t → mem_head(h_t) for memory
    F_i(h_t) → m̂_{t+i}, i={1,2}
    SmoothL1 to stopgrad(m_{t+i})
    Treat as weak baseline and collapse diagnostic.

2. Quantized future-memory prediction
    Assign codes directly from the model’s own memories:
        c_t = VQ(norm(m_t))
    Predict:
        m_t → p(c_{t+h})
    Use h={1,2,4,8}.
    Track code usage/perplexity.

3. Emitted chunk memories
    Insert <MEM> write positions every C tokens, or emit from every C-th token.
    M_j = mem_head(h_<MEM_j>)
    NTP remains on normal token states only.
    Train:
        M_j → p(code(M_{j+1}))
        M_j → p(code(M_{j+2}))
    Optional weak SmoothL1 only for h=1.

4. Chunked memory code prediction
    Quantize emitted chunk memories:
        C_j = VQ(norm(M_j))
    Predict:
        M_j → p(C_{j+h})
    Test chunk horizons h={1,2,4}.


### Open questions

1. "Memory is last latent of the backbone" vs. "memory is a seperate linear projection of the last latent of the backbone". 

                 ┌── lm_head(h_t)  → next-token prediction
x_≤t → backbone → h_t
                 └── mem_head(h_t) → m_t → memory-prediction objective

or 

x_≤t → backbone → h_t → m_t → lm_head(m_t)


2. Do we train a seperate (recurrent) dynamics head: $dynamics(h_t, e(x_{t+1})) → ĥ_{t+1}$ (NextLat/Deepseek mtp style).
