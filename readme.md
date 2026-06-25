# Next Memory Prediction 

## The Core Idea
Recently there has been a surge of interest in alternatives to reconstruction-based language and image modelling. The key intuition motivating this line of reasearch is that reconstruction-based methods are inherently fit "surface variation". For images this variation might be just noise, or it might be something hard (and useless) to predict like the specific leaf-placements in an image of a tree. Fameously, Shannon argued that natural language has an inherent entropy to it, which a language model would do well to avoid trying to predict. 

Methods like JEPA and NextLat avoid the pixel and token level targets of reconstruction-based losses by (also, in the case of NextLat) having the model predict it's own future latent embeddings. Prediction in a latent space is a clean way to selectively

*predict what is (1) predictable and (2) predictive*. 

Training with an objective satisfying desiderata 1 disposes the model to find real (non-random) patterns in the data. When the objective satisfies desiderata 2 as well, the model is disposed find patterns that are themselves predictive of other patterns. Such learned abstraction is, in deep learning, often learned implicitly; doing general prediction well often requires that the model learns patterns of patterns. One of the foundational motivations that lead to the invension of *deep* learning was that of an assumed (representational) hierarchy in patterns from real data. However, using a loss-function that (also) incentivises fitting of noisy or pointless detail wastes the representational and predictive capacity of the learned models.

**Learning Better Abstract Representations.** In the NextLat paper an argument is given as to why predicting just the next latent (and token) should lead to better latent representations. It goes something like; predicting the next latent $l_{t+1}$ at time $t$ means predicting a latent ($l_{t+1}$) that is itself predictive of the latent $l_{t+2}$ at time $t+2$ and so on. In principle we therefore train the latents to be predictive indefinitely into the future just by predicting the next latent. Crucially, this makes the latents informationally markovian, since the predictive signal between any two latents is mediated by the inbetween latents. This inductive bias is great for learning markovian world-models (which is the focus of the paper), but I think we can do better for general language modelling. My suggestion? Train the model to emit persistent "latent" memories $m_{1}...m_{t}$ that can be accessed directly for prediction future tokens ($x_{>t}$) and future memories ($m_{>t}$). The idea is that if we can put direct pressure on the latent memories $m_{1}...m_{t}$ to carry information that is useful for prediction over long ranges, the latents are more prone to more and better abstraction. 

**Multi-Pass Transformer Training (mptt)**. More concretely, I propose that we use the multipass transformer training method I have developed previously to train the memory-states. Much like the NextLat paper, we can then add suitable auxillary losses for predicting future memories. 

**Chunked mptt**. One way increase the amount of pressure put on the memory-states by having fewer of them. So, if we do not emit a latent memory for each token, the model will be further incentivised to predict a compact representation of continuations and memories. Such bottle-neck approaches always risk decreased performance from just "learning less of everyting". 

### Datasets
I will start with training on the TinyStories dataset, 
- https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean

and perhaps on the ROCStories dataset also
- https://huggingface.co/datasets/mintujupally/ROCStories. 

We should train 
1. A causal transformer model with NTP. 
2. An mptt model using NTP. 
3. An mptt model using NTP + next memory matching (and perhaps a consistency loss as in the nextlat paper). 


## To Do 
1. Reimplement the mptt memory-tape model in this directory (use https://github.com/PeterBjerreHansen/multi-pass-transformer-training) and the usual transformer models.
2. Implement a working training and eval pipeline for the TinyStories task following the experimental setup used in the NextLat paper for a comparison baseline. 
3. Implement and test the chunked mptt methods (use the archive dir for inspiration perhaps, or refer to the chunked_mptt dir for a reference implementation).
4. Eventually, we should implement the chunked memory version of the mptt to see if bottlenecks improve performance. 