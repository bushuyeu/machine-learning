# ECE 645 - Machine Learning: Topics & Key Concepts Guide

## 1. Soft Start — Foundations (Weeks 1-2)

### 1.1 Linear Regression, Ridge Regression & LASSO
- **Ordinary Least Squares (OLS):** closed-form solution (normal equations), geometric interpretation as projection
- **Ridge Regression (L2):** regularization term λ||w||², bias-variance tradeoff, shrinkage towards zero, closed-form solution (X^TX + λI)^{-1}X^Ty
- **LASSO (L1):** regularization term λ||w||₁, sparsity-inducing property, feature selection, no closed-form (requires iterative solvers like coordinate descent)
- **Key comparisons:** why L1 yields sparsity vs L2 does not (geometry of constraint regions), elastic net as combination, cross-validation for hyperparameter selection
- **Bayesian interpretation:** Ridge as MAP with Gaussian prior, LASSO as MAP with Laplacian prior

### 1.2 Neural Networks & Basic Architectures
- **Perceptrons:** single-layer limitations (XOR problem), activation functions (ReLU, sigmoid, tanh, softmax)
- **Multilayer networks:** universal approximation theorem, forward propagation, backpropagation derivation (chain rule through computational graph)
- **Convolutional Neural Networks (CNNs):** convolution operation, filters/kernels, pooling (max, average), translation invariance/equivariance, typical architectures (LeNet, VGG, ResNet skip connections)
- **Autoencoders:** encoder-decoder structure, bottleneck representation, reconstruction loss, variational autoencoders (VAE) and the reparameterization trick, latent space structure
- **Stochastic Gradient Descent (SGD):** mini-batch SGD, learning rate schedules, momentum, Adam optimizer, convergence guarantees for convex vs non-convex objectives

### 1.3 Linear Classification — Fisher Linear Discriminant
- **Fisher's criterion:** maximize between-class variance / within-class variance
- **Derivation:** projection direction w = S_w^{-1}(μ₁ - μ₂), connection to generalized eigenvalue problem
- **Multi-class extension:** multiple discriminant directions, dimensionality reduction to (C-1) dimensions
- **Relation to LDA:** Fisher discriminant as special case of Linear Discriminant Analysis with Gaussian class-conditional assumptions

### 1.4 Linear Projections
- **Orthogonal projections:** projection matrices (P² = P, P^T = P), column space and null space
- **Dimensionality reduction motivation:** curse of dimensionality, intrinsic dimensionality
- **Random projections:** Johnson-Lindenstrauss lemma (preview for Week 6)

### 1.5 PCA & Singular Value Decomposition
- **PCA:** maximum variance formulation, eigenvectors of covariance matrix, proportion of variance explained, scree plots
- **SVD:** A = UΣV^T, relationship between SVD and PCA, truncated SVD for low-rank approximation (Eckart-Young theorem)
- **Practical concerns:** centering/scaling data, choosing number of components, kernel PCA for nonlinear structure

### 1.6 Evaluation Metrics
- **Regression:** MSE, RMSE, MAE, R² (coefficient of determination)
- **Classification:** accuracy, precision, recall (sensitivity), specificity, F1-score
- **ROC curves:** true positive rate vs false positive rate, AUC-ROC interpretation, threshold selection
- **Precision-Recall curves:** preferred when classes are imbalanced, average precision
- **Confusion matrix:** true/false positives/negatives, multi-class extension
- **Domain-specific terminology:** sensitivity/specificity in medical diagnostics, positive/negative predictive value

---

## 2. Kernel Methods (Weeks 3-5)

### 2.1 Kernel Theory
- **The kernel trick:** computing inner products in high-dimensional (possibly infinite) feature space without explicit mapping, φ(x)^T φ(x') = k(x, x')
- **Mercer's theorem:** conditions for valid kernels (positive semi-definite), Mercer kernels
- **Common kernels:** linear, polynomial (degree d), Gaussian/RBF (k(x,x') = exp(-||x-x'||²/2σ²)), sigmoid
- **Kernel construction:** sums, products, composition rules for building valid kernels
- **Reproducing Kernel Hilbert Spaces (RKHS):** representer theorem (solution lies in span of kernel evaluations at data points), function norms in RKHS

### 2.2 Support Vector Machines (SVM)
- **Hard-margin SVM:** maximum margin classifier, support vectors, dual formulation (quadratic programming)
- **Soft-margin SVM:** slack variables ξ_i, C parameter tradeoff, hinge loss interpretation
- **Kernel SVM:** nonlinear decision boundaries via kernel trick in the dual, kernel matrix (Gram matrix)
- **SVM vs logistic regression:** margin maximization vs likelihood maximization, sparsity of solution (only support vectors matter)

### 2.3 Kernel PCA
- **Nonlinear PCA:** performing PCA in feature space via kernel matrix eigendecomposition
- **Centering in feature space:** kernel matrix centering formula K̃ = K - 1_nK - K1_n + 1_nK1_n
- **Applications:** nonlinear dimensionality reduction, denoising, pre-image problem

### 2.4 Gaussian Processes
- **GP definition:** distribution over functions, fully specified by mean function m(x) and covariance function k(x,x')
- **GP regression:** posterior predictive distribution (closed-form Gaussian), uncertainty quantification, predictive mean and variance
- **Kernel choice impact:** smoothness, periodicity, lengthscale — how kernel hyperparameters shape the function prior
- **Marginal likelihood:** model selection and hyperparameter optimization via log marginal likelihood
- **Computational cost:** O(n³) for exact inference, scalability challenges

### 2.5 Random Fourier Features
- **Bochner's theorem:** stationary kernels as Fourier transforms of non-negative measures
- **Approximation:** k(x,x') ≈ z(x)^T z(x') where z(x) is a randomized low-dimensional feature map
- **Practical impact:** reduces kernel method complexity from O(n²) or O(n³) to linear in n, enables scalable kernel methods
- **Connection to neural networks:** random features as single hidden-layer networks with fixed random weights

---

## 3. High-Dimensional Gaussians (Week 6)

### 3.1 Gaussians in High Dimensions
- **Concentration of measure:** most mass of a high-dimensional Gaussian lies on a thin shell at radius ≈ √d from the mean
- **Curse of dimensionality:** volume of hypersphere vs hypercube, nearest-neighbor distances becoming uniform, sparsity of high-dimensional space
- **Covariance estimation:** sample covariance issues when p >> n, shrinkage estimators

### 3.2 Johnson-Lindenstrauss Lemma
- **Statement:** n points in R^d can be embedded into R^k (k = O(log(n)/ε²)) while preserving pairwise distances within factor (1±ε)
- **Random projection matrices:** Gaussian, sparse, and structured (e.g., subsampled Hadamard) constructions
- **Applications:** dimensionality reduction, approximate nearest neighbors, compressed sensing connections, sketching algorithms

### 3.3 L1 and L2 Regularization (Deeper Dive)
- **Geometric perspective:** L1 ball (diamond) vs L2 ball (sphere), why corners of L1 ball induce sparsity
- **Optimization landscape:** L2 is smooth and differentiable everywhere; L1 has non-differentiable points requiring subgradient methods
- **High-dimensional statistics:** L1 regularization for sparse recovery (compressed sensing), restricted isometry property (RIP), basis pursuit
- **Elastic net:** combining L1 + L2 for grouped sparsity

---

## 4. Large Language Models & Transformers (Weeks 7-9)

### 4.1 Topic Models
- **Latent Dirichlet Allocation (LDA):** generative model for documents, Dirichlet priors on topic and word distributions
- **Inference:** variational inference, Gibbs sampling, collapsed Gibbs
- **Evaluation:** perplexity, topic coherence metrics

### 4.2 Non-Negative Matrix Factorization (NMF)
- **Formulation:** V ≈ WH where W, H ≥ 0, parts-based representation
- **Algorithms:** multiplicative update rules, alternating least squares
- **Comparison with SVD/PCA:** NMF gives interpretable, additive parts; PCA allows subtractive combinations
- **Applications:** text mining (topic extraction), image decomposition, audio source separation

### 4.3 Attention Mechanism
- **Motivation:** limitations of fixed-length encodings in seq2seq models, alignment problem
- **Scaled dot-product attention:** Attention(Q,K,V) = softmax(QK^T/√d_k)V
- **Multi-head attention:** parallel attention heads capturing different relationship types, concatenation and linear projection
- **Self-attention vs cross-attention:** attending to own sequence vs attending to another sequence

### 4.4 Transformer Architecture
- **Encoder-decoder structure:** stack of identical layers, residual connections, layer normalization
- **Positional encoding:** sinusoidal encodings, learned positional embeddings, rotary position embeddings (RoPE)
- **Feed-forward sublayers:** position-wise fully connected networks, expansion ratio
- **Training:** masked self-attention in decoder (causal masking), teacher forcing
- **Key innovations over RNNs:** parallelizable training, no vanishing gradient over long sequences, O(n²) attention complexity

### 4.5 Large Language Models
- **Pre-training paradigms:** autoregressive (GPT family), masked language modeling (BERT), encoder-decoder (T5)
- **Scaling laws:** relationship between model size, data, compute, and performance (Chinchilla scaling)
- **Emergent abilities:** few-shot learning, chain-of-thought reasoning, in-context learning
- **Fine-tuning:** supervised fine-tuning (SFT), RLHF, LoRA and parameter-efficient fine-tuning (PEFT)
- **Tokenization:** BPE, WordPiece, SentencePiece

### 4.6 In-Context Learning
- **Definition:** model performs tasks by conditioning on examples in the prompt without weight updates
- **Theoretical perspectives:** implicit Bayesian inference, implicit gradient descent, induction heads
- **Prompt engineering:** zero-shot, few-shot, chain-of-thought prompting
- **Limitations:** sensitivity to example order, format, and selection

### 4.7 Transformers for Non-Language Tasks
- **Vision Transformers (ViT):** patch embeddings, image classification without convolutions
- **Audio/Speech:** Whisper, audio spectrogram transformers
- **Time series:** temporal attention, forecasting
- **Scientific applications:** protein structure (AlphaFold), molecular property prediction
- **Multimodal models:** CLIP, image-text alignment

---

## 5. Theory of Learning (Weeks 10-13)

### 5.1 PAC Learning
- **PAC framework:** Probably Approximately Correct — with probability ≥ 1-δ, learner finds hypothesis with error ≤ ε
- **Sample complexity:** number of samples needed as function of ε, δ, and hypothesis class complexity
- **Realizable vs agnostic setting:** realizable assumes target in hypothesis class; agnostic allows arbitrary target
- **No Free Lunch theorem:** no single algorithm dominates across all distributions

### 5.2 VC Dimension
- **Shattering:** a hypothesis class H shatters a set S if it can realize all 2^|S| labelings
- **VC dimension definition:** largest set size that H can shatter
- **Examples:** VC dim of linear classifiers in R^d is d+1, VC dim of intervals on R is 2
- **Fundamental theorem of statistical learning:** finite VC dimension ⟺ PAC learnable (for binary classification)
- **VC generalization bound:** with probability ≥ 1-δ, |R(h) - R̂(h)| ≤ O(√(VCdim · log(n) / n))
- **Growth function:** Sauer-Shelah lemma, polynomial growth when VC dim is finite

### 5.3 Rademacher Complexity
- **Definition:** R_n(H) = E_σ[sup_{h∈H} (1/n)Σσᵢh(xᵢ)], measures how well H can fit random noise
- **Generalization bounds:** tighter and data-dependent (unlike VC dimension)
- **Relationship to VC dimension:** Rademacher complexity can be bounded by VC dimension but is often tighter
- **Kernel methods connection:** Rademacher complexity of RKHS balls, norm-based bounds

### 5.4 PAC-Bayes Methods
- **PAC-Bayes bound:** for any posterior Q over hypotheses, generalization bound depends on KL(Q || P) where P is a prior
- **Key insight:** trades off empirical performance with how much the learned posterior deviates from prior
- **Tighter than uniform bounds:** incorporates prior knowledge, can give non-vacuous bounds for neural networks
- **Applications:** model selection, compression-based generalization arguments

### 5.5 Self-Certified Neural Networks
- **Motivation:** neural network generalization bounds are typically vacuous; self-certification aims for practical bounds
- **PAC-Bayes for NNs:** using learned weight distributions, noise injection for stochastic networks
- **Compression and flat minima:** connection between compressibility of weights and generalization, sharpness-aware minimization
- **Practical implications:** networks that can certify their own generalization performance

---

## 6. Reinforcement Learning (Weeks 14-15)

### 6.1 Foundations
- **MDP framework:** states, actions, transition probabilities, reward function, discount factor γ
- **Value functions:** state-value V^π(s), action-value Q^π(s,a), Bellman equations (expectation and optimality)
- **Policy:** deterministic vs stochastic, policy as mapping from states to action distributions

### 6.2 Dynamic Programming
- **Policy evaluation:** iterative computation of V^π using Bellman expectation equation
- **Policy improvement:** greedy policy with respect to current value function
- **Policy iteration:** alternating evaluation and improvement until convergence
- **Value iteration:** directly computing optimal value function via Bellman optimality equation
- **Limitations:** requires known model (transition probabilities), computational cost for large state spaces

### 6.3 Model-Free Methods
- **Monte Carlo methods:** learning from complete episodes, first-visit vs every-visit MC
- **Temporal Difference (TD) learning:** TD(0), bootstrapping, TD vs MC bias-variance tradeoff
- **Q-Learning:** off-policy TD control, Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- **SARSA:** on-policy TD control
- **Eligibility traces:** TD(λ), bridging MC and TD

### 6.4 Deep Reinforcement Learning
- **Deep Q-Networks (DQN):** function approximation with neural nets, experience replay, target networks, double DQN
- **Policy gradient methods:** REINFORCE, log-probability trick, high variance problem
- **Actor-Critic:** combining value function (critic) with policy (actor), advantage function A(s,a) = Q(s,a) - V(s)
- **Advanced algorithms:** PPO (clipped surrogate objective), A3C/A2C, SAC (entropy-regularized)
- **Challenges:** sample efficiency, exploration vs exploitation, reward shaping, sim-to-real transfer

---

## References

- Shalev-Shwartz & Ben-David — *Understanding Machine Learning: From Theory to Algorithms*
- Hardt & Recht — *Patterns, Predictions and Actions*
- Bach — *Learning Theory from First Principles*
- Zhang et al. — *Dive into Deep Learning*
- Anthony & Bartlett — *Neural Network Learning: Theoretical Foundations*
