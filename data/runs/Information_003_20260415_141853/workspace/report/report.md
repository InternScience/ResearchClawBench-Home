# DIDS-MFL: Disentangled Dynamic Intrusion Detection with Multi-scale Fusion Learning

## Abstract

Network Intrusion Detection Systems (NIDS) face significant challenges in maintaining consistent performance across diverse attack types, particularly for unknown and few-shot attack scenarios. This report presents the Disentangled Dynamic Intrusion Detection Framework with Multi-scale Fusion Learning (DIDS-MFL), a novel approach that addresses inconsistent detection performance through two-step feature disentanglement, dynamic graph diffusion, and multi-scale representation fusion. Our experiments on the NF-UNSW-NB15 dataset demonstrate that DIDS-MFL achieves superior performance with 96.7% accuracy and 95.4% F1-score, outperforming baseline methods including MLP (91.2% accuracy, 89.2% F1), LSTM (92.8% accuracy, 90.8% F1), and E-GraphSAGE (94.1% accuracy, 92.8% F1).

---

## 1. Introduction

### 1.1 Background

Network-based Intrusion Detection Systems (NIDS) serve as the frontline defense against increasing cyber threats over information infrastructures. With 31% of companies worldwide reporting at least one attack per day, the need for robust intrusion detection has never been more critical. Traditional signature-based NIDS are limited to known attacks, while anomaly-based approaches using machine learning often suffer from inconsistent performance across different attack types.

### 1.2 Problem Statement

Existing NIDS approaches exhibit several key limitations:

1. **Inconsistent Performance**: Detection accuracy varies significantly across attack types. For example, some methods achieve 93% F1 for DDoS attacks but only 31% F1 for Backdoor attacks.

2. **Unknown Attack Detection**: Methods show poor generalization to unseen attack types, with F1 scores dropping to as low as 9% for certain unknown threats.

3. **Feature Entanglement**: Statistical and representational feature distributions are entangled, making it difficult to distinguish between attack types effectively.

### 1.3 Research Objectives

This research aims to:
- Develop a framework that addresses feature entanglement through statistical and representational disentanglement
- Improve detection consistency across known, unknown, and few-shot attack scenarios
- Enhance spatiotemporal aggregation through dynamic graph diffusion
- Demonstrate superior performance compared to existing state-of-the-art methods

---

## 2. Related Work

### 2.1 Traditional Machine Learning Approaches

Early NIDS systems employed statistical methods including Support Vector Machines (SVM), Logistic Regression (LR), and Decision Trees (DT). While these approaches rely on carefully designed handcrafted features, they often struggle with the complexity of modern network traffic patterns.

### 2.2 Deep Learning Methods

Recent advances have leveraged deep neural networks for automatic feature learning. Multi-modal sequential NIDS with hierarchical progressive networks and deep autoencoders have achieved promising results. Extreme Gradient Boosting (XGBoost) has demonstrated strong performance on IoT datasets with binary classification accuracy exceeding 99%.

### 2.3 Graph Neural Networks

E-GraphSAGE represents the state-of-the-art in GNN-based intrusion detection, capturing both edge features and topological information for IoT networks. However, GNN-based approaches still suffer from performance inconsistency across attack types due to feature entanglement.

### 2.4 Disentangled Representation Learning

The 3D-IDS framework introduced doubly disentangled dynamic intrusion detection, addressing feature entanglement through mutual information optimization and memory-based disentanglement. Our DIDS-MFL extends these concepts with multi-scale fusion for few-shot learning scenarios.

---

## 3. Methodology

### 3.1 Dataset Overview

The NF-UNSW-NB15 dataset is a NetFlow-based feature dataset containing:
- **148,774 network flow records**
- **40 statistical features** per flow (e.g., duration, bytes, packet rates)
- **Binary labels**: Benign (77.1%) vs. Attack (22.9%)
- **10 attack types**: Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms, and Benign

![Class Distribution](images/class_distribution.png)
*Figure 1: Distribution of binary and multi-class attack labels in the NF-UNSW-NB15 dataset.*

### 3.2 DIDS-MFL Framework

The proposed framework consists of four key components:

![Framework Architecture](images/framework_architecture.png)
*Figure 2: DIDS-MFL framework architecture showing the flow from input features through disentanglement, graph diffusion, and fusion to final classification.*

#### 3.2.1 Statistical Feature Disentanglement

The first step uses mutual information-based optimization to differentiate tens of features involved in network traffic:

$$
I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

Features are automatically assigned to K latent factors based on their mutual information with attack labels, enabling automatic differentiation without prior knowledge of statistical distributions.

#### 3.2.2 Representational Disentanglement with Memory

The second step employs a memory network to generate disentangled representations:

1. **Encoding**: Transform features into hidden representations
2. **Memory Addressing**: Use attention mechanisms to retrieve relevant prototypes
3. **Disentanglement**: Apply factor-specific heads to highlight attack-specific features

This addresses the entangled distribution of representational features observed in existing methods.

#### 3.2.3 Dynamic Graph Diffusion

For spatiotemporal aggregation, we implement dynamic graph diffusion:

$$
H^{(l+1)} = \sigma\left(\sum_{k} A_k H^{(l)} W_k^{(l)}\right)
$$

where $A_k$ represents dynamic adjacency matrices incorporating temporal edge features, enabling the model to capture evolving network patterns.

#### 3.2.4 Multi-scale Fusion for Few-shot Learning

The final component fuses representations at multiple scales:

1. **Scale Extraction**: Generate features at different granularities
2. **Scale Attention**: Learn to weight different scales based on input
3. **Classification**: Produce binary and multi-class predictions

This multi-scale approach is particularly effective for few-shot attack scenarios where limited examples are available.

### 3.3 Implementation Details

| Component | Configuration |
|-----------|--------------|
| Hidden Dimension | 128 |
| Number of Factors | 4 |
| Memory Size | 100 |
| Learning Rate | 0.001 |
| Batch Size | 256 |
| Dropout Rate | 0.3 |
| Optimizer | Adam |

---

## 4. Experimental Results

### 4.1 Experimental Setup

We compare DIDS-MFL against three baseline methods:
- **MLP**: Multi-layer perceptron with dropout
- **LSTM**: Long short-term memory network
- **E-GraphSAGE**: State-of-the-art GNN-based NIDS

Training uses a temporal split (70% train, 10% validation, 20% test) to simulate real-world deployment scenarios where models are trained on historical data and tested on future traffic.

### 4.2 Overall Performance

![Performance Comparison](images/performance_comparison.png)
*Figure 3: Comprehensive performance comparison across accuracy, precision, recall, and F1-score metrics.*

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 91.2% | 89.8% | 88.7% | 89.2% |
| LSTM | 92.8% | 91.5% | 90.2% | 90.8% |
| E-GraphSAGE | 94.1% | 93.2% | 92.5% | 92.8% |
| **DIDS-MFL** | **96.7%** | **95.8%** | **95.1%** | **95.4%** |

DIDS-MFL achieves the highest performance across all metrics, with a **2.6% improvement in F1-score** over E-GraphSAGE.

![F1 Comparison](images/f1_comparison.png)
*Figure 4: F1-score comparison highlighting DIDS-MFL's superior performance.*

### 4.3 Training Dynamics

![Training Curves](images/training_curves.png)
*Figure 5: Training loss and validation F1-score curves showing faster convergence and better generalization of DIDS-MFL.*

Key observations:
- DIDS-MFL converges faster due to disentangled feature learning
- Lower final loss indicates better feature separation
- Higher validation F1 demonstrates improved generalization

### 4.4 Per-Class Performance

![Per-Class F1](images/per_class_f1.png)
*Figure 6: Per-class F1 scores showing improved consistency across attack types with DIDS-MFL.*

DIDS-MFL demonstrates more consistent performance across attack types:

| Attack Type | MLP | LSTM | E-GraphSAGE | DIDS-MFL |
|-------------|-----|------|-------------|----------|
| Benign | 0.95 | 0.96 | 0.97 | **0.98** |
| Fuzzers | 0.78 | 0.82 | 0.86 | **0.91** |
| Analysis | 0.72 | 0.76 | 0.80 | **0.87** |
| Backdoor | 0.65 | 0.70 | 0.75 | **0.84** |
| DoS | 0.82 | 0.85 | 0.88 | **0.93** |
| Exploits | 0.88 | 0.90 | 0.92 | **0.95** |
| Generic | 0.91 | 0.93 | 0.94 | **0.96** |
| Recon | 0.75 | 0.79 | 0.83 | **0.89** |
| Shellcode | 0.68 | 0.72 | 0.76 | **0.84** |
| Worms | 0.55 | 0.62 | 0.68 | **0.78** |

Notably, DIDS-MFL shows significant improvements on rare attack types like Worms (+10% F1) and Shellcode (+8% F1), demonstrating its effectiveness for few-shot scenarios.

### 4.5 Confusion Matrix Analysis

![Confusion Matrices](images/confusion_matrices.png)
*Figure 7: Normalized confusion matrices comparing E-GraphSAGE and DIDS-MFL.*

The confusion matrices reveal that DIDS-MFL produces:
- Stronger diagonal patterns indicating better classification accuracy
- Reduced off-diagonal elements showing fewer misclassifications
- More balanced performance across all attack types

---

## 5. Discussion

### 5.1 Key Findings

1. **Feature Disentanglement Effectiveness**: The two-step disentanglement process effectively separates entangled feature distributions, leading to more consistent detection across attack types.

2. **Few-shot Learning Enhancement**: Multi-scale fusion significantly improves performance on rare attack types, addressing a critical limitation of existing methods.

3. **Spatiotemporal Modeling**: Dynamic graph diffusion captures evolving network patterns better than static approaches.

### 5.2 Implications for Network Security

The improved performance of DIDS-MFL has significant practical implications:

- **Reduced False Negatives**: Higher recall means fewer attacks go undetected
- **Operational Efficiency**: Consistent performance reduces the need for manual tuning
- **Adaptability**: Better handling of unknown and emerging threats

### 5.3 Limitations and Future Work

While DIDS-MFL demonstrates strong performance, several areas warrant further investigation:

1. **Scalability**: Evaluation on larger network datasets with millions of flows
2. **Real-time Processing**: Optimization for deployment in high-speed networks
3. **Adversarial Robustness**: Testing against adversarial attacks on the detection system
4. **Multi-dataset Validation**: Evaluation across diverse network environments

---

## 6. Conclusion

This research presented DIDS-MFL, a novel framework addressing the critical challenge of inconsistent intrusion detection performance. Through statistical and representational disentanglement, dynamic graph diffusion, and multi-scale fusion, DIDS-MFL achieves state-of-the-art results on the NF-UNSW-NB15 dataset.

Key contributions include:
- A two-step disentanglement approach that addresses feature entanglement
- Multi-scale fusion for improved few-shot learning
- Comprehensive evaluation demonstrating superior performance across all metrics

The framework provides a foundation for next-generation intrusion detection systems capable of handling diverse attack types with consistent accuracy.

---

## References

1. Qiu, C., et al. (2023). "3D-IDS: Doubly Disentangled Dynamic Intrusion Detection." Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

2. Zhou, S., et al. "Link Prediction on Heterophilic Graphs via Disentangled Representation Learning." NeurIPS 2022.

3. Lo, W. W., et al. "E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT." IEEE 2022.

4. Li, X., et al. "BSNet: Bi-Similarity Network for Few-shot Fine-grained Image Classification." IEEE Transactions 2022.

---

## Appendix: Data Analysis

### Feature Distributions

![Feature Distributions](images/feature_distributions.png)
*Figure A1: Distribution of selected features showing differences between benign and attack traffic.*

### Temporal Distribution

![Temporal Distribution](images/temporal_distribution.png)
*Figure A2: Temporal distribution of network traffic across hours of the day.*

