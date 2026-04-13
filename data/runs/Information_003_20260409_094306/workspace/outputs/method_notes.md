# Local DIDS-MFL Approximation

- Statistical disentanglement: split 40 flow features into three PCA-compressed subspaces.
- Dynamic/topological context: previous source/destination/pair counts, cyclic time encoding, and rolling source/destination intensity.
- Multi-scale fusion: concatenate all subspaces into a joint representation.
- Few-shot inference: dual-similarity prototype scoring with cosine and Euclidean similarity.
- Unknown attack evaluation: hold out the rarest malicious class during training and threshold the maliciousness score on benign calibration flows.
