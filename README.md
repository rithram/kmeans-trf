# Transformer Circuits Can Realize Clustering Algorithms

<div align="center">

### Kenneth L. Clarkson<sup>🏢</sup> · Lior Horesh<sup>🏢</sup> · Takuya Ito<sup>🏢</sup> · Charlotte Park<sup>🎓</sup> · Parikshit Ram<sup>🏢</sup>

<sub><sup>🏢</sup> <strong>IBM Research</strong> &nbsp;&nbsp; <sup>🎓</sup> <strong>EECS, MIT</strong></sub>

</div>

<div align="center">

📄 **[Read the Paper (PDF)](assets/Clarkson_2026_Transformer_Clustering.pdf)** 📄

[![arXiv](https://img.shields.io/badge/arXiv-2506.19125-b31b1b.svg)](https://arxiv.org/abs/2506.19125)

### 🚀 Preliminary version on arXiv
## **[arXiv:2506.19125](https://arxiv.org/abs/2506.19125)**

</div>

> [!NOTE]
>  ⚠️ 🚧 **Under construction** <br>
Code, results and plotting scripts coming soon!

**Abstract.**
Although transformers are most commonly optimized as statistical sequence models, it is unclear to what extent they can implement and learn exact algorithmic computations. Here, we specify a transformer implementation from first principles that executes a fundamental and widely used method for $k$-means clustering: Lloyd's algorithm. We theoretically prove and empirically demonstrate that this implementation of a transformer architecture, which we term the _$k$-means transformer_, exactly implements Lloyd's algorithm for $k$-means clustering using the standard circuit mechanisms of modern transformers: attention block, residual connections, and feed-forward block. In learning experiments, we find that training this base architecture on $k$-means clustering yields a generalizable clustering algorithm that surpasses Lloyd's algorithm in terms of clustering quality. Finally, we demonstrate that interpretable alterations (e.g., inclusion of layer normalizations) to this architecture yields diverse and novel variants of clustering algorithms, including soft $k$-means, spherical $k$-means, trimmed $k$-means. Overall, our results show that transformer circuit mechanisms can instantiate exact algorithmic routines for clustering, while simultaneously providing an effective learnable model.

## Main Results

<div align="center">
  <img src="assets/klic-overview.png" alt="KLIC Overview" width="800"/>
  <p><em>Overview</em></p>
</div>




## Empirical Evaluations

### Environment setup

Assuming CUDA is properly setup on the machine, we will be using python version 3.13

```
> cd kmeans-trf
> conda create -n kmt
> conda activate kmt
> conda install python=3.13 pip>25.0
> conda install cudatoolkit -c anaconda  # <== OPTIONAL: if we have access to a GPU
> pip install -r requirements.txt
```

### Experimental details

The details on running the scripts for the training and evaluation is in [expts.md](./expts.md). 