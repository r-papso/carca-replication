# CARCA: Context and Attribute-Aware Next-Item Recommendation via Cross-Attention

Replication of a state-of-the-art sequential recommendation method presented on RecSys2022. CARCA [1] is a context and attribute aware sequential model utilizing attention mechanism and Transformer [2] architecture to capture complex non-linear relations between all items in the sequence and candidate items to be recommended.

## Requirements

```bash
python==3.9.15
numpy==1.23.5
pytorch==1.13.0
```

## References

[1] Ahmed Rashed, Shereen Elsayed, and Lars Schmidt-Thieme. 2022. Context and Attribute-Aware Sequential Recommendation via Cross-Attention. In Proceedings of the 16th ACM Conference on Recommender Systems (RecSys '22). Association for Computing Machinery, New York, NY, USA, 71–80. https://doi.org/10.1145/3523227.3546777

[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 6000–6010.

### Citation
In case of using the code or a part of the code, feel free to cite the paper [Complementary Product Recommendation for Long-tail Products](https://dl.acm.org/doi/10.1145/3604915.3608864).

@inproceedings{10.1145/3604915.3608864,
author = {Papso, Rastislav},
title = {Complementary Product Recommendation for Long-Tail Products},
year = {2023},
isbn = {9798400702419},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3604915.3608864},
doi = {10.1145/3604915.3608864},
abstract = {Identifying complementary relations between products plays a key role in e-commerce Recommender Systems (RS). Existing methods in Complementary Product Recommendation (CPR), however, focus only on identifying complementary relations in huge and data-rich catalogs, while none of them considers real-world scenarios of small and medium e-commerce platforms with limited number of interactions. In this paper, we discuss our research proposal that addresses the problem of identifying complementary relations in such sparse settings. To overcome the data sparsity problem, we propose to first learn complementary relations in large and data-rich catalogs and then transfer learned knowledge to small and scarce ones. To be able to map individual products across different catalogs and thus transfer learned relations between them, we propose to create Product Universal Embedding Space (PUES) using textual and visual product meta-data, which serves as a common ground for the products from arbitrary catalog.},
booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
pages = {1305–1311},
numpages = {7},
keywords = {Personalization, Product embedding, Complementary Product Recommendation, E-commerce},
location = {Singapore, Singapore},
series = {RecSys '23}
}
