# CARCA: Context and Attribute-Aware Next-Item Recommendation via Cross-Attention

Replication of a state-of-the-art sequential recommendation method presented on RecSys2022. CARCA [1] is a context and attribute aware sequential model utilizing attention mechanism and Transformer [2] architecture to capture complex non-linear relations between all items in the sequence and candidate items to be recommended.

## Requirements

```bash
python==3.9.15
numpy==1.23.5
pytorch==1.13.0
```

## Citation
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

booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},

pages = {1305–1311},

numpages = {7},

keywords = {Personalization, Product embedding, Complementary Product Recommendation, E-commerce},

location = {Singapore, Singapore},

series = {RecSys '23}
}

## References

[1] Ahmed Rashed, Shereen Elsayed, and Lars Schmidt-Thieme. 2022. Context and Attribute-Aware Sequential Recommendation via Cross-Attention. In Proceedings of the 16th ACM Conference on Recommender Systems (RecSys '22). Association for Computing Machinery, New York, NY, USA, 71–80. https://doi.org/10.1145/3523227.3546777

[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 6000–6010.
