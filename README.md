# CARCA

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