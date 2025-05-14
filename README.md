# Abstract

Visual transformers have achieved remarkable performance in image classification tasks, but this performance gain has come at the cost of interpretability. One of the main obstacles to the interpretation of transformers is the self-attention mechanism, which mixes visual information across the whole image in a complex way. In this paper, we propose Hindered Transformer (HiT), a novel interpretable by design architecture inspired by visual transformers. Our proposed architecture rethinks the design of transformers to better disentangle patch influences at the classification stage. Ultimately, HiT can be interpreted as a linear combination of patch-level information. We show that the advantages of our approach in terms of explicability come with a reasonable trade-off in performance, making it an attractive alternative for applications where interpretability is paramount.


---

## 📚 Citation

If you find this work useful, please cite us:

```bibtex
@InProceedings{jeanneret2025disentangling,
    author = {Jeanneret, Guillaume and Simon, Lo{"i}c and Jurie, Fr{'e}d{'e}ric},
    title = {Disentangling Visual Transformers: Patch-level Interpretability for Image Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2025}
}
```
