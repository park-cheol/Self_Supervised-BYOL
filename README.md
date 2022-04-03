# Bootstrap your own latent: A new approach to self-supervised Learning

1. **Online Network**와 **Teacher Network**를 통해 새로운 방법으로 Representation Learning 수행
2. **Negative Examples**을 사용하지 않고 학습 - Batch Size/Augmentation에 Robust

    - Contrastive Learning은 Negative Examples을 사용하므로 Batch Size, Augmentation에 민감[[SimCLR](https://arxiv.org/abs/2002.05709)]

3. Online Network Parameter만 Gradient를 구해서 Update / Teacher Network Parameter는 **Online Network Parameter의 slow-moving average**
통해 Update

* BYOL하고 비슷한 컨셉인 [Mean-Teachers](https://arxiv.org/abs/1703.01780) 참고

# Method

![im1](https://user-images.githubusercontent.com/76771847/161025114-35b73611-1137-4fbf-8e9b-061c66aa3b86.png)

- f(base): ResNetx50
- Projection, Prediction: MLP-Norm-RELU-MLP

**Loss & Update Parameter**

![Loss](https://user-images.githubusercontent.com/76771847/161025113-4216b071-e4eb-47cf-aad6-08970e868a1d.png)
![Update](https://user-images.githubusercontent.com/76771847/161025109-3b02c6da-7c6a-4c78-ae86-92399932252b.png)

- Online Network Output이 Teacher Network Output Predict(MSE)
- Online Parameter: Gradient Update
- Teacher Parameter: slow-moving average Update

# Implementation

- **Train BYOL**
> **python main.py --gpu 0 --batch-size 128**

- **Linear Evaluation**
> **python linear_evaluation.py --gpu 0 --batch-size 512 --resume saved_models/checkpoint.pth**

# Reference
[BYOL Paper](https://arxiv.org/abs/2006.07733)

[BYOL Official Code](https://github.com/deepmind/deepmind-research/tree/master/byol)

[BYOL Code 1](https://github.com/lucidrains/byol-pytorch)

[BYOL Code2 ](https://github.com/sthalles/PyTorch-BYOL)
