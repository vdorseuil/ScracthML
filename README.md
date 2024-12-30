# ScratchML

Welcome to **ScratchML** a repository showcasing machine learning architectures implemented entirely from scratch (some only in Numpy). This project is a journey to deepen my understanding of Python, machine learning, and various libraries by implementing popular ML architectures from scratch.

## Disclaimer

The goal of this repository is not to create state-of-the-art models, but to demonstrate the implementation of various machine learning architectures. Due to computation time and limited resources, some implementations may lead to suboptimal results, especially for larger models.

## Repository Structure

The repository is organized into subfolders, each representing a different machine learning architecture or category. Below is an overview of the available architectures:

### 1. [Neural Networks](./neural_networks/)
- Basic feedforward neural networks just using numpy.
- Backpropagation algorithm implementation from scratch, with optimizer implementation.
- Trained and evaluated on MNIST.

<!-- ### 2. [Convolutional Neural Networks (CNNs)](./cnns/)
- Basic convolutional layers
- Pooling layers
- CNN architectures like LeNet, AlexNet, and more

### 3. [Recurrent Neural Networks (RNNs)](./rnns/)
- Vanilla RNNs
- LSTMs and GRUs
- Sequence-to-sequence models -->

### 2. [Transformers](./transformers/)
- Full transformer implemenation in pytorch from scratch.
- Encoder-decoder structure with attention mechanism.
- Trained on a basic translation task.

More implementations are comming soon !
<!-- ### 5. [Generative Models](./generative_models/)
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Normalizing Flows -->

<!-- ### 6. [Graph Neural Networks (GNNs)](./gnns/)
- Graph Convolutional Networks (GCNs)
- Graph Attention Networks (GATs)
- Applications to node classification and graph embeddings

### 7. [Reinforcement Learning Architectures](./reinforcement_learning/)
- Deep Q-Networks (DQN)
- Policy Gradient methods
- Actor-Critic models -->

## Getting Started

### Prerequisites

To run the code in this repository, ensure you have the following installed:

- Python 3.8+
- Common ML libraries: NumPy, Pandas, Scikit-learn
- Deep learning frameworks: PyTorch
- Visualization tools: Matplotlib, Seaborn

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

### Running an Example
1. Navigate to the desired architecture folder.
2. Follow the instructions in the folder-specific `README.md` or `notebook`.
3. Run the scripts to see the implementation in action.

## Contributing
Contributions are welcome! If you want to add a new architecture or improve an existing one, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeatureName`.
3. Commit your changes: `git commit -m 'Add your feature'`.
4. Push to the branch: `git push origin feature/YourFeatureName`.
5. Submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact
For questions or suggestions, feel free to reach out:
- **Email:** valentin.dorseuil@gmail.com
- **LinkedIn:** [Valentin Dorseuil](https://linkedin.com/in/valentin-dorseuil)

---

Happy learning and coding!
