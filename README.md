# LLM4SNLI

LLM Training for SNLI

# TODO

## Basic Requirement

- [ ] Develop an adversarial training framework that supports BERT pre-trained LLM architectures, integrating the Fast Gradient Method (FGM) as the core perturbation generation technique to ensure effective adversarial sample generation during training.
- [ ] Achieve an accuracy improvement of at least 5% on the SNLI and MNLI datasets while reducing the adversarial sample error rate by over 10%, thereby validating the enhancement in model robustness.
- [ ] Enable loading and preprocessing for the Spatial Eval dataset as well as custom spatial reasoning datasets, with the development of data interfaces to ensure cross-dataset compatibility and consistency.
- [ ] Provide a basic performance evaluation module that generates statistical reports on accuracy, F1-score, and adversarial sample success rate to facilitate experimental analysis and comparison.

## Ideal Requirements

- [ ] Develop an interactive web interface featuring an input section (supporting NLI pairs or spatial descriptions), an output section (displaying prediction results and confidence scores), and a visualization section (depicting spatial relationship graphs or adversarial sample transformations) to enhance user experience.
- [ ] Implement an automated adversarial sample generation tool with configurable parameters (e.g., perturbation type and frequency) and batch processing capabilities to meet the demands of large-scale experimentation by researchers.
- [ ] If feasible, conduct a needs analysis by soliciting feedback from 5–10 computer science students or peers to refine interface design and functional prioritization.
