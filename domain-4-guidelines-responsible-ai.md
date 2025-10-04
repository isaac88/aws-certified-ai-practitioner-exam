# Domain 4: Guidelines for Responsible AI

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)

## 📋 Overview

This domain covers the ethical considerations and best practices for developing and deploying AI systems responsibly. Key topics include fairness, accountability, transparency, and the societal impact of AI technologies.

## 🎯 Key Topics Covered

### 4.1: Explain the development of AI systems that are responsible.
Objectives:
- Identify features of responsible AI (for example, bias, fairness, inclusivity,
robustness, safety, veracity).
- Understand how to use tools to identify features of responsible AI (for
example, Guardrails for Amazon Bedrock).
- Understand responsible practices to select a model (for example,
environmental considerations, sustainability).
- Identify legal risks of working with generative AI (for example, intellectual
property infringement claims, biased model outputs, loss of customer trust,
end user risk, hallucinations).
- Identify characteristics of datasets (for example, inclusivity, diversity,
curated data sources, balanced datasets).
- Understand effects of bias and variance (for example, effects on
demographic groups, inaccuracy, overfitting, underfitting).
- Describe tools to detect and monitor bias, trustworthiness, and truthfulness
(for example, analyzing label quality, human audits, subgroup analysis,
Amazon SageMaker Clarify, SageMaker Model Monitor, Amazon Augmented
AI [Amazon A2I]).

#### Describe tools to detect and monitor bias, trustworthiness, and truthfulness
- **SageMaker Model Monitor**: Continuously monitors machine learning models **deployed** in **production** to detect and analyze data drift, bias, and other issues that may affect model **performance**.
- **Amazon Augmented AI (Amazon A2I)**: A service that makes it easy to build workflows for **human review of machine learning predictions**. It helps ensure the accuracy and reliability of AI systems by incorporating human judgment into the decision-making process.
- **Sagemaker Clarify**: A tool that helps **detect bias** in machine learning models and datasets. It provides insights into model behavior and **helps ensure fairness and transparency** in AI systems.

#### Identify features of responsible AI
- [Transform responsible AI from theory into practice](https://aws.amazon.com/ai/responsible-ai/)
- Sagemaker governance tools to ensure models are used responsibly:
  - **Amazon SageMaker Model Cards**: Provide a standardized way to document important information about machine learning models, including their intended use, performance metrics, and limitations. This helps ensure transparency and accountability in AI systems.
  - **SageMaker Model Dashboards**: Offer visualizations and insights into model performance, helping users monitor and manage their machine learning models effectively.
  - **SageMaker Role Manager**: Simplifies the management of permissions and access control for SageMaker resources, ensuring that only authorized users can interact with sensitive data and models.

### 4.2: Recognize the importance of transparent and explainable
models.
Objectives:
- Understand the differences between models that are transparent and
explainable and models that are not transparent and explainable.
- Understand the tools to identify transparent and explainable models (for
example, Amazon SageMaker Model Cards, open source models, data,
licensing).
- Identify tradeoffs between model safety and transparency (for example,
measure interpretability and performance).
- Understand principles of human-centered design for explainable AI.
