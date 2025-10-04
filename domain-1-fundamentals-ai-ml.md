# Domain 1: Fundamentals of AI and ML (20%)

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)

## 📋 Overview

This domain covers the fundamental concepts of Artificial Intelligence and Machine Learning, representing **20%** of the AWS Certified AI Practitioner exam. Understanding these core concepts is essential for working with AI/ML solutions on AWS.

## 🎯 Key Topics Covered

### 1.1: Explain basic AI concepts and terminologies
Objectives:
- Define basic AI terms (for example, AI, ML, deep learning, neural networks,
    computer vision, natural language processing [NLP], model, algorithm,
    training and inferencing, bias, fairness, fit, large language model [LLM]).
- Describe the similarities and differences between AI, ML, and deep learning.
- Describe various types of inferencing (for example, batch, real-time).
- Describe the different types of data in AI models (for example, labeled and
    unlabeled, tabular, time-series, image, text, structured and unstructured).
- Describe supervised learning, unsupervised learning, and reinforcement
    learning.

#### Define basic AI terms
- [**Artificial Intelligence (AI)**](https://aws.amazon.com/what-is/artificial-intelligence/)
- [**Machine Learning (ML)**](https://aws.amazon.com/what-is/machine-learning/)
    - Machine Learning allows machines to learn patterns from data and make predictions or decisions without being explicitly programmed.
- [**Deep Learning**](https://aws.amazon.com/what-is/deep-learning/)
    - [**What is the difference between machine learning and deep learning?**](https://aws.amazon.com/what-is/deep-learning/#ams#what-isc2#pattern-data)
- [**Neural Networks**](https://aws.amazon.com/what-is/neural-network/)
- [**Computer Vision**](https://aws.amazon.com/what-is/computer-vision/)
- [**Natural Language Processing (NLP)**](https://aws.amazon.com/what-is/nlp/)
- **Models**
    - [**What are Foundation Models?**](https://aws.amazon.com/what-is/foundation-models/)
    - [**What are Large Language Models?**](https://aws.amazon.com/what-is/large-language-model/)
    - [**How do generative AI models work?**](https://aws.amazon.com/what-is/generative-ai/#ams#what-isc9#pattern-data)
    - [**Foundation Models vs Large Language Models**](https://www.openxcell.com/blog/foundation-model-vs-llm/)
- [**What are the types of machine learning algorithms?**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc8#pattern-data)
    - Supervised machine learning
    - Unsupervised machine learning
    - Reinforcement learning
    - Semi-supervised learning
    - Deep learning
- [**Training and Inference in Machine Learning**](https://www.clarifai.com/blog/training-vs-inference)
    - [AI inference vs. training: What is AI inference?](https://www.cloudflare.com/learning/ai/inference-vs-training/)
- [**Bias in AI and Machine Learning**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc11#pattern-data)
- [**Fairness in Machine Learning**](https://pages.awscloud.com/rs/112-TZM-766/images/Amazon.AI.Fairness.and.Explainability.Whitepaper.pdf)
    An algorithm is fair if it makes predictions that do not favor or discriminate against certain individuals or groups based on sensitive characteristics.
    - [Bias vs Fairness vs Explainability in AI](https://www.seldon.io/bias-vs-fairness-vs-explainability-in-ai/)
    - https://medium.com/@nay1228/model-fitting-bias-and-fairness-in-ai-aws-practitioners-guide-456df4720497
- **Fit in Machine Learning**
    - [**What is Overfitting?**](https://aws.amazon.com/what-is/overfitting/#ams#what-isc1#pattern-data)
    Overfit models experience **high variance** — they give accurate results for the training set but not for the test set
    - [**What is Underfitting?**](https://aws.amazon.com/what-is/overfitting/#ams#what-isc5#pattern-data)
    Underfit models experience **high bias** — they give inaccurate results for both the training data and test set.
    **Balanced** models experience low bias and low variance.
        Neither overfitting or underfitting is desirable.
- [**MLU-EXPLAIN**](https://mlu-explain.github.io/)
- [What is Transfer Learning?](https://aws.amazon.com/what-is/transfer-learning/)

#### Describe the similarities and differences between AI, ML, and deep learning
- [**What’s the Difference Between AI and Machine Learning?**](https://aws.amazon.com/compare/the-difference-between-artificial-intelligence-and-machine-learning/)
- [**What is the difference between machine learning, deep learning, and artificial intelligence?**](https://aws.amazon.com/what-is/artificial-intelligence/#ams#what-isc3#pattern-data)

#### Describe various types of inferencing (for example, batch, real-time)

- [Inference options in Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model-options.html)

#### Describe the different types of data in AI models (for example, labeled and unlabeled, tabular, time-series, image, text, structured and unstructured)
- [**Labeled**](https://aws.amazon.com/what-is/data-labeling/)
- [**Unlabeled**](https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/)

Structured Data:
Data that adheres to a predefined schema or format, making it easily searchable and analyzable. Examples include:
- Tabular Data
- Time-Series Data

Unstructured Data:
Data that does not have a predefined structure or format, making it more complex to analyze. Examples include:
- Image Data (Photos, Videos, Medical Imaging, etc.)
- Audio Data (Voice Recordings, Music, etc.)
- Text Data (Articles, Social Media, Reviews, etc.)

#### Describe supervised learning, unsupervised learning, and reinforcement learning
- [**Supervised Learning**](https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/)
- [**Unsupervised Learning**](https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/)
- [**Reinforcement Learning**](https://aws.amazon.com/what-is/reinforcement-learning/)

### 1.2: Identify practical use cases for AI

Objectives:
- Recognize applications where AI/ML can provide value (for example, assist
human decision making, solution scalability, automation).
- Determine when AI/ML solutions are not appropriate (for example, costbenefit analyses, situations when a specific outcome is needed instead of a
prediction).
- Select the appropriate ML techniques for specific use cases (for example,
regression, classification, clustering).
- Identify examples of real-world AI applications (for example, computer
vision, NLP, speech recognition, recommendation systems, fraud detection,
forecasting).
- Explain the capabilities of AWS managed AI/ML services (for example,
SageMaker, Amazon Transcribe, Amazon Translate, Amazon Comprehend,
Amazon Lex, Amazon Polly).

#### Recognize applications where AI/ML can provide value
- [**What are the benefits of AI for business transformation?**](https://aws.amazon.com/what-is/artificial-intelligence/#ams#what-isc9#pattern-data)
- [**What are the benefits of machine learning?**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc6#pattern-data)
- [**What are machine learning use cases?**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc7#pattern-data)
- [**AI Use Cases**](https://aws.amazon.com/machine-learning/ai-use-cases/)

#### Determine when AI/ML solutions are not appropriate
- [**What are the challenges in artificial intelligence implementation?**](https://aws.amazon.com/what-is/artificial-intelligence/#ams#what-isc12#pattern-data)
- [**What are the challenges in machine learning implementation?**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc11#pattern-data)
- [**How can you implement machine learning in your organization?**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc10#pattern-data)
- [**Are machine learning models deterministic?**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc9#pattern-data)

#### Select the appropriate ML techniques for specific use cases
- [**What are the types of machine learning algorithms?**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc8#pattern-data)
- [**Types of Algorithms**](https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html)
- [**Built-in algorithms and pretrained models in Amazon SageMaker**](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
- [**Problem types for the basic machine learning paradigms**](https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html#algorithms-choose-problem-types)

#### Identify examples of real-world AI applications
- [**What are machine learning use cases?**](https://aws.amazon.com/what-is/machine-learning/#what-are-machine-learning-use-cases)
- [**What is Computer Vision?**](https://aws.amazon.com/what-is/computer-vision/)
- [**What is Natural Language Processing (NLP)?**](https://aws.amazon.com/what-is/nlp/)
- [**What is Speech to Text?**](https://aws.amazon.com/what-is/speech-to-text/)
- [**AI Use Cases**](https://aws.amazon.com/machine-learning/ai-use-cases/)

#### Explain the capabilities of AWS managed AI/ML services
- [**AI Services Overview**](https://aws.amazon.com/ai/services/)
- [**Amazon SageMaker**](https://aws.amazon.com/sagemaker/)
    - [**Amazon SageMaker Ground Truth**](https://aws.amazon.com/sagemaker/groundtruth/)
- [**Amazon Transcribe**](https://aws.amazon.com/transcribe/)
- [**Amazon Translate**](https://aws.amazon.com/translate/)
- [**Amazon Comprehend**](https://aws.amazon.com/comprehend/)
    - [Real-time analysis using the API](https://docs.aws.amazon.com/comprehend/latest/dg/using-api-sync.html)
    - [Custom Classification](https://docs.aws.amazon.com/comprehend/latest/dg/how-document-classification.html)
    - [Detect specific entities](https://docs.aws.amazon.com/comprehend/latest/dg/how-entities.html)
    - [Identifying the sentiment](https://docs.aws.amazon.com/comprehend/latest/dg/how-sentiment.html)
- [**Amazon Lex**](https://aws.amazon.com/lex/)
    - [Session Attributes](https://docs.aws.amazon.com/lexv2/latest/dg/context-mgmt-session-attribs.html)
- [**Amazon Polly**](https://aws.amazon.com/polly/)
    - [Speech Synthesis Markup Language (SSML)](https://docs.aws.amazon.com/polly/latest/dg/ssml.html)
- [***Amazon Kendra**](https://docs.aws.amazon.com/kendra/latest/dg/what-is-kendra.html)
  - [Semantically ranking a search service's results](https://docs.aws.amazon.com/kendra/latest/dg/search-service-rerank.html)
  - [Data Sources](https://docs.aws.amazon.com/kendra/latest/dg/hiw-data-source.html)
    - ❌ AWS DynamoDB is not a valid data source for Amazon Kendra.
- [**Amazon Personalize**](https://aws.amazon.com/personalize/)
- [**Amazon Textract**](https://aws.amazon.com/textract/)
    - Confidence scores
    - Form extraction: Form extraction in Amazon Textract is used to extract data from forms and documents that have a structured layout, such as applications, tax forms, and surveys.
    - Key-value pairs extraction(Form Analysis): Key-value pair extraction in Amazon Textract identifies and extracts pairs of related information from documents, such as "Name: John Doe" or "Date: 01/01/2023". Key-value pairs extraction in Amazon Textract is specifically designed to extract structured data from documents by identifying relationships between keys (labels) and their corresponding values.
- [**Amazon Fraud Detector**](https://docs.aws.amazon.com/frauddetector/latest/ug/what-is-frauddetector.html)
    - [Online Fraud Insights](https://docs.aws.amazon.com/frauddetector/latest/ug/online-fraud-insights.html)
- [** Amazon Rekognition**](https://docs.aws.amazon.com/rekognition/latest/dg/what-is.html)
    - [Searching faces in a collection](https://docs.aws.amazon.com/rekognition/latest/dg/collections.html)
- [**Amazon Macie**](https://docs.aws.amazon.com/macie/latest/user/what-is-macie.html)
    - Automated Data Discovery: Amazon Macie uses machine learning to automatically discover, classify, and protect sensitive data in AWS.

### 1.3: Describe the ML development lifecycle
Objectives:
- Describe components of an ML pipeline (for example, data collection,
exploratory data analysis [EDA], data pre-processing, feature engineering,
model training, hyperparameter tuning, evaluation, deployment,
monitoring).
- Understand sources of ML models (for example, open source pre-trained
models, training custom models).
- Describe methods to use a model in production (for example, managed API
service, self-hosted API).
- Identify relevant AWS services and features for each stage of an ML pipeline
(for example, SageMaker, Amazon SageMaker Data Wrangler, Amazon
SageMaker Feature Store, Amazon SageMaker Model Monitor).
- Understand fundamental concepts of ML operations (MLOps) (for example,
experimentation, repeatable processes, scalable systems, managing
technical debt, achieving production readiness, model monitoring, model
re-training).
- Understand model performance metrics (for example, accuracy, Area Under
the ROC Curve [AUC], F1 score) and business metrics (for example, cost per
user, development costs, customer feedback, return on investment [ROI]) to
evaluate ML models.

#### Describe components of an ML pipeline
- **Amazon SageMaker AI pipeline** is a series of interconnected steps in directed acyclic graph (DAG).
- **Data collection**: Data preparation in machine learning refers to the process of collecting, preprocessing, and organizing raw data to make it suitable for analysis and modeling.
- **Data Cleaning**: The process of identifying and correcting (or removing) errors and inconsistencies in data to improve its quality.(missing values and outliers)
- **Exploratory data analysis (EDA)**:
    - [Perform exploratory data analysis (EDA)](https://docs.aws.amazon.com/sagemaker/latest/dg/canvas-analyses.html)
    - https://medium.com/@tantabase/your-guide-to-exploratory-data-analysis-9234aa4bd775
    - https://medium.com/@tantabase/aws-certified-machine-learning-cheat-sheet-eda-02262f1ee26e
- **Data pre-processing**: fill in missing values, normalize numerical data, or split data into the train, validation, and test datasets.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/data-prep.html#data-prep-choose-recommended
- **Feature engineering**: The process of using domain knowledge to select and transform raw data into meaningful features.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
- **Model training**: Configures hyperparameters and trains a model from the preprocessed input data.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html
- **Hyperparameter tuning**: https://aws.amazon.com/what-is/hyperparameter-tuning/
    - Amazon SageMaker AI automatic model tuning (AMT) is also known as hyperparameter tuning
    - https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-considerations.html
- **Evaluation**: After training your machine learning model, you need to evaluate its performance using various metrics.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/model-explainability.html
    - https://docs.aws.amazon.com/sagemaker/latest/dg/canvas-evaluate-model.html
- **Deployment**: After you train your machine learning model, you can deploy it using Amazon SageMaker AI to get predictions.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html
    - https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html
    - [Inference options in Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model-options.html)
- **Monitoring**: Once your model is deployed, you can monitor its performance and accuracy over time using Amazon SageMaker AI Model Monitor.
    - Amazon SageMaker Model Monitor
    - https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-mlops.html

#### Understand sources of ML models (for example, open source pre-trained models, training custom models)
- Pre-trained model: A model that has been previously trained on a large dataset and can be fine-tuned for specific tasks.
- SageMaker JumpStart
- https://aws.amazon.com/marketplace/solutions/machine-learning/pre-trained-models
- https://aws.amazon.com/marketplace/b/c3714653-8485-4e34-b35b-82c2203e81c1?category=c3714653-8485-4e34-b35b-82c2203e81c1&PRICING_MODEL=FREE&filters=PRICING_MODEL
- https://docs.aws.amazon.com/sagemaker/latest/dg/canvas-build-model.html

#### Describe methods to use a model in production (for example, managed API service, self-hosted API)
- https://docs.aws.amazon.com/sagemaker/latest/dg/model-deploy-mlops.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/model-ab-testing.html
#### Identify relevant AWS services and features for each stage of an ML pipeline
- Amazon SageMaker AI: https://aws.amazon.com/sagemaker/ai/
- Amazon SageMaker Studio: https://aws.amazon.com/sagemaker/ai/studio/
- Amazon SageMaker Canvas: Is a visual low-code environment for building, training, and deploying machine learning models in SageMaker AI
    - https://aws.amazon.com/sagemaker/ai/canvas/
- Amazon SageMaker Data Wrangler: Simplifies the process of data preparation and feature engineering.
    - https://aws.amazon.com/sagemaker/ai/data-wrangler/
- Amazon SageMaker Feature Store: A fully managed repository to store, update, retrieve, and share machine learning features.
    - https://aws.amazon.com/sagemaker/ai/feature-store
- Amazon SageMaker Model Monitor: Continuously monitors the quality of machine learning models in production.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html

#### Understand fundamental concepts of ML operations (MLOps)
- [**What is MLOps?**](https://aws.amazon.com/what-is/mlops/)
- [**Amazon SageMaker for MLOps**](https://aws.amazon.com/sagemaker/mlops/)
- [**SageMaker Pipelines**](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [**SageMaker Model Registry**](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [**SageMaker Model Monitor**](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-overview.html)
- [**SageMaker MLflow**](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html)

#### Understand model performance metrics and business metrics to evaluate ML models
- [**How does machine learning work?**](https://aws.amazon.com/what-is/machine-learning/#how-does-machine-learning-work)
- [**SageMaker Model Quality Monitoring**](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html)
- [**Post-training Data and Model Bias Metrics**](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-post-training-bias.html)
- [**Model Quality Metrics and CloudWatch Monitoring**](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
- Model performance metrics:
    - [**Accuracy**](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html#evaluations)
    - [**Precision and Recall**](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html#evaluations)
    - [**F1 Score**](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html#evaluations)
    - [**Area Under the ROC Curve (AUC)**](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html#evaluations)
- Business metrics:
    - [**Cost per User**]
    - [**Development Costs**]
    - [**Customer Feedback**]
    - [**Return on Investment (ROI)**]

## 🔗 Related Domains

- [Domain 2: Fundamentals of Generative AI](domain-2-fundamentals-generative-ai.md) *(Coming Soon)*
- [Domain 3: Applications of Foundation Models](domain-3-applications-foundation-models.md) *(Coming Soon)*
- [Domain 4: Guidelines for Responsible AI](domain-4-guidelines-responsible-ai.md) *(Coming Soon)*
- [Domain 5: Security, Compliance, and Governance](domain-5-security-compliance-governance.md) *(Coming Soon)*

---

**📝 Note**: This study guide is continuously updated. Check back regularly for new content and improvements!

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)