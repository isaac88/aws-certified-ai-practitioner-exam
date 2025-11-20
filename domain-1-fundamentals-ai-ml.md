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
    - Computer Vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos, and other visual inputs, and take actions or make recommendations based on that information.
        - Object detection: Identifying and locating objects within an image or video.
        - CNN (Convolutional Neural Networks): Used for **single-image** analysis.
            - Examples: Image classification, object detection, image segmentation.
        - RNN (Recurrent Neural Networks): Used for **video analysis**.
            - Examples: Activity recognition, video classification, video captioning.
- [**Natural Language Processing (NLP)**](https://aws.amazon.com/what-is/nlp/)
- **Models**
    - [**What are Foundation Models?**](https://aws.amazon.com/what-is/foundation-models/)
        - **Self-supervised learning** -> **Foundation models use self-supervised learning.** Foundation models use self-supervised learning to create labels from unlabeled input data. In self-supervised learning, models are provided vast amounts of raw completely unlabeled data and then the **models generate the labels themselves**. This means no one has instructed or trained the model with labeled training data sets.
    - [**What are Large Language Models?**](https://aws.amazon.com/what-is/large-language-model/)
    - [**How do generative AI models work?**](https://aws.amazon.com/what-is/generative-ai/#ams#what-isc9#pattern-data)
    - [**Foundation Models vs Large Language Models**](https://www.openxcell.com/blog/foundation-model-vs-llm/)
- [**What are the types of machine learning algorithms?**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc8#pattern-data)
    - **Supervised machine learning**
        - KNN (K-Nearest Neighbors): A simple, instance-based learning algorithm used for **classification** and regression tasks. It classifies new data points based on the majority class of their k-nearest neighbors in the feature space.
        - SVM (Support Vector Machines): A supervised learning algorithm used for **classification** and regression tasks. It works by finding the optimal hyperplane that separates data points of different classes in the feature space.
    - **Unsupervised machine learning**
        - K-Means Clustering: A popular unsupervised learning algorithm **used for clustering tasks**. It partitions a dataset into k distinct clusters based on feature similarity, where each data point belongs to the cluster with the nearest mean.
        It's not use for prediction classification or regression tasks.
    - **Reinforcement learning**
        - Agent: The entity that learns and makes decisions in a reinforcement learning environment based on the received rewards.
        - Reward: A feedback signal that indicates the success or failure of an agent's actions in achieving its goals within the environment.
        - AWS DeepRacer use case. The AWS DeepRacer vehicle is a Wi-Fi-enabled 1/18th scale autonomous race car designed to test reinforcement learning (RL) models by racing them around a physical or virtual track.
    - **Semi-supervised learning**: Semi-supervised learning is when you apply both supervised and unsupervised learning techniques to a common problem.
        - Sentiment analysis: Example of semi-supervised learning.
        - Fraud detection: Example of semi-supervised learning.
    - **Deep learning**
- [**Training and Inference in Machine Learning**](https://www.clarifai.com/blog/training-vs-inference)
    - [AI inference vs. training: What is AI inference?](https://www.cloudflare.com/learning/ai/inference-vs-training/)
    - **Inference**: The process of using a trained machine learning model to make predictions or decisions based on new, unseen data. The model uses its trained parameters to generate a prediction or output based on new input data provided by the user.
- [**Bias in AI and Machine Learning**](https://aws.amazon.com/what-is/machine-learning/#ams#what-isc11#pattern-data)
    - **Sampling bias**: Occurs when the training data is not representative of the overall population, leading to skewed model predictions.
    - **Measurement bias**: Arises from inaccuracies or inconsistencies in the data collection process, which can introduce errors into the training data.
    - **Observer bias**: Happens when human judgment or subjective interpretation influences the labeling or annotation of training data.
    - **Confirmation bias**: Occurs when the model is trained on data that reinforces existing beliefs or stereotypes, leading to biased predictions.
- [**Fairness in Machine Learning**](https://pages.awscloud.com/rs/112-TZM-766/images/Amazon.AI.Fairness.and.Explainability.Whitepaper.pdf)
    An algorithm is fair if it makes predictions that do not favor or discriminate against certain individuals or groups based on sensitive characteristics.
    - [Bias vs Fairness vs Explainability in AI](https://www.seldon.io/bias-vs-fairness-vs-explainability-in-ai/)
    - https://medium.com/@nay1228/model-fitting-bias-and-fairness-in-ai-aws-practitioners-guide-456df4720497
    - Shapley Values: 
        - A method from cooperative game theory used to explain the output of machine learning models by attributing the contribution of each feature to the final prediction.
        - Use Shapley values to explain individual predictions.
    - Partial Dependence Plots (PDPs): 
        - A graphical representation that shows the relationship between a feature and the predicted outcome of a machine learning model, while averaging out the effects of other features.
        - PDP to understand the model's behavior at a dataset level.
- **Fit in Machine Learning**
    - [**What is Overfitting?**](https://aws.amazon.com/what-is/overfitting/#ams#what-isc1#pattern-data)
    Overfit models experience **high variance** — they give accurate results for the training set but not for the test set
    - [**What is Underfitting?**](https://aws.amazon.com/what-is/overfitting/#ams#what-isc5#pattern-data)
    Underfit models experience **high bias** — they give inaccurate results for both the training data and test set.
    **Balanced** models experience low bias and low variance.
        Neither overfitting or underfitting is desirable.
- [**MLU-EXPLAIN**](https://mlu-explain.github.io/)
- [What is Transfer Learning?](https://aws.amazon.com/what-is/transfer-learning/)
- **GAN** (Generative Adversarial Networks): A class of machine learning frameworks designed for generative modeling tasks, where two neural networks (the **generator** and the **discriminator**) compete against each other to produce realistic data samples.
- **GPT** (Generative Pre-trained Transformer): A type of large language model (LLM) developed by OpenAI that uses transformer architecture to generate human-like text based on the input it receives.


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
    - **Logic regression**: A statistical method used in supervised learning for binary classification tasks, where the goal is to predict the probability of a binary outcome (e.g., yes/no, true/false) based on one or more input features.
    - **Decision trees**: A supervised learning algorithm that uses a tree-like model of decisions and their possible consequences to make **predictions** or **classifications** based on input data.
    A supervised learning algorithm that uses a tree-like model of decisions and their possible consequences to make predictions or **classifications** based on input data.
    - **Linear regression**: A supervised learning algorithm used for **predicting** a **continuous target variable** based on one or more input features by fitting a **linear relationship** between the input and output variables.
    - **Neural Networks**: A supervised learning algorithm inspired by the structure and function of the human brain, consisting of interconnected nodes (neurons) that process and learn from input data to make predictions or classifications.
- [**Unsupervised Learning**](https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/)
    - **Clustering**: An unsupervised learning technique that groups similar data points together based on their features or characteristics, allowing for the discovery of patterns or structures within the data without predefined labels.
    - **Dimensionality Reduction**: An unsupervised learning technique that reduces the number of features or variables in a dataset while preserving its essential structure and information, making it easier to analyze and visualize.
    - **Association Rule Learning**: An unsupervised learning technique that identifies relationships or patterns between different items or variables in a dataset, often used in market basket analysis to discover co-occurrence patterns.
    - **Document classification**: An unsupervised learning technique that categorizes documents into predefined classes or topics based on their content, without the need for labeled training data.
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
    - Algorithms selection is a manual process that requires understanding the problem type, data characteristics, and desired outcomes. **There is no automatic algorithm selection in Amazon SageMaker.**
    - [**Amazon SageMaker Ground Truth**](https://aws.amazon.com/sagemaker/groundtruth/)
        - Build highly accurate training datasets for machine learning quickly. 
        - Amazon SageMaker Ground Truth offers easy access to human labelers and provides them with built-in workflows and interfaces for common labeling tasks.
    - [**Amazon SageMaker Autopilot**](https://aws.amazon.com/sagemaker/ai/autopilot/)
- [**Amazon Transcribe**](https://aws.amazon.com/transcribe/)
  - [**Amazon Transcribe Medical**](https://aws.amazon.com/transcribe/medical/)
- [**Amazon Translate**](https://aws.amazon.com/translate/)
- [**Amazon Comprehend**](https://aws.amazon.com/comprehend/)
    - Amazon Comprehend ML capabilities can be used to detect and redact personally identifiable information (PII) in customer emails, support tickets, product reviews, social media..
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
    - OCR (Optical Character Recognition): The process of converting different types of documents, such as scanned paper documents, PDFs, or images captured by a digital camera, into editable and searchable data.
    - Confidence scores
    - Form extraction: Form extraction in Amazon Textract is used to extract data from forms and documents that have a structured layout, such as applications, tax forms, and surveys.
    - Key-value pairs extraction(Form Analysis): Key-value pair extraction in Amazon Textract identifies and extracts pairs of related information from documents, such as "Name: John Doe" or "Date: 01/01/2023". Key-value pairs extraction in Amazon Textract is specifically designed to extract structured data from documents by identifying relationships between keys (labels) and their corresponding values.
- [**Amazon Fraud Detector**](https://docs.aws.amazon.com/frauddetector/latest/ug/what-is-frauddetector.html)
    - [Online Fraud Insights](https://docs.aws.amazon.com/frauddetector/latest/ug/online-fraud-insights.html)
- [**Amazon Rekognition**](https://docs.aws.amazon.com/rekognition/latest/dg/what-is.html)
    - [Searching faces in a collection](https://docs.aws.amazon.com/rekognition/latest/dg/collections.html)
    - [Labeling images](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-labeling-images.html)
    - Amazon Rekognition offers pre-trained and customizable computer vision (CV) capabilities to extract information and insights from images and videos.
- [**Amazon Macie**](https://docs.aws.amazon.com/macie/latest/user/what-is-macie.html)
    - Automated Data Discovery: Amazon Macie uses machine learning to automatically discover, classify, and protect sensitive data in AWS.
- [**Amazon SageMaker JumpStart**](https://aws.amazon.com/sagemaker/ai/jumpstart/)
    - Pre-trained models are fully customizable and can be fine-tuned with your own data to meet specific use cases.
    - Continued Pre-Training: You can further pre-train foundation models on your own domain-specific data to improve their performance for specialized tasks.
    - We can compare models based on various factors, including accuracy, size, and inference latency.
- [**Amazon SageMaker Clarify**](https://aws.amazon.com/sagemaker/ai/clarify/)
    - Identify imbalances in data
    - Detect bias in models
    - Explain model predictions
    - Evaluation wizard and reports
- [**Amazon Forecast**](https://aws.amazon.com/forecast/)
- [**Amazon Augmented AI (A2I)**](https://aws.amazon.com/augmented-ai/)
    - Implement human review of ML predictions

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
        -  Involves examining the data through statistical summaries and visualizations to identify patterns, detect anomalies, and form hypotheses. This phase is crucial for understanding the dataset’s structure and characteristics, making it the most appropriate description of the current activities.
    - https://medium.com/@tantabase/your-guide-to-exploratory-data-analysis-9234aa4bd775
    - https://medium.com/@tantabase/aws-certified-machine-learning-cheat-sheet-eda-02262f1ee26e
- **Data pre-processing**: fill in missing values, normalize numerical data, or split data into the train, validation, and test datasets.
    - https://aws.amazon.com/blogs/machine-learning/create-train-test-and-validation-splits-on-your-data-for-machine-learning-with-amazon-sagemaker-data-wrangler/
    - https://docs.aws.amazon.com/sagemaker/latest/dg/data-prep.html#data-prep-choose-recommended
- **Feature engineering**: The process of using domain knowledge to select and transform raw data into meaningful features.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
- **Model training**: Model Training is the stage where the **data is split** into **training** and **validation sets**, and the model is fine-tuned to optimize its performance.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html
    -  **Training set**: Used to train the model.
    -  **Validation set (Optional)**: Used to tune the model's hyperparameters and evaluate its performance during training.
    -  **Test set**: Used to assess the final performance of the trained model on unseen data.
- **Hyperparameter tuning**: https://aws.amazon.com/what-is/hyperparameter-tuning/
    - **Amazon SageMaker AI automatic model tuning (AMT)** is also known as hyperparameter tuning.
        - Amazon SageMaker Automatic Model Tuning can automatically choose hyperparameter ranges, search strategy, maximum runtime of a tuning job, early stopping type for training jobs, number of times to retry a training job, and model convergence flag to stop a tuning job, based on the objective metric you provide.
    - https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-considerations.html
    - **Hyperparameter tuning techniques**
        - https://aws.amazon.com/what-is/hyperparameter-tuning/#ams#what-isc6#pattern-data
        - **Grid Search**: A systematic approach that exhaustively searches through a predefined set of hyperparameter combinations to find the optimal configuration for a machine learning model.
        - **Random Search**: A hyperparameter tuning technique that randomly samples combinations of hyperparameters from a specified range or distribution to find the best-performing configuration for a machine learning model.
        - **Bayesian Optimization**: A probabilistic model-based approach to hyperparameter tuning that uses Bayesian inference to iteratively select and evaluate hyperparameter configurations, aiming to find the optimal settings for a machine learning model with fewer evaluations compared to traditional methods like grid search or random search.
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
- Amazon SageMaker Data Wrangler: Simplifies the process of **data preparation ( prepare, and transform data )** and feature engineering.
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
    - Use MLflow with Amazon SageMaker to track and **manage machine learning experiments and models**.

- **Experimentation**: The process of systematically testing and evaluating different machine learning models, algorithms, and hyperparameters to identify the best-performing solution for a given problem.

#### Understand model performance metrics and business metrics to evaluate ML models
- [**How does machine learning work?**](https://aws.amazon.com/what-is/machine-learning/#how-does-machine-learning-work)
- [**SageMaker Model Quality Monitoring**](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html)
- [**Post-training Data and Model Bias Metrics**](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-post-training-bias.html)
- [**Model Quality Metrics and CloudWatch Monitoring**](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
- Model performance metrics:
    - [**Accuracy**](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html#evaluations)
        - To understand the proportion of correct outcomes in a **binary classification** problem. It provides a straightforward measure of how often the model correctly predicts the positive and negative classes.
    - **RMSE (Root Mean Square Error)**: A metric used to measure the average magnitude of errors between predicted and actual values in regression tasks, providing insight into the model's accuracy. **It's not used for classification tasks.**
    - [**Precision and Recall**](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html#evaluations)
        - Precision is the ratio of true positive predictions to the total predicted positives, indicating the accuracy of positive predictions.
        - Recall is the ratio of true positive predictions to the total actual positives, measuring the model's ability to identify all relevant instances.
    - [**F1 Score**](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html#evaluations)
        - The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both aspects of model performance. It gives equal weight to both precision and recall, making it useful for evaluating models where false positives and false negatives are equally important.
        - Precision, Recall, and F1-Score are standard performance metrics used to evaluate the effectiveness of a classification system
    - [**Area Under the ROC Curve (AUC)**](https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html#evaluations)
        - AUC measures the ability of a **binary classification** model to distinguish between positive and negative classes across different threshold settings. It provides an aggregate measure of performance across all possible classification thresholds.
        - AUC is particularly useful when dealing with imbalanced datasets, as it evaluates the model's performance across various decision thresholds rather than relying on a single threshold.
    **Other metrics:**
    - **Precision**: Precision is a metric that measures the **exact matches** between the candidate text generated by the AI model and the reference text. It calculates the proportion of relevant instances among the retrieved instances.
    - **ROUGE**: ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to **evaluate the quality of summarie** by **comparing them to reference summaries**.
    - **BLEU**: (Bilingual Evaluation Understudy) is a metric used to **evaluate the quality of text generated by natural language processing** models, particularly in machine **translation tasks**, by **comparing** the generated text to one or more reference texts based on **n-gram overlap**.
            - **N-grams** (Sequences of N words used in BLEU and ROUGE metrics): N-grams are contiguous sequences of N items (words or characters) extracted from a given text, used in natural language processing tasks to analyze and evaluate the quality of generated text by comparing it to reference texts based on overlapping sequences.
    - **BERTScore**: (Bidirectional Encoder Representations from Transformers Score) is a metric that **evaluates the quality of text generated** by natural language processing models by **comparing** it to reference texts **using contextual embeddings from transformer-based** models like BERT.
    - **Perplexity**: A metric used to evaluate the performance of language models by measuring how well the model predicts a sample. It quantifies the uncertainty of the model when **predicting the next word** in a sequence, with lower perplexity indicating better predictive performance.
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
