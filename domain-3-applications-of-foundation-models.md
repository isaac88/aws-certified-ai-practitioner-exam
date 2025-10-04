# Domain 3: Applications of Foundation Models

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)

## 📋 Overview
This domain covers the practical applications of foundation models in various industries and use cases. Understanding these applications is crucial for leveraging AI/ML solutions on AWS effectively. **Weight: 28% of the exam.**

## 🎯 Key Topics Covered

### 3.1: Describe design considerations for applications that use foundation models
Objectives:
- Identify selection criteria to choose pre-trained models (for example, cost,
modality, latency, multi-lingual, model size, model complexity,
customization, input/output length).
- Understand the effect of inference parameters on model responses (for
example, temperature, input/output length).
- Define Retrieval Augmented Generation (RAG) and describe its business
applications (for example, Amazon Bedrock, knowledge base).
- Identify AWS services that help store embeddings within vector databases
(for example, Amazon OpenSearch Service, Amazon Aurora, Amazon
Neptune, Amazon DocumentDB [with MongoDB compatibility], Amazon
RDS for PostgreSQL).
- Explain the cost tradeoffs of various approaches to foundation model
customization (for example, pre-training, fine-tuning, in-context learning,
RAG).
- Understand the role of agents in multi-step tasks (for example, Agents for
Amazon Bedrock).

#### Understand the effect of inference parameters on model responses
- **Temperature**: A parameter that controls the **randomness** of the model's output. Higher values (e.g., 0.8) result in more diverse and creative responses, while lower values (e.g., 0.2) produce more focused and deterministic outputs.
- **Input/Output Length**: The maximum number of tokens the model can process in a single request. Longer input lengths allow for more context, while output length determines how much text the model can generate in response.
- **Top-p (nucleus sampling)**: A parameter that controls the diversity of the model's output by limiting the selection of tokens to a subset with a cumulative probability above a certain threshold (p). This helps balance creativity and coherence in generated text.
- **Top-k sampling**: A parameter that restricts the model's token selection to the **top k most-likely candidates** tokens at each step, promoting more focused and relevant outputs.
  - It filter out lower-probability responses, ensuring that only the most probable tokens are considered for generation.
This parameter limits the number of **most-likely candidates words** the model considers for the next word in a fixed way.

#### Define Retrieval Augmented Generation (RAG) and describe its business applications
- **Retrieval Augmented Generation (RAG)**: A technique that combines pre-trained foundation models with external knowledge bases to enhance the model's ability to generate accurate and contextually relevant responses. RAG retrieves relevant information from a knowledge base and incorporates it into the generation process.
  - Reranking algorithms: Techniques used to reorder retrieved documents based on their relevance to the input query, improving the quality of information provided to the foundation model for generation.
- **AWS Bedrock**: A fully managed service that makes it easy to build and scale generative AI applications using foundation models from leading AI startups and Amazon. It supports RAG by allowing developers to integrate external knowledge bases into their applications.
  - **Primarily responsability** provide access to foundation models(LLM) and manage infrastructure.

#### Identify AWS services that help store embeddings within vector databases
- **Amazon OpenSearch Service**: A managed service that makes it easy to deploy, operate, and scale OpenSearch clusters in the AWS Cloud. It supports vector search capabilities for storing and querying embeddings.
- **Amazon Aurora PostgreSQL**: A relational database service that is compatible with PostgreSQL and offers high performance and availability. It supports extensions for vector search, enabling efficient storage and retrieval of embeddings.
- **Amazon DocumentDB (with MongoDB compatibility)**: A fully managed document database service that supports MongoDB workloads. It can be used to store and query vector embeddings for AI applications.
- **Amazon Netptune**: Serverless graph database service for connected data and improved AI accuracy. Neptune captures context that improves accuracy and explainability of generative AI applications. It's not designed for vector search.
#### Explain the cost tradeoffs of various approaches to foundation model customization
- AWS Trainium instances: These instances are optimized for training large machine learning models, including foundation models. They offer high performance and cost-efficiency for training workloads. It's reducing its carbon footprint, highest energy efficiency for training ML models on AWS.
- AWS Inferentia instances: These instances are designed for high-performance inference (generative AI inference applications) of machine learning models, including foundation models. They provide cost-effective and scalable solutions for deploying AI applications.

### 3.2: Choose effective prompt engineering techniques
Objectives:
- Describe the concepts and constructs of prompt engineering (for example,
context, instruction, negative prompts, model latent space).
- Understand techniques for prompt engineering (for example, chain-ofthought, zero-shot, single-shot, few-shot, prompt templates).
- Understand the benefits and best practices for prompt engineering (for
example, response quality improvement, experimentation, guardrails,
discovery, specificity and concision, using multiple comments).
- Define potential risks and limitations of prompt engineering (for example,
exposure, poisoning, hijacking, jailbreaking).

#### Describe the concepts and constructs of prompt engineering
- Prompt engineering: The process of designing and refining prompts to effectively communicate tasks and instructions to foundation models (FMs) in order to elicit desired responses.
- Prompt tunning: The iterative process of refining and optimizing prompts to improve the quality and relevance of the model's responses.

---
- **Input Data**: The information or content provided to the foundation model to generate a response.
- **Output Data**: The response or result generated by the foundation model based on the input data and prompt.
- **Context**: The **background information** or setting provided to the model to help it understand the task or generate relevant responses.
- **Instruction**: A clear and specific directive given to the model to guide its response or behavior.
- **Negative prompts**: Instructions that specify what the model should avoid or not include in its response.
  - It doesn't provide a way to limit the ouput to a specific length.
- **Model latent space**: The high-dimensional space in which the model represents and processes information, capturing relationships and patterns learned during training.

#### Understand techniques for prompt engineering
- **Chain-of-thought**: This technique involves providing a model with a series of reasoning steps or a logical progression of ideas to guide it toward generating a more accurate and coherent response.
- **Zero-shot**: The zero-example request technique requires FMs to generate a response without providing explicit examples of the desired behavior, relying solely on their pre-training.
- **Single-shot**: This technique involves providing the model with a single example of the desired output format or behavior within the prompt itself to help guide its response.
- **Few-shot**: This technique involves providing the model with a few examples of the desired input-output pairs within the prompt. These examples guide the model to understand the pattern or format expected in its responses, improving its ability to generalize to new, similar tasks.
- **Prompt templates**: This technique involves creating structured templates that can be filled in with specific information or variables to generate prompts dynamically.

#### Understand the benefits and best practices for prompt engineering

#### Define potential risks and limitations of prompt engineering
- [Common prompt injection attacks](https://docs.aws.amazon.com/prescriptive-guidance/latest/llm-prompt-engineering-best-practices/common-attacks.html)
- **Prompt injection**: Refers to influencing the outputs by embedding specific instructions within the prompts themselves.
- **Prompt leaking**: It tries to extract the original prompt used to generate a response, potentially revealing sensitive or proprietary information.
- **Exposure**: Refers to the risk of exposing sensitive or confidential information to a model during training or inference. The model can then reveal this sensitive data from their training corpus, leading to potential data leaks or privacy violations.
- **Poisoning**: Malicious inputs could manipulate the model's behavior or outputs.
- **Hijacking**: Attackers manipulate an AI system to serve malicious purposes or to misbehave in unintended ways.
- **Jailbreaking**: Attempts to bypass the built-in restrictions and safety measures of AI systems to unlock restricted functionalities or generate prohibited content.

### 3.3: Describe the training and fine-tuning process for foundation models
Objectives:
- Describe the key elements of training a foundation model (for example,
pre-training, fine-tuning, continuous pre-training).
- Define methods for fine-tuning a foundation model (for example,
instruction tuning, adapting models for specific domains, transfer learning,
continuous pre-training).
- Describe how to prepare data to fine-tune a foundation model (for
example, data curation, governance, size, labeling, representativeness,
reinforcement learning from human feedback [RLHF]). 

#### Describe the key elements of training a foundation model
- Foundation models use **unlabeled** training data sets for **self-supervised** learning

- **Pre-training**: The initial phase where a foundation model is trained on a large and diverse dataset to learn general patterns and representations.
- **Fine-tuning**: The process of adapting a pre-trained foundation model to a specific task or domain by training it on a smaller, task-specific dataset.
- **Continuous pre-training**: An ongoing process where a foundation model is periodically updated with new data to maintain its relevance and performance.
- **Pruning**: The process of removing unnecessary parameters from a model to reduce its size and improve efficiency without significantly impacting performance.
- **Quantization**: The process of reducing the precision of the model's weights and activations to decrease memory usage and increase inference speed.
- **Distillation**: The process of transferring knowledge from a larger, more complex model (teacher) to a smaller, more efficient model (student) while retaining performance.

#### Describe how to prepare data to fine-tune a foundation model
- **Representativeness**: Ensuring that the fine-tuning dataset accurately reflects the diversity and characteristics of the target domain or task. This helps the model generalize better to real-world scenarios.
- **Reinforcement learning from human feedback [RLHF]**: A technique where human feedback is used to guide the fine-tuning process, helping the model learn to generate more accurate and relevant responses based on human preferences.

### 3.4: Describe methods to evaluate foundation model performance
Objectives:
- Understand approaches to evaluate foundation model performance (for
example, human evaluation, benchmark datasets).
- Identify relevant metrics to assess foundation model performance (for
example, Recall-Oriented Understudy for Gisting Evaluation [ROUGE],
Bilingual Evaluation Understudy [BLEU], BERTScore).
- Determine whether a foundation model effectively meets business
objectives (for example, productivity, user engagement, task engineering). 


#### Understand approaches to evaluate foundation model performance
- Metric to determine the optimal number of epochs: 
    - Validateion ouput accuracy: A measure of how well the model performs on unseen data. It helps in determining when to stop training to avoid overfitting.
    - Validation Loss: A measure of the model's error on the validation dataset. Monitoring validation loss helps identify the point at which the model starts to overfit the training data.
#### Identify relevant metrics to assess foundation model performance

## 🔗 Related Domains

- [Domain 4: Guidelines for Responsible AI](domain-4-guidelines-responsible-ai.md) *(Coming Soon)*
- [Domain 5: Security, Compliance, and Governance](domain-5-security-compliance-governance.md) *(Coming Soon)*

---

**📝 Note**: This study guide is continuously updated. Check back regularly for new content and improvements!

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)