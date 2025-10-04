# Domain 2: Fundamentals of Generative AI

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)

## 📋 Overview

This domain covers the fundamental concepts of generative AI, including the capabilities and limitations of foundation models, as well as methods for improving their performance. **Weight: 24% of the exam.**

## 🎯 Key Topics Covered

### 2.1: Explain the basic concepts of generative AI
Objectives:
- Understand foundational generative AI concepts (for example, tokens,
chunking, embeddings, vectors, prompt engineering, transformer-based
LLMs, foundation models, multi-modal models, diffusion models).
- Identify potential use cases for generative AI models (for example, image,
video, and audio generation; summarization; chatbots; translation; code
generation; customer service agents; search; recommendation engines).
- Describe the foundation model lifecycle (for example, data selection, model
selection, pre-training, fine-tuning, evaluation, deployment, feedback).

#### Understanding Generative AI Concepts
- [Tokens](https://aws.amazon.com/blogs/machine-learning/optimizing-costs-of-generative-ai-applications-on-aws/)
  - Tokens are the basic units of text(single unit of meaning) that a generative AI model processes. They can represent words, subwords, or characters, depending on the tokenization method used.
  - Tokenizer: A tokenizer is a tool that converts text into tokens, which are the basic units of meaning that a generative AI model processes. Tokenizers can break down text into words, subwords, or characters, depending on the tokenization method used.
- [Chunking](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-chunking.html)
- [Embeddings](https://aws.amazon.com/what-is/embeddings-in-machine-learning/)
  - Embedding is a vector of numerical values that represents a piece of data, such as a word, sentence, or image, in a continuous vector space. Embeddings capture semantic relationships between data points, allowing machine learning models to understand and process complex information more effectively.
- [Vectors](https://aws.amazon.com/what-is/vector-databases/#ams#what-isc3#pattern-data)
  - Vectors are used in large language models to represent the numerical meaning of words. Each word or token is converted into a high-dimensional vector, capturing its semantic relationships with other words.
  - Word2Vec: A popular technique for generating word embeddings using neural networks. It captures semantic relationships between words by training on large text corpora.
- [Prompt Engineering](https://aws.amazon.com/what-is/prompt-engineering/)
- [Transformer-based LLMs](https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/)
  - **Self-attention mechanism**: A key component of transformer models that allows them to weigh the importance of different words in a sequence when generating predictions. It enables the model to capture long-range dependencies and contextual relationships between words.
- [LLM](https://aws.amazon.com/what-is/large-language-model/)
- [Foundation Models](https://aws.amazon.com/what-is/foundation-models/)
    - [What are examples of foundation models?](https://aws.amazon.com/what-is/foundation-models/#ams#what-isc6#pattern-data)
- [Multi-modal Models](https://aws.amazon.com/blogs/machine-learning/generative-ai-and-multi-modal-agents-in-aws-the-key-to-unlocking-new-value-in-financial-markets/)
  - Multi-modal embedding model: A model that can process and generate multiple types of data (e.g., text, images, audio) simultaneously, enabling more comprehensive understanding and generation capabilities.
  - **Amazon Titan Multimodal Embeddings G1 model**: A foundation model that generates high-quality embeddings from text and images, enabling improved performance across various tasks such as image retrieval, text retrieval, and multimodal classification.
- [Diffusion Models](hhttps://aws.amazon.com/what-is/stable-diffusion/)
    - https://aws.amazon.com/blogs/machine-learning/safe-image-generation-and-diffusion-models-with-amazon-ai-content-moderation-services/
    - Diffusion models work by first corrupting data with noise through a forward diffusion process and then learning to reverse this process to denoise the data.
    - **Forward diffusion**: Add Gaussian noise to the input data over a series of time steps until it becomes pure noise.
    - They use neural networks to predict and remove the noise step by step, ultimately generating new, structured data from random noise.

#### Identify potential use cases for generative AI models
- [Use Cases for Generative AI](https://aws.amazon.com/generative-ai/use-cases/)
- [What are generative AI examples?](https://aws.amazon.com/what-is/generative-ai/#ams#what-isc5#pattern-data)

#### Describe the foundation model lifecycle
- [Data selection](https://docs.aws.amazon.com/sagemaker/latest/dg/data-prep.html)
- [Model Selection](https://aws.amazon.com/blogs/machine-learning/beyond-the-basics-a-comprehensive-foundation-model-selection-framework-for-generative-ai/)
- [Pre-training](https://www.youtube.com/watch?v=4cuHNMhU_QY)
- [Fine-tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-fine-tuning.html)
  - Parameter-Efficient Fine-Tuning (PEFT): A technique that allows you to fine-tune large foundation models with a smaller number of trainable parameters, reducing computational resources and time required for training.
    - Use case:
      - Adapting a large language model to a specific domain or task without the need for extensive computational resources.
- [Evaluation](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate-whatis.html)
  - https://aws.amazon.com/bedrock/evaluations/
- [Deployment](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-marketplace-deploy-a-model.html)
  - A/B Testing: A method of comparing two versions of a model or application to determine which one performs better based on predefined metrics.
- [Feedback](https://aws.amazon.com/what-is/reinforcement-learning-from-human-feedback/)

### 2.2: Understand the capabilities and limitations of generative AI for solving business problems
Objectives:
- Describe the advantages of generative AI (for example, adaptability,
responsiveness, simplicity).
- Identify disadvantages of generative AI solutions (for example,
hallucinations, interpretability, inaccuracy, nondeterminism).
- Understand various factors to select appropriate generative AI models (for
example, model types, performance requirements, capabilities, constraints,
compliance).
- Determine business value and metrics for generative AI applications (for
example, cross-domain performance, efficiency, conversion rate, average
revenue per user, accuracy, customer lifetime value).

#### Describe the advantages of generative AI

- [kodekloud - Capabilities and Limitations of Generative AI Applications](https://notes.kodekloud.com/docs/AWS-Certified-AI-Practitioner/Fundamentals-of-Generative-AI/Capabilities-and-Limitations-of-Generative-AI-Applications)

- Adaptability
- Responsiveness
- Simplicity
- Creativity and exploration
- Data efficiency
- Personalization
- Scalability

#### Identify disadvantages of generative AI solutions

- Regulatory violations
- Social risks
- Data security and privacy concerns
- Toxicity
- Hallucinations
- Interpretability: Simpler models like decision trees offer higher interpretability, while complex models like deep neural networks are often considered "black boxes" due to their intricate architectures.
  - Interpretability is about **understanding the internal mechanisms** of a machine learning model.
- Explainability: Explainability focuses on providing clear and understandable reasons for a **model's predictions** or **decisions to end-users**.
- Nondeterminism
- Plagiarism and cheating

#### Understand various factors to select appropriate generative AI models
- [Best Practices for Generative AI Applications on AWS: Model Selection and Implementation Strategies](https://aws.amazon.com/es/awstv/watch/2e92fd37882/)
- [Amazon SageMaker Model Cards](https://docs.aws.amazon.com/sagemaker/latest/dg/model-cards.html)
- [Choosing an AWS generative AI service](https://docs.aws.amazon.com/generative-ai-on-aws-how-to-choose/)
- [Why model choice matters: Flexible AI unlocks freedom to innovate](https://aws.amazon.com/blogs/aws-insights/why-model-choice-matters-flexible-ai-unlocks-freedom-to-innovate/)

#### Determine business value and metrics for generative AI applications
- Cross-domain performance
- Efficiency: Measures how cost-effectively and quickly the AI model can be deployed, focusing on resource utilization and time to market.
- Conversion rate: The conversion rate is a key business  that directly measures how well the AI solution drives desired user actions, such as making a purchase or signing up for a service. (This metric can be directly influenced by the **quality of AI**.)
- User Satisfaction: A measure of how satisfied users are with the AI-generated outputs, often assessed through surveys or feedback mechanisms. (This metric can be directly influenced by the **quality of AI**.)
- Average
- Average Revenue per user(ARPU): Average revenue per user, or unit, is a metric used by app businesses to calculate how much money they generate from a user during a specific, set time period.
- Accuracy
- Customer lifetime value
- Transfer learning: Technique where a pre-trained model is adapted to a new task or domain by fine-tuning it with a smaller dataset specific to the new task.

### 2.3: Describe AWS infrastructure and technologies for building generative AI applications
Objectives:
- Identify AWS services and features to develop generative AI applications
(for example, Amazon SageMaker JumpStart; Amazon Bedrock; PartyRock,
an Amazon Bedrock Playground; Amazon Q).
- Describe the advantages of using AWS generative AI services to build
applications (for example, accessibility, lower barrier to entry, efficiency,
cost-effectiveness, speed to market, ability to meet business objectives).
- Understand the benefits of AWS infrastructure for generative AI
applications (for example, security, compliance, responsibility, safety).
- Understand cost tradeoffs of AWS generative AI services (for example,
responsiveness, availability, redundancy, performance, regional coverage,
token-based pricing, provision throughput, custom models).

#### Identify AWS services and features to develop generative AI applications
- **Amazon Bedrock**: A fully managed service that makes it easy to build and scale **generative AI** applications using foundation models from leading AI startups and Amazon.
  - **Action group**: used to define the specific tasks the agent should perform, such as making API calls or invoking Lambda functions to carry out.
  - Bedrock fine-tunes pre-trained models NO re-train fundation models with new data.
  - **Customize models** with **Fine-tuning** and or **Continued pre-training**: Purchase **provision throughput** to **fine-tune** or **continue pre-training** foundation models on 
  your own data.
  - **Watermark detection**: Security feature in Amazon Bedrock that identifies if an image was created by the Amazon Titan Image Generator model on Bedrock.
  - [Submit prompts and generate responses using the API](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-api.html)
  - Use Cases:
    - Content generation:
      - Generate captions for images stored in Amazon S3.
    - Personalized recommendations
    - Customer support
    - Data augmentation
- https://caylent.com/blog/amazon-bedrock-vs-sage-maker-jumpstart
- **Amazon SageMaker JumpStart**: Designed to simplify and expedite the machine learning process by offering pre-trained models, solution templates, and example notebooks that enable users to start machine learning projects quickly without needing to build models from scratch.
- **Amazon Q Developer**: It helps you understand and manage your cloud infrastructure on AWS. With this capability, you can list and describe your AWS resources using natural language prompts, minimizing friction in navigating the AWS Management Console and compiling all information from documentation pages.
  - It can get answers to AWS cost-related questions using natural language.
  - It can coding, testing, and upgrading applications using generative AI.
- **Amazon Q in Connect**: Amazon Q in Connect is a generative **AI-powered assistant for customer service** that delivers end-customers and agents information and actions to solve issues in real time.
- **Amazon Connect**: A cloud-based contact center service that enables businesses to deliver better customer service at lower costs. It integrates with generative AI services to enhance customer interactions.
  - **Amazon Connect Contact Lens**: A feature of Amazon Connect that uses natural language processing (NLP) and machine learning to analyze customer interactions, providing insights into customer sentiment, trends, and agent performance.
#### Describe the advantages of using AWS generative AI services to build applications

#### Understand the benefits of AWS infrastructure for generative AI applications

#### Understand cost tradeoffs of AWS generative AI services
- Responsiveness
- Availability
- Token-based pricing
  - Context windows: The context window refers to the maximum number of tokens (words or subwords) that a generative AI model can process in a single input. Models with larger context windows can understand and generate more coherent and contextually relevant responses, especially for longer texts.
- Provision throughput: When you configure Provisioned Throughput for a model, you receive a level of throughput at a fixed cost.(The total number of input tokens per minute , The total number of output tokens per minute)
- Custom models

## 🔗 Related Domains

- [Domain 3: Applications of Foundation Models](domain-3-applications-foundation-models.md)
- [Domain 4: Guidelines for Responsible AI](domain-4-guidelines-responsible-ai.md) *(Coming Soon)*
- [Domain 5: Security, Compliance, and Governance](domain-5-security-compliance-governance.md) *(Coming Soon)*

---

**📝 Note**: This study guide is continuously updated. Check back regularly for new content and improvements!

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)