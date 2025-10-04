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
  - Tokens are the basic units of text that a generative AI model processes. They can represent words, subwords, or characters, depending on the tokenization method used.
- [Chunking](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-chunking.html)
- [Embeddings](https://aws.amazon.com/what-is/embeddings-in-machine-learning/)
- [Vectors](https://aws.amazon.com/what-is/vector-databases/#ams#what-isc3#pattern-data)
- [Prompt Engineering](https://aws.amazon.com/what-is/prompt-engineering/)
- [Transformer-based LLMs](https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/)
- [LLM](https://aws.amazon.com/what-is/large-language-model/)
- [Foundation Models](https://aws.amazon.com/what-is/foundation-models/)
    - [What are examples of foundation models?](https://aws.amazon.com/what-is/foundation-models/#ams#what-isc6#pattern-data)
- [Multi-modal Models](https://aws.amazon.com/blogs/machine-learning/generative-ai-and-multi-modal-agents-in-aws-the-key-to-unlocking-new-value-in-financial-markets/)
- [Diffusion Models](hhttps://aws.amazon.com/what-is/stable-diffusion/)
    - https://aws.amazon.com/blogs/machine-learning/safe-image-generation-and-diffusion-models-with-amazon-ai-content-moderation-services/

#### Identify potential use cases for generative AI models
- [Use Cases for Generative AI](https://aws.amazon.com/generative-ai/use-cases/)
- [What are generative AI examples?](https://aws.amazon.com/what-is/generative-ai/#ams#what-isc5#pattern-data)

#### Describe the foundation model lifecycle
- [Data selection](https://docs.aws.amazon.com/sagemaker/latest/dg/data-prep.html)
- [Model Selection](https://aws.amazon.com/blogs/machine-learning/beyond-the-basics-a-comprehensive-foundation-model-selection-framework-for-generative-ai/)
- [Pre-training](https://www.youtube.com/watch?v=4cuHNMhU_QY)
- [Fine-tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-fine-tuning.html)
- [Evaluation](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate-whatis.html)
  - https://aws.amazon.com/bedrock/evaluations/
- [Deployment](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-marketplace-deploy-a-model.html)
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
- Conversion rate: The conversion rate is a key business metric that directly measures how well the AI solution drives desired user actions, such as making a purchase or signing up for a service.
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
- https://caylent.com/blog/amazon-bedrock-vs-sage-maker-jumpstart

#### Describe the advantages of using AWS generative AI services to build applications

#### Understand the benefits of AWS infrastructure for generative AI applications

#### Understand cost tradeoffs of AWS generative AI services
- Responsiveness
- Availability
- Token-based pricing
- Provision throughput: When you configure Provisioned Throughput for a model, you receive a level of throughput at a fixed cost.(The total number of input tokens per minute , The total number of output tokens per minute)
- Custom models

## 🔗 Related Domains

- [Domain 3: Applications of Foundation Models](domain-3-applications-foundation-models.md)
- [Domain 4: Guidelines for Responsible AI](domain-4-guidelines-responsible-ai.md) *(Coming Soon)*
- [Domain 5: Security, Compliance, and Governance](domain-5-security-compliance-governance.md) *(Coming Soon)*

---

**📝 Note**: This study guide is continuously updated. Check back regularly for new content and improvements!

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)