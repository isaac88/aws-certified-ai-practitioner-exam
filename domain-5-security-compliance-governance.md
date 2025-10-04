# Domain 5: Security, Compliance, and Governance

[![Back to Main](https://img.shields.io/badge/←-Back%20to%20Main-blue?style=flat)](README.md)

## 📋 Overview

This domain focuses on the essential aspects of securing AI systems, ensuring compliance with relevant regulations, and implementing governance frameworks. Key topics include methods to secure AI systems, governance practices, and compliance regulations.

## 🎯 Key Topics Covered

### 5.1: Explain methods to secure AI systems.
Objectives:
- Identify AWS services and features to secure AI systems (for example, IAM
roles, policies, and permissions; encryption; Amazon Macie; AWS
PrivateLink; AWS shared responsibility model).
- Understand the concept of source citation and documenting data origins
(for example, data lineage, data cataloging, SageMaker Model Cards).
- Describe best practices for secure data engineering (for example, assessing
data quality, implementing privacy-enhancing technologies, data access
control, data integrity).
- Understand security and privacy considerations for AI systems (for example,
application security, threat detection, vulnerability management,
infrastructure protection, prompt injection, encryption at rest and in
transit).

#### Identify AWS services and features to secure AI systems

#### Understand the concept of source citation and documenting data origins
- **Data lineage**: The tracking and documentation of the origin, movement, and transformation of data throughout its lifecycle. This is crucial for ensuring data privacy, security, and compliance with regulatory standards.
- **SageMaker Model Cards**: Provide a standardized way to document important information about machine learning models, including their intended use, performance metrics, and limitations. This helps ensure transparency and accountability in AI systems. 
The intended uses of a model go beyond technical details and **describe how a model should be used in production**, the scenarios in which is appropriate to use a model, and additional considerations such as the type of data to use with the model or any assumptions made during development.

#### Describe best practices for secure data engineering

#### Understand security and privacy considerations for AI systems

- **Threat Detection**: Implementing systems to identify potential security threats to AI systems, such as unauthorized access or data breaches.

- **Securing generative AI: An introduction to the Generative AI Security Scoping Matrix**:
    - Governance and Compliance: The policies, procedures, and reporting needed to empower the business while minimizing risk.
    - Legal & Privacy: The specific regulatory, legal, and privacy requirements for using or creating generative AI solutions.
    - Risk Management: Identification of potential threats to generative AI solutions and recommended mitigations.
    - Controls: The implementation of security controls that are used to mitigate risk.
    - Resilience: How to architect generative AI solutions to maintain availability and meet business SLAs.
    - https://aws.amazon.com/blogs/security/securing-generative-ai-an-introduction-to-the-generative-ai-security-scoping-matrix/

### 5.2: Recognize governance and compliance regulations for AI
systems.
Objectives:
- Identify regulatory compliance standards for AI systems (for example,
International Organization for Standardization [ISO], System and
Organization Controls [SOC], algorithm accountability laws).
- Identify AWS services and features to assist with governance and regulation
compliance (for example, AWS Config, Amazon Inspector, AWS Audit
Manager, AWS Artifact, AWS CloudTrail, AWS Trusted Advisor).
- Describe data governance strategies (for example, data lifecycles, logging,
residency, monitoring, observation, retention).
- Describe processes to follow governance protocols (for example, policies,
review cadence, review strategies, governance frameworks such as the
Generative AI Security Scoping Matrix, transparency standards, team
training requirements).

#### Identify regulatory compliance standards for AI systems

#### Identify AWS services and features to assist with governance and regulation compliance
- **AWS CloudTrail Lake**: A managed data lake that enables you to aggregate, immutably store, and **query your AWS CloudTrail logs**. It helps with governance, compliance, and operational and risk auditing of your AWS account.

#### Describe data governance strategies
- **Data lifecycles**: The stages through which data passes, from creation and initial storage to the time when it becomes obsolete and is deleted. Effective data lifecycle management ensures data integrity, security, and compliance with regulations.

#### Describe processes to follow governance protocols
