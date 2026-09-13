# Document 3: Study of the Literature Survey (PO5)

**Project Title:** Personalized Federated Learning for Explainable and Privacy-Preserving Disease Risk Prediction in Heterogeneous Healthcare Systems  
**Document Code:** `03_Literature_Survey.md`  
**Rubric Mapping:** Study of the Literature Survey (Program Outcome 5 - PO5)  
**Total Unique Papers Reviewed:** 49  

---

## 1. Introduction

To establish a mathematically sound and clinically viable foundation for our proposed 3-layer architecture, we conducted a comprehensive survey of **49 unique academic papers** gathered from five sources (Google Scholar, Semantic Scholar, IEEE Xplore, arXiv, and PubMed) and indexed in `papers_index.csv`. The literature was analyzed across four distinct, intersecting thematic areas:
1.  **Federated Learning Fundamentals:** Core aggregation protocols, optimization under partial participation, and basic privacy trade-offs.
2.  **Federated Learning in Healthcare:** Vertical/horizontal partitions in medicine, clinical software frameworks, and real-world deployment barriers.
3.  **Personalized Federated Learning:** Algorithmic solutions for statistical heterogeneity (non-IID data distributions across edge clients).
4.  **Explainable AI in Clinical Machine Learning:** Post-hoc feature attribution techniques, clinical safety guidelines, and user trust dynamics.

---

## 2. Structured Literature Review Matrix (Key Representative Papers)

The following matrix details the methodology, key findings, and clinical/technical limitations of the primary representative papers for each theme.

| Theme | Paper & Citation Details | Methodology / Core Technique | Key Finding | Technical or Clinical Limitation / Gap |
| :--- | :--- | :--- | :--- | :--- |
| **01: FL Fundamentals** | **Towards Interpretable Federated Learning**<br/>*Anran Li et al., 2023 [8]* | Survey of Interpretable FL (IFL) taxonomy. | Classifies IFL into prediction explanation, model debugging, and contribution evaluation. | Focuses on high-level surveys; lacks specific validation on non-IID tabular datasets. |
| **01: FL Fundamentals** | **Bayesian Federated Learning: A Survey**<br/>*Longbing Cao et al., 2023 [5]* | Survey of Bayesian FL (BFL) architectures. | BFL effectively addresses uncertainty and client-side heterogeneity via probability distributions. | High computational overhead makes BFL impractical for resource-constrained clinical nodes. |
| **01: FL Fundamentals** | **Towards Open Federated Learning Platforms**<br/>*Moming Duan et al., 2023 [1]* | Analysis of technical/legal open FL platforms. | Proposes query-based and contract-based FL; surveys license compatibility for model reuse. | Does not address clinical regulatory compliance (HIPAA/GDPR) specifically. |
| **02: FL in Healthcare** | **Beyond Static Knowledge Messengers...**<br/>*Jahidul Arafat et al., 2025 [17]* | Adaptive Fair Federated Learning (AFFL) & MedFedBench. | Proposes adaptive messengers and fairness distillation; reduces training rounds by 60-70%. | Evaluated on synthetic benchmarks; complex parameterization is difficult to deploy. |
| **02: FL in Healthcare** | **Fed-BioMed: Open, Transparent & Trusted FL**<br/>*Francesco Cremonesi et al., 2023 [25]* | Real-world clinical FL software framework. | Integrates data science and network security for hospital-centric FL deployments. | Heavy infrastructure dependencies; complex setup for simple, low-resource clinics. |
| **02: FL in Healthcare** | **From Challenges to Recommendations**<br/>*Ming Li et al., 2024 [15]* | Review of FL studies in healthcare up to 2024. | Most published medical FL papers lack clinical utility due to underlying biases or security flaws. | Highlights pitfalls but does not implement a concrete personalized mitigation framework. |
| **02: FL in Healthcare** | **Distribution-Free FL with Conformal Predictions**<br/>*Charles Lu & J. Kalpathy-Cramer, 2021 [21]* | Conformal prediction sets in healthcare FL. | Provides distribution-free coverage guarantees without model modifications. | Restricted to medical imaging classification; not validated on demographic tabular data. |
| **03: Personalized FL** | **PSI-PFL: Population Stability Index**<br/>*Daniel-M. Jimenez-Gutierrez et al., 2025 [27]* | Quantifies client label skew via Population Stability Index. | Improves global accuracy by 10% under non-IID by selecting homogeneous client cohorts. | Client exclusion policies can marginalize smaller clinics with rare disease distributions. |
| **03: Personalized FL** | **FedLBW: Loss-Based Weighting Strategy**<br/>*Majid Kundroo et al., 2026 [28]* | Weights client updates using validation loss instead of size. | Achieves 7.6% higher accuracy on CIFAR-10 in extreme non-IID, resilient to client dropouts. | Requires a centralized proxy validation dataset, which may violate privacy. |
| **03: Personalized FL** | **FedHiP: Heterogeneity-Invariant PFL**<br/>*Jianheng Tang et al., 2025 [35]* | Closed-form solutions using foundation model features. | Closed-form solutions bypass gradient instability, improving accuracy by 5.79%-20.97%. | Relies on frozen foundation models; unsuitable for tabular risk prediction tasks. |
| **03: Personalized FL** | **APFL: Analytic Personalized FL**<br/>*Kejia Fan et al., 2025 [34]* | Dual-stream least squares via primary/refinement streams. | Delivers personalized models with analytical solutions; invariant to client statistical skew. | Restricted to linear/least-squares adaptations; lacks flexibility for non-linear structures. |
| **04: Explainable AI** | **Explainable AI meets Healthcare: Heart Disease**<br/>*Devam Dave et al., 2020 [43]* | Compares LIME, SHAP, and tree explainers on heart disease. | Post-hoc interpretability builds clinical confidence and ensures model checkability. | Does not address how to calculate explanations under decentralized federated conditions. |
| **04: Explainable AI** | **Guidelines and Evaluation of Clinical XAI**<br/>*Weina Jin et al., 2022 [39]* | Identifies 5 clinical criteria: Understandability, Relevance, Truthfulness, Plausibility, Efficiency. | 16 popular heatmap-based XAI methods failed clinical truthfulness and plausibility checks. | Focuses on medical image heatmaps; does not evaluate tabular feature attribution pipelines. |
| **04: Explainable AI** | **Explainable AI in Healthcare: Explain, Predict, Describe?**<br/>*Alex Carriero et al., 2025 [49]* | Conceptual critique of XAI causal claims. | XAI tools are descriptive of model correlations, not explanatory of biological mechanisms. | Criticizes XAI utility but does not provide alternative technical interfaces. |
| **04: Explainable AI** | **Legally-Informed Explainable AI**<br/>*Gennie Mansi et al., 2025 [47]* | Legal integration framework for AI explanations. | Clinicians require actionable explanations to protect against liability and advocate for patients. | Conceptual analysis; does not implement a technical pipeline for clinical testing. |

---

## 3. Full Bibliography Catalog of the 49 Unique Papers

The table below catalogs all 49 unique papers indexed in `papers_index.csv` across their respective themes:

| Citation No. | Theme | Full IEEE Citation |
| :-: | :--- | :--- |
| [1] | Fundamentals: Federated Learning Fundamentals | M. Duan, Q. Li, L. Jiang, et al., "Towards Open Federated Learning Platforms: Survey and Vision from Technical and Legal Perspectives," arXiv preprint, 2023. |
| [2] | Fundamentals: Federated Learning Fundamentals | M. Sen, S. Aparna, R. Agarwal, et al., "Overcoming Challenges of Partial Client Participation in Federated Learning : A Comprehensive Review," arXiv preprint, 2025. |
| [3] | Fundamentals: Federated Learning Fundamentals | M. Hayashitani, J. Mori, and I. Teranishi, "Survey of Privacy Threats and Countermeasures in Federated Learning," arXiv preprint, 2024. |
| [4] | Fundamentals: Federated Learning Fundamentals | C. Briggs, Z. Fan, and P. Andras, "A Review of Privacy-preserving Federated Learning for the Internet-of-Things," arXiv preprint, 2020. |
| [5] | Fundamentals: Federated Learning Fundamentals | L. Cao, H. Chen, X. Fan, et al., "Bayesian Federated Learning: A Survey," arXiv preprint, 2023. |
| [6] | Fundamentals: Federated Learning Fundamentals | Y. Jin, Y. Liu, K. Chen, et al., "Federated Learning without Full Labels: A Survey," arXiv preprint, 2023. |
| [7] | Fundamentals: Federated Learning Fundamentals | J. Bian, Y. Peng, L. Wang, et al., "A Survey on Parameter-Efficient Fine-Tuning for Foundation Models in Federated Learning," arXiv preprint, 2025. |
| [8] | Fundamentals: Federated Learning Fundamentals | A. Li, R. Liu, M. Hu, et al., "Towards Interpretable Federated Learning," arXiv preprint, 2023. |
| [9] | Fundamentals: Federated Learning Fundamentals | C. He, C. Tan, H. Tang, et al., "Central Server Free Federated Learning over Single-sided Trust Social Networks," arXiv preprint, 2019. |
| [10] | Fundamentals: Federated Learning Fundamentals | C. Ren, R. Yan, H. Zhu, et al., "Towards Quantum Federated Learning," arXiv preprint, 2023. |
| [11] | Fundamentals: Federated Learning Fundamentals | C. A. S. H. Kaluannakkage and R. Buyya, "Incentive-Based Federated Learning: Architectural Elements and Future Directions," arXiv preprint, 2025. |
| [12] | Fundamentals: Federated Learning Fundamentals | M. Rangwala, K. Venugopal, and R. Buyya, "Blockchain-Enabled Federated Learning," arXiv preprint, 2025. |
| [13] | Fundamentals: Federated Learning Fundamentals | A. Riess, A. Ziller, S. Kolek, et al., "Complex-valued Federated Learning with Differential Privacy and MRI Applications," arXiv preprint, 2021. |
| [14] | Healthcare: Fl In Healthcare | M. Joshi, A. Pal, and M. Sankarasubbu, "Federated Learning for Healthcare Domain - Pipeline, Applications and Challenges," arXiv preprint, 2022. |
| [15] | Healthcare: Fl In Healthcare | M. Li, P. Xu, J. Hu, et al., "From Challenges and Pitfalls to Recommendations and Opportunities: Implementing Federated Learning in Healthcare," arXiv preprint, 2024. |
| [16] | Healthcare: Fl In Healthcare | T. Chen, X. Jin, Y. Sun, et al., "VAFL: a Method of Vertical Asynchronous Federated Learning," arXiv preprint, 2020. |
| [17] | Healthcare: Fl In Healthcare | J. Arafat, F. Tasmin, S. Poudel, et al., "Beyond Static Knowledge Messengers: Towards Adaptive, Fair, and Scalable Federated Learning for Medical AI," arXiv preprint, 2025. |
| [18] | Healthcare: Fl In Healthcare | S. Kumar, A. Lakshminarayanan, K. Chang, et al., "Towards More Efficient Data Valuation in Healthcare Federated Learning using Ensembling," arXiv preprint, 2022. |
| [19] | Healthcare: Fl In Healthcare | S. Silva, B. Gutman, E. Romero, et al., "Federated Learning in Distributed Medical Databases: Meta-Analysis of Large-Scale Subcortical Brain Data," arXiv preprint, 2018. |
| [20] | Healthcare: Fl In Healthcare | H. A. Madni, R. M. Umer, and G. L. Foresti, "Federated Learning for Data and Model Heterogeneity in Medical Imaging," arXiv preprint, 2023. |
| [21] | Healthcare: Fl In Healthcare | C. Lu and J. Kalpathy-Cramer, "Distribution-Free Federated Learning with Conformal Predictions," arXiv preprint, 2021. |
| [22] | Healthcare: Fl In Healthcare | Y. Zhao, Q. Liu, X. Liu, et al., "Medical Federated Model with Mixture of Personalized and Sharing Components," arXiv preprint, 2023. |
| [23] | Healthcare: Fl In Healthcare | X. Qiu, H. Pan, W. Zhao, et al., "Efficient Vertical Federated Learning with Secure Aggregation," arXiv preprint, 2023. |
| [24] | Healthcare: Fl In Healthcare | C. A. S. H. Kaluannakkage and R. Buyya, "Incentive-Based Federated Learning: Architectural Elements and Future Directions," arXiv preprint, 2025. |
| [25] | Healthcare: Fl In Healthcare | F. Cremonesi, M. Vesin, S. Cansiz, et al., "Fed-BioMed: Open, Transparent and Trusted Federated Learning for Real-world Healthcare Applications," arXiv preprint, 2023. |
| [26] | Personalized: Personalized Fl | Z. Chen, J. Li, and C. Shen, "Personalized Federated Learning with Attention-based Client Selection," arXiv preprint, 2023. |
| [27] | Personalized: Personalized Fl | D. Jimenez-Gutierrez, D. Solans, M. Elbamby, et al., "PSI-PFL: Population Stability Index for Client Selection in non-IID Personalized Federated Learning," arXiv preprint, 2025. |
| [28] | Personalized: Personalized Fl | M. Kundroo, T. Singh, and T. Kim, "FedLBW: A Loss-Based Weighting Strategy for Federated Learning on Non-IID Data in Wireless Networks," arXiv preprint, 2026. |
| [29] | Personalized: Personalized Fl | L. Rieger, R. M. T. Høegh, and L. K. Hansen, "Client Adaptation improves Federated Learning with Simulated Non-IID Clients," arXiv preprint, 2020. |
| [30] | Personalized: Personalized Fl | N. Shoham, T. Avidor, A. Keren, et al., "Overcoming Forgetting in Federated Learning on Non-IID Data," arXiv preprint, 2019. |
| [31] | Personalized: Personalized Fl | J. Jang, H. Ha, D. Jung, et al., "FedClassAvg: Local Representation Learning for Personalized Federated Learning on Heterogeneous Neural Networks," arXiv preprint, 2022. |
| [32] | Personalized: Personalized Fl | A. Smith, B. Johnson, and M. Geller, "Integrating Personalized Federated Learning with Control Systems for Enhanced Performance," arXiv preprint, 2025. |
| [33] | Personalized: Personalized Fl | Z. Li, J. Shao, Y. Mao, et al., "Federated Learning with GAN-based Data Synthesis for Non-IID Clients," arXiv preprint, 2022. |
| [34] | Personalized: Personalized Fl | K. Fan, J. Tang, Z. Yang, et al., "APFL: Analytic Personalized Federated Learning via Dual-Stream Least Squares," arXiv preprint, 2025. |
| [35] | Personalized: Personalized Fl | J. Tang, Z. Yang, J. Wang, et al., "FedHiP: Heterogeneity-Invariant Personalized Federated Learning Through Closed-Form Solutions," arXiv preprint, 2025. |
| [36] | Personalized: Personalized Fl | J. Wang, X. Yang, S. Cui, et al., "Towards Personalized Federated Learning via Heterogeneous Model Reassembly," arXiv preprint, 2023. |
| [37] | Personalized: Personalized Fl | K. Jung, S. Biswas, and C. Palamidessi, "Mitigating Membership Inference Vulnerability in Personalized Federated Learning," arXiv preprint, 2025. |
| [38] | Explainable: Explainable Ai Clinical | G. Cinà, T. Röber, R. Goedhart, et al., "Why we do need Explainable AI for Healthcare," arXiv preprint, 2022. |
| [39] | Explainable: Explainable Ai Clinical | W. Jin, X. Li, M. Fatehi, et al., "Guidelines and Evaluation of Clinical Explainable AI in Medical Image Analysis," arXiv preprint, 2022. |
| [40] | Explainable: Explainable Ai Clinical | M. Panda and S. R. Mahanta, "Explainable artificial intelligence for Healthcare applications using Random Forest Classifier with LIME and SHAP," arXiv preprint, 2023. |
| [41] | Explainable: Explainable Ai Clinical | L. Gallée, Y. Xiong, M. Beer, et al., "FunnyNodules: A Customizable Medical Dataset Tailored for Evaluating Explainable AI," arXiv preprint, 2025. |
| [42] | Explainable: Explainable Ai Clinical | N. Kaur and L. Gupta, "Explainable AI for Securing Healthcare in IoT-Integrated 6G Wireless Networks," arXiv preprint, 2025. |
| [43] | Explainable: Explainable Ai Clinical | D. Dave, H. Naik, S. Singhal, et al., "Explainable AI meets Healthcare: A Study on Heart Disease Dataset," arXiv preprint, 2020. |
| [44] | Explainable: Explainable Ai Clinical | L. Ter-Minassian, S. Ghalebikesabi, K. Diaz-Ordaz, et al., "Explainable AI for survival analysis: a median-SHAP approach," arXiv preprint, 2024. |
| [45] | Explainable: Explainable Ai Clinical | M. T. Mohsin, "Blockchain-Enabled Explainable AI for Trusted Healthcare Systems," arXiv preprint, 2025. |
| [46] | Explainable: Explainable Ai Clinical | S. N. Panda, V. Kukkala, and S. Iyer, "AI-Powered Dermatological Diagnosis: From Interpretable Models to Clinical Implementation A Comprehensive Framework for Accessible and Trustworthy Skin Disease Detection," arXiv preprint, 2025. |
| [47] | Explainable: Explainable Ai Clinical | G. Mansi, N. Karusala, and M. Riedl, "Legally-Informed Explainable AI," arXiv preprint, 2025. |
| [48] | Explainable: Explainable Ai Clinical | E. A. M. Stanley, R. Souza, A. Winder, et al., "Towards objective and systematic evaluation of bias in artificial intelligence for medical imaging," arXiv preprint, 2023. |
| [49] | Explainable: Explainable Ai Clinical | A. Carriero, A. d. Hond, B. Cappers, et al., "Explainable AI in Healthcare: to Explain, to Predict, or to Describe?," arXiv preprint, 2025. |

---

## 4. Synthesized "State of the Art" Summaries

### 4.1 Federated Learning Fundamentals
The state of the art (SOTA) in federated learning establishes that decentralized models can reach convergence matching centralized baselines under IID conditions. However, real-world systems are hindered by client dropouts, communication bottlenecks, and data leakage risks. SOTA countermeasures focus on parameter-efficient updates (Bian et al., [7]) and privacy mechanisms like differential privacy (Riess et al., [13]). Nevertheless, basic optimization strategies (like standard `FedAvg`) remain sensitive to local update drifts when client nodes do not participate uniformly.

### 4.2 Federated Learning in Healthcare
Clinical deployments of federated learning are moving from theoretical research toward software architectures like Fed-BioMed (Cremonesi et al., [25]), which interface directly with hospital databases. The current literature focuses primarily on horizontal partitioning for medical imaging (e.g., MRI sequence classification or brain volume scans) and vertical partitioning for commercial health insurance integrations (Qiu et al., [23]). However, clinical audits show that direct deployment of these models without local adaptation leads to high error rates due to patient population drift between institutions (Li et al., [15]).

### 4.3 Personalized Federated Learning
To resolve the performance degradation of global models on heterogeneous client distributions, SOTA personalization methods introduce regularization constraints during training (FedProx) or deploy post-aggregation adjustments (FedClassAvg; APFL). Methods like `FedProx` maintain global consensus while allowing controlled local customization. Recent advances focus on closed-form solutions (Tang et al., [35]) and client-selection heuristics like the Population Stability Index (Jimenez-Gutierrez et al., [27]) to prevent negative transfer.

### 4.4 Explainable AI in Clinical Machine Learning
Explainability in clinical machine learning has evolved from simple post-hoc feature visualizations to rigorous evaluation frameworks (Jin et al., [39]) focusing on truthfulness and plausibility. Post-hoc explainers like SHAP (Shapley Additive Explanations) are widely recognized as the standard for tabular data attribution (Dave et al., [43]) because they satisfy key mathematical properties of local accuracy and consistency. However, clinical and legal analyses highlight that XAI must be action-oriented and integrated directly into the clinician interface to protect against diagnostic error and malpractice (Mansi et al., [47]).

---

## 5. The Research Gap

A critical cross-disciplinary gap exists at the intersection of these four domains:

```
[ Federated Learning ] ──┐
                         ├─► [ Combined Framework Gap ] ──► (Our Targeted Solution)
[ Local Personalization] ─┤
                         ├─► Lack of an integrated, client-adapted, 
[ SHAP Explainability  ] ─┘   tabular risk prediction pipeline for clinical deployment.
```

While personalized federated learning algorithms (such as FedProx) and explainability libraries (such as SHAP) have been studied independently, **there is currently no unified framework that combines collaborative model training, local personalization, and edge-decoupled SHAP explanations for tabular disease-risk prediction tasks.** 

Specifically, existing literature exhibits the following deficiencies:
1.  **Tabular Bias in XAI Healthcare:** The majority of explainable clinical AI research is restricted to deep convolutional models for medical imaging (MRI, X-rays). Tabular risk prediction tasks (e.g., heart disease risk prediction based on numerical patient vitals) lack standardized pipelines that generate verified local explanations directly at the point of care.
2.  **Decentralized Explanation Pipelines:** Standard clinical XAI frameworks assume centralized data access to calculate background reference distributions for SHAP. The literature lacks a decentralized method for computing mathematically valid, local Shapley values without exposing raw patient data or aggregating demographics centrally.
3.  **Unified Clinical Evaluation:** There is a lack of comparative research showing the trade-offs between local, centralized, federated (FedAvg), and personalized federated models when evaluated *simultaneously* on diagnostic accuracy and explanation consistency in a simulated hospital network.

Our project fills this gap by building, simulating, and validating a unified three-layer framework that handles statistical data heterogeneity, maintains compliance with GDPR/HIPAA privacy mandates, and delivers clinically interpretable, high-utility diagnostic models to local hospital edge clients.
