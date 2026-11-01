"""
Builds paper/references.bib and paper/literature_matrix.csv from one verified source list.

Every entry was checked against the arXiv API (title, first author, year) or Crossref (DOI, venue, volume,
pages) on 2026-11-01. Author lists are abbreviated to the first author plus "others"; fill them in from the DOI
or arXiv page before submission if the journal style needs every author.

    python paper/build_refs.py
"""
import csv
import os

# key, type, first author, title, venue, year, publisher, volume, pages, doi, arxiv, theme,
# what they did, key finding, limitation (relative to our question), how we use it
R = [
 # ---------------- Theme A: shortcut learning and group robustness ----------------
 ("geirhos2020shortcut", "article", "Geirhos, Robert", "Shortcut learning in deep neural networks", "Nature Machine Intelligence", 2020, "Springer Nature", "2", "665--673", "10.1038/s42256-020-00257-z", "2004.07780", "A1 Shortcut theory",
  "Perspective unifying shortcut learning across vision, language and RL", "Decision rules that exploit training-set cues fail under distribution shift; o.o.d. tests are needed", "Conceptual; no architectural analysis", "Defines the core construct; motivates o.o.d. group evaluation"),
 ("lapuschkin2019clever", "article", "Lapuschkin, Sebastian", "Unmasking Clever Hans predictors and assessing what machines really learn", "Nature Communications", 2019, "Springer Nature", "10", "1096", "10.1038/s41467-019-08987-4", "", "A1 Shortcut theory",
  "Spectral relevance analysis of explanations over a dataset", "High-accuracy models can rely on watermarks and source tags (Clever Hans)", "CNN classifiers; no LLM-based VLMs", "Precedent for a watermark shortcut; our Watermark task operationalises it"),
 ("anders2022finding", "article", "Anders, Christopher J.", "Finding and removing Clever Hans: Using explanation methods to debug and improve deep models", "Information Fusion", 2022, "Elsevier", "77", "261--295", "10.1016/j.inffus.2021.07.015", "", "A1 Shortcut theory",
  "Explanation-guided detection and unlearning of artefact reliance", "Artefacts can be found and removed post hoc", "Requires explanation tooling; classifiers only", "Contrasts post-hoc removal with our architectural cause"),
 ("degrave2021covid", "article", "DeGrave, Alex J.", "AI for radiographic COVID-19 detection selects shortcuts over signal", "Nature Machine Intelligence", 2021, "Springer Nature", "3", "610--619", "10.1038/s42256-021-00338-7", "", "A1 Shortcut theory",
  "Audit of COVID-19 chest X-ray classifiers across data sources", "Models used text markers and source artefacts, failing on external data", "Single domain", "Shows the high stakes of shortcut reliance (hook of the introduction)"),
 ("brown2023shortcut", "article", "Brown, Alexander", "Detecting shortcut learning for fair medical AI using shortcut testing", "Nature Communications", 2023, "Springer Nature", "14", "4314", "10.1038/s41467-023-39902-7", "", "A1 Shortcut theory",
  "Multitask test that varies how much a model encodes a sensitive attribute", "Shortcut reliance can be measured, not just suspected", "Medical imaging only", "Supports measuring reliance with controlled interventions"),
 ("ongly2024shortcut", "article", "Ong Ly, Cathy", "Shortcut learning in medical AI hinders generalization: method for estimating AI model generalization without external data", "npj Digital Medicine", 2024, "Springer Nature", "7", "124", "10.1038/s41746-024-01118-4", "", "A1 Shortcut theory",
  "Estimate generalisation gap from shortcut indicators", "In-distribution accuracy over-states real performance", "Medical imaging only", "Motivates reporting ID/OOD gap"),
 ("sagawa2020groupdro", "inproceedings", "Sagawa, Shiori", "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization", "International Conference on Learning Representations (ICLR)", 2020, "", "", "", "", "1911.08731", "A2 Group robustness methods",
  "GroupDRO; introduced the Waterbirds benchmark", "Worst-group accuracy collapses under ERM; group-aware training and regularisation fix it", "Needs group labels; ResNet classifiers", "Waterbirds, WGA metric, group definition"),
 ("sagawa2020overparam", "inproceedings", "Sagawa, Shiori", "An Investigation of Why Overparameterization Exacerbates Spurious Correlations", "International Conference on Machine Learning (ICML)", 2020, "PMLR", "", "", "", "2005.04345", "A2 Group robustness methods",
  "Theory and experiments on model size vs spurious reliance", "Larger models can memorise minority examples and rely more on spurious features", "Classifiers; capacity not connector design", "Explains why more capacity (SigLIP-large) need not remove reliance"),
 ("arjovsky2019irm", "misc", "Arjovsky, Martin", "Invariant Risk Minimization", "arXiv preprint", 2019, "", "", "", "", "1907.02893", "A2 Group robustness methods",
  "Learn representations whose optimal classifier is invariant across environments", "Invariance is a principled target against spurious features", "Needs environments; mixed empirical success", "Theoretical anchor: a robust model must use invariant (core) features"),
 ("nam2020lff", "inproceedings", "Nam, Junhyun", "Learning from Failure: Training Debiased Classifier from Biased Classifier", "Advances in Neural Information Processing Systems (NeurIPS)", 2020, "", "", "", "", "2007.02561", "A2 Group robustness methods",
  "Train a deliberately biased model and reweight the debiased one", "Bias can be inferred without group labels", "Classifiers only", "Group-label-free family that JTT belongs to"),
 ("liu2021jtt", "inproceedings", "Liu, Evan Zheran", "Just Train Twice: Improving Group Robustness without Training Group Information", "International Conference on Machine Learning (ICML)", 2021, "PMLR", "", "", "", "2107.09044", "A2 Group robustness methods",
  "Upweight the errors of an ERM model and retrain", "Strong WGA with group labels only for selection", "Sensitive to hyper-parameters", "Connector-JTT in stage 3 (did not help)"),
 ("zhang2022cnc", "inproceedings", "Zhang, Michael", "Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations", "International Conference on Machine Learning (ICML)", 2022, "PMLR", "", "", "", "2203.01517", "A2 Group robustness methods",
  "Contrastive learning between same-class samples the ERM model separates", "Aligning representations within class improves WGA", "Classifiers only", "Related method family; not used"),
 ("idrissi2022balancing", "inproceedings", "Idrissi, Badr Youbi", "Simple data balancing achieves competitive worst-group-accuracy", "Conference on Causal Learning and Reasoning (CLeaR)", 2022, "PMLR", "", "", "", "2110.14503", "A2 Group robustness methods",
  "Subsampling and reweighting baselines", "Simple balancing rivals complex robust methods", "Needs group labels", "Group-balanced sampling baseline"),
 ("izmailov2022feature", "inproceedings", "Izmailov, Pavel", "On Feature Learning in the Presence of Spurious Correlations", "Advances in Neural Information Processing Systems (NeurIPS)", 2022, "", "", "", "", "2210.11369", "A2 Group robustness methods",
  "Probe how much core information ERM features keep", "Features often contain the core signal; readout is the bottleneck", "Vision classifiers", "Linear probes on our frozen features separate feature vs readout failure"),
 ("kirichenko2023dfr", "inproceedings", "Kirichenko, Polina", "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations", "International Conference on Learning Representations (ICLR)", 2023, "", "", "", "", "2204.02937", "A2 Group robustness methods",
  "Deep Feature Reweighting: retrain the last layer on a small balanced set", "Last-layer retraining recovers most robustness", "Classifier heads only", "Linear DFR baseline; inspires connector-DFR"),
 ("labonte2023llr", "inproceedings", "LaBonte, Tyler", "Towards Last-layer Retraining for Group Robustness with Fewer Annotations", "Advances in Neural Information Processing Systems (NeurIPS)", 2023, "", "", "", "", "2309.08534", "A2 Group robustness methods",
  "Last-layer retraining with few or no group annotations", "Class-balanced held-out data already helps", "Classifiers", "Supports the cheap, small-data fix"),
 ("qiu2023afr", "inproceedings", "Qiu, Shikai", "Simple and Fast Group Robustness by Automatic Feature Reweighting", "International Conference on Machine Learning (ICML)", 2023, "PMLR", "", "", "", "2306.11074", "A2 Group robustness methods",
  "Reweight held-out points by ERM loss for last-layer retraining", "Fast WGA gains without group labels in training", "Classifiers", "Related readout fix"),
 ("noohdani2024decompose", "inproceedings", "Hosseini Noohdani, Fahimeh", "Decompose-and-Compose: A Compositional Approach to Mitigating Spurious Correlation", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2024, "IEEE", "", "27652--27661", "10.1109/CVPR52733.2024.02612", "", "A2 Group robustness methods",
  "Compose image parts to break spurious combinations", "Data-level recombination improves WGA", "Image classifiers", "Recent CVPR evidence that data/readout fixes dominate"),
 ("chakraborty2024exmap", "inproceedings", "Chakraborty, Rwiddhi", "ExMap: Leveraging Explainability Heatmaps for Unsupervised Group Robustness to Spurious Correlations", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2024, "IEEE", "", "12017--12026", "10.1109/CVPR52733.2024.01142", "", "A2 Group robustness methods",
  "Cluster explanation heatmaps to infer groups", "Attention/explanations can reveal spurious groups", "Classifiers", "Contrast: our attention analysis shows attention alone does not indicate reliance"),
 ("koh2021wilds", "inproceedings", "Koh, Pang Wei", "WILDS: A Benchmark of in-the-Wild Distribution Shifts", "International Conference on Machine Learning (ICML)", 2021, "PMLR", "", "", "", "2012.07421", "A3 Evaluation of shift",
  "Benchmark suite of real distribution shifts", "Report ID and OOD performance and their gap", "Not VLM-specific", "ID/OOD reporting"),
 ("gulrajani2021domainbed", "inproceedings", "Gulrajani, Ishaan", "In Search of Lost Domain Generalization", "International Conference on Learning Representations (ICLR)", 2021, "", "", "", "", "2007.01434", "A3 Evaluation of shift",
  "Controlled re-evaluation of domain-generalisation methods", "With fair model selection, ERM matches most methods", "DG, not group shift", "Justifies fixed protocols and explicit model-selection rules"),
 ("zhou2022dgsurvey", "article", "Zhou, Kaiyang", "Domain Generalization: A Survey", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2022, "IEEE", "", "", "10.1109/TPAMI.2022.3195549", "", "A3 Evaluation of shift",
  "Survey of domain generalisation", "Taxonomy of o.o.d. methods and protocols", "Pre-LLM era", "Background on o.o.d. generalisation"),
 ("wang2022dgsurvey", "article", "Wang, Jindong", "Generalizing to Unseen Domains: A Survey on Domain Generalization", "IEEE Transactions on Knowledge and Data Engineering", 2022, "IEEE", "", "", "10.1109/TKDE.2022.3178128", "", "A3 Evaluation of shift",
  "Survey of domain generalisation", "Theory and methods of o.o.d. generalisation", "Pre-LLM era", "Background on o.o.d. generalisation"),
 ("liu2015celeba", "inproceedings", "Liu, Ziwei", "Deep Learning Face Attributes in the Wild", "IEEE International Conference on Computer Vision (ICCV)", 2015, "IEEE", "", "3730--3738", "10.1109/ICCV.2015.425", "1411.7766", "A3 Evaluation of shift",
  "CelebA face-attribute dataset", "Natural attribute correlations (blond hair and gender)", "Dataset paper", "Third task"),
 ("zhou2018places", "article", "Zhou, Bolei", "Places: A 10 Million Image Database for Scene Recognition", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2018, "IEEE", "40", "1452--1464", "10.1109/TPAMI.2017.2723009", "", "A3 Evaluation of shift",
  "Scene recognition dataset", "Source of Waterbirds backgrounds", "Dataset paper", "Waterbirds construction"),
 ("lin2014coco", "inproceedings", "Lin, Tsung-Yi", "Microsoft COCO: Common Objects in Context", "European Conference on Computer Vision (ECCV)", 2014, "Springer", "", "740--755", "10.1007/978-3-319-10602-1_48", "1405.0312", "A3 Evaluation of shift",
  "Object detection and captioning dataset", "Captions and object labels", "Dataset paper", "Alignment captions and Watermark task images"),

 # ---------------- Theme B: shortcuts and bias in vision-language models ----------------
 ("agrawal2018vqacp", "inproceedings", "Agrawal, Aishwarya", "Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2018, "IEEE", "", "", "", "1712.00377", "B1 Language priors in VQA",
  "VQA-CP: answer distributions changed between train and test", "VQA models answer from language priors, ignoring the image", "Pre-LLM VQA models", "Precedent: multimodal models under-use vision"),
 ("goyal2019vqav2", "article", "Goyal, Yash", "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering", "International Journal of Computer Vision", 2019, "Springer", "127", "398--414", "10.1007/s11263-018-1116-0", "", "B1 Language priors in VQA",
  "Balanced VQA v2 with complementary image pairs", "Balancing data forces models to look at the image", "Data fix, not architecture", "Shows data balancing as the classic lever"),
 ("yang2022deconfounded", "article", "Yang, Xu", "Deconfounded Image Captioning: A Causal Retrospect", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2022, "IEEE", "", "", "10.1109/TPAMI.2021.3121705", "", "B1 Language priors in VQA",
  "Causal intervention against dataset confounders in captioning", "Spurious co-occurrences bias captions", "Encoder-decoder captioners", "Causal framing of spurious correlations in V+L"),
 ("liu2023crossmodal", "article", "Liu, Yang", "Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2023, "IEEE", "45", "11624--11641", "10.1109/TPAMI.2023.3284038", "", "B1 Language priors in VQA",
  "Causal front-door/back-door adjustment for video QA", "Cross-modal spurious correlations hurt reasoning", "Specialised VQA models", "Background on cross-modal spurious correlation"),
 ("jiang2023cib", "article", "Jiang, Jingjing", "Correlation Information Bottleneck: Towards Adapting Pretrained Multimodal Models for Robust Visual Question Answering", "International Journal of Computer Vision", 2023, "Springer", "132", "185--207", "10.1007/s11263-023-01858-y", "", "B1 Language priors in VQA",
  "Information bottleneck when adapting pretrained multimodal models", "Compressing redundant correlations improves robustness", "Pre-LLM multimodal models", "Bottleneck view of the connector"),
 ("radford2021clip", "inproceedings", "Radford, Alec", "Learning Transferable Visual Models From Natural Language Supervision", "International Conference on Machine Learning (ICML)", 2021, "PMLR", "", "", "", "2103.00020", "B2 Bias in contrastive VLMs",
  "CLIP: contrastive image-text pretraining", "Zero-shot transfer from language supervision", "Inherits web biases", "Background on vision encoders"),
 ("zhang2022adapters", "inproceedings", "Zhang, Michael", "Contrastive Adapters for Foundation Model Group Robustness", "Advances in Neural Information Processing Systems (NeurIPS)", 2022, "", "", "", "", "2207.07180", "B2 Bias in contrastive VLMs",
  "Small adapters on frozen CLIP embeddings trained contrastively", "Adapting a light module improves WGA of foundation models", "Contrastive models, not generative VLMs", "Closest precedent for fixing robustness in a small module (our connector)"),
 ("yang2023mitigating", "inproceedings", "Yang, Yu", "Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning", "International Conference on Machine Learning (ICML)", 2023, "PMLR", "", "", "", "2304.03916", "B2 Bias in contrastive VLMs",
  "Use language to detect and suppress spurious features in CLIP fine-tuning", "Multimodal fine-tuning can be steered away from spurious cues", "Contrastive models", "Related fine-tuning fix"),
 ("chuang2023biasedprompts", "misc", "Chuang, Ching-Yao", "Debiasing Vision-Language Models via Biased Prompts", "arXiv preprint", 2023, "", "", "", "", "2302.00070", "B2 Bias in contrastive VLMs",
  "Project out biased directions in text embeddings", "Training-free debiasing of CLIP", "Contrastive models", "Contrast with training-time fixes"),
 ("you2024calibrating", "inproceedings", "You, Chenyu", "Calibrating Multi-modal Representations: A Pursuit of Group Robustness without Annotations", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2024, "IEEE", "", "26140--26150", "10.1109/CVPR52733.2024.02470", "", "B2 Bias in contrastive VLMs",
  "Calibrate CLIP representations without group labels", "Representation calibration (a DFR relative) improves WGA", "Contrastive models", "Forward-citation of DFR in VLMs; nearest to connector-DFR"),
 ("liang2022modalitygap", "inproceedings", "Liang, Weixin", "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning", "Advances in Neural Information Processing Systems (NeurIPS)", 2022, "", "", "", "", "2203.02053", "B2 Bias in contrastive VLMs",
  "Analysis of the gap between image and text embeddings", "Modalities occupy separate cones of the embedding space", "Contrastive models", "Why a connector is needed to map vision into the LLM space"),
 ("materzynska2022disentangling", "inproceedings", "Materzynska, Joanna", "Disentangling visual and written concepts in CLIP", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2022, "", "", "", "", "2206.07835", "B2 Bias in contrastive VLMs",
  "Separate the reading of text in images from visual concepts", "CLIP strongly reads rendered text", "Contrastive models", "Our Watermark shortcut is partly textual; explains its strength"),
 ("bleeker2024shortcuts", "misc", "Bleeker, Maurits", "Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning", "arXiv preprint", 2024, "", "", "", "", "2402.17510", "B3 Shortcuts in VLM training",
  "Synthetic shortcuts in image-caption training; LTD and IFM mitigations", "Shortcuts can almost erase retrieval performance; LTD helps partly", "Contrastive models", "Controlled-shortcut design; LTD ablation"),
 ("hosseini2025spurlens", "misc", "Hosseini, Parsa", "SpurLens: Automatic Detection of Spurious Cues in Multimodal LLMs", "arXiv preprint", 2025, "", "", "", "", "2503.08884", "B3 Shortcuts in VLM training",
  "Automatic discovery of spurious cues in MLLMs", "LLaVA-class models fail on minority groups zero-shot", "Detection, not cause", "Motivates native zero-shot baseline"),
 ("yang2025spuriverse", "misc", "Yang, Yiwei", "Escaping the SpuriVerse: Can Large Vision-Language Models Generalize Beyond Seen Spurious Correlations?", "arXiv preprint", 2025, "", "", "", "", "2506.18322", "B3 Shortcuts in VLM training",
  "Benchmark of 124 real spurious correlations for LVLMs", "Fine-tuning on spurious cases trades off ordinary accuracy", "Benchmark; architecture not varied", "Reporting average accuracy with WGA"),
 ("sarker2026purge", "misc", "Sarker, Aditi", "Partition-Aware Unlearning for Removing Spurious Correlations in Large Vision-Language Models", "arXiv preprint", 2026, "", "", "", "", "2608.29996", "B3 Shortcuts in VLM training",
  "Unlearning object-background correlations in LLaVA and Qwen models", "Post-hoc unlearning reduces reliance", "Connector held fixed", "Gap evidence: connector unexamined"),
 ("sarridis2026scaling", "misc", "Sarridis, Ioannis", "Scaling Vision-Language Models Is Not Enough to Mitigate Bias", "arXiv preprint", 2026, "", "", "", "", "2607.28211", "B3 Shortcuts in VLM training",
  "194 contrastive models; predictors of bias", "Data, not scale, drives robustness", "Contrastive models; connector not studied", "Agrees with our near-null connector-type effect"),
 ("zhang2024oodmllm", "misc", "Zhang, Xingxuan", "On the Out-Of-Distribution Generalization of Multimodal Large Language Models", "arXiv preprint", 2024, "", "", "", "", "2402.06599", "B3 Shortcuts in VLM training",
  "Evaluation of MLLMs under distribution shift", "MLLMs generalise poorly beyond training domains; mapping deficiency suspected", "Evaluation only", "Supports the connector as a candidate cause"),
 ("tong2024eyes", "inproceedings", "Tong, Shengbang", "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2024, "", "", "", "", "2401.06209", "B3 Shortcuts in VLM training",
  "CLIP-blind pairs and MMVP benchmark", "MLLM failures trace to visual representations", "Encoder-focused", "Contradiction to resolve: encoder vs connector as the cause"),
 ("li2023pope", "inproceedings", "Li, Yifan", "Evaluating Object Hallucination in Large Vision-Language Models", "Conference on Empirical Methods in Natural Language Processing (EMNLP)", 2023, "", "", "", "", "2305.10355", "B3 Shortcuts in VLM training",
  "POPE benchmark for object hallucination", "LVLMs answer from co-occurrence priors", "Hallucination, not group robustness", "Related symptom of under-used vision"),
 ("leng2024vcd", "inproceedings", "Leng, Sicong", "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2024, "IEEE", "", "", "10.1109/CVPR52733.2024.01316", "", "B3 Shortcuts in VLM training",
  "Contrast outputs with and without distorted image", "Over-reliance on language priors causes hallucinations", "Decoding-time fix", "Related evidence that LVLMs under-use visual input"),
 ("xu2025lvlmehub", "article", "Xu, Peng", "LVLM-EHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2025, "IEEE", "47", "1877--1893", "10.1109/TPAMI.2024.3507000", "", "B3 Shortcuts in VLM training",
  "Evaluation hub for LVLMs", "LVLMs vary widely; hallucination and over-fitting common", "Benchmark", "Journal-level evidence of LVLM robustness gaps"),

 # ---------------- Theme C: connectors and multimodal LLM design ----------------
 ("alayrac2022flamingo", "inproceedings", "Alayrac, Jean-Baptiste", "Flamingo: a Visual Language Model for Few-Shot Learning", "Advances in Neural Information Processing Systems (NeurIPS)", 2022, "", "", "", "", "2204.14198", "C1 Connector families",
  "Frozen LM with tanh-gated cross-attention layers, gate initialised at 0", "Gated cross-attention preserves the LM at initialisation", "Gate dynamics not analysed", "Learned-gate connector design"),
 ("laurencon2023obelics", "inproceedings", "Laurençon, Hugo", "OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents", "Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks", 2023, "", "", "", "", "2306.16527", "C1 Connector families",
  "Open interleaved data; IDEFICS 9B/80B Flamingo reproduction", "Open gated-cross-attention checkpoints", "Gate values not reported", "We read IDEFICS gates from the checkpoints"),
 ("laurencon2024matters", "inproceedings", "Laurençon, Hugo", "What matters when building vision-language models?", "Advances in Neural Information Processing Systems (NeurIPS)", 2024, "", "", "", "", "2405.02246", "C1 Connector families",
  "Controlled ablations; cross-attention vs fully autoregressive (Idefics2)", "Cross-attention wins with frozen backbones; the fully autoregressive design wins once backbones are trained", "Benchmark accuracy, not robustness", "Conflicting evidence on cross-attention; we add the robustness dimension"),
 ("li2023blip2", "inproceedings", "Li, Junnan", "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", "International Conference on Machine Learning (ICML)", 2023, "PMLR", "", "", "", "2301.12597", "C1 Connector families",
  "Q-Former with learned queries", "Few query tokens bridge frozen encoder and LLM", "Robustness not measured", "Q-Former connector"),
 ("dai2023instructblip", "inproceedings", "Dai, Wenliang", "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning", "Advances in Neural Information Processing Systems (NeurIPS)", 2023, "", "", "", "", "2305.06500", "C1 Connector families",
  "Instruction-aware Q-Former", "Q-Former competitive after instruction tuning", "Robustness not measured", "Q-Former background"),
 ("liu2023llava", "inproceedings", "Liu, Haotian", "Visual Instruction Tuning", "Advances in Neural Information Processing Systems (NeurIPS)", 2023, "", "", "", "", "2304.08485", "C1 Connector families",
  "Linear projector, two-stage training (LLaVA)", "Simple projection plus instruction data works", "Robustness not measured", "MLP projector and two-stage protocol"),
 ("liu2024llava15", "inproceedings", "Liu, Haotian", "Improved Baselines with Visual Instruction Tuning", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2024, "IEEE", "", "26286--26296", "10.1109/CVPR52733.2024.02484", "2310.03744", "C1 Connector families",
  "Two-layer MLP projector (LLaVA-1.5)", "MLP projector is a strong default", "Robustness not measured", "MLP connector"),
 ("bai2023qwenvl", "misc", "Bai, Jinze", "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond", "arXiv preprint", 2023, "", "", "", "", "2308.12966", "C1 Connector families",
  "Single-layer cross-attention adapter compressing to fixed queries", "Resampler-style connector at scale", "Robustness not measured", "Qwen VLM lineage"),
 ("beyer2024paligemma", "misc", "Beyer, Lucas", "PaliGemma: A versatile 3B VLM for transfer", "arXiv preprint", 2024, "", "", "", "", "2407.07726", "C1 Connector families",
  "Linear projection of SigLIP tokens into Gemma", "Simple linear connector suffices for transfer", "Robustness not measured", "SigLIP + simple connector precedent"),
 ("grattafiori2024llama3", "misc", "Grattafiori, Aaron", "The Llama 3 Herd of Models", "arXiv preprint", 2024, "", "", "", "", "2407.21783", "C1 Connector families",
  "Llama 3 including cross-attention vision adapters", "Cross-attention adapters used in a frontier model family", "Gate behaviour not reported", "Shows cross-attention injection is current practice"),
 ("cha2024honeybee", "inproceedings", "Cha, Junbum", "Honeybee: Locality-enhanced Projector for Multimodal LLM", "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", 2024, "", "", "", "", "2312.06742", "C2 Connector design studies",
  "Locality-preserving abstractors", "Token count and locality trade off", "Benchmark accuracy", "Token-count ablation"),
 ("li2025tokenpacker", "article", "Li, Wentong", "TokenPacker: Efficient Visual Projector for Multimodal LLM", "International Journal of Computer Vision", 2025, "Springer", "133", "6794--6812", "10.1007/s11263-025-02491-7", "2407.02392", "C2 Connector design studies",
  "Coarse-to-fine token compression projector", "Fewer tokens keep accuracy", "Benchmark accuracy", "Efficiency-driven connector design"),
 ("mckinzie2024mm1", "inproceedings", "McKinzie, Brandon", "MM1: Methods, Analysis and Insights from Multimodal LLM Pre-training", "European Conference on Computer Vision (ECCV)", 2024, "Springer", "", "304--323", "10.1007/978-3-031-73397-0_18", "2403.09611", "C2 Connector design studies",
  "Large ablation of encoder, connector and data", "Connector type matters little; token count and resolution matter", "Benchmark accuracy", "Agrees with our near-null connector-type effect (on robustness)"),
 ("karamcheti2024prismatic", "inproceedings", "Karamcheti, Siddharth", "Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models", "International Conference on Machine Learning (ICML)", 2024, "PMLR", "", "", "", "2402.07865", "C2 Connector design studies",
  "Controlled design-space study of VLMs", "Single-stage training and encoder fusion matter", "Benchmark accuracy", "Template for controlled comparisons"),
 ("tong2024cambrian", "inproceedings", "Tong, Shengbang", "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs", "Advances in Neural Information Processing Systems (NeurIPS)", 2024, "", "", "", "", "2406.16860", "C2 Connector design studies",
  "Vision-centric study of encoders and connectors", "Encoder choice strongly affects MLLMs", "Benchmark accuracy", "Motivates encoder-scale check"),
 ("huang2024mir", "misc", "Huang, Qidong", "Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate", "arXiv preprint", 2024, "", "", "", "", "2410.07167", "C2 Connector design studies",
  "Alignment metric; MLP vs Q-Former", "MLP aligns better than Q-Former", "Benchmark accuracy", "Contradiction: no MLP vs Q-Former robustness difference here"),
 ("groger2026rethinking", "misc", "Gröger, Fabian", "Rethinking Projector Training in Multimodal LLMs", "arXiv preprint", 2026, "", "", "", "", "2610.18598", "C2 Connector design studies",
  "What projector alignment learns; training-free projector", "Alignment is coarse; small data suffices", "Benchmark accuracy", "Justifies short alignment"),
 ("farazi2026gate", "misc", "Farazi, Moshiur", "The Gate Always Closes: On Injecting Auxiliary Signals into Frozen Vision-Language Models", "arXiv preprint", 2026, "", "", "", "", "2607.23335", "C3 Gating and injection",
  "Five learnable-gate injection designs in frozen VLMs", "Learnable gates end up closed; fixed injection recommended", "Shortcut reliance not measured", "Basis of H1; we test its consequence for robustness"),
 ("bachlechner2021rezero", "inproceedings", "Bachlechner, Thomas", "ReZero is All You Need: Fast Convergence at Large Depth", "Conference on Uncertainty in Artificial Intelligence (UAI)", 2021, "PMLR", "", "", "", "2003.04887", "C3 Gating and injection",
  "Zero-initialised learnable residual scalars", "Zero-init gates speed convergence in deep nets", "Unimodal", "Contrast with fixed-gate design"),
 ("yin2024mllmsurvey", "article", "Yin, Shukang", "A survey on multimodal large language models", "National Science Review", 2024, "Oxford University Press", "11", "nwae403", "10.1093/nsr/nwae403", "2306.13549", "C4 Surveys",
  "Survey of MLLM architectures and training", "Connector taxonomy: projection, query-based, fusion", "Survey", "Connector taxonomy"),
 ("zhang2024vlmsurvey", "article", "Zhang, Jingyi", "Vision-Language Models for Vision Tasks: A Survey", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2024, "IEEE", "46", "5625--5644", "10.1109/TPAMI.2024.3369699", "", "C4 Surveys",
  "Survey of VLM pretraining and transfer", "Taxonomy of VLM objectives and evaluation", "Survey", "Background"),
 ("chen2023vlpsurvey", "article", "Chen, Feilong", "VLP: A Survey on Vision-language Pre-training", "Machine Intelligence Research", 2023, "Springer", "20", "38--56", "10.1007/s11633-022-1369-5", "", "C4 Surveys",
  "Survey of vision-language pretraining", "Fusion architectures: single- vs dual-stream, cross-attention", "Survey", "Background on fusion design"),
 ("awais2025foundation", "article", "Awais, Muhammad", "Foundation Models Defining a New Era in Vision: A Survey and Outlook", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2025, "IEEE", "47", "2245--2264", "10.1109/TPAMI.2024.3506283", "", "C4 Surveys",
  "Survey of vision foundation models", "Robustness and bias listed as open challenges", "Survey", "Hook: significance of VLM robustness"),
 ("jin2025efficientmllm", "article", "Jin, Yizhang", "Efficient multimodal large language models: a survey", "Visual Intelligence", 2025, "Springer", "3", "27", "10.1007/s44267-025-00099-6", "", "C4 Surveys",
  "Survey of efficient MLLMs", "Connector and token-compression designs for efficiency", "Survey", "Compute discussion"),

 # ---------------- Theme D: models and efficient training ----------------
 ("dosovitskiy2021vit", "inproceedings", "Dosovitskiy, Alexey", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "International Conference on Learning Representations (ICLR)", 2021, "", "", "", "", "2010.11929", "D Models and training",
  "Vision Transformer", "Patch tokens as image representation", "", "Patch features fed to connectors"),
 ("zhai2023siglip", "inproceedings", "Zhai, Xiaohua", "Sigmoid Loss for Language Image Pre-Training", "IEEE/CVF International Conference on Computer Vision (ICCV)", 2023, "", "", "", "", "2303.15343", "D Models and training",
  "SigLIP", "Strong encoders at small batch", "", "Frozen encoders (base, large)"),
 ("abouelenin2025phi4mini", "misc", "Microsoft", "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs", "arXiv preprint", 2025, "", "", "", "", "2503.01743", "D Models and training",
  "Phi-4-mini language model", "Compact LLM", "", "Second LLM"),
 ("hu2022lora", "inproceedings", "Hu, Edward J.", "LoRA: Low-Rank Adaptation of Large Language Models", "International Conference on Learning Representations (ICLR)", 2022, "", "", "", "", "2106.09685", "D Models and training",
  "Low-rank adapters", "Cheap fine-tuning", "", "LoRA r=16"),
 ("dettmers2023qlora", "inproceedings", "Dettmers, Tim", "QLoRA: Efficient Finetuning of Quantized LLMs", "Advances in Neural Information Processing Systems (NeurIPS)", 2023, "", "", "", "", "2305.14314", "D Models and training",
  "NF4 quantisation with LoRA", "4-bit fine-tuning at full quality", "", "NF4 + paged optimiser"),
 ("ding2023peft", "article", "Ding, Ning", "Parameter-efficient fine-tuning of large-scale pre-trained language models", "Nature Machine Intelligence", 2023, "Springer Nature", "5", "220--235", "10.1038/s42256-023-00626-4", "", "D Models and training",
  "Delta-tuning analysis", "PEFT approaches full fine-tuning", "", "Justifies LoRA"),
 ("xu2026peftreview", "article", "Xu, Lingling", "Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2026, "IEEE", "48", "6107--6126", "10.1109/TPAMI.2026.3657354", "", "D Models and training",
  "Review of PEFT", "LoRA-family trade-offs", "", "Justifies LoRA"),
 ("chen2016checkpoint", "misc", "Chen, Tianqi", "Training Deep Nets with Sublinear Memory Cost", "arXiv preprint", 2016, "", "", "", "", "1604.06174", "D Models and training",
  "Gradient checkpointing", "Memory-compute trade-off", "", "SigLIP-large runs"),

 # ---------------- Theme E: interpretation, statistics and reproducibility ----------------
 ("jain2019attention", "inproceedings", "Jain, Sarthak", "Attention is not Explanation", "Conference of the North American Chapter of the ACL (NAACL)", 2019, "", "", "", "", "1902.10186", "E1 Attention as evidence",
  "Attention vs gradient importance", "Attention weights often do not explain predictions", "", "Explains why both gates attend to the watermark yet only one relies on it"),
 ("wiegreffe2019attention", "inproceedings", "Wiegreffe, Sarah", "Attention is not not Explanation", "Conference on Empirical Methods in Natural Language Processing (EMNLP)", 2019, "", "", "", "", "1908.04626", "E1 Attention as evidence",
  "Counter-analysis of attention explanations", "Attention can be informative under conditions", "", "Balanced reading of our attention analysis"),
 ("chormai2024disentangled", "article", "Chormai, Pattarawat", "Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces", "IEEE Transactions on Pattern Analysis and Machine Intelligence", 2024, "IEEE", "46", "7283--7299", "10.1109/TPAMI.2024.3388275", "", "E1 Attention as evidence",
  "Subspace explanations; cites DFR", "Explanations can isolate Clever-Hans subspaces", "", "Future work: subspace analysis of the connector"),
 ("henderson2018matters", "inproceedings", "Henderson, Peter", "Deep Reinforcement Learning That Matters", "AAAI Conference on Artificial Intelligence", 2018, "AAAI", "32", "", "10.1609/aaai.v32i1.11694", "1709.06560", "E2 Seeds and statistics",
  "Variance analysis of deep RL results", "Random seeds alone change conclusions", "", "Justifies 13-15 seeds and per-seed reporting"),
 ("bouthillier2021variance", "inproceedings", "Bouthillier, Xavier", "Accounting for Variance in Machine Learning Benchmarks", "Conference on Machine Learning and Systems (MLSys)", 2021, "", "", "", "", "2103.03098", "E2 Seeds and statistics",
  "Sources of variance in ML benchmarks", "Randomising more sources of variation improves conclusions", "", "Seeds vary data order, LoRA init and sampling"),
 ("picard2021seed", "misc", "Picard, David", "Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision", "arXiv preprint", 2021, "", "", "", "", "2109.08203", "E2 Seeds and statistics",
  "Seed sweep on vision models", "Seed choice changes accuracy by more than many method gains", "", "Supports H2 (variance) as a first-class outcome"),
 ("dror2018hitchhiker", "inproceedings", "Dror, Rotem", "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing", "Annual Meeting of the Association for Computational Linguistics (ACL)", 2018, "ACL", "", "1383--1392", "10.18653/v1/P18-1128", "", "E2 Seeds and statistics",
  "Guide to significance testing", "Choose paired / non-parametric tests by data properties", "", "Paired t-test + Wilcoxon"),
 ("schuirmann1987tost", "article", "Schuirmann, Donald J.", "A comparison of the Two One-Sided Tests Procedure and the Power Approach for assessing the equivalence of average bioavailability", "Journal of Pharmacokinetics and Biopharmaceutics", 1987, "Springer", "15", "657--680", "10.1007/BF01068419", "", "E2 Seeds and statistics",
  "TOST equivalence testing", "Equivalence needs its own test, not a non-significant difference", "", "Equivalence claims for connector type"),
]

BOOKS = r"""
@book{efron1993bootstrap,
  author = {Efron, Bradley and Tibshirani, Robert J.},
  title = {An Introduction to the Bootstrap},
  publisher = {Chapman and Hall/CRC},
  year = {1993}
}

@article{holm1979simple,
  author = {Holm, Sture},
  title = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal = {Scandinavian Journal of Statistics},
  volume = {6},
  number = {2},
  pages = {65--70},
  year = {1979}
}

@book{cohen1988power,
  author = {Cohen, Jacob},
  title = {Statistical Power Analysis for the Behavioral Sciences},
  edition = {2},
  publisher = {Lawrence Erlbaum Associates},
  year = {1988}
}

@techreport{wah2011cub,
  author = {Wah, Catherine and Branson, Steve and Welinder, Peter and Perona, Pietro and Belongie, Serge},
  title = {The Caltech-UCSD Birds-200-2011 Dataset},
  institution = {California Institute of Technology},
  number = {CNS-TR-2011-001},
  year = {2011}
}

@misc{qwen2026qwen35,
  author = {{Qwen Team}},
  title = {Qwen3.5-4B (model card)},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3.5-4B}},
  year = {2026}
}
"""


def bib(r):
    key, typ, a, title, venue, year, pub, vol, pages, doi, ax = r[:11]
    f = [f"  author = {{{a + ' and others' if a != 'Microsoft' else '{Microsoft}'}}}", f"  title = {{{{{title}}}}}", f"  year = {{{year}}}"]
    if typ == "article":
        f.append(f"  journal = {{{venue}}}")
    elif typ == "inproceedings":
        f.append(f"  booktitle = {{{venue}}}")
    else:
        f.append(f"  howpublished = {{arXiv:{ax}}}")
    for k, v in (("publisher", pub), ("volume", vol), ("pages", pages), ("doi", doi)):
        if v:
            f.append(f"  {k} = {{{v}}}")
    if ax:
        f += [f"  eprint = {{{ax}}}", "  archivePrefix = {arXiv}"]
    return f"@{typ}{{{key},\n" + ",\n".join(f) + "\n}\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "references.bib"), "w") as fh:
        fh.write("% Generated by paper/build_refs.py. Every entry verified against arXiv or Crossref on 2026-11-01.\n\n")
        fh.write("\n".join(bib(r) for r in R) + BOOKS)
    cols = ["key", "first_author", "year", "title", "venue", "venue_type", "publisher", "doi", "arxiv", "theme",
            "objective_method", "key_finding", "limitation_for_our_question", "use_in_this_study"]
    with open(os.path.join(here, "literature_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in R:
            key, typ, a, title, venue, year, pub, vol, pages, doi, ax, theme, did, found, lim, use = r
            vt = {"article": "journal", "inproceedings": "conference"}.get(typ, "preprint")
            w.writerow([key, a, year, title, venue, vt, pub, doi, ax, theme, did, found, lim, use])
    n_j = sum(r[1] == "article" for r in R)
    print(f"{len(R) + 5} references ({n_j} journal articles, {sum(r[1] == 'inproceedings' for r in R)} conference, "
          f"{sum(r[1] == 'misc' for r in R)} preprints, 5 books/other)")


if __name__ == "__main__":
    main()
