# AUTOHALLUSION: Automatic Generation of Hallucination Benchmarks for Vision-Language Models [[EMNLP 2024 Findings]](https://2024.emnlp.org/)

[Xiyang Wu*](https://wuxiyang1996.github.io/), [Tianrui Guan*](https://tianruiguan.phd), Dianqi Li, Shuaiyi Huang, Xiaoyu Liu, Xijun Wang, Ruiqi Xian, Abhinav Shrivastava, Furong Huang, Jordan Lee Boyd-Graber, Tianyi Zhou, Dinesh Manocha

**[Paper](https://arxiv.org/abs/2406.10900)** | **[Twitter](https://x.com/terryguan97/status/1814757000066253120)** | **[Website](https://wuxiyang1996.github.io/autohallusion_page/)** | **[Code](https://github.com/wuxiyang1996/AutoHallusion)**

<p align="center" >
      <img src="./imgs/teaser.png" alt="Teaser" class="center" width="800"/>
</p>

## Updates
- [09/20] Our **[AUTOHALLUSION](https://arxiv.org/abs/2406.10900)** is accepted by **[EMNLP 2024 Findings](https://2024.emnlp.org/)**.
- [06/15] We release our early version of **[AUTOHALLUSION](https://arxiv.org/abs/2406.10900)**, as an extension of our prior work **[HallusionBench](https://arxiv.org/abs/2310.14566)**.

## We delightedly welcome everyone in our community to continue exploring the mechanisms causing hallucinations of Large Multimodal Models (GPT-4V) and their mitigations!
Large vision-language models (LVLMs) hallucinate: certain context cues in an image may trigger the language module's overconfident and incorrect reasoning on abnormal or hypothetical objects. Though a few benchmarks have been developed to investigate LVLM hallucinations, they mainly rely on hand-crafted corner cases whose fail patterns may hardly generalize, and finetuning on them could undermine their validity. These motivate us to develop the first automatic benchmark generation approach, AUTOHALLUSION, that harnesses a few principal strategies to create diverse hallucination examples. It probes the language modules in LVLMs for context cues and uses them to synthesize images by: (1) adding objects abnormal to the context cues; (2) for two co-occurring objects, keeping one and excluding the other; or (3) removing objects closely tied to the context cues. It then generates image-based questions whose ground-truth answers contradict the language module's prior. A model has to overcome contextual biases and distractions to reach correct answers, while incorrect or inconsistent answers indicate hallucinations. AUTOHALLUSION enables us to create new benchmarks at the minimum cost and thus overcomes the fragility of hand-crafted benchmarks. It also reveals common failure patterns and reasons, providing key insights to detect, avoid, or control hallucinations. Comprehensive evaluations of top-tier LVLMs, e.g., GPT-4V(ision), Gemini Pro Vision, Claude 3, and LLaVA-1.5, show a 97.7% and 98.7% success rate of hallucination induction on synthetic and real-world datasets of AUTOHALLUSION, paving the way for a long battle against hallucinations.

If you find our paper useful, please cite our paper:
```bibtex
@misc{wu2024autohallusionautomaticgenerationhallucination,
      title={AUTOHALLUSION: Automatic Generation of Hallucination Benchmarks for Vision-Language Models}, 
      author={Xiyang Wu and Tianrui Guan and Dianqi Li and Shuaiyi Huang and Xiaoyu Liu and Xijun Wang and Ruiqi Xian and Abhinav Shrivastava and Furong Huang and Jordan Lee Boyd-Graber and Tianyi Zhou and Dinesh Manocha},
      year={2024},
      eprint={2406.10900},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.10900}, 
}

@misc{guan2023hallusionbench,
      title={HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination & Visual Illusion in Large Vision-Language Models}, 
      author={Tianrui Guan and Fuxiao Liu and Xiyang Wu and Ruiqi Xian and Zongxia Li and Xiaoyu Liu and Xijun Wang and Lichang Chen and Furong Huang and Yaser Yacoob and Dinesh Manocha and Tianyi Zhou},
      year={2023},
      eprint={2310.14566},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Dependency

## Demo

## Usage

## Leaderboard

## License
This repository is under [BSD 3-Clause License](LICENSE.md). 