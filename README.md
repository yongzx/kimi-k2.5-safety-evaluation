# An Independent Safety Evaluation of Kimi K2.5

This repository contains the paper **"An Independent Safety Evaluation of Kimi K2.5"**.

The paper evaluates Kimi K2.5 across several risk domains, including **CBRNE misuse risk, cybersecurity risk, misalignment, political censorship, bias, and harmlessness**, in both agentic and non-agentic settings. We intend for our work to provide a useful foundation for fostering future releases of safety reports for open-weight models, as well as empowering research institutions around the world to do so in a constructive way.

## Paper

[Read the paper (PDF)](./kimi_k25_safety_report.pdf)

## Summary

Kimi K2.5 is an open-weight model with frontier capabilities across a range of benchmarks, but it was released without a corresponding public safety evaluation report.
In this work, we conduct a preliminary independent assessment of the model's dual-use capabilities and safety behaviors in both agentic and non-agentic settings, with a focus on risks that may be amplified by the powerful open-weight models.

## Key findings
- **CBRNE:** similar dual-use capabilities as frontier closed-source models but weaker refusals on harmful requests.
- **Cybersecurity:** strong cybersecurity knowledge, but no clear evidence of frontier-level autonomous offensive cyber capability
- **Misalignment-related behaviors:** concerning levels of misalignment such as sabotage and self-replication tendencies, but no clear evidence of being a scheming model.
- **Political bias and censorship:** notable narrow censorship and political bias, especially in Chinese
- **Harmlessness:** less refusal towards queries relevant to national security and public safety, disinformation and copyright infringement, but rarely reinforces delusional beliefs when interacting with vulnerable users.

## Reproduction

To reproduce the Kimi K2.5 benchmark results, fork or clone this repository, install `uv`, set your API keys, and run the one-button script:

```bash
git clone https://github.com/yongzx/kimi-k2.5-safety-evaluation.git
cd kimi-k2.5-safety-evaluation
curl -LsSf https://astral.sh/uv/install.sh | sh
export OPENROUTER_API_KEY="..."
export RUNPOD_API_KEY="..."
./run_kimi_k25_paper_reprod.sh
```

## Citation

```bibtex
@misc{yong2026kimik25safety,
  title={An Independent Safety Evaluation of Kimi K2.5},
  author={Zheng-Xin Yong and Parv Mahajan and Andy Wang and Ida Caspary and Yernat Yestekov and Zora Che and Mosh Levy and Elle Najt and Dennis Murphy and Prashant Kulkarni and Lev McKinney and Kei Nishimura-Gasparian and Ram Potham and Aengus Lynch and Michael L. Chen},
  year={2026},
  note={Preprint}
}
```
