# An Independent Safety Evaluation of Kimi K2.5

This repository contains the paper **"An Independent Safety Evaluation of Kimi K2.5"**.

The paper evaluates Kimi K2.5 across several risk domains, including **CBRNE misuse risk, cybersecurity risk, misalignment, political censorship, bias, and harmlessness**, in both agentic and non-agentic settings. We intend for our work to provide a useful foundation for fostering future releases of safety reports for open-weight models, as well as empowering research institutions around the world to do so in a constructive way.

## Paper

[Read the paper (PDF)](./kimi_k25_safety_report.pdf)

## Summary

Kimi K2.5 is an open-weight model with frontier capabilities across a range of benchmarks, but it was released without a corresponding public safety evaluation report.
In this work, we conduct a preliminary independent assessment of the model's dual-use capabilities and safety behaviors in both agentic and non-agentic settings, with a focus on risks that may be amplified by the powerful open-weight release.

## Key findings
- **CBRNE:** similar dual-use capabilities as frontier closed-source models but weaker refusals on harmful requests.
- **Cybersecurity:** strong cybersecurity knowledge, but no clear evidence of frontier-level autonomous offensive cyber capability
- **Misalignment-related behaviors:** concerning levels of misalignment such as sabotage and self-replication tendencies, but no clear evidence of being a scheming model.
- **Political bias and censorship:** notable narrow censorship and political bias, especially in Chinese
- **Harmlessness:** less refusal towards queries relevant to national security and public safety, disinformation and copyright infringement, but rarely reinforces delusional beliefs when interacting with vulnerable users.

## Repository contents

This repository is intended to host the paper PDF and citation information for the work. arXiv preprint release is coming soon.

## Citation

```bibtex
@misc{yong2026kimik25safety,
  title={An Independent Safety Evaluation of Kimi K2.5},
  author={Zheng-Xin Yong and Parv Mahajan and Andy Wang and Ida Caspary and Yernat Yestekov and Zora Che and Mosh Levy and Elle Najt and Dennis Murphy and Prashant Kulkarni and Lev McKinney and Kei Nishimura-Gasparian and Ram Potham and Aengus Lynch and Michael L. Chen},
  year={2026},
  note={Preprint}
}
