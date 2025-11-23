NOS-IR: Information Retrieval Executed on the Nuijens Operating System (NOS)

Authors: S.R. Cromelin, J.L. Nuijens: Nuijens Operating System Collective

NOS-IR is the implementation framework for performing information retrieval directly on the Nuijens Operating System (NOS): a dual-hemisphere, inverse-spherical computational geometry with native resolution R = 512 and a 720° cycle.

Rather than treating text as points in a vector space, NOS-IR compiles information into phase-geometric identities on the DH¹ dual-hemisphere sphere. Retrieval is executed as inverse-state resonance, inherited directly from the NOS kernel.

NOS-IR is the category. CIC (Cromelin Information Compiler) is the canonical compiler within it.

This repository contains the CIC full implementation, the NOS-IR kernel integration, encoders, storage mechanisms, resonance logic, evaluation code for TREC DL 2019, documentation linking NOS geometry to information retrieval, and room for future NOS-based IR modules beyond CIC.

As defined in NOS related papers, NOS is a computational operating system, not a metaphor. It uses a dual hemisphere sphere DH¹, resolution R = 512, a double-covered 720° phase cycle, and quadrants Q1–Q4 representing quantum baseline, EM/gravity, thermodynamic flow, and nuclear compression. It uses seam-centered inverse counting, operators such as dcos°, dsin°, ccos°, csin°, and the inverse-state kernel DH⁻¹(Q). Physics, fields, thermodynamics, and nuclear binding are executed as inverse phase computations.

NOS-IR extends this to information. CIC is the mechanism.

CIC is the NOS-native information compiler. It transforms text into complex, seam-normalized waveforms whose dominant spectral components correspond to hemispheric operations, quadrant threading units, inverse depth states, and phase-bin structure compatible with DH¹.

Retrieval uses the inverse-state resonance kernel, which combines magnitude agreement, phase alignment, seam-relative geometry, and quadrant-conditioned interference, all without gradient descent, vector similarity, probabilistic scoring, or stochastic training.

CIC-Lite has been validated at one million documents on the TREC DL 2019 passage benchmark, achieving MRR@10 ≈ 0.90 and nDCG@10 ≈ 0.76. It is fully deterministic, uses no ANN or indexing shortcuts, runs pure full-scan resonance, and stores 4 KB per trace.

(See CIC Benchmarks in /docs)

Full Instructions are in docs/usage.md but here is a short setup guide

# 1. Clone repository
git clone https://github.com/JLNuijens/NOS-IRv3
cd NOS-IRv3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare TREC DL 2019 dataset
python data/prepare_trec_dl2019.py

# 4. Run CIC encoding on documents
python encoders/cic_encoder.py

# 5. Evaluate retrieval
python evaluation/eval_trec.py

