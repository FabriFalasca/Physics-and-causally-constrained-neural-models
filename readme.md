# Codes for "Physics and causally constrained discrete-time neural models of turbulent dynamical systems"
(Work in progress)

"Every morning, a software engineer in Santa Clara wakes up and starts hunting for bad code written by physicists..." 
I. Newton; March 30, 1727

Find the paper in https://arxiv.org/abs/2602.13847

- Folder CdV
Codes to replicate the results in Section IIIA of the paper on the Charney-DeVore model. 
Here you find 3 folders: 
-- Numerical: codes to run the ground truth simulation and responses to impulse and step function forcings.
-- score_matching: codes to infer the score of the system, responses via the Fluctuation-Dissipation Theorem and the causal adjacency matrix from data.
-- neural_models: Codes for the Physics constrained and "Physics & Causal" consttrained neural models to fit from data.

- Folder L96
Same as for the Folder CdV but for the Symmetry Broken Lorenz-96 considered in Section IIIB of the paper.

- Folder Splitting-Approximation-Examples
Here find the simple experiments defined in Section II of the Supplemental Material. We show how, given a small $\Delta t$ we can approximate the 
Lorenz-63 and a triad model using the splitting procedure considered in the main paper. In the case of the Lorenz-63 system 
this leads to a nice sequential reformulation of the dynamics.
