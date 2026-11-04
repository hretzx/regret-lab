# Adaptive Clinical Trials using Multi-Armed Bandits

A simulation-based project that compares different patient-allocation strategies in clinical trials. The goal is to understand how adaptive algorithms perform when treatment outcomes are uncertain, difficult to distinguish, or change over time.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Exploration vs Exploitation](#exploration-vs-exploitation)
- [Algorithms Implemented](#algorithms-implemented)
- [Experiments](#experiments)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Validation](#validation)
- [Limitations](#limitations)
- [References](#references)

---

## Problem Statement

A clinical trial compares multiple treatments, where each patient receives one treatment and has a binary outcome: cured ($1$) or not cured ($0$).

Each treatment has an unknown probability of success. Traditional approaches distribute patients equally among treatments, while adaptive approaches use previous outcomes to decide where to allocate the next patient.

This project compares both approaches and studies how they perform under different conditions.

---

## Exploration vs Exploitation

Adaptive algorithms face two choices:

* **Exploration:** Try different treatments to learn how effective they are.
* **Exploitation:** Allocate more patients to treatments that appear to perform better.

The challenge is to balance both without allocating too many patients to less effective treatments.

The project uses pseudo-regret to measure this trade-off. It represents the difference between the best possible treatment reward and the reward from the treatment actually selected.

---

## Algorithms Implemented

* **Equal Split:** Distributes patients equally among all treatments.
* **Epsilon-Greedy:** Explores randomly $10\%$ of the time and otherwise selects the treatment with the highest observed success rate.
* **UCB1:** Uses observed success rates and uncertainty to balance exploration and exploitation.
* **Thompson Sampling:** Uses Beta probability distributions to select treatments based on their estimated success rates.
* **Sliding-Window UCB:** Uses only recent observations to adapt when treatment effectiveness changes.
* **Discounted UCB:** Gives more importance to recent observations by reducing the influence of older ones.

---

## Experiments

The project includes four experiments:

1. **Baseline Comparison:** Compares algorithm performance across different treatment scenarios.
2. **Log-Regret Analysis:** Studies how cumulative regret changes as the number of patients increases.
3. **Gap Sensitivity:** Examines how differences in treatment success rates affect algorithm performance.
4. **Non-Stationary Environment:** Tests how algorithms respond when treatment effectiveness changes over time.

---

## Tech Stack

* Python
* NumPy
* Matplotlib
* Streamlit
* SciPy

---

## How to Run

1. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the sanity tests:**
   ```bash
   python -m tests.test_sanity
   ```

3. **Run individual experiments:**
   ```bash
   python -m experiments.exp1_baseline --seed 0
   python -m experiments.exp2_log_regret_check --seed 0
   python -m experiments.exp3_gap_sensitivity --seed 0
   python -m experiments.exp4_drift --seed 0
   ```

4. **Launch the interactive dashboard:**
   ```bash
   streamlit run app.py
   ```

5. **For a quicker test run:**
   ```bash
   python -m experiments.exp1_baseline --trials 30 --horizon 500
   ```

> **Note:** Generated figures are saved in the `results/` directory.

---

## Validation

The project includes sanity tests to check:

* Zero regret when only one treatment is available.
* Exploration behaviour of UCB1.
* Non-negative and non-decreasing cumulative regret.
* Reproducibility with the same random seed.
* Thompson Sampling performance on a simple scenario.
* Confidence interval calculations.

---

## Limitations

* The project is simulation-based and does not use real patient data.
* It assumes binary outcomes and immediate feedback.
* Real clinical trials involve additional factors such as ethics, patient safety, and statistical significance.
* Adaptive allocation can affect statistical power and the interpretation of treatment comparisons.
* The drift experiment focuses on an abrupt change in treatment effectiveness rather than gradual changes.

---

## References

* Auer, Cesa-Bianchi and Fischer (2002). *Finite-time Analysis of the Multiarmed Bandit Problem*. Machine Learning.
* Thompson (1933). *On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples*. Biometrika.
* Garivier and Moulines (2011). *On Upper-Confidence Bound Policies for Switching Bandit Problems*.
* Sutton and Barto. *Reinforcement Learning: An Introduction*, 2nd Edition, Chapter 2.