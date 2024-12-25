# Quantifying Entropy-Phase Dynamics: Comparative Simulations of Active Time Hypothesis and Quantum Mechanics

Dr. Maher Abdelsamie<br>maherabdelsamie@gmail.com<br>

### Abstract
The Active Time Hypothesis (ATH) [1] proposes a novel mechanism of entropy-phase coupling that deviates from standard quantum mechanics (QM). To investigate this hypothesis, we developed a comparative simulation framework that evaluates entropy, coherence, and phase variance metrics under both ATH and QM models across a range of parameters. Our results reveal significant differences in phase-sensitive behaviors, enhanced coherence persistence, and entropy-phase coupling under ATH. These findings offer a compelling basis for ATH’s theoretical validity and pave the way for experimental validation. The simulation framework and its results are discussed in detail.

## References

1. Abdelsamie, Maher, Redefining Gravity and Bridging Quantum Mechanics and Classical Physics: The Active Time Theory
 (March 12, 2024). Available at SSRN: http://dx.doi.org/10.2139/ssrn.4762792

---
# Installation

The simulation is implemented in Python and requires the following libraries:

- `numpy`
- `matplotlib`
- `qiskit`
- `scipy`
- `typing`
- `logging`
- `bluequbit` (for interfacing with the quantum computing platform)

To set up the environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**:

   You can install the required libraries using `pip`. Run the following command in the root of the repository:

   ```bash
   pip install numpy matplotlib qiskit scipy seaborn bluequbit
   ```

3. **BlueQubit Authentication**:
   The simulation requires a BlueQubit authentication token to connect to the quantum computing backend. To obtain the token, sign up for an account on [BlueQubit’s website](https://www.bluequbit.io/) and retrieve your API key. Store the token in a secure place, as you’ll need to input it when running the simulation.

4. **Running the Simulation**:
   Once the dependencies are installed, you can run the main script using:

   ```bash
   python main.py
   ```

This will initiate the simulation, run validation tests, and produce the results and visualizations as specified in the code.

---

## License

See the LICENSE.md file for details.

## Citing This Work

You can cite it using the information provided in the `CITATION.cff` file available in this repository.
