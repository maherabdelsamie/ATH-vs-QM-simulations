import bluequbit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy.stats import entropy, ttest_ind, pearsonr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for experiments
class ExperimentConfig:
    def __init__(self, num_qubits_range, energy_scales, evolution_times, shots_per_test, repetitions, initial_states):
        self.num_qubits_range = num_qubits_range
        self.energy_scales = energy_scales
        self.evolution_times = evolution_times
        self.shots_per_test = shots_per_test
        self.repetitions = repetitions
        self.initial_states = initial_states

# Experiment class
class ComparativeExperiment:
    def __init__(self, bluequbit_token: str, config: ExperimentConfig):
        self.bq = bluequbit.init(bluequbit_token)
        self.config = config

    def create_qm_circuit(self, num_qubits: int, energy_scale: float, evolution_time: float, initial_state: List[int]) -> QuantumCircuit:
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)

        for i, state in enumerate(initial_state):
            if state == 1:
                circuit.x(i)

        for i in range(num_qubits):
            circuit.h(i)
            phase = energy_scale * evolution_time * np.pi
            circuit.rz(phase, i)

        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)

        circuit.measure(qr, cr)
        return circuit

    def create_ath_circuit(self, num_qubits: int, energy_scale: float, evolution_time: float, initial_state: List[int]) -> QuantumCircuit:
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)

        for i, state in enumerate(initial_state):
            if state == 1:
                circuit.x(i)

        for i in range(num_qubits):
            circuit.h(i)
            phase = energy_scale * (i + 1) * evolution_time * np.pi
            circuit.rz(phase, i)

        num_layers = max(3, int(evolution_time * 20))

        for layer in range(num_layers):
            for i in range(num_qubits):
                phase = energy_scale * evolution_time * np.pi / num_layers
                circuit.rz(phase * (i + 1), i)
                circuit.rx(evolution_time * np.pi / (2 * num_layers), i)

            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rz(energy_scale * evolution_time / num_layers, i + 1)

            if layer % 2 == 0:
                for i in range(0, num_qubits - 2, 2):
                    circuit.cz(i, i + 2)

        circuit.measure(qr, cr)
        return circuit

    def analyze_results(self, counts: Dict, label: str, num_qubits: int) -> Dict:
        total_shots = sum(counts.values())

        excitation_probs = np.zeros(num_qubits)
        for bitstring, count in counts.items():
            for i, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    excitation_probs[i] += count / total_shots

        probabilities = np.array(list(counts.values())) / total_shots
        state_entropy = entropy(probabilities)

        coherence = np.sum(probabilities ** 2)
        phase_variance = np.var([np.angle(complex(np.cos(prob), np.sin(prob))) for prob in probabilities])

        return {
            'excitation_probabilities': excitation_probs.tolist(),
            'state_entropy': state_entropy,
            'coherence': coherence,
            'phase_variance': phase_variance,
            'label': label
        }

    def run_experiment(self, num_qubits: int, energy_scale: float, evolution_time: float, initial_state: List[int]) -> Dict:
        ath_circuit = self.create_ath_circuit(num_qubits, energy_scale, evolution_time, initial_state)
        qm_circuit = self.create_qm_circuit(num_qubits, energy_scale, evolution_time, initial_state)

        ath_result = self.bq.run(ath_circuit, shots=self.config.shots_per_test, device='cpu')
        qm_result = self.bq.run(qm_circuit, shots=self.config.shots_per_test, device='cpu')

        ath_analysis = self.analyze_results(ath_result.get_counts(), 'ATH', num_qubits)
        qm_analysis = self.analyze_results(qm_result.get_counts(), 'QM', num_qubits)

        return {
            'ath': ath_analysis,
            'qm': qm_analysis
        }

    def plot_results(self, results):
        evolution_times = self.config.evolution_times

        avg_ath_entropy = {t: [] for t in evolution_times}
        avg_qm_entropy = {t: [] for t in evolution_times}
        avg_ath_coherence = {t: [] for t in evolution_times}
        avg_qm_coherence = {t: [] for t in evolution_times}
        avg_ath_phase_variance = {t: [] for t in evolution_times}
        avg_qm_phase_variance = {t: [] for t in evolution_times}

        for res in results:
            for t_idx, t in enumerate(self.config.evolution_times):
                avg_ath_entropy[t].append(res['ath']['state_entropy'])
                avg_qm_entropy[t].append(res['qm']['state_entropy'])
                avg_ath_coherence[t].append(res['ath']['coherence'])
                avg_qm_coherence[t].append(res['qm']['coherence'])
                avg_ath_phase_variance[t].append(res['ath']['phase_variance'])
                avg_qm_phase_variance[t].append(res['qm']['phase_variance'])

        plt.figure(figsize=(10, 5))
        plt.plot(
            evolution_times,
            [np.mean(avg_ath_entropy[t]) for t in evolution_times],
            label='ATH Entropy', marker='o'
        )
        plt.plot(
            evolution_times,
            [np.mean(avg_qm_entropy[t]) for t in evolution_times],
            label='QM Entropy', marker='x'
        )
        plt.xlabel('Evolution Time')
        plt.ylabel('Average Entropy')
        plt.title('Entropy Trends: ATH vs QM')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(
            evolution_times,
            [np.mean(avg_ath_coherence[t]) for t in evolution_times],
            label='ATH Coherence', marker='o'
        )
        plt.plot(
            evolution_times,
            [np.mean(avg_qm_coherence[t]) for t in evolution_times],
            label='QM Coherence', marker='x'
        )
        plt.xlabel('Evolution Time')
        plt.ylabel('Average Coherence')
        plt.title('Coherence Trends: ATH vs QM')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(
            evolution_times,
            [np.mean(avg_ath_phase_variance[t]) for t in evolution_times],
            label='ATH Phase Variance', marker='o'
        )
        plt.plot(
            evolution_times,
            [np.mean(avg_qm_phase_variance[t]) for t in evolution_times],
            label='QM Phase Variance', marker='x'
        )
        plt.xlabel('Evolution Time')
        plt.ylabel('Average Phase Variance')
        plt.title('Phase Variance Trends: ATH vs QM')
        plt.legend()
        plt.grid()
        plt.show()

    def compute_phase_entropy_correlation(self, results):
        ath_entropies = [res['ath']['state_entropy'] for res in results]
        ath_phase_variances = [res['ath']['phase_variance'] for res in results]
        qm_entropies = [res['qm']['state_entropy'] for res in results]
        qm_phase_variances = [res['qm']['phase_variance'] for res in results]

        ath_corr, ath_pval = pearsonr(ath_entropies, ath_phase_variances)
        qm_corr, qm_pval = pearsonr(qm_entropies, qm_phase_variances)

        logger.info(f"ATH Phase-Entropy Correlation: {ath_corr:.4f} (p-value: {ath_pval:.4e})")
        logger.info(f"QM Phase-Entropy Correlation: {qm_corr:.4f} (p-value: {qm_pval:.4e})")

        print("=== Phase-Entropy Correlation ===")
        print(f"ATH Phase-Entropy Correlation: {ath_corr:.4f} (p-value: {ath_pval:.4e})")
        print(f"QM Phase-Entropy Correlation: {qm_corr:.4f} (p-value: {qm_pval:.4e})")

    def summarize_results(self, results):
        for result in results:
            ath = result['ath']
            qm = result['qm']
            entropy_ratio = ath['state_entropy'] / qm['state_entropy']
            coherence_ratio = ath['coherence'] / qm['coherence']
            dominance_measure = ath['coherence'] - qm['coherence']

            print("=== Comparative Analysis: ATH vs Standard QM ===")
            print(f"Entropy Ratio (ATH/QM): {entropy_ratio:.4f}")
            print(f"Coherence Ratio (ATH/QM): {coherence_ratio:.4f}")
            print(f"ATH Dominance Measure: {dominance_measure:.4f}")
            print(f"ATH Phase Variance: {ath['phase_variance']:.4f}")
            print(f"QM Phase Variance: {qm['phase_variance']:.4f}")
            print()

    def run_full_experiment(self):
        results = []

        for num_qubits in self.config.num_qubits_range:
            for initial_state in self.config.initial_states:
                if len(initial_state) > num_qubits:
                    continue
                for energy_scale in self.config.energy_scales:
                    for evolution_time in self.config.evolution_times:
                        for _ in range(self.config.repetitions):
                            result = self.run_experiment(num_qubits, energy_scale, evolution_time, initial_state[:num_qubits])
                            results.append(result)

        self.plot_results(results)
        self.compute_phase_entropy_correlation(results)
        self.summarize_results(results)

if __name__ == "__main__":
    config = ExperimentConfig(
        num_qubits_range=range(3, 7),  # Simulate for 3 to 6 qubits
        energy_scales=[0.1, 0.5, 1.0, 1.5, 2.0],
        evolution_times=[0.25, 0.5, 1.0, 1.5, 2.0],
        shots_per_test=1000,
        repetitions=3,
        initial_states=[[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1]]
    )

    BLUEQUBIT_TOKEN = "your_bluequbit_token_here"

    experiment = ComparativeExperiment(BLUEQUBIT_TOKEN, config)
    logger.info("Starting the ATH vs QM experiment...")
    experiment.run_full_experiment()
    logger.info("Experiment complete.")
