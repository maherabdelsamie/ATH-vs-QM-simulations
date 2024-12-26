import bluequbit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr
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

        ath_result = self.bq.run(ath_circuit, shots=self.config.shots_per_test, device='quantum')  # Use Quantum Device
        qm_result = self.bq.run(qm_circuit, shots=self.config.shots_per_test, device='quantum')  # Use Quantum Device

        ath_analysis = self.analyze_results(ath_result.get_counts(), 'ATH', num_qubits)
        qm_analysis = self.analyze_results(qm_result.get_counts(), 'QM', num_qubits)

        return {
            'ath': ath_analysis,
            'qm': qm_analysis
        }

    def plot_results(self, results):
        ath_entropy = [res['ath']['state_entropy'] for res in results]
        qm_entropy = [res['qm']['state_entropy'] for res in results]
        ath_coherence = [res['ath']['coherence'] for res in results]
        qm_coherence = [res['qm']['coherence'] for res in results]
        ath_phase_variance = [res['ath']['phase_variance'] for res in results]
        qm_phase_variance = [res['qm']['phase_variance'] for res in results]

        plt.figure(figsize=(10, 5))
        plt.plot(ath_entropy, label="ATH Entropy", marker='o')
        plt.plot(qm_entropy, label="QM Entropy", marker='x')
        plt.title("Entropy Comparison: ATH vs QM")
        plt.xlabel("Experiment Index")
        plt.ylabel("Entropy")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(ath_coherence, label="ATH Coherence", marker='o')
        plt.plot(qm_coherence, label="QM Coherence", marker='x')
        plt.title("Coherence Comparison: ATH vs QM")
        plt.xlabel("Experiment Index")
        plt.ylabel("Coherence")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(ath_phase_variance, label="ATH Phase Variance", marker='o')
        plt.plot(qm_phase_variance, label="QM Phase Variance", marker='x')
        plt.title("Phase Variance Comparison: ATH vs QM")
        plt.xlabel("Experiment Index")
        plt.ylabel("Phase Variance")
        plt.legend()
        plt.grid()
        plt.show()

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
        self.summarize_results(results)

# Main program
if __name__ == "__main__":
    config = ExperimentConfig(
        num_qubits_range=range(3, 5),  # 3 and 4 qubits
        energy_scales=[0.1, 1.0],  # Two energy scales
        evolution_times=[1.0],  # Reduced to one evolution time
        shots_per_test=1000,  # Increased shots for better accuracy
        repetitions=1,  # Single repetition
        initial_states=[[0, 0, 0], [1, 0, 1], [1, 1, 0]]  # Three initial states
    )

    BLUEQUBIT_TOKEN = "your_bluequbit_token_here"

    experiment = ComparativeExperiment(BLUEQUBIT_TOKEN, config)
    logger.info("Starting the optimized ATH vs QM experiment on Bluequbit Quantum Device...")
    experiment.run_full_experiment()
    logger.info("Experiment complete.")
