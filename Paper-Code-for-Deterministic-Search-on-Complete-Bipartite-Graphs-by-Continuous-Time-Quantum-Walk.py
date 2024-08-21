from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates.z import ZGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit.primitives.sampler import Sampler
from qiskit.visualization import circuit_drawer, plot_histogram, plot_state_city
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt

qc = QuantumCircuit(4)

# control_Z gate
control001 = ZGate().control(3, None, '001')
control010 = ZGate().control(3, None, '010')
control011 = ZGate().control(3, None, '011')
control100 = ZGate().control(3, None, '100')
control101 = ZGate().control(3, None, '101')
control110 = ZGate().control(3, None, '110')
control111 = ZGate().control(3, None, '111')

# control_H gate
control2 = HGate().control(1, None, '0')
control3 = HGate().control(2, None, '10')
control4 = HGate().control(3, None, '000')

# iteration number
itr = math.ceil(np.pi/4*math.sqrt(4)-1/2)
# evolution time
t = 2/math.sqrt(8*4)*np.arcsin(math.sqrt(4)*np.sin(np.pi/(2*(2*itr+1))))

# phase gate
a = cmath.exp(-1j*math.sqrt(8*4)*t/2)
b = cmath.exp(-1j*math.sqrt(8*4)*t)
gate_matrix1 = [[a, 0], [0, a.conjugate()]]
gate_matrix2 = [[b, 0], [0, b.conjugate()]]
phase_gate1 = Operator(gate_matrix1)
phase_gate2 = Operator(gate_matrix2)
control5 = UnitaryGate(phase_gate1).control(3, None, '000')
control6 = UnitaryGate(phase_gate2).control(3, None, '000')

# preparing the initial state
qc.h(0)
qc.h(1)
qc.h(2)

# preprocessing (implementing exp(-iAt/2))
qc.append(control2, [3, 0])
qc.append(control2, [3, 1])
qc.append(control2, [3, 2])
qc.append(control3, [2, 3, 0])
qc.append(control3, [2, 3, 1])
qc.append(control4, [0, 1, 2, 3])

qc.append(control5, [0, 1, 2, 3])

qc.append(control4, [0, 1, 2, 3])
qc.append(control3, [2, 3, 1])
qc.append(control3, [2, 3, 0])
qc.append(control2, [3, 2])
qc.append(control2, [3, 1])
qc.append(control2, [3, 0])

i = 0
while i < itr:
    # oracle that recognizes the marked vertex
    qc.append(control001, [0, 1, 2, 3])
    qc.append(control010, [0, 1, 2, 3])
    qc.append(control011, [0, 1, 2, 3])
    qc.append(control100, [0, 1, 2, 3])
    qc.append(control101, [0, 1, 2, 3])
    qc.append(control110, [0, 1, 2, 3])
    qc.append(control111, [0, 1, 2, 3])
    # implementing exp(-iAt)
    qc.append(control2, [3, 0])
    qc.append(control2, [3, 1])
    qc.append(control2, [3, 2])
    qc.append(control3, [2, 3, 0])
    qc.append(control3, [2, 3, 1])
    qc.append(control4, [0, 1, 2, 3])
    qc.append(control6, [0, 1, 2, 3])
    qc.append(control4, [0, 1, 2, 3])
    qc.append(control3, [2, 3, 1])
    qc.append(control3, [2, 3, 0])
    qc.append(control2, [3, 2])
    qc.append(control2, [3, 1])
    qc.append(control2, [3, 0])
    i = i+1

# Sampler
qc_measured = qc.measure_all(inplace=False)
sampler = Sampler()
result = sampler.run(qc_measured, shots=1024).result()
print(f"Quasi probability distribution: {result.quasi_dists}")
qc.draw(output='mpl')
plt.show()

# qc.measure_all()
# # Transpile for simulator
# simulator = AerSimulator()
# qc = transpile(qc, simulator)
# # run and get counts
# result = simulator.run(qc).result()
# counts = result.get_counts(qc)
# print('counts:', counts)
# plot_histogram(counts, title='State Count')
# plt.show()
