from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector



# We now proceed to build the QAOA ansatz using the swapped cost layer
def build_optimal_qaoa_ansatz(num_qubits, qaoa_layers, swapped_cost_layer):
  qaoa_circuit = QuantumCircuit(num_qubits, num_qubits)

  # Add initial state -> equal superposition
  qaoa_circuit.h(range(num_qubits))

  # Re-parametrize the circuit
  gammas = ParameterVector("γ", qaoa_layers)
  betas = ParameterVector("β", qaoa_layers)

  # Define mixer layer
  mixer_layer = QuantumCircuit(num_qubits)
  mixer_layer.rx(-2 * betas[0], range(num_qubits))

  # iterate over number of qaoa layers
  # and alternate cost/reversed cost and mixer
  for layer in range(qaoa_layers):

    bind_dict = {swapped_cost_layer.parameters[0]: gammas[layer]}
    bound_swapped_cost_layer = swapped_cost_layer.assign_parameters(bind_dict)

    bind_dict = {mixer_layer.parameters[0]: betas[layer]}
    bound_mixer_layer = mixer_layer.assign_parameters(bind_dict)

    if layer % 2 == 0:
      # even layer -> append cost
      qaoa_circuit.compose(bound_swapped_cost_layer, range(num_qubits), inplace=True)
    else:
      # odd layer -> append reversed cost
      qaoa_circuit.compose(bound_swapped_cost_layer.reverse_ops(), range(num_qubits), inplace=True)

    # the mixer layer is not reversed
    qaoa_circuit.compose(bound_mixer_layer, range(num_qubits), inplace=True)

  # qaoa_circuit.barrier()

  if qaoa_layers % 2 == 1:
    # iterate over layout permutations to recover measurements
    for qidx, cidx in enumerate(swapped_cost_layer.layout.initial_index_layout()):
      qaoa_circuit.measure(qidx, cidx)
  else:
    for idx in range(num_qubits):
      qaoa_circuit.measure(idx, idx)

  return qaoa_circuit