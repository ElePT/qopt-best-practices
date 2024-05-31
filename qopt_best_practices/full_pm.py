from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

class QAOAPass(TransformationPass):

  def __init__(self, num_layers, num_qubits, init_state=None, mixer_layer=None):

    super().__init__()
    self.num_layers = num_layers
    self.num_qubits = num_qubits

    if init_state is None:
      # Add default initial state -> equal superposition
      self.init_state = QuantumCircuit(num_qubits)
      self.init_state.h(range(num_qubits))
    else:
      self.init_state = init_state

    if mixer_layer is None:
      # Define default mixer layer
      p = Parameter('p')
      self.mixer_layer = QuantumCircuit(num_qubits)
      self.mixer_layer.rx(-2 * p, range(num_qubits))
    else:
      self.mixer_layer = mixer_layer

    # self.requires = False
    # self.preserves = False

  def run(self, cost_layer_dag):

    cost_layer = dag_to_circuit(cost_layer_dag)
    qaoa_circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
    # Re-parametrize the circuit
    gammas = ParameterVector("γ", self.num_layers)
    betas = ParameterVector("β", self.num_layers)

    # Add initial state
    qaoa_circuit.compose(self.init_state, inplace=True)

    # iterate over number of qaoa layers
    # and alternate cost/reversed cost and mixer
    for layer in range(self.num_layers):

      bind_dict = {cost_layer.parameters[0]: gammas[layer]}
      bound_cost_layer = cost_layer.assign_parameters(bind_dict)

      bind_dict = {self.mixer_layer.parameters[0]: betas[layer]}
      bound_mixer_layer = self.mixer_layer.assign_parameters(bind_dict)

      if layer % 2 == 0:
        # even layer -> append cost
        qaoa_circuit.compose(bound_cost_layer, range(self.num_qubits), inplace=True)
      else:
        # odd layer -> append reversed cost
        qaoa_circuit.compose(bound_cost_layer.reverse_ops(), range(self.num_qubits), inplace=True)

      # the mixer layer is not reversed
      qaoa_circuit.compose(bound_mixer_layer, range(self.num_qubits), inplace=True)

    if self.num_layers % 2 == 1:
      # iterate over layout permutations to recover measurements
      if self.property_set["layout"]:
        for cidx, qidx in self.property_set["layout"].get_physical_bits().items():
          qaoa_circuit.measure(qidx, cidx)
      else:
        print("layout not found, assigining trivial layout")
        for idx in range(self.num_qubits):
          qaoa_circuit.measure(idx, idx)
    else:
      for idx in range(self.num_qubits):
        qaoa_circuit.measure(idx, idx)

    return circuit_to_dag(qaoa_circuit)


