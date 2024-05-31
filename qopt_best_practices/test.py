from networkx import barabasi_albert_graph, draw

from qopt_best_practices.utils import build_max_cut_paulis
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    FindCommutingPauliEvolutions,
    Commuting2qGateRouter,
)
from qiskit.transpiler.passes import (
    BasisTranslator,
    UnrollCustomDefinitions,
    CommutativeCancellation,
    Decompose,
    CXCancellation
)
import time
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qopt_best_practices.qubit_selection import BackendEvaluator

from split_pm import build_optimal_qaoa_ansatz
from qiskit.transpiler import Layout
from qiskit.transpiler.passes import (
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout,
    SetLayout,
)
from qiskit.circuit.library.standard_gates.equivalence_library import _sel
from qiskit.transpiler.passes import HighLevelSynthesis, InverseCancellation
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.circuit.library import CXGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

graph = barabasi_albert_graph(n=10, m=6, seed=42)

local_correlators = build_max_cut_paulis(graph)
cost_operator = SparsePauliOp.from_list(local_correlators)
print(cost_operator)

num_qubits = cost_operator.num_qubits
print(num_qubits)

qaoa_layers = 3

cmap = CouplingMap.from_heavy_hex(distance=3)
print(cmap.size())
backend = GenericBackendV2(num_qubits = 19, coupling_map = cmap, basis_gates = ["x", "sx", "cz", "id", "rz"], seed=0)


dummy_initial_state = QuantumCircuit(num_qubits)  # the real initial state is defined later
dummy_mixer_operator = QuantumCircuit(num_qubits)  # the real mixer is defined later

cost_layer = QAOAAnsatz(
    cost_operator,
    reps=1,
    initial_state=dummy_initial_state,
    mixer_operator=dummy_mixer_operator,
    name="QAOA cost block",
)

# ------------
# Split PM:
# ------------

# 1. choose swap strategy (in this case -> line)
swap_strategy = SwapStrategy.from_line([i for i in range(num_qubits)])
edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(num_qubits)}


# 2. define pass manager for cost layer
pm_pre = PassManager(
    [
        Decompose(),
        FindCommutingPauliEvolutions(),
        Commuting2qGateRouter(
            swap_strategy,
            edge_coloring,
        ),
        Decompose(),
        Decompose(),
        CXCancellation(),
    ]
)

# 3. run pass manager for cost layer
t0_pre = time.time()
swapped_cost_layer = pm_pre.run(cost_layer)
t1_pre = time.time()
print(f"transpilation time: {t1_pre - t0_pre} (s)")

print("ops after first transpilation", swapped_cost_layer.count_ops())

qaoa_circuit = build_optimal_qaoa_ansatz(num_qubits, qaoa_layers, swapped_cost_layer)

# The backend evaluator finds the line of qubits with the best fidelity to map the circuit to
path_finder = BackendEvaluator(backend)
path, fidelity, num_subsets = path_finder.evaluate(num_qubits)
# We use the obtained path to define the initial layout
initial_layout = Layout.from_intlist(path, qaoa_circuit.qregs[0])  # needs qaoa_circ


pm_post = PassManager(
    [
        UnrollCustomDefinitions(_sel, basis_gates=backend.operation_names),
        BasisTranslator(_sel, target_basis=backend.operation_names),
        CommutativeCancellation(target=backend.target),
        SetLayout(initial_layout),
        FullAncillaAllocation(backend.target),
        EnlargeWithAncilla(),
        ApplyLayout(),
        CommutativeCancellation(target=backend.target)
        # SabreLayout(CouplingMap(backend.coupling_map), seed=0, swap_trials=32, layout_trials=32),
    ]
)

t0_post = time.time()
optimally_transpiled_qaoa = pm_post.run(qaoa_circuit)
t1_post = time.time()
print(f"transpilation time: {t1_post - t0_post} (s)")
print("ops after second transpilation", optimally_transpiled_qaoa.count_ops())
print("depth", optimally_transpiled_qaoa.depth())


# ------------
# United PM:
# ------------
from full_pm import QAOAPass

pre_init = PassManager(
            [HighLevelSynthesis(basis_gates=['PauliEvolution']),
             FindCommutingPauliEvolutions(),
             Commuting2qGateRouter(
                    swap_strategy,
                    edge_coloring,
                ),
             HighLevelSynthesis(basis_gates=["x", "cx", "sx", "rz", "id"]),
             InverseCancellation(gates_to_cancel=[CXGate()]),
            ])

init = PassManager([QAOAPass(num_layers=3, num_qubits=10)])

post_init = PassManager(
    [
        UnrollCustomDefinitions(_sel, basis_gates=backend.operation_names),
        BasisTranslator(_sel, target_basis=backend.operation_names),
    ]
)

layout = PassManager(
    [
        SetLayout(initial_layout),
        FullAncillaAllocation(backend.target),
        EnlargeWithAncilla(),
        ApplyLayout(),
    ]
)

optimization = PassManager(
    [
      CommutativeCancellation(target=backend.target)
    ]
)
staged_pm = StagedPassManager(stages=["init", "layout", "optimization"], pre_init=pre_init, init=init, post_init=post_init, layout=layout, optimization=optimization)

t0 = time.time()
out = staged_pm.run(cost_layer)
t1 = time.time()

print("ops after second transpilation", out.count_ops())
print("depth", out.depth())
print("time", t1 - t0)

# ------------
# Naive
# ------------
from qiskit.circuit import ParameterVector

# Initial state = equal superposition
initial_state = QuantumCircuit(num_qubits)
initial_state.h(range(num_qubits))

# Mixer operator = rx rotations
betas = ParameterVector("β", qaoa_layers)
mixer_operator = QuantumCircuit(num_qubits)
mixer_operator.rx(-2*betas[0], range(num_qubits))

# Use off-the-shelf qiskit QAOAAnsatz
qaoa_ansatz = QAOAAnsatz(
    cost_operator,
    initial_state = initial_state,
    mixer_operator = mixer_operator,
    reps = qaoa_layers,
)
qaoa_ansatz.measure_all()

naive_pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
t0 = time.time()
naively_transpiled_qaoa = naive_pm.run(qaoa_ansatz)
t1 = time.time()

print("ops after naive transpilation", naively_transpiled_qaoa.count_ops())
print("naive depth", naively_transpiled_qaoa.depth())
print("time", t1 - t0)