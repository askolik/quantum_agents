
import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

from config import Envs, EncType
from src.utils.utils import generate_all_bitstrings_of_size


class ScalableDataReuploadingController(tf.keras.layers.Layer):
    def __init__(self, input_dim, theta_dim, encoding_dim, angles, tanh=False, name="scalable_data_reuploading"):
        super(ScalableDataReuploadingController, self).__init__(name=name)
        self.input_dim = input_dim
        self.theta_dim = theta_dim
        self.encoding_dim = encoding_dim

        theta_init = tf.random_uniform_initializer(minval=0., maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, theta_dim), dtype=tf.dtypes.float32),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(1, input_dim*encoding_dim))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype=tf.dtypes.float32,
            trainable=True, name="lambdas"
        )

        alphabetical_angles = sorted(angles)
        self.indices = tf.constant([alphabetical_angles.index(a) for a in angles])
        self.tanh = tanh

    def call(self, inputs):
        output = tf.repeat(self.theta, repeats=tf.shape(inputs)[0], axis=0)
        repeat_inputs = tf.repeat(inputs, repeats=self.encoding_dim, axis=1)
        repeat_lmbd = tf.repeat(self.lmbd, repeats=tf.shape(inputs)[0], axis=0)

        if self.tanh:
            output = tf.concat([output, tf.keras.layers.Activation('tanh')(tf.math.multiply(repeat_inputs, repeat_lmbd))], 1)
        else:
            output = tf.concat([output, tf.math.multiply(repeat_inputs, repeat_lmbd)], 1)
        output = tf.gather(output, self.indices, axis=1)
        return output


class TrainableRescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(TrainableRescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype=tf.dtypes.float32,
            trainable=True, name="obs-weights")

    def call(self, inputs):
        # scale expectation values to [0, 1] and multiply with weights
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w, repeats=tf.shape(inputs)[0], axis=0))


def state_to_circuit(state, depth=2, env_name=None, enc_type=None):
    circuit = None
    if env_name == Envs.CARTPOLE:
        circuit = state_to_circuit_continuous(state, depth, enc_type=enc_type)
    elif env_name == Envs.FROZENLAKE:
        circuit = state_to_circuit_discrete(state, 4)

    return circuit


def state_to_circuit_discrete(state, observation_space_size):
    enumerated_bitstrings = generate_all_bitstrings_of_size(observation_space_size)
    state_vec = enumerated_bitstrings[state]
    circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(0, i) for i in range(len(state_vec))]

    for i, x in enumerate(state_vec):
        if x == 1:
            circuit += cirq.X(qubits[i])

    return circuit


def state_to_circuit_continuous(state_vec, depth, enc_type, scale_range=True):
    if scale_range:
        scaled_state = [np.arctan(x) for x in state_vec]
    else:
        scaled_state = state_vec

    qubits = [cirq.GridQubit(0, i) for i in range(len(state_vec))]

    if enc_type == EncType.HIDDEN_SHIFT:
        circuit = state_to_circuit_hidden_shift_boolean(scaled_state, depth, qubits)
    elif enc_type == EncType.CONT_X:
        circuit = state_to_circuit_cont_x(scaled_state, depth, qubits)
    elif enc_type == EncType.CONT_YZ:
        circuit = state_to_circuit_cont_yz(scaled_state, depth, qubits)

    return circuit


def state_to_circuit_hidden_shift_boolean(state, depth, qubits):
    circuit = cirq.Circuit()
    for layer in range(depth):
        circuit += cirq.H.on_each(qubits)
        for i, value in enumerate(state):
            circuit += cirq.rz(value)(qubits[i])
        for i, value in enumerate(state):
            for j in range(i):
                phi = state[i] * state[j]
                circuit += cirq.CNOT(qubits[j], qubits[i])
                circuit += cirq.rz(phi)(qubits[i])
                circuit += cirq.CNOT(qubits[j], qubits[i])

    return circuit


def state_to_circuit_cont_x(state, depth, qubits):
    circuit = cirq.Circuit()
    for layer in range(depth):
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(state[i])(qubit))

    return circuit


def state_to_circuit_cont_yz(state, depth, qubits):
    circuit = cirq.Circuit()
    for layer in range(depth):
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(state[0])(qubit))
            circuit.append(cirq.rz(state[1])(qubit))
            circuit.append(cirq.ry(state[2])(qubit))
            circuit.append(cirq.rz(state[3])(qubit))

    return circuit


def create_q_circuit(n_qubits, n_layers=5):
    circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    symbols = [sympy.Symbol(str(i)) for i in (range(n_qubits * 2 * n_layers + (n_qubits * 2)))]
    return_symb = [sympy.Symbol(str(i)) for i in (range(n_qubits * 2 * n_layers + (n_qubits * 2)))]
    for i in range(n_layers):
        for j in range(n_qubits):
            circuit.append(cirq.rz(symbols.pop())(qubits[j]))
            circuit.append(cirq.ry(symbols.pop())(qubits[j]))

        for j in range(n_qubits):
            for k in range(j):
                circuit.append(cirq.CZ(qubits[k], qubits[j]))

    for j in range(n_qubits):
        circuit.append(cirq.rz(symbols.pop())(qubits[j]))
        circuit.append(cirq.ry(symbols.pop())(qubits[j]))

    return circuit, return_symb


def hwe_layer(qubits, symbols):
    circuit = cirq.Circuit()
    symbols = list(symbols)[::-1]

    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(symbols.pop())(qubit))
        circuit.append(cirq.rz(symbols.pop())(qubit))

    for i in range(len(qubits)):
        circuit.append(cirq.CZ(qubits[i], qubits[(i + 1) % len(qubits)]))

    return circuit


def generate_circuit(n_qubits, n_layers, qubits):
    theta_dim = 2 * n_qubits * n_layers
    params = sympy.symbols('theta(0:' + str(theta_dim) + ')')
    inputs = sympy.symbols(
        'x(0:' + str(n_qubits) + ')' + '(0:' + str(n_layers) + ')')

    circuit = cirq.Circuit()
    for l in range(n_layers):
        for i in range(n_qubits):
            circuit += cirq.rx(inputs[l + i * n_layers])(qubits[i])
        circuit += hwe_layer(qubits, params[l * n_qubits * 2:(l + 1) * n_qubits * 2])

    return circuit, theta_dim, params, inputs


def generate_model(n_qubits, n_layers, circuit, theta_dim, params, inputs, observables, target):
    input_tensor = tf.keras.Input(shape=(n_qubits), dtype=tf.dtypes.float32, name='input')

    # Define input quantum state
    input_q_state = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_state')

    # Define encoding layer
    encoding_layer = ScalableDataReuploadingController(
        input_dim=n_qubits, theta_dim=theta_dim, encoding_dim=n_layers,
        angles=[str(param) for param in params] + [str(x) for x in inputs], tanh=True)

    # Define Controlled PQC
    expectation_layer = tfq.layers.ControlledPQC(
        circuit, differentiator=tfq.differentiators.Adjoint(),
        operators=observables, name="PQC")

    prepend = ""
    if target:
        prepend = "Target"

    # Define post-processing
    expectation_values = expectation_layer(
        [input_q_state, encoding_layer(input_tensor)])

    process = tf.keras.Sequential([
        TrainableRescaling(len(observables))
    ], name=prepend + "Q-values")

    Q_values = process(expectation_values)

    # Define model
    model = tf.keras.Model(
        inputs=[input_q_state, input_tensor],
        outputs=Q_values,
        name=prepend + "Q-function")

    if not target:
        model.summary()

    return model


def empty_circuits(n):
    return tfq.convert_to_tensor([cirq.Circuit()]*n)


def construct_readout_ops(qubits, env_name):
    if env_name == Envs.CARTPOLE:
        readout_op = [cirq.PauliString(cirq.Z(qubits[0]) * cirq.Z(qubits[1])),
                      cirq.PauliString(cirq.Z(qubits[2]) * cirq.Z(qubits[3]))]

        # readout_op = [
        #     cirq.PauliString(
        #         cirq.Z(qubits[0]) * cirq.Z(qubits[1]) * cirq.Z(qubits[2]) * cirq.Z(qubits[3]))]
    elif env_name == Envs.FROZENLAKE:
        readout_op = [cirq.PauliString(cirq.Z(qubit)) for qubit in qubits]

    return readout_op


def perform_action(
        state, model, circuit, symbols, ops, env, encoding_depth,
        multiply_output_by, epsilon=0.3, env_name=None, gamma=0.9):
    action_type = 'random'
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        q = q_val(state, model, circuit, symbols, ops, encoding_depth, multiply_output_by, env_name)
        print("\tQ values:", q)
        action = np.argmax(q)
        action_type = 'argmax'

    return action, action_type


def q_val(state, model, circuit, symbols, ops, encoding_depth, multiply_output_by, env_name):
    params = model.trainable_variables
    state_circ = state_to_circuit(state, encoding_depth, env_name)
    sample_circuit = state_circ + circuit

    if env_name == Envs.CARTPOLE:
        in_state = tfq.convert_to_tensor([state_circ])
        model_prediction = model(in_state)
        scaled_preds = tf.divide(tf.add(model_prediction, 1), 2)
        action_preds = tf.multiply(
            scaled_preds,
            np.asarray([
                multiply_output_by,
                multiply_output_by]))

        output = action_preds.numpy()[0]

        # expectation_layer = tfq.layers.Expectation()
        # expectation_output = expectation_layer(
        #     sample_circuit, symbol_names=symbols,
        #     symbol_values=params, operators=ops)
        #
        # output = ((expectation_output.numpy()[0] + 1) / 2) * multiply_output_by

    elif env_name == Envs.FROZENLAKE:
        in_state = tfq.convert_to_tensor([state_circ])
        model_prediction = model(in_state)
        scaled_preds = tf.divide(tf.add(model_prediction, 1), 2)
        action_preds = tf.multiply(
            scaled_preds,
            np.asarray([
                multiply_output_by,
                multiply_output_by,
                multiply_output_by,
                multiply_output_by]))

        output = action_preds.numpy()[0]

    return output


def q_val_frozenlake(state, gamma):
    q0 = [gamma ** 6, gamma ** 5, gamma ** 5, gamma ** 6]
    q1 = [gamma ** 5, 0, gamma ** 4, gamma ** 5]
    q2 = [gamma ** 5, gamma ** 3, gamma ** 5, gamma ** 4]
    q3 = [gamma ** 4, 0, gamma ** 5, gamma ** 5]
    q4 = [gamma ** 5, gamma ** 4, 0, gamma ** 6]
    q6 = [0, gamma ** 2, 0, gamma ** 4]
    q8 = [gamma ** 4, 0, gamma ** 3, gamma ** 5]
    q9 = [gamma ** 4, gamma ** 2, gamma ** 2, 0]
    q10 = [gamma ** 3, gamma, 0, gamma ** 3]
    q13 = [0, gamma ** 2, gamma, gamma ** 3]
    q14 = [gamma ** 2, gamma, 1, gamma ** 2]

    true_q = [q0, q1, q2, q3, q4, 0, q6, 0, q8, q9, q10, 0, 0, q13, q14, 0]

    return true_q[state]


def add_to_memory(memory, state, action, reward, next_state, memory_len, fixed_memory=False):
    if len(memory) >= memory_len:
        if not fixed_memory:
            random_ix = np.random.randint(0, memory_len)
            memory[random_ix] = (state, action, reward, next_state)
    else:
        memory.append((state, action, reward, next_state))
    return memory


def build_models(circuit, readout_op, opt, loss=None, train_readout=False):
    if loss is None:
        loss = scaled_mse_loss

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string))
    model.add(
        tfq.layers.PQC(
            model_circuit=circuit,
            operators=readout_op,
            differentiator=tfq.differentiators.ForwardDifference(),
            initializer=tf.keras.initializers.RandomUniform))

    model.summary()

    model.compile(
        loss=loss,
        optimizer=opt)

    target_model = tf.keras.Sequential()
    target_model.add(tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string))
    target_model.add(
        tfq.layers.PQC(
            model_circuit=circuit,
            operators=readout_op,
            differentiator=tfq.differentiators.ForwardDifference(),
            initializer=tf.keras.initializers.RandomUniform))

    target_model.compile(
        loss=loss,
        optimizer=opt)

    return model, target_model


def scaled_mse_loss(y_true, y_predicted):
    scaled_preds = tf.divide(tf.add(y_predicted, 1), 2)
    scaled_to_probs = tf.multiply(scaled_preds, np.asarray([-1, 1]))
    probs = tf.add(scaled_to_probs, np.asarray([1, 0]))
    return tf.keras.losses.mse(y_true, probs)