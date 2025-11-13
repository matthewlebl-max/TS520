let model = null;

// Transpose helper because PyTorch stores weights differently
function transpose(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const out = [];
    for (let c = 0; c < cols; c++) {
        const row = [];
        for (let r = 0; r < rows; r++) {
            row.push(matrix[r][c]);
        }
        out.push(row);
    }
    return out;
}

// Load JSON model exported from Python
async function loadModel() {
    if (!model) {
        const response = await fetch("model.json");
        model = await response.json();

        // Fix weight orientation to match JS math
        model.fc1_weight = transpose(model.fc1_weight);
        model.fc2_weight = transpose(model.fc2_weight);
        model.fc3_weight = transpose(model.fc3_weight);
    }
}

// Simple dense layer: y = ReLU(Wx + b)
function denseLayer(inputVec, weights, biases, useRelu = true) {
    const out = new Array(weights[0].length).fill(0);

    for (let j = 0; j < weights[0].length; j++) {
        let sum = biases[j];

        for (let i = 0; i < inputVec.length; i++) {
            sum += inputVec[i] * weights[i][j];
        }

        out[j] = useRelu ? Math.max(0, sum) : sum;
    }

    return out;
}

// Normalize using saved means and stds
function normalize(vec) {
    let out = [];
    for (let i = 0; i < vec.length; i++) {
        out.push((vec[i] - model.means[i]) / model.stds[i]);
    }
    return out;
}

async function predict() {
    await loadModel();

    const cyl = parseFloat(document.getElementById("cyl").value);
    const disp = parseFloat(document.getElementById("disp").value);
    const hp = parseFloat(document.getElementById("hp").value);
    const weight = parseFloat(document.getElementById("weight").value);
    const acc = parseFloat(document.getElementById("acc").value);
    const year = parseFloat(document.getElementById("year").value);
    const origin = parseFloat(document.getElementById("origin").value);

    const input = [cyl, disp, hp, weight, acc, year, origin];

    // Normalize input
    const x1 = normalize(input);

    // Layer 1
    const h1 = denseLayer(x1, model.fc1_weight, model.fc1_bias, true);

    // Layer 2
    const h2 = denseLayer(h1, model.fc2_weight, model.fc2_bias, true);

    // Layer 3 (output)
    const out = denseLayer(h2, model.fc3_weight, model.fc3_bias, false);

    const mpg = out[0];

    document.getElementById("result").textContent =
        "Predicted MPG: " + mpg.toFixed(2);
}
