let model = null;

// Load JSON model once
async function loadModel() {
    if (model) return;

    const response = await fetch("model.json");
    model = await response.json();
}

// Matrix multiply helper
function matmul(a, b) {
    const result = new Array(a.length)
        .fill(0)
        .map(() => new Array(b[0].length).fill(0));

    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < b[0].length; j++) {
            for (let k = 0; k < b.length; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

// Add bias
function addBias(mat, bias) {
    return mat.map((row, i) =>
        row.map((v, j) => v + bias[j])
    );
}

// ReLU
function relu(mat) {
    return mat.map(row => row.map(v => Math.max(0, v)));
}

async function predict() {
    await loadModel();

    const inputs = [
        parseFloat(document.getElementById("cyl").value),
        parseFloat(document.getElementById("disp").value),
        parseFloat(document.getElementById("hp").value),
        parseFloat(document.getElementById("weight").value),
        parseFloat(document.getElementById("acc").value),
        parseFloat(document.getElementById("year").value),
        parseFloat(document.getElementById("origin").value)
    ];

    // Normalize
    const x = inputs.map((v, i) => (v - model.means[i]) / model.stds[i]);
    let layer = [x];  // shape [1,7]

    // fc1
    layer = matmul(layer, model.fc1_weight);
    layer = addBias(layer, model.fc1_bias);
    layer = relu(layer);

    // fc2
    layer = matmul(layer, model.fc2_weight);
    layer = addBias(layer, model.fc2_bias);
    layer = relu(layer);

    // fc3
    layer = matmul(layer, model.fc3_weight);
    layer = addBias(layer, model.fc3_bias);

    const mpg = layer[0][0];

    document.getElementById("result").innerHTML =
        "Predicted MPG: " + mpg.toFixed(2);
}
