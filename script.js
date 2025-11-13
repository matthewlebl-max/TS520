let model = null;

// Load model.json
async function loadModel() {
    if (!model) {
        const response = await fetch("model.json");
        model = await response.json();
    }
}

function normalizeInput(values, means, stds) {
    const out = [];
    for (let i = 0; i < values.length; i++) {
        out.push((values[i] - means[i]) / stds[i]);
    }
    return out;
}

function denseLayer(input, weights, bias) {
    const out = [];
    for (let i = 0; i < bias.length; i++) {
        let sum = 0;
        for (let j = 0; j < input.length; j++) {
            sum += input[j] * weights[i][j];
        }
        out.push(sum + bias[i]);
    }
    return out;
}

function relu(v) {
    return v.map(x => Math.max(0, x));
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

    let input = [cyl, disp, hp, weight, acc, year, origin];

    // Normalize input to match PyTorch model
    input = normalizeInput(input, model.means, model.stds);

    // Forward pass
    let x = denseLayer(input, model.fc1_weight, model.fc1_bias);
    x = relu(x);

    x = denseLayer(x, model.fc2_weight, model.fc2_bias);
    x = relu(x);

    x = denseLayer(x, model.fc3_weight, model.fc3_bias);

    const mpg = x[0];

    document.getElementById("result").innerHTML =
        "Predicted MPG: <strong>" + mpg.toFixed(2) + "</strong>";
}
