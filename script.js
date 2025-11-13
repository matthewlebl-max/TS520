let model = null;

// Load JSON model once
async function loadModel() {
    if (model) return;

    try {
        const response = await fetch("model.json");
        model = await response.json();
        console.log("Model loaded:", model);
    } catch (err) {
        console.error("Failed to load model.json", err);
    }
}

// Ensure the input is a valid number
function readValue(id) {
    const v = document.getElementById(id).value;
    if (v === "" || isNaN(parseFloat(v))) {
        throw new Error("Input " + id + " is empty or invalid");
    }
    return parseFloat(v);
}

// Matrix multiply
function matmul(a, b) {
    const rowsA = a.length;
    const colsA = a[0].length;
    const rowsB = b.length;
    const colsB = b[0].length;

    if (colsA !== rowsB) {
        console.error("Shape mismatch:", rowsA, colsA, "x", rowsB, colsB);
        throw new Error("Matrix shape mismatch");
    }

    const result = Array(rowsA)
        .fill(0)
        .map(() => Array(colsB).fill(0));

    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            for (let k = 0; k < colsA; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

// Add bias vector
function addBias(mat, bias) {
    return mat.map(row => row.map((v, j) => v + bias[j]));
}

// ReLU
function relu(mat) {
    return mat.map(row => row.map(v => Math.max(0, v)));
}

async function predict() {
    try {
        await loadModel();

        if (!model) {
            alert("Model not loaded. Check model.json path.");
            return;
        }

        // Read inputs safely
        const inputs = [
            readValue("cyl"),
            readValue("disp"),
            readValue("hp"),
            readValue("weight"),
            readValue("acc"),
            readValue("year"),
            readValue("origin")
        ];

        // Normalize
        const x = inputs.map((v, i) => {
            const z = (v - model.means[i]) / model.stds[i];
            if (isNaN(z)) console.error("Normalization NaN at index", i);
            return z;
        });

        console.log("Normalized input:", x);

        let layer = [x]; // Shape [1,7]

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

        if (isNaN(mpg)) {
            console.error("Final MPG is NaN. Layer dump:", layer);
            document.getElementById("result").innerHTML =
                "Error: MPG computed as NaN. Check console.";
            return;
        }

        // Success
        document.getElementById("result").innerHTML =
            "Predicted MPG: " + mpg.toFixed(2);

    } catch (err) {
        console.error(err);
        document.getElementById("result").innerHTML =
            "Error: " + err.message;
    }
}
