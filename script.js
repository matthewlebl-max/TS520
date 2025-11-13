// =====================================================
// Shared functions
// =====================================================
function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function matmul(input, weights) {
    const out = new Array(weights.length).fill(0);
    for (let r = 0; r < weights.length; r++) {
        out[r] = dot(input, weights[r]);
    }
    return out;
}

function relu(arr) {
    return arr.map(v => Math.max(0, v));
}



// =====================================================
// MPG REGRESSION
// =====================================================
let mpgModel = null;

async function loadMPG() {
    if (!mpgModel) {
        const res = await fetch("model.json");
        mpgModel = await res.json();
    }
}

async function predictMPG() {

    await loadMPG();

    const x = [
        parseFloat(document.getElementById("r_cyl").value),
        parseFloat(document.getElementById("r_disp").value),
        parseFloat(document.getElementById("r_hp").value),
        parseFloat(document.getElementById("r_weight").value),
        parseFloat(document.getElementById("r_acc").value),
        parseFloat(document.getElementById("r_year").value),
        parseFloat(document.getElementById("r_origin").value)
    ];

    // normalize
    const norm = x.map((v, i) => (v - mpgModel.means[i]) / mpgModel.stds[i]);

    // forward pass
    let h1 = matmul(norm, mpgModel.fc1_weight).map((v,i)=>v+mpgModel.fc1_bias[i]);
    h1 = relu(h1);

    let h2 = matmul(h1, mpgModel.fc2_weight).map((v,i)=>v+mpgModel.fc2_bias[i]);
    h2 = relu(h2);

    let out = matmul(h2, mpgModel.fc3_weight)[0] + mpgModel.fc3_bias[0];

    document.getElementById("mpg_result").innerHTML =
        "Predicted MPG: <strong>" + out.toFixed(2) + "</strong>";
}



// =====================================================
// CYLINDER CLASSIFICATION
// =====================================================
let cylModel = null;

async function loadCyl() {
    if (!cylModel) {
        const res = await fetch("cylinders.json");
        cylModel = await res.json();
    }
}

function softmax(arr) {
    const e = arr.map(Math.exp);
    const s = e.reduce((a,c)=>a+c,0);
    return e.map(v => v / s);
}

async function predictCylinders() {

    await loadCyl();

    const x = [
        parseFloat(document.getElementById("c_mpg").value),
        parseFloat(document.getElementById("c_disp").value),
        parseFloat(document.getElementById("c_hp").value),
        parseFloat(document.getElementById("c_weight").value),
        parseFloat(document.getElementById("c_acc").value),
        parseFloat(document.getElementById("c_year").value),
        parseFloat(document.getElementById("c_origin").value)
    ];

    // normalize
    const norm = x.map((v,i)=>(v - cylModel.means[i]) / cylModel.stds[i]);

    // fc1
    let h1 = matmul(norm, cylModel.fc1_weight).map((v,i)=>v+cylModel.fc1_bias[i]);
    h1 = relu(h1);

    // fc2
    let h2 = matmul(h1, cylModel.fc2_weight).map((v,i)=>v+cylModel.fc2_bias[i]);
    h2 = relu(h2);

    // fc3 logits
    let h3 = matmul(h2, cylModel.fc3_weight).map((v,i)=>v+cylModel.fc3_bias[i]);

    const probs = softmax(h3);

    const best = probs.indexOf(Math.max(...probs));
    const predicted = cylModel.classes[best];

    document.getElementById("cyl_result").innerHTML =
        "Predicted Cylinders: <strong>" + predicted + "</strong>";
}
