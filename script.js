// Load ONNX Runtime Web
const ort = window.ort;

let session = null;

// Load the ONNX model once
async function initModel() {
    if (!session) {
        session = await ort.InferenceSession.create("mpg_pytorch.onnx");
    }
}

async function predict() {

    await initModel();

    // Read your existing inputs exactly as you already have them
    const cyl = parseFloat(document.getElementById("cyl").value);
    const disp = parseFloat(document.getElementById("disp").value);
    const hp = parseFloat(document.getElementById("hp").value);
    const w = parseFloat(document.getElementById("weight").value);
    const acc = parseFloat(document.getElementById("acc").value);
    const year = parseFloat(document.getElementById("year").value);
    const origin = parseFloat(document.getElementById("origin").value);

    // Create a Float32Array with your seven inputs
    const inputData = Float32Array.from([cyl, disp, hp, w, acc, year, origin]);

    // Build a tensor shaped like [1,7]
    const tensor = new ort.Tensor("float32", inputData, [1, 7]);

    // Prepare feeds
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;

    // Run inference
    const results = await session.run(feeds);

    const mpgValue = results[session.outputNames[0]].data[0];

    // Update your existing result box
    document.getElementById("result").innerHTML = 
        "Predicted MPG: <strong>" + mpgValue.toFixed(2) + "</strong>";
}
