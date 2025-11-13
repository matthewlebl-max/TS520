// Load ONNX Runtime Web
const ort = window.ort;

// Will hold the model session
let session = null;

// Load the ONNX model when needed
async function initModel() {
    if (!session) {
        session = await ort.InferenceSession.create("mpg_pytorch.onnx");
    }
}

async function predict() {

    // Ensure model is loaded
    await initModel();

    // Read user inputs
    const cyl = parseFloat(document.getElementById("cyl").value);
    const disp = parseFloat(document.getElementById("disp").value);
    const hp = parseFloat(document.getElementById("hp").value);
    const weight = parseFloat(document.getElementById("weight").value);
    const acc = parseFloat(document.getElementById("acc").value);
    const year = parseFloat(document.getElementById("year").value);
    const origin = parseFloat(document.getElementById("origin").value);

    // Pack into Float32Array [1,7]
    const inputData = Float32Array.from([
        cyl, disp, hp, weight, acc, year, origin
    ]);

    // Create tensor with shape (1, 7)
    const inputTensor = new ort.Tensor("float32", inputData, [1, 7]);

    // Prepare input
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;

    // Run inference
    const results = await session.run(feeds);

    const mpg = results[session.outputNames[0]].data[0];

    // Update result on page
    document.getElementById("result").innerHTML =
        "Predicted MPG: <strong>" + mpg.toFixed(2) + "</strong>";
}
