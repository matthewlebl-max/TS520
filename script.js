// Load ORT single global instance
let session = null;

// Load the ONNX model once
async function initModel() {
    if (!session) {
        session = await ort.InferenceSession.create("mpg_pytorch.onnx");
    }
}

async function predict() {

    await initModel();

    const cyl = parseFloat(document.getElementById("cyl").value);
    const disp = parseFloat(document.getElementById("disp").value);
    const hp = parseFloat(document.getElementById("hp").value);
    const weight = parseFloat(document.getElementById("weight").value);
    const acc = parseFloat(document.getElementById("acc").value);
    const year = parseFloat(document.getElementById("year").value);
    const origin = parseFloat(document.getElementById("origin").value);

    const inputData = Float32Array.from([
        cyl, disp, hp, weight, acc, year, origin
    ]);

    const tensor = new ort.Tensor("float32", inputData, [1, 7]);

    const feeds = {};
    feeds[session.inputNames[0]] = tensor;

    const results = await session.run(feeds);
    const mpg = results[session.outputNames[0]].data[0];

    document.getElementById("result").innerHTML =
        "Predicted MPG: <strong>" + mpg.toFixed(2) + "</strong>";
}
