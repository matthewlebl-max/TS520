async function predict() {
    const inputs = [
        parseFloat(document.getElementById("cyl").value),
        parseFloat(document.getElementById("disp").value),
        parseFloat(document.getElementById("hp").value),
        parseFloat(document.getElementById("weight").value),
        parseFloat(document.getElementById("acc").value),
        parseFloat(document.getElementById("year").value),
        parseFloat(document.getElementById("orig").value)
    ];

    const apiUrl = "http://localhost:5000/predict";  // adjust if you deploy elsewhere

    try {
        const res = await fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ inputs: inputs })
        });

        const data = await res.json();
        document.getElementById("result").innerText = "Predicted MPG: " + data.mpg;
    } catch (err) {
        document.getElementById("result").innerText = "Error contacting API.";
        console.error(err);
    }
}
