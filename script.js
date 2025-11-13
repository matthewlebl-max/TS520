async function predict() {
    const inputs = [
        parseFloat(document.getElementById("cyl").value),
        parseFloat(document.getElementById("disp").value),
        parseFloat(document.getElementById("hp").value),
        parseFloat(document.getElementById("weight").value),
        parseFloat(document.getElementById("acc").value),
        parseFloat(document.getElementById("year").value),
        parseFloat(document.getElementById("origin").value)
    ];

    const apiUrl = "http://localhost:5000/predict";

    try {
        const res = await fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ inputs: inputs })
        });

        const data = await res.json();
        document.getElementById("result").innerText = "Predicted MPG: " + data.mpg;
    } catch (error) {
        document.getElementById("result").innerText = "Error connecting to backend.";
        console.error(error);
    }
}
