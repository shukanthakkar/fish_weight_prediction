function predictWeight() {
    var length1 = document.getElementById("length1").value;
    var length2 = document.getElementById("length2").value;
    var length3 = document.getElementById("length3").value;
    var height = document.getElementById("height").value;
    var width = document.getElementById("width").value;

    // Send input data to the server for prediction
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            length1: length1,
            length2: length2,
            length3: length3,
            height: height,
            width: width
        }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML = "Predicted Weight: " + data.weight;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
