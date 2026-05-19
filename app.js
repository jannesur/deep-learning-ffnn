console.log("TensorFlow.js geladen:", tf);
console.log("Plotly geladen:", Plotly);

//Generieren von N=100 zufälligen, gleich-verteilten x Werte aus dem Intervall [-2,+2]

const DATASET_SIZE = 100;

const xValues = [];

for (let i = 0; i < DATASET_SIZE; i++) {

    const x = Math.random() * 4 - 2;

    xValues.push(x);
}

console.log(xValues);