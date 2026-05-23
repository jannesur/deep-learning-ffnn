console.log("TensorFlow.js geladen:", tf);
console.log("Plotly geladen:", Plotly);


// Generieren von N=100 zufälligen, gleich-verteilten x-Werten aus dem Intervall [-2,+2]
// Berechnen der dazugehörigen y-Werte

const DATASET_SIZE = 100;

const xValues = [];

function calculateY(x) {

    return (
        0.5 *
        (x + 0.8) *
        (x + 1.8) *
        (x - 0.2) *
        (x - 0.3) *
        (x - 1.9)
        + 1
    );
}

for (let i = 0; i < DATASET_SIZE; i++) {

    const x = Math.random() * 4 - 2;

    xValues.push(x);
}

const dataset = [];

for (const x of xValues) {

    const y = calculateY(x);

    dataset.push({
        x: x,
        y: y
    });
}


// Datensatz zufällig mischen und in Trainingsdaten und Testdaten aufteilen

dataset.sort(() => Math.random() - 0.5);

const trainingData = dataset.slice(0, DATASET_SIZE / 2);

const testData = dataset.slice(DATASET_SIZE / 2);

console.log("Training Data:", trainingData);

console.log("Test Data:", testData);


// Gaussian Noise erzeugen

function generateGaussianNoise(mean = 0, variance = 0.05) {

    const standardDeviation = Math.sqrt(variance);

    const randomValue1 = Math.random();

    const randomValue2 = Math.random();

    const randomNormalValue =
        Math.sqrt(-2 * Math.log(randomValue1)) *
        Math.cos(2 * Math.PI * randomValue2);

    return mean + standardDeviation * randomNormalValue;
}


// Trainingsdaten und Testdaten künstlich verrauschen

const noisyTrainingData = trainingData.map(point => {

    return {
        x: point.x,
        y: point.y + generateGaussianNoise()
    };
});

const noisyTestData = testData.map(point => {

    return {
        x: point.x,
        y: point.y + generateGaussianNoise()
    };
});

console.log("Noisy Training Data:", noisyTrainingData);

console.log("Noisy Test Data:", noisyTestData);


// Trainingsdaten für TensorFlow vorbereiten

const trainingInputs = tf.tensor2d(
    noisyTrainingData.map(point => [point.x])
);

const trainingLabels = tf.tensor2d(
    noisyTrainingData.map(point => [point.y])
);

console.log(trainingInputs);

console.log(trainingLabels);


// Feed-Forward Neural Network erstellen

const model = tf.sequential();

model.add(tf.layers.dense({

    units: 100,

    activation: "relu",

    inputShape: [1]

}));

model.add(tf.layers.dense({

    units: 100,

    activation: "relu"

}));

model.add(tf.layers.dense({

    units: 1,

    activation: "linear"

}));


// Modell konfigurieren

model.compile({

    optimizer: tf.train.adam(0.01),

    loss: "meanSquaredError"

});

console.log(model);


// Modell trainieren

async function trainModel() {

    await model.fit(

        trainingInputs,
        trainingLabels,

        {
            epochs: 100,
            batchSize: 32,
            shuffle: true
        }

    );

    console.log("Training abgeschlossen");


    // Vorhersagen mit dem trainierten Modell erzeugen

    const trainingPredictions = model.predict(trainingInputs);

    console.log("Training Predictions:");

    trainingPredictions.print();


    // Vorhersagen visualisieren

    const predictionValues = await trainingPredictions.array();

    Plotly.newPlot("prediction-plot", [

        {
            x: noisyTrainingData.map(point => point.x),

            y: noisyTrainingData.map(point => point.y),

            mode: "markers",

            type: "scatter",

            name: "Trainingsdaten"
        },

        {
            x: noisyTrainingData.map(point => point.x),

            y: predictionValues.map(value => value[0]),

            mode: "markers",

            type: "scatter",

            name: "Vorhersage"
        }

    ], {

        title: "Erste Vorhersage des trainierten Modells",

        xaxis: {
            title: "x"
        },

        yaxis: {
            title: "y"
        }

    });

}

trainModel();


// Unverrauschten Datensatz mit Plotly visualisieren

Plotly.newPlot("dataset-plot", [

    {
        x: trainingData.map(point => point.x),
        y: trainingData.map(point => point.y),

        mode: "markers",
        type: "scatter",

        name: "Trainingsdaten"
    },

    {
        x: testData.map(point => point.x),
        y: testData.map(point => point.y),

        mode: "markers",
        type: "scatter",

        name: "Testdaten"
    }

], {

    title: "Unverrauschter Datensatz",

    xaxis: {
        title: "x"
    },

    yaxis: {
        title: "y"
    }

});


// Verrauschten Datensatz mit Plotly visualisieren

Plotly.newPlot("noisy-dataset-plot", [

    {
        x: noisyTrainingData.map(point => point.x),
        y: noisyTrainingData.map(point => point.y),

        mode: "markers",
        type: "scatter",

        name: "Trainingsdaten"
    },

    {
        x: noisyTestData.map(point => point.x),
        y: noisyTestData.map(point => point.y),

        mode: "markers",
        type: "scatter",

        name: "Testdaten"
    }

], {

    title: "Verrauschter Datensatz",

    xaxis: {
        title: "x"
    },

    yaxis: {
        title: "y"
    }

});