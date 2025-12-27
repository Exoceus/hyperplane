import {useEffect, useState} from "react";
import Plot from "react-plotly.js";
import "./App.css";
import image from "./assets/logo.png";
import init, {LinearlySeparableDatasetGenerator, Perceptron, type PerceptronStep} from "./wasm";

type ModelType = null | "perceptron";

function App() {
    const [ready, setReady] = useState(false);

    // Dataset generation parameters
    const [datasetSize, setDatasetSize] = useState(25);
    const [w1, setW1] = useState(0.5);
    const [w2, setW2] = useState(0.5);
    const [b, setB] = useState(0);
    const [dataset, setDataset] = useState<Array<{data_point: {features: number[]}; label: number}>>([]);

    // Model selection
    const [selectedModel, setSelectedModel] = useState<ModelType>(null);

    // Perceptron parameters
    const [margin, setMargin] = useState(0);
    const [perceptronW1, setPerceptronW1] = useState(0);
    const [perceptronW2, setPerceptronW2] = useState(0);
    const [perceptronB, setPerceptronB] = useState(0);
    const [maxEpochs, setMaxEpochs] = useState(50);
    const [trainingSteps, setTrainingSteps] = useState<PerceptronStep[]>([]);
    const [currentTrainingIter, setCurrentTrainingIter] = useState(0);
    const [isTraining, setIsTraining] = useState(false);
    const [isAnimating, setIsAnimating] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [trainingConverged, setTrainingConverged] = useState(false);
    const [animationSpeed, setAnimationSpeed] = useState(500); // milliseconds per epoch

    useEffect(() => {
        init().then(() => setReady(true));
    }, []);

    // Generate dataset whenever parameters change
    useEffect(() => {
        if (!ready) return;
        const datasetGenerator = new LinearlySeparableDatasetGenerator(datasetSize, 2);
        const generated = datasetGenerator.generateFixed(new Float64Array([w1, w2]), b);
        setDataset(generated);
    }, [ready, datasetSize, w1, w2, b]);

    // Handle animation loop
    useEffect(() => {
        if (!isAnimating || isPaused || trainingSteps.length === 0) return;

        const interval = setInterval(() => {
            setCurrentTrainingIter((prev) => {
                const nextIndex = prev + 1;
                if (nextIndex >= trainingSteps.length) {
                    setIsAnimating(false);
                    return prev;
                }
                return nextIndex;
            });
        }, animationSpeed);

        return () => clearInterval(interval);
    }, [isAnimating, isPaused, trainingSteps.length, animationSpeed]);

    const handleSelectModel = (model: ModelType) => {
        setSelectedModel(model);
        // Always clear perceptron training state
        setTrainingSteps([]);
        setCurrentTrainingIter(0);
        setIsAnimating(false);
        setIsPaused(false);
    };

    const handleTrainPerceptron = () => {
        if (!ready || dataset.length === 0) return;
        setIsTraining(true);
        setIsPaused(false);
        try {
            const perceptron = new Perceptron(
                margin,
                new Float64Array([perceptronW1, perceptronW2]),
                perceptronB,
                maxEpochs
            );
            const result = perceptron.train(dataset);

            setTrainingSteps(result.steps);
            setTrainingConverged(result.converged);
            setCurrentTrainingIter(0);
            perceptron.free();

            // Start animation if there are multiple steps
            if (result.steps.length > 1) {
                setIsAnimating(true);
            }
        } finally {
            setIsTraining(false);
        }
    };

    if (!ready) return <div>Loading WASM...</div>;

    // Get current weights and bias based on what we're displaying
    let displayWeights = [w1, w2]; // Default to dataset generation weights
    let displayBias = b;

    // If we're in perceptron mode, use perceptron weights
    if (selectedModel === "perceptron") {
        displayWeights = [perceptronW1, perceptronW2];
        displayBias = perceptronB;

        // If training has occurred, use the current epoch's weights
        if (trainingSteps.length > 0 && currentTrainingIter >= 0 && currentTrainingIter < trainingSteps.length) {
            const step = trainingSteps[currentTrainingIter];
            displayWeights = step.weights;
            displayBias = step.bias;
        }
    }

    // Separate data points by label for coloring
    const positivePoints = dataset.filter((p) => p.label === 1);
    const negativePoints = dataset.filter((p) => p.label === -1);

    // Generate line data for decision boundary
    const lineX = Array.from({length: 100}, (_, i) => -10 + (i * 20) / 99);
    const lineY = lineX.map((x1) =>
        displayWeights[1] !== 0 ? -(displayWeights[0] * x1 + displayBias) / displayWeights[1] : 0
    );

    return (
        <div style={{padding: "20px", maxWidth: "1400px", margin: "0 auto"}}>
            <img src={image} alt="Logo" style={{maxWidth: "250px", marginBottom: "20px"}} />

            {/* Dataset Generation Controls - Always Visible */}
            <div
                style={{
                    marginBottom: "10px",
                    padding: "20px",
                    backgroundColor: "rgba(100, 100, 150, 0.1)",
                    borderRadius: "8px",
                }}
            >
                <h2>Dataset Configuration</h2>
                <p>Adjust parameters to see the generated dataset update in real-time</p>

                <div style={{marginBottom: "20px"}}>
                    <div style={{marginBottom: "15px"}}>
                        <label>
                            n (dataset size): <strong>{datasetSize.toFixed(0)}</strong>{" "}
                            <input
                                type="range"
                                min="5"
                                max="500"
                                step="1"
                                value={datasetSize}
                                onChange={(e) => setDatasetSize(parseFloat(e.target.value))}
                                style={{width: "300px"}}
                            />
                        </label>
                    </div>

                    <div style={{marginBottom: "15px"}}>
                        <label>
                            w1 (weight for x1): <strong>{w1.toFixed(2)}</strong>{" "}
                            <input
                                type="range"
                                min="-1"
                                max="1"
                                step="0.01"
                                value={w1}
                                onChange={(e) => setW1(parseFloat(e.target.value))}
                                style={{width: "300px"}}
                            />
                        </label>
                    </div>

                    <div style={{marginBottom: "15px"}}>
                        <label>
                            w2 (weight for x2): <strong>{w2.toFixed(2)}</strong>{" "}
                            <input
                                type="range"
                                min="-1"
                                max="1"
                                step="0.01"
                                value={w2}
                                onChange={(e) => setW2(parseFloat(e.target.value))}
                                style={{width: "300px"}}
                            />
                        </label>
                    </div>

                    <div style={{marginBottom: "15px"}}>
                        <label>
                            b (bias): <strong>{b.toFixed(2)}</strong>{" "}
                            <input
                                type="range"
                                min="-10"
                                max="10"
                                step="0.01"
                                value={b}
                                onChange={(e) => setB(parseFloat(e.target.value))}
                                style={{width: "300px"}}
                            />
                        </label>
                    </div>
                </div>
            </div>

            {/* Dataset Visualization - Only show if no model selected */}
            {!selectedModel && (
                <div>
                    <Plot
                        data={[
                            {
                                x: positivePoints.map((p) => p.data_point.features[0]),
                                y: positivePoints.map((p) => p.data_point.features[1]),
                                mode: "markers",
                                type: "scatter",
                                name: "Label +1",
                                marker: {color: "lightblue", size: 6, symbol: "cross"},
                            },
                            {
                                x: negativePoints.map((p) => p.data_point.features[0]),
                                y: negativePoints.map((p) => p.data_point.features[1]),
                                mode: "markers",
                                type: "scatter",
                                name: "Label -1",
                                marker: {color: "lightcoral", size: 6, symbol: "square"},
                            },
                            {
                                x: lineX,
                                y: lineY,
                                mode: "lines",
                                type: "scatter",
                                name: "True Boundary",
                                line: {color: "green", width: 3},
                            },
                        ]}
                        layout={{
                            title: "Dataset with True Boundary",
                            xaxis: {title: "x1", range: [-10.5, 10.5]},
                            yaxis: {title: "x2", range: [-10.5, 10.5]},
                            width: 700,
                            height: 700,
                            hovermode: "closest",
                            plot_bgcolor: "rgba(0,0,0,0)",
                            paper_bgcolor: "rgba(0,0,0,0)",
                            font: {color: "white", family: "Inter, sans-serif"},
                        }}
                    />
                </div>
            )}

            {/* Model Selection - Only if no model selected */}
            {!selectedModel && (
                <div
                    style={{
                        marginBottom: "10px",
                        padding: "20px",
                        backgroundColor: "rgba(150, 148, 100, 0.1)",
                        borderRadius: "8px",
                    }}
                >
                    <h2>Model Type</h2>
                    <p>Select one of the model types to train on the generated data:</p>

                    <button
                        onClick={() => handleSelectModel("perceptron")}
                        style={{
                            padding: "10px 20px",
                            fontSize: "16px",
                            backgroundColor: "#2196F3",
                            color: "white",
                            border: "none",
                            borderRadius: "4px",
                            cursor: "pointer",
                        }}
                    >
                        Perceptron
                    </button>
                </div>
            )}

            {/* Perceptron Training Section - Only show if perceptron selected */}
            {selectedModel === "perceptron" && (
                <>
                    <div
                        style={{
                            marginBottom: "40px",
                            padding: "20px",
                            backgroundColor: "rgba(150, 100, 150, 0.1)",
                            borderRadius: "8px",
                        }}
                    >
                        <h2>Perceptron Configuration</h2>
                        <p>Configure perceptron parameters and train on the dataset</p>

                        <div style={{marginBottom: "20px"}}>
                            <div style={{marginBottom: "15px"}}>
                                <label>
                                    margin: <strong>{margin.toFixed(2)}</strong>{" "}
                                    <input
                                        type="range"
                                        min="0"
                                        max="2"
                                        step="0.01"
                                        value={margin}
                                        onChange={(e) => setMargin(parseFloat(e.target.value))}
                                        style={{width: "300px"}}
                                    />
                                </label>
                            </div>

                            <div style={{marginBottom: "15px"}}>
                                <label>
                                    initial w1: <strong>{perceptronW1.toFixed(2)}</strong>{" "}
                                    <input
                                        type="range"
                                        min="-1"
                                        max="1"
                                        step="0.01"
                                        value={perceptronW1}
                                        onChange={(e) => setPerceptronW1(parseFloat(e.target.value))}
                                        style={{width: "300px"}}
                                    />
                                </label>
                            </div>

                            <div style={{marginBottom: "15px"}}>
                                <label>
                                    initial w2: <strong>{perceptronW2.toFixed(2)}</strong>{" "}
                                    <input
                                        type="range"
                                        min="-1"
                                        max="1"
                                        step="0.01"
                                        value={perceptronW2}
                                        onChange={(e) => setPerceptronW2(parseFloat(e.target.value))}
                                        style={{width: "300px"}}
                                    />
                                </label>
                            </div>

                            <div style={{marginBottom: "15px"}}>
                                <label>
                                    initial bias: <strong>{perceptronB.toFixed(2)}</strong>{" "}
                                    <input
                                        type="range"
                                        min="-10"
                                        max="10"
                                        step="0.01"
                                        value={perceptronB}
                                        onChange={(e) => setPerceptronB(parseFloat(e.target.value))}
                                        style={{width: "300px"}}
                                    />
                                </label>
                            </div>

                            <div style={{marginBottom: "15px"}}>
                                <label>
                                    max epochs: <strong>{maxEpochs.toFixed(0)}</strong>{" "}
                                    <input
                                        type="range"
                                        min="1"
                                        max="200"
                                        step="1"
                                        value={maxEpochs}
                                        onChange={(e) => setMaxEpochs(parseFloat(e.target.value))}
                                        style={{width: "300px"}}
                                    />
                                </label>
                            </div>
                        </div>

                        <div style={{marginBottom: "15px"}}>
                            <button
                                onClick={handleTrainPerceptron}
                                disabled={isTraining}
                                style={{
                                    padding: "10px 20px",
                                    fontSize: "16px",
                                    backgroundColor: isTraining ? "#cccccc" : "#4CAF50",
                                    color: "white",
                                    border: "none",
                                    borderRadius: "4px",
                                    cursor: isTraining ? "not-allowed" : "pointer",
                                    marginRight: "10px",
                                }}
                            >
                                {isTraining ? "Training..." : "Train Perceptron"}
                            </button>
                            {isAnimating && (
                                <button
                                    onClick={() => setIsPaused(!isPaused)}
                                    style={{
                                        padding: "10px 20px",
                                        fontSize: "16px",
                                        backgroundColor: isPaused ? "#FFA500" : "#FF6B6B",
                                        color: "white",
                                        border: "none",
                                        borderRadius: "4px",
                                        cursor: "pointer",
                                        marginRight: "10px",
                                    }}
                                >
                                    {isPaused ? "Resume" : "Pause"}
                                </button>
                            )}
                            <button
                                onClick={() => handleSelectModel(null)}
                                style={{
                                    padding: "10px 20px",
                                    fontSize: "16px",
                                    backgroundColor: "#f44336",
                                    color: "white",
                                    border: "none",
                                    borderRadius: "4px",
                                    cursor: "pointer",
                                }}
                            >
                                Back to Dataset
                            </button>
                        </div>

                        {trainingSteps.length > 0 && (
                            <p style={{color: "#4CAF50"}}>
                                ✓ Training completed ({trainingSteps.length} steps)
                                {!trainingConverged && " - Did not converge within max epochs"}
                                {isAnimating && " - Animating..."}
                            </p>
                        )}
                    </div>

                    {/* Training Visualization */}
                    {trainingSteps.length > 0 && (
                        <div style={{marginBottom: "40px"}}>
                            <h2>Training Progress</h2>
                            <div style={{marginBottom: "20px"}}>
                                <label>
                                    Iter: <strong>{currentTrainingIter}</strong> (Epoch{" "}
                                    {trainingSteps[currentTrainingIter].epoch}) / {trainingSteps.length - 1}{" "}
                                    <input
                                        type="range"
                                        min="0"
                                        max={trainingSteps.length - 1}
                                        step="1"
                                        value={currentTrainingIter}
                                        onChange={(e) => setCurrentTrainingIter(parseFloat(e.target.value))}
                                        disabled={isAnimating}
                                        style={{width: "300px", opacity: isAnimating ? 0.5 : 1}}
                                    />
                                </label>
                                <p style={{fontSize: "14px", color: "rgba(255, 255, 255, 0.7)", marginTop: "10px"}}>
                                    w1: {displayWeights[0].toFixed(3)}, w2: {displayWeights[1].toFixed(3)}, b:{" "}
                                    {displayBias.toFixed(3)}
                                </p>
                            </div>

                            <div style={{marginBottom: "20px"}}>
                                <label>
                                    animation speed (ms per epoch): <strong>{animationSpeed.toFixed(0)}</strong>{" "}
                                    <input
                                        type="range"
                                        min="100"
                                        max="2000"
                                        step="100"
                                        value={animationSpeed}
                                        onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
                                        style={{width: "300px"}}
                                    />
                                </label>
                            </div>

                            <Plot
                                data={[
                                    {
                                        x: positivePoints.map((p) => p.data_point.features[0]),
                                        y: positivePoints.map((p) => p.data_point.features[1]),
                                        mode: "markers",
                                        type: "scatter",
                                        name: "Label +1",
                                        marker: {color: "lightblue", size: 6, symbol: "cross"},
                                    },
                                    {
                                        x: negativePoints.map((p) => p.data_point.features[0]),
                                        y: negativePoints.map((p) => p.data_point.features[1]),
                                        mode: "markers",
                                        type: "scatter",
                                        name: "Label -1",
                                        marker: {color: "lightcoral", size: 6, symbol: "square"},
                                    },
                                    {
                                        x: lineX,
                                        y: lineY,
                                        mode: "lines",
                                        type: "scatter",
                                        name: "Learned Boundary",
                                        line: {color: "orange", width: 3},
                                    },
                                ]}
                                layout={{
                                    title: `Perceptron Decision Boundary (Epoch ${currentTrainingIter})`,
                                    xaxis: {title: "x1", range: [-10.5, 10.5]},
                                    yaxis: {title: "x2", range: [-10.5, 10.5]},
                                    width: 700,
                                    height: 700,
                                    hovermode: "closest",
                                    plot_bgcolor: "rgba(0,0,0,0)",
                                    paper_bgcolor: "rgba(0,0,0,0)",
                                    font: {color: "white", family: "Inter, sans-serif"},
                                }}
                            />
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

export default App;
