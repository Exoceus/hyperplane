import {useCallback, useEffect, useState} from "react";
import Plot from "react-plotly.js";
import "./App.css";
import init, {type DataPoint, LinearlySeparableDatasetGenerator} from "./wasm";

function App() {
    const [ready, setReady] = useState(false);
    const [w1, setW1] = useState(0.5);
    const [w2, setW2] = useState(0.5);
    const [b, setB] = useState(0);
    const [dataset, setDataset] = useState<DataPoint[]>([]);

    useEffect(() => {
        init().then(() => setReady(true));
    }, []);

    const generateDataset = useCallback(() => {
        const datasetGenerator = new LinearlySeparableDatasetGenerator(500);
        return datasetGenerator.generateFixed(w1, w2, b);
    }, [w1, w2, b]);

    useEffect(() => {
        if (!ready) return;

        setDataset(generateDataset());
    }, [ready, generateDataset]);

    if (!ready) return <div>Loading WASM...</div>;

    // Separate data points by label for coloring
    const positivePoints = dataset.filter((p) => p.label === 1);
    const negativePoints = dataset.filter((p) => p.label === -1);

    // Generate line data: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
    const lineX = Array.from({length: 100}, (_, i) => -10 + (i * 20) / 99);
    const lineY = lineX.map((x1) => (w2 !== 0 ? -(w1 * x1 + b) / w2 : 0));

    return (
        <div style={{padding: "20px"}}>
            <h1>Linearly Separable Dataset Visualizer</h1>

            <div style={{marginBottom: "20px"}}>
                <div style={{marginBottom: "15px"}}>
                    <label>
                        w1 (weight for x1): <strong>{w1.toFixed(2)}</strong>
                        <br />
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
                        w2 (weight for x2): <strong>{w2.toFixed(2)}</strong>
                        <br />
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
                        b (bias): <strong>{b.toFixed(2)}</strong>
                        <br />
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

            <Plot
                data={[
                    {
                        x: positivePoints.map((p) => p.x1),
                        y: positivePoints.map((p) => p.x2),
                        mode: "markers",
                        type: "scatter",
                        name: "Label +1",
                        marker: {color: "blue", size: 6, symbol: "cross"},
                    },
                    {
                        x: negativePoints.map((p) => p.x1),
                        y: negativePoints.map((p) => p.x2),
                        mode: "markers",
                        type: "scatter",
                        name: "Label -1",
                        marker: {color: "red", size: 6, symbol: "square"},
                    },
                    {
                        x: lineX,
                        y: lineY,
                        mode: "lines",
                        type: "scatter",
                        name: "Decision Boundary",
                        line: {color: "green", width: 3},
                    },
                ]}
                layout={{
                    title: "Dataset Visualization",
                    xaxis: {title: "x1", range: [-10, 10]},
                    yaxis: {title: "x2", range: [-10, 10]},
                    width: 700,
                    height: 700,
                    hovermode: "closest",
                }}
            />
        </div>
    );
}

export default App;
