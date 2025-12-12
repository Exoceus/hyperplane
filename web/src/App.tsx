import {useEffect, useState} from "react";
import init, {LinearlySeparableDatasetGenerator} from "./wasm";

function App() {
    const [ready, setReady] = useState(false);

    useEffect(() => {
        init().then(() => setReady(true));
    }, []);

    if (!ready) return <div>Loading WASM...</div>;

    // Now you can call your_rust_function with full type safety!
    const datasetGenerator = new LinearlySeparableDatasetGenerator(100000);
    const dataset = datasetGenerator.generateRandom(14n).slice(0, 100);

    return (
        <div>
            Dataset:
            {dataset.slice(0, 10).map((datapoint) => (
                <p>
                    {datapoint.label}: ({datapoint.x1}, {datapoint.x2})
                </p>
            ))}
        </div>
    );
}

export default App;
