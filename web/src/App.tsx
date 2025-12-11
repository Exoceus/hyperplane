import {useEffect, useState} from "react";
import init, {add} from "./wasm";

function App() {
    const [ready, setReady] = useState(false);

    useEffect(() => {
        init().then(() => setReady(true));
    }, []);

    if (!ready) return <div>Loading WASM...</div>;

    // Now you can call your_rust_function with full type safety!
    const result = add(4, 2);

    return <div>Testing: {result}</div>;
}

export default App;
