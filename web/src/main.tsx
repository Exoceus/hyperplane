import {StrictMode} from "react";
import {createRoot} from "react-dom/client";
import {BrowserRouter} from "react-router";
import App from "./App";
import {initAnalytics} from "./lib/analytics";
import {watchForStaleDeploys} from "./lib/staleDeploy";
import "./styles/global.css";

initAnalytics();
watchForStaleDeploys();

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <BrowserRouter>
            <App />
        </BrowserRouter>
    </StrictMode>
);
