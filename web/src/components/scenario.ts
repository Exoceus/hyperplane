import {createContext} from "react";

/** Lets a rig offer "reset scenario" without every instrument wiring it up. */
export const ScenarioContext = createContext<{reset: () => void} | null>(null);
