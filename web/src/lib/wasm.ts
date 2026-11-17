import init from "../wasm/hyperplane_wasm.js";

export * from "../wasm/hyperplane_wasm.js";

/** Resolves once the WebAssembly module is instantiated. Started eagerly. */
export const wasmReady: Promise<unknown> = init();

/** Fresh random seed for dataset generators. */
export function randomSeed(): number {
    return Math.floor(Math.random() * 2 ** 31);
}
