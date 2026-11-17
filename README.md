# Hyperplane

Interactive, visual explanations of machine-learning algorithms. Drag the data, nudge the parameters, step through training, and build intuition for _why_ the math works. Try it: [hyperplane.jatinm.com](https://hyperplane.jatinm.com/)

Everything runs in the browser. The algorithms live in a Rust crate compiled to WebAssembly, and the UI is a static React site, so there is no server and nothing to pay for.

## Lessons

| Track          | Lesson               | What you can play with                                                                   |
| -------------- | -------------------- | ---------------------------------------------------------------------------------------- |
| Foundations    | Linear regression    | Residuals drawn as literal squares; the loss bowl in (m, b) space; feature centering      |
| Foundations    | Gradient descent     | 1D step-size regimes; GD vs. momentum vs. RMSProp vs. Adam on five landscapes             |
| Classification | The perceptron       | Draggable weight vector; each update shown in data space _and_ weight space; Novikoff bound |
| Classification | Logistic regression  | Sigmoid and per-point cross-entropy; probability contours; why ‖w‖ blows up without L2   |
| Classification | k-nearest neighbors  | Live decision regions, neighborhood shapes per metric, train vs. test accuracy over k    |
| Classification | Neural networks      | A single neuron, a trainable network with per-neuron maps, and layers that bend the plane |

More are on the way (k-means, PCA, SVMs, kernels, decision trees, backpropagation…).

## Architecture

```
core/   Rust crate → WebAssembly (wasm-bindgen). All numerical work:
        datasets, perceptron, linear/logistic regression, optimizers,
        marching-squares contours, k-NN, k-means, PCA, a small MLP.
web/    Vite + React + TypeScript site.
        src/lib/painter.ts    canvas drawing in data coordinates
        src/components/       Plot, Figure, controls, KaTeX math, charts
        src/lessons/          one folder per lesson + registry.ts
```

The JS/WASM boundary uses flat `Float64Array` buffers with documented strides (for example, labelled 2D data is `[x, y, class, …]`) instead of serialized objects. That keeps calls cheap enough to retrain models and repaint decision regions every animation frame.

### Adding a lesson

1. Put any heavy computation in a new module in `core/src/` and export it with `#[wasm_bindgen]`. Add unit tests (`cargo test`).
2. Create `web/src/lessons/<slug>/<Name>.tsx`. Build figures from `Figure`, `Plot`, the controls in `components/Controls.tsx`, and `useTimeline` for step-through animations.
3. Register it in `web/src/lessons/registry.ts`.

## Development

```sh
npm install --include=dev
npm run dev:all        # builds the wasm package, then runs Vite and rebuilds wasm on Rust changes
```

Other useful commands:

```sh
cd core && cargo test --release     # Rust unit tests
npm run build                       # full production build (wasm + site), as on Netlify
```

Requires Rust (with the `wasm32-unknown-unknown` target), [wasm-pack](https://github.com/drager/wasm-pack) and Node 24.

## Deployment

Netlify builds and deploys `main` automatically (`netlify.toml`). Changes land through pull requests that auto-merge once CI (Rust tests, wasm build, lint, type-check and site build) passes.

## Style guide

- [Conventional commits](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13#types)
