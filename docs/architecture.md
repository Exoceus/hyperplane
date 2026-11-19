# Architecture

Hyperplane is a static site. Every algorithm runs in the visitor's browser, so there is no server to pay for and nothing to keep alive.

```
core/   Rust crate, compiled to WebAssembly with wasm-pack. All the math.
web/    Vite + React + TypeScript. Layout, controls and canvas drawing.
```

## The Rust core (`core/`)

Each algorithm is a module in `core/src/` (`kmeans.rs`, `svm.rs`, `tsne.rs`, `rl.rs`, …) exported to JavaScript with `#[wasm_bindgen]`. The modules have unit tests (`cargo test --release`), which CI runs on every PR.

Two conventions keep the boundary fast and simple:

- **Flat buffers, not objects.** Data crosses the JS/WASM boundary as `Float64Array`s with documented strides instead of serialized structs. A labelled 2D dataset is `[x, y, class, x, y, class, …]` (stride 3); points or regression data use stride 2; grids sample a rectangle at cell centers, row-major, with row 0 at the top so they map straight onto an image. This is cheap enough to retrain a model and repaint its decision regions every animation frame.
- **Record a trace, replay it.** Iterative algorithms (k-means, EM, SMO, gradient descent, tree growth, power iteration, …) return every intermediate state in one call. The UI then scrubs through that trace, so it can step backwards, change speed, or jump to the end without recomputing anything. Simulations that never settle, such as network training, t-SNE, Q-learning and bandits, are stateful WASM objects that the UI steps a few times per frame.

`web/src/lib/wasm.ts` initializes the module once (`wasmReady`), and the app suspends until it is ready.

## The site (`web/`)

```
src/pages/            Home, PlaygroundPage, NotFound
src/playgrounds/      one folder per playground, plus registry.ts
src/components/       shared UI
src/lib/              hooks, drawing, colors, the speed setting
```

**A playground** (`src/playgrounds/<slug>/`) exports a `<Playground scenarios explainer>`. Each scenario is an id (kept in the URL as `?s=…` so it can be shared), a title, a one-line "Try this" prompt, and a `render()` that returns a preset *instrument*. `registry.ts` lists the playgrounds, their categories and lazy imports.

**An instrument** is a component built on one shared layout, `<Rig>` (`components/Rig.tsx`):

- **visual**: what you look at (one or more `Plot`s or a `Plot3D`);
- **caption** and **hint**: what is happening now, and how to interact with the visual directly;
- **controls**: the control deck. The transport comes first (`Player` for a recorded run, `LiveControls` for a live simulation), then `<Group label="…">` blocks of sliders, toggles and buttons. The deck adds *Reset scenario* automatically;
- **readouts**: stats, charts and legends, kept apart from the things you can change.

**Animation.** `useTimeline(length, pace, {autoplay, runKey, duration})` (`lib/hooks.ts`) drives a scrubbable run. Time is continuous, so a single step is a short tween and you can watch each quantity move. With `autoplay` the run replays from the start whenever `runKey` (usually the memoized trace) changes, so editing any input visibly reruns the algorithm. Speed is one site-wide multiplier (`lib/speed.ts`, ¼× to 10×), saved in `localStorage` and applied on top of each playground's natural pace by every timeline and live simulation. Space plays and pauses; ← and → step.

**Drawing.** `components/Plot.tsx` is a responsive, retina-aware canvas that redraws whenever it re-renders and reports pointer events in data coordinates. Drawing goes through `lib/painter.ts`, which works in data coordinates and provides the shared visual language: points, arrows, half-planes, heatmaps, contours. 3D views use `lib/scene3d.ts` and `components/Plot3D.tsx`: an orbit camera, depth-sorted shaded meshes and picking, all on a 2D canvas with no WebGL or three.js. Colors come from `lib/theme.ts`, so "the yellow 𝐰" in a picture is the same yellow as 𝐰 in the formulas, which are rendered with KaTeX.

## Analytics

Visits are counted with [Umami](https://umami.is): cookieless, no personal data, and it honors the browser's Do Not Track setting. `web/src/lib/analytics.ts` adds the tracker script only when `VITE_UMAMI_WEBSITE_ID` is set at build time, so local builds, deploy previews and forks load nothing. Besides pageviews (including client-side navigation), the site records two custom events: `scenario` (which playground and scenario someone picks) and `speed` (the speed they choose).

The website ID is configured in Netlify, not in the repo. It isn't a secret, since every analytics ID ships in the page, but keeping it out of the repo stops forks from reporting into this site's dashboard:

| Variable | Where | Value |
| --- | --- | --- |
| `VITE_UMAMI_WEBSITE_ID` | Netlify → Site configuration → Environment variables, **Production** context only | the website ID from Umami |
| `VITE_UMAMI_DOMAINS` (optional) | same | `hyperplane.jatinm.com`, an extra guard so only the real domain is counted |
| `VITE_UMAMI_SRC` (optional) | same | a self-hosted tracker URL, if you don't use Umami Cloud |

Vite bakes these in at build time, so redeploy after changing them.

## Build and deploy

- `npm run build` installs Rust if needed, builds the WASM package, copies it into `web/src/wasm/`, then type-checks and bundles the site.
- CI (`.github/workflows/ci.yml`) runs the Rust tests, the WASM build, ESLint, and the TypeScript build on every PR.
- Netlify (`netlify.toml`) builds and deploys `main` automatically, and creates a deploy preview for every PR.
