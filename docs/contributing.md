# Contributing

New playgrounds, new scenarios for existing ones, and fixes are all welcome. Read [architecture](architecture.md) first for the big picture.

## Setup

You need Node 24, Rust with the `wasm32-unknown-unknown` target, and [wasm-pack](https://github.com/drager/wasm-pack).

```sh
rustup target add wasm32-unknown-unknown
npm install --include=dev
npm run dev:all        # wasm build + Vite dev server; rebuilds wasm when Rust changes
```

Checks that CI runs on every PR:

```sh
cd core && cargo test --release    # Rust unit tests
cd web && npx eslint src           # lint
npm run build                      # wasm + type-check + production bundle
```

## Adding a playground

The best playgrounds pick one algorithm, let it visibly *run* step by step, and come with scenarios that show where it shines and where it breaks. Aim for pictures, not paragraphs.

### 1. The algorithm, in Rust

Add `core/src/<name>.rs`, declare it in `core/src/lib.rs`, and export what the UI needs with `#[wasm_bindgen]`.

- Take and return flat `f64` buffers with documented strides (labelled 2D data is `[x, y, class, …]`).
- For iterative algorithms, **record the trace**: return every iteration's state so the UI can scrub through it. For open-ended simulations, export a struct with a `step(n)` method instead.
- Take a `seed` for anything random, so a scenario looks the same on every visit.
- Add unit tests at the bottom of the module.

### 2. An instrument

Create `web/src/playgrounds/<slug>/<Name>.tsx`. An instrument is a component that takes preset props (the starting data and parameters) and renders a `<Rig>`:

```tsx
const run = useMemo(() => my_algorithm(data, k, seed), [data, k, seed]);
const tl = useTimeline(run.steps, 1, {autoplay: true, runKey: run, duration: 6});

return (
    <Rig
        hint="Drag the points"
        visual={<Plot label="…" view={VIEW} draw={(p) => { /* draw state at tl.t */ }} onPointer={…} />}
        caption={`Iteration ${Math.floor(tl.t)} of ${run.steps}`}
        controls={
            <>
                <Player timeline={tl} />
                <Group label="Data">…</Group>
                <Group label="Model">…</Group>
            </>
        }
        readouts={<Stats>…</Stats>}
    />
);
```

Conventions that keep the playgrounds consistent:

- **Every animation is controllable.** Use `Player` for recorded runs and `LiveControls` for live simulations. Both include the site-wide speed picker, so scale any per-frame work by `getSpeed()` (see `takeSteps` in `lib/speed.ts`).
- **Rerun on change.** Pass the memoized trace as `runKey` so editing any input replays the run from the start.
- **Controls go in the deck, outputs go in readouts.** Put the transport first, then `Group`s named for what they change (*Data*, *Model*, *Kernel*, …). Direct manipulation of the canvas goes in `hint`.
- **Use the theme's data colors** from `lib/theme.ts`, and the same color for a quantity everywhere it appears (plot, slider, formula).
- **Pause offscreen work.** Live simulations should run under `useInView` so they stop when scrolled away.

### 3. Scenarios

Export a `<Playground scenarios={…} explainer={…} />`. Each scenario is:

```tsx
{
    id: "bad-start",                       // becomes ?s=bad-start
    title: "Bad start",
    prompt: <>All four centroids start in one corner. Play it, then switch to k-means++.</>,
    render: () => <LloydFigure init={[…]} />,
}
```

Write three to eight scenarios. The first should run itself and look good with no input. The rest should each make one point, often a failure mode. Keep each prompt to one or two sentences: what to do, and what to look for. The `explainer` is a short "How it works": a few bullets and the key formula.

### 4. Register it

- Add it to `PLAYGROUNDS` and `PLAYGROUND_COMPONENTS` in `web/src/playgrounds/registry.ts`.
- Draw a thumbnail for the home page in `web/src/components/Thumb.tsx` (a few lines of painter calls).
- Add it to the [playground guide](playgrounds.md).

## How changes ship

- Work in small PRs: one playground, one feature or one polish pass each.
- Use [conventional commits](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13#types) (`feat(svm): …`, `fix: …`, `docs: …`).
- Every PR gets CI and a Netlify deploy preview. Once both are green it is merged, and Netlify deploys `main` to [hyperplane.jatinm.com](https://hyperplane.jatinm.com/).
