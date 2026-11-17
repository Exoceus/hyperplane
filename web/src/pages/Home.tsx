import {useEffect, useMemo, useRef, useState} from "react";
import {Link} from "react-router";
import {Plot} from "../components/Plot";
import {Thumb} from "../components/Thumb";
import {probabilityColor} from "../lib/colors";
import {useAnimationFrame, useInView} from "../lib/hooks";
import {heatImage} from "../lib/painter";
import {C, classColor} from "../lib/theme";
import {contour_segments, make_shape, Mlp, randomSeed} from "../lib/wasm";
import {CATEGORIES, LIVE, PLAYGROUNDS, type CategoryId} from "../playgrounds/registry";

const HERO_VIEW = {xmin: -1.65, xmax: 1.65, ymin: -1.05, ymax: 1.05};
const SHAPES = ["spiral", "moons", "circles", "xor"] as const;
const GRID = {nx: 110, ny: 70};

/** A small neural network learning a toy dataset, live, right on the landing page. */
function HeroCanvas() {
    const [ref, visible] = useInView<HTMLDivElement>("0px");
    const sim = useRef<{net: Mlp; data: Float64Array; frames: number; shape: number; done: number} | null>(null);
    const [frame, setFrame] = useState<{img: HTMLCanvasElement; contour: Float64Array; loss: number; steps: number; shape: string} | null>(null);

    const reset = (shape: number) => {
        sim.current?.net.free();
        const kind = SHAPES[shape % SHAPES.length];
        const data = make_shape(kind, 260, kind === "spiral" ? 0.12 : 0.15, randomSeed());
        sim.current = {net: new Mlp(new Uint32Array([2, 10, 10, 1]), 0, randomSeed()), data, frames: 0, shape, done: 0};
    };

    useEffect(() => {
        reset(0);
        return () => {
            sim.current?.net.free();
            sim.current = null;
        };
    }, []);

    useAnimationFrame(() => {
        const s = sim.current;
        if (!s) return;
        s.frames++;
        const loss = s.net.train(s.data, 3, 0.02, 0, 1, 0);
        if (loss < 0.05 || s.frames > 2400) s.done++;
        if (s.done > 120) {
            reset(s.shape + 1);
            return;
        }
        if (s.frames % 2 === 0) {
            const g = s.net.predict_grid(HERO_VIEW.xmin, HERO_VIEW.xmax, HERO_VIEW.ymin, HERO_VIEW.ymax, GRID.nx, GRID.ny);
            setFrame({
                img: heatImage(g, GRID.nx, GRID.ny, (p) => probabilityColor(p, 0.22)),
                contour: contour_segments(g, GRID.nx, GRID.ny, HERO_VIEW.xmin, HERO_VIEW.xmax, HERO_VIEW.ymin, HERO_VIEW.ymax, new Float64Array([0.5])),
                loss,
                steps: s.net.steps,
                shape: SHAPES[s.shape % SHAPES.length],
            });
        }
    }, visible);

    return (
        <div ref={ref} className="hero-canvas">
            <Plot
                label="A neural network learning to separate two classes, live"
                view={HERO_VIEW}
                draw={(p) => {
                    p.clear();
                    if (frame) p.image(frame.img);
                    p.grid(0.25, {axes: false});
                    if (frame) p.contours(frame.contour, () => ({color: C.white, width: 2, alpha: 0.9}));
                    const d = sim.current?.data;
                    if (d) for (let i = 0; i < d.length; i += 3) p.point(d[i], d[i + 1], classColor(d[i + 2]), {r: 3.6});
                }}
            />
            <div className="hero-canvas-caption">
                <span>
                    <span className="live-dot" /> A 2-10-10-1 network learning <em>{frame?.shape ?? "spiral"}</em>, live
                </span>
                <span className="mono">
                    step {frame?.steps ?? 0} · loss {(frame?.loss ?? 0.69).toFixed(3)}
                </span>
            </div>
        </div>
    );
}

export default function Home() {
    useEffect(() => {
        document.title = "Hyperplane · interactive machine learning playgrounds";
    }, []);
    const [filter, setFilter] = useState<CategoryId | "all">("all");
    const shown = useMemo(() => PLAYGROUNDS.filter((p) => filter === "all" || p.category === filter), [filter]);
    const counts = useMemo(() => Object.fromEntries(CATEGORIES.map((c) => [c.id, PLAYGROUNDS.filter((p) => p.category === c.id).length])), []);

    return (
        <div className="home">
            <section className="hero">
                <div className="hero-text">
                    <div className="eyebrow">Machine learning playgrounds</div>
                    <h1>
                        Poke the algorithms. <span className="grad">Watch the math move.</span>
                    </h1>
                    <p className="hero-lede">
                        {LIVE.length} interactive sandboxes. Drag data, tweak parameters, and run scenarios that show where each algorithm shines
                        and where it breaks. Everything runs live in your browser, powered by Rust and WebAssembly.
                    </p>
                    <div className="hero-ctas">
                        <a className="btn btn-primary btn-lg" href="#playgrounds">
                            Pick a playground ↓
                        </a>
                    </div>
                </div>
                <HeroCanvas />
            </section>

            <section id="playgrounds" className="catalog">
                <div className="filters" role="tablist" aria-label="Filter by category">
                    <button type="button" className={`chip ${filter === "all" ? "is-active" : ""}`} onClick={() => setFilter("all")}>
                        All <span className="chip-count">{PLAYGROUNDS.length}</span>
                    </button>
                    {CATEGORIES.map((c) => (
                        <button key={c.id} type="button" className={`chip ${filter === c.id ? "is-active" : ""}`} onClick={() => setFilter(c.id)}>
                            {c.title} <span className="chip-count">{counts[c.id]}</span>
                        </button>
                    ))}
                </div>
                <div className="cards">
                    {shown.map((p) => {
                        const cat = CATEGORIES.find((c) => c.id === p.category)!;
                        return p.status === "live" ? (
                            <Link key={p.slug} to={`/play/${p.slug}`} className="card">
                                <div className="card-thumb">
                                    <Thumb slug={p.slug} />
                                </div>
                                <div className="card-body">
                                    <div className="card-kicker">{cat.title}</div>
                                    <h3>{p.title}</h3>
                                    <p>{p.tagline}</p>
                                </div>
                            </Link>
                        ) : (
                            <div key={p.slug} className="card card-soon" aria-disabled>
                                <div className="card-body">
                                    <div className="card-kicker">{cat.title} · coming soon</div>
                                    <h3>{p.title}</h3>
                                    <p>{p.tagline}</p>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </section>
        </div>
    );
}
