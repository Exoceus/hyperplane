import {useEffect, useMemo, useRef, useState} from "react";
import {Link} from "react-router";
import {FeatureArt} from "../components/FeatureArt";
import {Plot} from "../components/Plot";
import {Thumb} from "../components/Thumb";
import {probabilityColor} from "../lib/colors";
import {useAnimationFrame, useInView} from "../lib/hooks";
import {heatImage} from "../lib/painter";
import {C, classColor} from "../lib/theme";
import {contour_segments, make_shape, Mlp, randomSeed} from "../lib/wasm";
import {CATEGORIES, LIVE, ORDERED, PLAYGROUNDS, type CategoryId} from "../playgrounds/registry";

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
        <figure ref={ref} className="hero-figure">
            <div className="hero-canvas">
                <span className="tick tick-tl" aria-hidden />
                <span className="tick tick-tr" aria-hidden />
                <span className="tick tick-bl" aria-hidden />
                <span className="tick tick-br" aria-hidden />
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
            </div>
            <figcaption className="hero-caption">
                <span className="fig-label">Fig. 0</span>
                <span>
                    A 2-10-10-1 network learning <em>{frame?.shape ?? "spiral"}</em>, live in your browser.{" "}
                    <Link to="/play/neural-networks">Open this playground →</Link>
                </span>
                <span className="mono hero-meta">
                    <span className="live-dot" />
                    step {frame?.steps ?? 0} · loss {(frame?.loss ?? 0.69).toFixed(3)}
                </span>
            </figcaption>
        </figure>
    );
}

interface Featured {
    slug: string;
    pitch: string;
    jumps: {id: string; title: string}[];
}

/** Two playgrounds worth starting with. */
const FEATURED: Featured[] = [
    {
        slug: "gradient-descent",
        pitch: "Drop four optimizers on a loss landscape and race them: plain gradient descent zig-zags, momentum overshoots, Adam adapts. Then orbit the same race in 3D.",
        jumps: [
            {id: "valley", title: "Narrow valley"},
            {id: "saddle", title: "Escape the saddle"},
            {id: "3d", title: "In 3D"},
        ],
    },
    {
        slug: "q-learning",
        pitch: "An agent that knows nothing explores a gridworld until value leaks back from the goal and its arrows line up into a route. Paint walls and pits while it learns.",
        jumps: [
            {id: "maze", title: "Find the goal"},
            {id: "cliff", title: "Cliff walk"},
            {id: "build", title: "Build a world"},
        ],
    },
];

function FeatureCard({f}: {f: Featured}) {
    const meta = PLAYGROUNDS.find((p) => p.slug === f.slug)!;
    const cat = CATEGORIES.find((c) => c.id === meta.category)!;
    return (
        <article className="feature">
            <Link to={`/play/${f.slug}`} className="feature-art" aria-label={`Open ${meta.title}`}>
                <FeatureArt slug={f.slug} label={`${meta.title}, animated preview`} />
                <span className="feature-live">
                    <span className="live-dot" /> live preview
                </span>
            </Link>
            <div className="feature-body">
                <div className="feature-kicker">{cat.title}</div>
                <h3>
                    <Link to={`/play/${f.slug}`}>{meta.title}</Link>
                </h3>
                <p>{f.pitch}</p>
                <div className="feature-jump">
                    <span className="feature-jump-label">Jump to</span>
                    {f.jumps.map((j) => (
                        <Link key={j.id} className="jump" to={`/play/${f.slug}?s=${j.id}`}>
                            {j.title}
                        </Link>
                    ))}
                </div>
            </div>
        </article>
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
                    <div className="eyebrow">Machine learning, by hand</div>
                    <h1>
                        Poke the algorithms. <em>Watch the math move.</em>
                    </h1>
                    <p className="hero-lede">
                        {LIVE.length} interactive playgrounds. Drag the data, turn the knobs, and step through scenarios that show where each
                        algorithm shines and where it breaks. Everything runs live in your browser on a Rust core compiled to WebAssembly.
                    </p>
                    <div className="hero-ctas">
                        <Link className="btn btn-primary btn-lg" to="/play/neural-networks">
                            Start with neural networks
                        </Link>
                        <a className="btn btn-ghost btn-lg" href="#playgrounds">
                            Browse all {LIVE.length} ↓
                        </a>
                    </div>
                </div>
                <HeroCanvas />
            </section>

            <section className="featured" aria-labelledby="featured-title">
                <header className="section-head">
                    <span className="section-num">§ 1</span>
                    <h2 id="featured-title">Start here</h2>
                    <p>Two playgrounds that show what this place is for.</p>
                </header>
                <div className="features">
                    {FEATURED.map((f) => (
                        <FeatureCard key={f.slug} f={f} />
                    ))}
                </div>
            </section>

            <section id="playgrounds" className="catalog" aria-labelledby="catalog-title">
                <header className="section-head">
                    <span className="section-num">§ 2</span>
                    <h2 id="catalog-title">Every playground</h2>
                    <p>Grouped by what the algorithm is for. Each one opens on a scenario that runs itself.</p>
                </header>
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
                        const n = ORDERED.indexOf(p) + 1;
                        return (
                            <Link key={p.slug} to={`/play/${p.slug}`} className="card">
                                <div className="card-thumb">
                                    <Thumb slug={p.slug} />
                                </div>
                                <div className="card-body">
                                    <div className="card-kicker">
                                        <span>{cat.title}</span>
                                        <span className="card-index">No. {String(n).padStart(2, "0")}</span>
                                    </div>
                                    <h3>{p.title}</h3>
                                    <p>{p.tagline}</p>
                                </div>
                            </Link>
                        );
                    })}
                </div>
            </section>
        </div>
    );
}
