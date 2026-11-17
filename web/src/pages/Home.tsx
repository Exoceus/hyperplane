import {useEffect, useMemo, useRef, useState} from "react";
import {Link} from "react-router";
import {Plot} from "../components/Plot";
import {Thumb} from "../components/Thumb";
import {probabilityColor} from "../lib/colors";
import {useAnimationFrame, useInView} from "../lib/hooks";
import {heatImage} from "../lib/painter";
import {C, classColor} from "../lib/theme";
import {contour_segments, make_shape, Mlp, randomSeed} from "../lib/wasm";
import {LESSONS, LIVE_LESSONS, TRACKS, lessonNumber} from "../lessons/registry";

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
                img: heatImage(g, GRID.nx, GRID.ny, (p) => probabilityColor(p, 0.3)),
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
        document.title = "Hyperplane — see the math behind machine learning";
    }, []);
    const first = LIVE_LESSONS[0];
    const byTrack = useMemo(() => TRACKS.map((t) => ({track: t, lessons: LESSONS.filter((l) => l.track === t.id)})), []);

    return (
        <div className="home">
            <section className="hero">
                <div className="hero-text">
                    <div className="eyebrow">Interactive machine learning</div>
                    <h1>
                        See the math <span className="grad">move.</span>
                    </h1>
                    <p className="hero-lede">
                        Hyperplane is a set of visual, hands-on explanations of the algorithms behind machine learning. Drag the data,
                        nudge the parameters, step through training, and build intuition for <em>why</em> the math works.
                    </p>
                    <div className="hero-ctas">
                        <Link className="btn btn-primary btn-lg" to={`/learn/${first.slug}`}>
                            Start with {first.title.toLowerCase()} →
                        </Link>
                        <a className="btn btn-ghost btn-lg" href="#lessons">
                            Browse {LIVE_LESSONS.length} lessons
                        </a>
                    </div>
                </div>
                <HeroCanvas />
            </section>

            <section className="pillars">
                <div>
                    <h3>Touch the math</h3>
                    <p>Every figure is a live instrument. Drag points, move weights, and change the step size to see what happens.</p>
                </div>
                <div>
                    <h3>Formulas in color</h3>
                    <p>Symbols in the equations share colors with the figures, so the yellow 𝐰 in a formula is the yellow arrow on screen.</p>
                </div>
                <div>
                    <h3>Real algorithms, locally</h3>
                    <p>The algorithms are written in Rust and compiled to WebAssembly. Training runs in your browser at 60 fps.</p>
                </div>
            </section>

            <section id="lessons" className="tracks">
                {byTrack.map(({track, lessons}) => (
                    <div key={track.id} className="track">
                        <div className="track-head">
                            <h2>{track.title}</h2>
                            <p>{track.blurb}</p>
                        </div>
                        <div className="cards">
                            {lessons.map((l) =>
                                l.status === "live" ? (
                                    <Link key={l.slug} to={`/learn/${l.slug}`} className="card">
                                        <div className="card-thumb">
                                            <Thumb slug={l.slug} />
                                        </div>
                                        <div className="card-body">
                                            <div className="card-kicker">
                                                Lesson {lessonNumber(l.slug)} · {l.minutes} min
                                            </div>
                                            <h3>{l.title}</h3>
                                            <p>{l.tagline}</p>
                                        </div>
                                    </Link>
                                ) : (
                                    <div key={l.slug} className="card card-soon" aria-disabled>
                                        <div className="card-body">
                                            <div className="card-kicker">Coming soon</div>
                                            <h3>{l.title}</h3>
                                            <p>{l.tagline}</p>
                                        </div>
                                    </div>
                                )
                            )}
                        </div>
                    </div>
                ))}
            </section>
        </div>
    );
}
