import {useEffect, useMemo, useRef, useState} from "react";
import {Button, Icon, Legend, Segmented, Slider, Stat, Stats, Toggle} from "../../components/Controls";
import {FigureLayout} from "../../components/Figure";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {Plot} from "../../components/Plot";
import {useAnimationFrame, useInView} from "../../lib/hooks";
import {C} from "../../lib/theme";
import {Bandit, randomSeed} from "../../lib/wasm";

const STRATEGIES = [
    {id: 0, name: "Greedy", color: C.white},
    {id: 1, name: "ε-greedy", color: C.blue},
    {id: 2, name: "UCB", color: C.green},
    {id: 3, name: "Thompson", color: C.pink},
];

/** log Γ(x) (Lanczos). */
function logGamma(x: number): number {
    const g = 7;
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    x -= 1;
    let a = c[0];
    const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betaPdf(p: number, a: number, b: number): number {
    if (p <= 0 || p >= 1) return 0;
    return Math.exp((a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p) + logGamma(a + b) - logGamma(a) - logGamma(b));
}

function randomMeans(k: number, close = false): number[] {
    return Array.from({length: k}, () => (close ? 0.42 + Math.random() * 0.16 : 0.08 + Math.random() * 0.84));
}

interface Snap {
    counts: Float64Array;
    wins: Float64Array;
    last: Float64Array;
    regret: Float64Array;
    t: number;
}

function BanditInstrument(props: {means?: number[]; focus?: number; epsilon?: number; speed?: number}) {
    const [means, setMeans] = useState<number[]>(props.means ?? [0.25, 0.5, 0.7]);
    const [focus, setFocus] = useState(props.focus ?? 3);
    const [epsilon, setEpsilon] = useState(props.epsilon ?? 0.1);
    const [ucbC, setUcbC] = useState(1);
    const [speed, setSpeed] = useState(props.speed ?? 20);
    const [running, setRunning] = useState(false);
    const [reveal, setReveal] = useState(false);
    const [version, setVersion] = useState(0);
    const [snap, setSnap] = useState<Snap | null>(null);
    const [ref, visible] = useInView<HTMLDivElement>();
    const sim = useRef<Bandit | null>(null);
    const acc = useRef(0);
    const k = means.length;

    useEffect(() => {
        const b = new Bandit(new Float64Array(means), randomSeed());
        sim.current = b;
        return () => {
            b.free();
            if (sim.current === b) sim.current = null;
        };
    }, [means, version]);

    useEffect(() => {
        sim.current?.set_params(epsilon, ucbC);
    }, [epsilon, ucbC, means, version]);

    const read = (b: Bandit): Snap => ({counts: b.counts(), wins: b.wins(), last: b.last(), regret: b.regret(), t: b.t});

    useAnimationFrame((dt) => {
        const b = sim.current;
        if (!b) return;
        if (running && b.t < 20000) {
            acc.current += dt * speed;
            const n = Math.floor(acc.current);
            acc.current -= n;
            if (n > 0) b.step(n);
        }
        if (running || !snap || snap.t !== b.t) setSnap(read(b));
    }, visible);

    const series = useMemo(() => {
        if (!snap) return [];
        const T = snap.regret.length / 4;
        return STRATEGIES.map((s) => {
            const v = new Float64Array(T + 1);
            for (let i = 0; i < T; i++) v[i + 1] = snap.regret[i * 4 + s.id];
            return {values: v, color: s.color, width: s.id === focus ? 2.6 : 1.4};
        });
    }, [snap, focus]);

    const best = means.indexOf(Math.max(...means));
    const counts = snap?.counts ?? new Float64Array(4 * k);
    const wins = snap?.wins ?? new Float64Array(4 * k);
    const total = snap?.t ?? 0;
    const strategy = STRATEGIES[focus];

    return (
        <Instrument hint="Press play: all four strategies pull arms on the same slot machines at the same time">
            <div ref={ref}>
                <FigureLayout
                    sideWidth={300}
                    main={
                        <>
                            <div className="panel-label">
                                <span>What {strategy.name} believes about each arm</span>
                                <span className="panel-label-right">pull {total}</span>
                            </div>
                            <Plot
                                label="Per-arm beliefs for the selected strategy"
                                view={{xmin: 0, xmax: k, ymin: -0.12, ymax: 1.06}}
                                aspect={0.5}
                                draw={(p) => {
                                    p.clear();
                                    for (let v = 0; v <= 1.001; v += 0.25) {
                                        p.line(0, v, k, v, {color: C.grid, width: 1});
                                        p.textPx(`${Math.round(v * 100)}%`, 4, p.sy(v) - 3, {color: C.muted, size: 10, font: "mono"});
                                    }
                                    for (let a = 0; a < k; a++) {
                                        const cx = a + 0.5;
                                        const n = counts[focus * k + a];
                                        const w = wins[focus * k + a];
                                        const share = total > 0 ? n / total : 0;
                                        // Pull share as a faint column.
                                        p.polyline([cx - 0.42, -0.12, cx + 0.42, -0.12, cx + 0.42, -0.12 + share * 1.18, cx - 0.42, -0.12 + share * 1.18], {
                                            close: true,
                                            fill: strategy.color,
                                            fillAlpha: 0.07,
                                            color: "none",
                                        });
                                        if (focus === 3) {
                                            // Thompson: the Beta posterior as a violin.
                                            const A = 1 + w;
                                            const B = 1 + n - w;
                                            let peak = 0;
                                            const ys: number[] = [];
                                            for (let i = 0; i <= 120; i++) {
                                                const y = i / 120;
                                                const d = betaPdf(Math.min(0.999, Math.max(0.001, y)), A, B);
                                                ys.push(d);
                                                peak = Math.max(peak, d);
                                            }
                                            const pts: number[] = [];
                                            ys.forEach((d, i) => pts.push(cx + (0.38 * d) / peak, i / 120));
                                            for (let i = ys.length - 1; i >= 0; i--) pts.push(cx - (0.38 * ys[i]) / peak, i / 120);
                                            p.polyline(pts, {close: true, fill: strategy.color, fillAlpha: 0.28, color: strategy.color, width: 1.5});
                                        } else if (n > 0) {
                                            const mean = w / n;
                                            const half = focus === 2 ? ucbC * Math.sqrt((2 * Math.log(Math.max(total, 1))) / n) : 1.96 * Math.sqrt(Math.max(mean * (1 - mean), 0.01) / n);
                                            const lo = focus === 2 ? mean : Math.max(0, mean - half);
                                            const hi = Math.min(1.05, mean + half);
                                            p.line(cx, lo, cx, hi, {color: strategy.color, width: 6, alpha: 0.4, cap: "butt"});
                                            p.circle(cx, mean, 6, {fill: strategy.color, stroke: C.canvas, width: 2});
                                        }
                                        if (reveal) p.line(cx - 0.3, means[a], cx + 0.3, means[a], {color: a === best ? C.yellow : C.text2, width: 2, dash: [5, 3]});
                                        const pulledNow = snap !== null && snap.last[focus] === a && running;
                                        p.text(pulledNow ? `▸ arm ${a + 1}` : `arm ${a + 1}`, cx, -0.06, {color: pulledNow ? C.yellow : C.text2, size: 11, align: "center", baseline: "middle", weight: pulledNow ? 700 : 500});
                                        p.text(`${n} pulls`, cx, 1.03, {color: C.muted, size: 10, align: "center", font: "mono"});
                                    }
                                }}
                            />
                            <div className="panel-label">
                                <span>Cumulative regret (lower is better)</span>
                            </div>
                            <LineChart label="Cumulative regret per strategy" series={series} height={110} yMin={0} xLabel="pulls" />
                            <Legend items={STRATEGIES.map((s) => ({color: s.color, label: s.name, shape: "line" as const}))} />
                        </>
                    }
                    side={
                        <div className="controls">
                            <Segmented label="Show beliefs of" value={focus} onChange={setFocus} options={STRATEGIES.map((s) => ({value: s.id, label: s.name}))} />
                            <div className="control-row">
                                <Button variant="primary" onClick={() => setRunning(!running)} title={running ? "Pause" : "Play"}>
                                    {running ? Icon.pause : Icon.play}
                                </Button>
                                <Button onClick={() => sim.current && (sim.current.step(1), setSnap(read(sim.current)))}>Pull once</Button>
                                <Button onClick={() => (setVersion(version + 1), setSnap(null), setRunning(false))} title="Reset">
                                    {Icon.reset}
                                </Button>
                            </div>
                            <Slider label="pulls per second" value={speed} min={2} max={400} log step={1} onChange={setSpeed} format={(v) => v.toFixed(0)} />
                            <Slider label={<>ε (ε-greedy)</>} value={epsilon} min={0} max={0.5} onChange={setEpsilon} color={C.blue} />
                            <Slider label="optimism c (UCB)" value={ucbC} min={0} max={3} onChange={setUcbC} color={C.green} />
                            <Toggle label="Reveal true payout rates" checked={reveal} onChange={setReveal} />
                            <Stats>
                                {STRATEGIES.map((s) => (
                                    <Stat key={s.id} label={`${s.name} → best arm`} value={total ? `${Math.round((counts[s.id * k + best] / total) * 100)}%` : "—"} color={s.color} />
                                ))}
                            </Stats>
                            <div className="control-row">
                                <Button onClick={() => (setMeans(randomMeans(k)), setSnap(null), setRunning(false))}>New arms</Button>
                                <Button onClick={() => (setMeans(randomMeans(k === 3 ? 6 : 3)), setSnap(null), setRunning(false))}>{k === 3 ? "6 arms" : "3 arms"}</Button>
                            </div>
                        </div>
                    }
                />
            </div>
        </Instrument>
    );
}

const SCENARIOS: Scenario[] = [
    {
        id: "thompson",
        title: "Three slot machines",
        prompt: <>Press play. Thompson sampling keeps a belief (violin) about each arm's payout and pulls whichever arm wins a random draw from those beliefs. Watch the beliefs about bad arms freeze as they stop being worth trying.</>,
        render: () => <BanditInstrument />,
    },
    {
        id: "greedy",
        title: "Greedy gets stuck",
        prompt: <>Greedy tries each arm once, then always picks the best-looking one. An unlucky first pull can lock it onto a bad arm forever. Hit reset a few times and watch its regret climb in a straight line.</>,
        render: () => <BanditInstrument focus={0} means={[0.55, 0.45, 0.7]} speed={60} />,
    },
    {
        id: "ucb",
        title: "Optimism (UCB)",
        prompt: <>UCB pulls the arm with the highest optimistic estimate: the mean plus a bonus that shrinks with every pull. Neglected arms' bars grow until they get revisited.</>,
        render: () => <BanditInstrument focus={2} speed={40} />,
    },
    {
        id: "close",
        title: "Close call",
        prompt: <>Payouts of 45–55%. Telling these arms apart takes hundreds of pulls. Watch how long the beliefs overlap, and how regret keeps growing slowly.</>,
        render: () => <BanditInstrument means={[0.46, 0.5, 0.55, 0.48]} speed={200} />,
    },
    {
        id: "many",
        title: "Many arms",
        prompt: <>Eight arms. Constant-ε exploration keeps wasting pulls on obviously bad arms forever. The adaptive strategies stop exploring once they're sure.</>,
        render: () => <BanditInstrument means={[0.3, 0.55, 0.2, 0.62, 0.45, 0.7, 0.35, 0.5]} focus={1} epsilon={0.2} speed={200} />,
    },
];

export default function Bandits() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            Each arm pays 1 with an unknown probability <M>{"\\mu_a"}</M>. Every pull is both a bet and an experiment, which is the{" "}
                            <strong>exploration–exploitation</strong> trade-off in its purest form.
                        </li>
                        <li>Performance is measured by <strong>regret</strong>, the expected reward lost relative to always pulling the best arm:</li>
                    </ul>
                    <MathBlock>{"R_T = \\sum_{t=1}^T \\big(\\mu^\\ast - \\mu_{a_t}\\big)"}</MathBlock>
                    <ul>
                        <li>
                            <strong>ε-greedy</strong> explores at random a fixed fraction of the time, so its regret grows linearly. <strong>UCB</strong> pulls{" "}
                            <M>{"\\arg\\max_a \\hat\\mu_a + c\\sqrt{2\\ln t / n_a}"}</M>: optimism in the face of uncertainty.
                        </li>
                        <li>
                            <strong>Thompson sampling</strong> keeps a Beta posterior <M>{"\\text{Beta}(1 + \\text{wins}, 1 + \\text{losses})"}</M> per arm,
                            draws one sample from each, and pulls the arm with the highest draw. UCB and Thompson both achieve logarithmic regret.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
