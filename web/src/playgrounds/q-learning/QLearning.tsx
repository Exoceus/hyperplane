import {useEffect, useMemo, useRef, useState} from "react";
import {Button, LiveControls, Player, Segmented, Slider, Stat, Stats, Toggle} from "../../components/Controls";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Playground, type Scenario} from "../../components/Playground";
import {Plot, type PlotPointer} from "../../components/Plot";
import {Actions, Group, Rig} from "../../components/Rig";
import {useAnimationFrame, useInView, useTimeline} from "../../lib/hooks";
import type {Painter} from "../../lib/painter";
import {useSpeed} from "../../lib/speed";
import {C, rgba} from "../../lib/theme";
import {GridWorld, randomSeed, value_iteration} from "../../lib/wasm";

// ---------------------------------------------------------------------------
// Maps
// ---------------------------------------------------------------------------

/** '.' empty, '#' wall, 'G' goal (+1), 'X' pit (−1), 'S' start, 'g' small goal (+0.3). */
const CODES: Record<string, number> = {".": 0, "#": 1, G: 2, X: 3, S: 4, g: 5};

const MAPS = {
    maze: ["S...#....", ".##.#.##.", ".#....#..", ".#.####.#", "...#.....", "##.#.###.", ".....#..G"],
    cliff: ["............", "............", "............", "SXXXXXXXXXXG"],
    twoGoals: [".........", ".g.......", "S.......G", ".........", "........."],
    open: ["..........", "..........", "..........", "S........G", "..........", "..........", ".........."],
};
type MapName = keyof typeof MAPS;

function parseMap(rows: string[]) {
    const h = rows.length;
    const w = rows[0].length;
    const cells = new Uint8Array(w * h);
    rows.forEach((row, y) => [...row].forEach((ch, x) => (cells[y * w + x] = CODES[ch] ?? 0)));
    return {w, h, cells};
}

const TERMINAL: Record<number, {label: string; color: string}> = {
    2: {label: "+1", color: C.green},
    3: {label: "−1", color: C.red},
    5: {label: "+0.3", color: C.teal},
};

const ARROW = [
    [0, 1],
    [1, 0],
    [0, -1],
    [-1, 0],
];

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

interface GridState {
    w: number;
    h: number;
    cells: Uint8Array;
    q: Float64Array;
    visits?: Uint32Array;
}

function cellCenter(w: number, h: number, i: number): [number, number] {
    return [(i % w) + 0.5, h - Math.floor(i / w) - 0.5];
}

function drawGrid(p: Painter, g: GridState, opts: {showQ: boolean; hideUnvisited?: boolean}) {
    const {w, h, cells, q} = g;
    p.clear();
    let maxAbs = 0.05;
    for (let s = 0; s < w * h; s++) for (let a = 0; a < 4; a++) maxAbs = Math.max(maxAbs, Math.abs(q[s * 4 + a]));
    const valueColor = (v: number) => rgba(v >= 0 ? C.green : C.red, Math.min(0.75, (Math.abs(v) / maxAbs) * 0.75));

    for (let s = 0; s < w * h; s++) {
        const x = s % w;
        const y = h - Math.floor(s / w) - 1;
        const box = [x, y, x + 1, y, x + 1, y + 1, x, y + 1];
        const code = cells[s];
        if (code === 1) {
            p.polyline(box, {close: true, fill: "#2c3447", color: "#3a4459", width: 1});
            for (let k = 1; k <= 3; k++) p.line(x + k * 0.25, y, x, y + k * 0.25, {color: "#3d4760", width: 1});
            for (let k = 1; k <= 3; k++) p.line(x + 1, y + k * 0.25, x + k * 0.25, y + 1, {color: "#3d4760", width: 1});
            continue;
        }
        if (TERMINAL[code]) {
            const t = TERMINAL[code];
            p.polyline(box, {close: true, fill: t.color, fillAlpha: 0.32, color: t.color, width: 1.5});
            p.text(t.label, x + 0.5, y + 0.5, {color: C.white, size: 13, align: "center", baseline: "middle", font: "mono", weight: 600});
            continue;
        }
        const qs = q.subarray(s * 4, s * 4 + 4);
        const v = Math.max(qs[0], qs[1], qs[2], qs[3]);
        const seen = !opts.hideUnvisited || !g.visits || g.visits[s] > 0;
        if (opts.showQ) {
            // Four triangles, one per action.
            const cx = x + 0.5;
            const cy = y + 0.5;
            const tri = [
                [x, y + 1, x + 1, y + 1],
                [x + 1, y + 1, x + 1, y],
                [x + 1, y, x, y],
                [x, y, x, y + 1],
            ];
            for (let a = 0; a < 4; a++) {
                const [ax, ay, bx, by] = tri[a];
                p.polyline([cx, cy, ax, ay, bx, by], {close: true, fill: valueColor(qs[a]), color: rgba(C.muted, 0.25), width: 0.6});
            }
        } else if (seen) {
            p.polyline(box, {close: true, fill: valueColor(v), color: "none"});
        }
        p.polyline(box, {close: true, color: rgba(C.muted, 0.18), width: 1});
        if (code === 4) p.text("S", x + 0.5, y + 0.5, {color: C.text2, size: 12, align: "center", baseline: "middle", weight: 700});
        // Greedy action arrow.
        const spread = Math.max(...qs) - Math.min(...qs);
        if (seen && spread > 1e-6 && !opts.showQ) {
            const best = qs.indexOf(v);
            const [dx, dy] = ARROW[best];
            p.arrow(x + 0.5 - dx * 0.22, y + 0.5 - dy * 0.22, x + 0.5 + dx * 0.25, y + 0.5 + dy * 0.25, {color: rgba(C.white, 0.85), width: 2, head: 7});
        }
    }
}

function paintIndex(e: PlotPointer, w: number, h: number): number {
    const x = Math.floor(e.x);
    const y = h - 1 - Math.floor(e.y);
    if (x < 0 || y < 0 || x >= w || y >= h) return -1;
    return y * w + x;
}

const TOOLS = [
    {value: 1, label: "Wall"},
    {value: 3, label: "Pit"},
    {value: 2, label: "Goal"},
    {value: 4, label: "Start"},
    {value: 0, label: "Erase"},
];

// ---------------------------------------------------------------------------
// Q-learning instrument
// ---------------------------------------------------------------------------

interface QProps {
    map?: MapName;
    epsilon?: number;
    gamma?: number;
    alpha?: number;
    slip?: number;
    stepReward?: number;
    speed?: number;
}

interface Snapshot extends GridState {
    pos: number;
    trail: Uint32Array;
    lengths: Float64Array;
    episode: number;
}

function QInstrument(props: QProps) {
    const map = useMemo(() => parseMap(MAPS[props.map ?? "maze"]), [props.map]);
    const [epsilon, setEpsilon] = useState(props.epsilon ?? 0.15);
    const [alpha, setAlpha] = useState(props.alpha ?? 0.5);
    const [gamma, setGamma] = useState(props.gamma ?? 0.95);
    const [slip, setSlip] = useState(props.slip ?? 0);
    const [stepReward, setStepReward] = useState(props.stepReward ?? -0.01);
    // Agent steps per second: this scenario's natural pace times the viewer's speed.
    const speed = (props.speed ?? 12) * useSpeed();
    const [running, setRunning] = useState(true);
    const [showQ, setShowQ] = useState(false);
    const [tool, setTool] = useState(1);
    const [painting, setPainting] = useState(false);
    const [snap, setSnap] = useState<Snapshot | null>(null);
    const [ref, visible] = useInView<HTMLDivElement>();
    const agent = useRef<GridWorld | null>(null);
    const motion = useRef({from: 0, to: 0, t: 1, acc: 0});

    const read = (g: GridWorld): Snapshot => ({
        w: map.w,
        h: map.h,
        cells: g.cells(),
        q: g.q(),
        visits: g.visits(),
        pos: g.pos,
        trail: g.trail(),
        lengths: g.lengths(),
        episode: g.episode,
    });

    useEffect(() => {
        const g = new GridWorld(map.w, map.h, map.cells, randomSeed());
        agent.current = g;
        return () => {
            g.free();
            if (agent.current === g) agent.current = null;
        };
    }, [map]);

    useEffect(() => {
        agent.current?.set_params(alpha, gamma, epsilon, stepReward, slip);
    }, [alpha, gamma, epsilon, stepReward, slip, map]);

    const advance = (g: GridWorld) => {
        const m = motion.current;
        const r = g.step();
        m.from = r[0];
        m.to = r[4] > 0 ? r[2] : g.pos;
        m.t = 0;
    };

    useAnimationFrame((dt) => {
        const g = agent.current;
        if (!g) return;
        const m = motion.current;
        if (running) {
            m.acc += dt * speed;
            let steps = Math.floor(m.acc);
            m.acc -= steps;
            steps = Math.min(steps, 5000);
            for (let i = 0; i < steps; i++) advance(g);
            m.t = Math.min(1, m.t + dt * Math.max(speed, 4));
        }
        if (running || !snap) setSnap(read(g));
    }, visible);

    const act = (fn: (g: GridWorld) => void) => {
        const g = agent.current;
        if (!g) return;
        fn(g);
        setSnap(read(g));
    };

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") setPainting(true);
        if (e.type === "up" || e.type === "leave") setPainting(false);
        if (e.type === "down" || (e.type === "move" && painting)) {
            const i = paintIndex(e, map.w, map.h);
            if (i >= 0 && snap && snap.cells[i] !== tool) act((g) => g.set_cell(i, tool));
        }
    };

    const lengths = useMemo(() => snap?.lengths ?? new Float64Array(), [snap]);
    const smooth = useMemo(() => {
        const out = new Float64Array(lengths.length);
        let acc = 0;
        for (let i = 0; i < lengths.length; i++) {
            acc = i === 0 ? lengths[0] : acc * 0.9 + lengths[i] * 0.1;
            out[i] = acc;
        }
        return out;
    }, [lengths]);

    const view = {xmin: 0, xmax: map.w, ymin: 0, ymax: map.h};
    const m = motion.current;
    const slowEnough = speed <= 30;

    return (
        <div ref={ref}>
            <Rig
                hint="The agent is exploring live · paint the grid to change its world"
                visual={
                    <Plot
                        label="Gridworld with learned values and greedy policy"
                        view={view}
                        maxHeight={560}
                        onPointer={onPointer}
                        cursor="cell"
                        draw={(p) => {
                            if (!snap) return p.clear();
                            drawGrid(p, snap, {showQ, hideUnvisited: true});
                            // Current episode's trail.
                            const trail: number[] = [];
                            for (const s of snap.trail) trail.push(...cellCenter(map.w, map.h, s));
                            if (slowEnough) p.polyline(trail, {color: rgba(C.yellow, 0.35), width: 2});
                            const [fx, fy] = cellCenter(map.w, map.h, m.from);
                            const [tx, ty] = cellCenter(map.w, map.h, slowEnough ? m.to : snap.pos);
                            const t = slowEnough ? m.t : 1;
                            p.circle(fx + (tx - fx) * t, fy + (ty - fy) * t, Math.min(p.kx, p.ky) * 0.22, {fill: C.yellow, stroke: C.canvas, width: 2.5});
                        }}
                    />
                }
                controls={
                    <>
                        <LiveControls
                            running={running}
                            onToggle={() => setRunning(!running)}
                            onStep={() =>
                                act((g) => {
                                    advance(g);
                                    motion.current.t = 1;
                                })
                            }
                            stepTitle="One move"
                            onReset={() => act((g) => g.reset())}
                            resetTitle="Forget everything it has learned"
                            status={<span className="mono">{Math.round(speed)} moves per second</span>}
                        >
                            <Button onClick={() => act((g) => g.train(100))}>Train 100 episodes</Button>
                        </LiveControls>
                        <Group label="Agent">
                            <Slider label={<>exploration <M>{"\\varepsilon"}</M></>} value={epsilon} min={0} max={1} onChange={setEpsilon} color={C.yellow} />
                            <Slider label={<>discount <M>{"\\gamma"}</M></>} value={gamma} min={0} max={0.99} onChange={setGamma} color={C.purple} />
                            <Slider label={<>learning rate <M>{"\\alpha"}</M></>} value={alpha} min={0.01} max={1} onChange={setAlpha} />
                        </Group>
                        <Group label="World">
                            <Slider label="slip (random wind)" value={slip} min={0} max={0.6} onChange={setSlip} color={C.red} format={(v) => `${Math.round(v * 100)}%`} />
                            <Slider label="reward per step" value={stepReward} min={-0.1} max={0} onChange={setStepReward} format={(v) => v.toFixed(3)} />
                            <Segmented label="Paint" value={tool} onChange={setTool} options={TOOLS} />
                            <Toggle label="Show all four Q-values per cell" checked={showQ} onChange={setShowQ} />
                        </Group>
                    </>
                }
                readouts={
                    <>
                        <Stats>
                            <Stat label="episodes" value={snap?.episode ?? 0} />
                            <Stat label="last episode" value={lengths.length ? `${lengths[lengths.length - 1]} steps` : "—"} />
                        </Stats>
                        <div className="panel-label">
                            <span>Steps per episode (smoothed)</span>
                            <span className="panel-label-right">episode {snap?.episode ?? 0}</span>
                        </div>
                        <LineChart label="Steps per episode" series={[{values: smooth, color: C.yellow}, {values: lengths, color: rgba(C.muted, 0.5), width: 1}]} height={90} yMin={0} />
                    </>
                }
            />
        </div>
    );
}

// ---------------------------------------------------------------------------
// Value iteration instrument
// ---------------------------------------------------------------------------

const SWEEPS = 30;

function ValueIterationInstrument(props: {map?: MapName}) {
    const base = useMemo(() => parseMap(MAPS[props.map ?? "maze"]), [props.map]);
    const [cells, setCells] = useState<Uint8Array>(base.cells);
    const [gamma, setGamma] = useState(0.9);
    const [slip, setSlip] = useState(0);
    const [stepReward, setStepReward] = useState(0);
    const [tool, setTool] = useState(1);
    const [painting, setPainting] = useState(false);
    const viKey = useMemo(() => ({cells, gamma, slip, stepReward}), [cells, gamma, slip, stepReward]);
    const tl = useTimeline(SWEEPS, 3, {autoplay: true, runKey: viKey});
    const sweep = Math.floor(tl.t + 1e-9);
    const q = useMemo(() => value_iteration(base.w, base.h, cells, gamma, stepReward, slip, sweep), [base, cells, gamma, stepReward, slip, sweep]);

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") setPainting(true);
        if (e.type === "up" || e.type === "leave") setPainting(false);
        if (e.type === "down" || (e.type === "move" && painting)) {
            const i = paintIndex(e, base.w, base.h);
            if (i < 0 || cells[i] === tool) return;
            const next = cells.slice();
            if (tool === 4) next.forEach((c, j) => c === 4 && (next[j] = 0));
            next[i] = tool;
            setCells(next);
        }
    };

    return (
        <Rig
            hint="Each sweep is one round of Bellman backups over every cell · paint to replan"
            visual={
                <Plot
                    label="Value iteration on a gridworld"
                    view={{xmin: 0, xmax: base.w, ymin: 0, ymax: base.h}}
                    maxHeight={560}
                    onPointer={onPointer}
                    cursor="cell"
                    draw={(p) => drawGrid(p, {w: base.w, h: base.h, cells, q}, {showQ: false})}
                />
            }
            caption={`Sweep ${sweep}: values have spread up to ${sweep} steps from every reward.`}
            controls={
                <>
                    <Player timeline={tl} />
                    <Group label="World">
                        <Slider label={<>discount <M>{"\\gamma"}</M></>} value={gamma} min={0} max={0.99} onChange={setGamma} color={C.purple} />
                        <Slider label="slip (random wind)" value={slip} min={0} max={0.6} onChange={setSlip} color={C.red} format={(v) => `${Math.round(v * 100)}%`} />
                        <Slider label="reward per step" value={stepReward} min={-0.1} max={0} onChange={setStepReward} format={(v) => v.toFixed(3)} />
                        <Segmented label="Paint" value={tool} onChange={setTool} options={TOOLS} />
                        <Actions>
                            <Button onClick={() => setCells(base.cells)}>Restore map</Button>
                        </Actions>
                    </Group>
                </>
            }
            readouts={
                <p className="note">
                    Value iteration knows the map and the slip probabilities, so it can <em>plan</em> without ever moving. Q-learning has to find the
                    same values by trial and error.
                </p>
            }
        />
    );
}

// ---------------------------------------------------------------------------
// Playground
// ---------------------------------------------------------------------------

const SCENARIOS: Scenario[] = [
    {
        id: "maze",
        title: "Find the goal",
        prompt: <>At first the agent stumbles around at random. Every time it reaches the goal, value leaks one step further back, and the arrows line up into a route. Hit "Train 100 episodes" to fast-forward.</>,
        render: () => <QInstrument />,
    },
    {
        id: "cliff",
        title: "Cliff walk",
        prompt: <>The shortest route hugs a cliff. Train, and the greedy arrows walk right along the edge. Now add slip (wind), reset and retrain: the learned path backs away from the drop.</>,
        render: () => <QInstrument map="cliff" epsilon={0.1} speed={60} />,
    },
    {
        id: "explore",
        title: "Explore vs. exploit",
        prompt: <>With ε = 0 the agent grabs the first reward it finds, the nearby +0.3, and never looks further. Raise ε, reset and train again until it discovers the +1.</>,
        render: () => <QInstrument map="twoGoals" epsilon={0} stepReward={0} speed={60} />,
    },
    {
        id: "discount",
        title: "Short-sighted agent",
        prompt: <>With γ = 0.6 the far +1 is discounted to almost nothing, so the agent settles for +0.3. Raise γ toward 0.95, reset and retrain, and it starts making the long trip.</>,
        render: () => <QInstrument map="twoGoals" gamma={0.6} epsilon={0.3} speed={600} />,
    },
    {
        id: "value-iteration",
        title: "Plan with value iteration",
        prompt: <>If you know the map you don't need to explore. Each sweep of Bellman backups spreads value one more step out from the goal. Paint walls and watch the plan reroute.</>,
        render: () => <ValueIterationInstrument />,
    },
    {
        id: "build",
        title: "Build a world",
        prompt: <>An empty room. Paint walls, pits and goals while the agent explores. Try a pit right next to the shortest path, or a second goal.</>,
        render: () => <QInstrument map="open" speed={60} />,
    },
];

export default function QLearning() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            The agent learns <M>{"Q(s, a)"}</M>, the expected discounted reward from taking action <M>{"a"}</M> in cell <M>{"s"}</M> and
                            acting well afterwards. After every step it nudges its estimate toward a one-step lookahead:
                        </li>
                    </ul>
                    <MathBlock>{"Q(s,a) \\leftarrow Q(s,a) + \\alpha\\big[r + \\gamma \\max_{a'} Q(s', a') - Q(s,a)\\big]"}</MathBlock>
                    <ul>
                        <li>
                            It acts <M>{"\\varepsilon"}</M>-greedily: a random move with probability <M>{"\\varepsilon"}</M>, otherwise the best-looking
                            one. Too little exploration and it settles for whatever it found first.
                        </li>
                        <li>
                            <M>{"\\gamma"}</M> sets how far ahead it cares: a reward <M>{"k"}</M> steps away is worth <M>{"\\gamma^k"}</M> of its face
                            value.
                        </li>
                        <li>
                            Colors show <M>{"V(s) = \\max_a Q(s,a)"}</M> and arrows show the greedy policy. Value iteration computes the same fixed point
                            directly when the model is known: <M>{"V(s) \\leftarrow \\max_a \\sum_{s'} P(s'\\mid s,a)\\,[r + \\gamma V(s')]"}</M>.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
