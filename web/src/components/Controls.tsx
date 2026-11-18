import {useEffect, useRef, type ReactNode} from "react";
import type {Timeline} from "../lib/hooks";
import {setSpeed, SPEEDS, speedLabel, useSpeed} from "../lib/speed";

// ---------------------------------------------------------------------------
// Slider
// ---------------------------------------------------------------------------

interface SliderProps {
    label: ReactNode;
    value: number;
    min: number;
    max: number;
    step?: number;
    onChange: (v: number) => void;
    format?: (v: number) => string;
    /** Logarithmic scale (min must be > 0). */
    log?: boolean;
    color?: string;
    disabled?: boolean;
}

export function Slider({label, value, min, max, step, onChange, format, log, color, disabled}: SliderProps) {
    const toPos = (v: number) => (log ? (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min)) : (v - min) / (max - min));
    const fromPos = (p: number) => (log ? Math.exp(Math.log(min) + p * (Math.log(max) - Math.log(min))) : min + p * (max - min));
    const pos = Math.min(1, Math.max(0, toPos(value)));
    const fmt = format ?? ((v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)));

    const handle = (raw: number) => {
        let v = fromPos(raw / 1000);
        if (!log && step) v = Math.round((v - min) / step) * step + min;
        v = Math.min(max, Math.max(min, v));
        onChange(+v.toPrecision(12));
    };

    return (
        <label className={`slider ${disabled ? "is-disabled" : ""}`} style={color ? ({"--accent": color} as React.CSSProperties) : undefined}>
            <span className="slider-head">
                <span className="slider-label">{label}</span>
                <span className="slider-value">{fmt(value)}</span>
            </span>
            <input
                type="range"
                min={0}
                max={1000}
                step={log || !step ? 1 : Math.max(1, (step / (max - min)) * 1000)}
                value={Math.round(pos * 1000)}
                onChange={(e) => handle(+e.target.value)}
                disabled={disabled}
                style={{"--pos": `${pos * 100}%`} as React.CSSProperties}
            />
        </label>
    );
}

// ---------------------------------------------------------------------------
// Buttons & toggles
// ---------------------------------------------------------------------------

interface ButtonProps {
    children: ReactNode;
    onClick: () => void;
    variant?: "primary" | "ghost" | "subtle";
    disabled?: boolean;
    title?: string;
    active?: boolean;
}

export function Button({children, onClick, variant = "ghost", disabled, title, active}: ButtonProps) {
    return (
        <button type="button" className={`btn btn-${variant} ${active ? "is-active" : ""}`} onClick={onClick} disabled={disabled} title={title} aria-label={title}>
            {children}
        </button>
    );
}

interface SegmentedProps<T extends string | number> {
    options: {value: T; label: ReactNode; title?: string}[];
    value: T;
    onChange: (v: T) => void;
    label?: ReactNode;
}

export function Segmented<T extends string | number>({options, value, onChange, label}: SegmentedProps<T>) {
    return (
        <div className="segmented-wrap">
            {label && <span className="control-label">{label}</span>}
            <div className="segmented" role="radiogroup">
                {options.map((o) => (
                    <button
                        key={String(o.value)}
                        type="button"
                        role="radio"
                        aria-checked={o.value === value}
                        className={o.value === value ? "is-active" : ""}
                        onClick={() => onChange(o.value)}
                        title={o.title}
                    >
                        {o.label}
                    </button>
                ))}
            </div>
        </div>
    );
}

export function Toggle({label, checked, onChange}: {label: ReactNode; checked: boolean; onChange: (v: boolean) => void}) {
    return (
        <label className="toggle">
            <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
            <span className="toggle-track" aria-hidden />
            <span className="toggle-label">{label}</span>
        </label>
    );
}

// ---------------------------------------------------------------------------
// Readouts
// ---------------------------------------------------------------------------

export function Stat({label, value, color, sub}: {label: ReactNode; value: ReactNode; color?: string; sub?: ReactNode}) {
    return (
        <div className="stat">
            <span className="stat-label">{label}</span>
            <span className="stat-value" style={color ? {color} : undefined}>
                {value}
            </span>
            {sub && <span className="stat-sub">{sub}</span>}
        </div>
    );
}

export function Stats({children}: {children: ReactNode}) {
    return <div className="stats">{children}</div>;
}

/** Horizontal bar comparing a value to a maximum. */
export function Meter({value, max, color, label, right}: {value: number; max: number; color: string; label?: ReactNode; right?: ReactNode}) {
    const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
    return (
        <div className="meter">
            {(label || right) && (
                <div className="meter-head">
                    <span>{label}</span>
                    <span className="meter-right">{right}</span>
                </div>
            )}
            <div className="meter-track">
                <div className="meter-fill" style={{width: `${pct}%`, background: color}} />
            </div>
        </div>
    );
}

export function Legend({items}: {items: {color: string; label: ReactNode; shape?: "dot" | "line" | "dash" | "square"}[]}) {
    return (
        <div className="legend">
            {items.map((it, i) => (
                <span key={i} className="legend-item">
                    <span className={`legend-swatch legend-${it.shape ?? "dot"}`} style={{"--c": it.color} as React.CSSProperties} />
                    {it.label}
                </span>
            ))}
        </div>
    );
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

const Icon = {
    play: (
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
            <path d="M7 4.5v15l13-7.5z" fill="currentColor" />
        </svg>
    ),
    pause: (
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
            <path d="M6 4h4v16H6zM14 4h4v16h-4z" fill="currentColor" />
        </svg>
    ),
    next: (
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
            <path d="M5 4.5v15l10-7.5zM17 4h3v16h-3z" fill="currentColor" />
        </svg>
    ),
    prev: (
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
            <path d="M19 4.5v15L9 12zM4 4h3v16H4z" fill="currentColor" />
        </svg>
    ),
    reset: (
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
            <path
                d="M12 5a7 7 0 1 1-6.6 4.7"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
            />
            <path d="M4 3.5v6h6z" fill="currentColor" />
        </svg>
    ),
};

export {Icon};

/** Only the most recently mounted transport answers the keyboard. */
const keyTargets: {current: TransportKeys}[] = [];

interface TransportKeys {
    toggle?: () => void;
    forward?: () => void;
    back?: () => void;
}

/** Space plays/pauses, ← / → step, unless the viewer is typing or using a slider. */
function useTransportKeys(keys: TransportKeys) {
    const ref = useRef(keys);
    useEffect(() => {
        ref.current = keys;
    });
    useEffect(() => {
        const entry = ref;
        keyTargets.push(entry);
        const onKey = (e: KeyboardEvent) => {
            if (keyTargets[keyTargets.length - 1] !== entry) return;
            if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
            const el = e.target instanceof HTMLElement ? e.target : null;
            if (el?.closest("input, textarea, select, [contenteditable='true']")) return;
            const k = entry.current;
            if (e.key === " " && k.toggle && !el?.closest("button, a, summary")) {
                e.preventDefault();
                k.toggle();
            } else if (e.key === "ArrowRight" && k.forward) {
                e.preventDefault();
                k.forward();
            } else if (e.key === "ArrowLeft" && k.back) {
                e.preventDefault();
                k.back();
            }
        };
        window.addEventListener("keydown", onKey);
        return () => {
            window.removeEventListener("keydown", onKey);
            keyTargets.splice(keyTargets.indexOf(entry), 1);
        };
    }, []);
}

/** The site-wide playback speed. Every animation scales its own pace by it. */
export function SpeedPicker() {
    const speed = useSpeed();
    return (
        <div className="speed-picker" role="radiogroup" aria-label="Playback speed" title="Playback speed (remembered across playgrounds)">
            {SPEEDS.map((s) => (
                <button key={s} type="button" role="radio" aria-checked={speed === s} className={speed === s ? "is-active" : ""} onClick={() => setSpeed(s)}>
                    {speedLabel(s)}
                </button>
            ))}
        </div>
    );
}

interface PlayerProps {
    timeline: Timeline;
    /** Label for the current step, e.g. "update 3 / 12". */
    status?: ReactNode;
}

/** Transport for a recorded run: reset, step, play, scrub and speed. */
export function Player({timeline: tl, status}: PlayerProps) {
    const pos = tl.length > 0 ? (tl.t / tl.length) * 100 : 0;
    useTransportKeys({toggle: tl.length > 0 ? tl.toggle : undefined, forward: tl.stepForward, back: tl.stepBack});
    return (
        <div className="player">
            <div className="player-buttons">
                <Button onClick={tl.reset} title="Back to the start" variant="subtle">
                    {Icon.reset}
                </Button>
                <Button onClick={tl.stepBack} title="Step back (←)" variant="subtle" disabled={tl.t <= 0}>
                    {Icon.prev}
                </Button>
                <Button onClick={tl.toggle} title={tl.playing ? "Pause (space)" : "Play (space)"} variant="primary" disabled={tl.length === 0}>
                    {tl.playing ? Icon.pause : Icon.play}
                </Button>
                <Button onClick={tl.stepForward} title="Step forward (→)" variant="subtle" disabled={tl.atEnd}>
                    {Icon.next}
                </Button>
            </div>
            <input
                className="player-scrub"
                type="range"
                min={0}
                max={Math.max(1, tl.length) * 100}
                value={Math.round(tl.t * 100)}
                onChange={(e) => tl.seek(+e.target.value / 100)}
                aria-label="Scrub through steps"
                style={{"--pos": `${pos}%`} as React.CSSProperties}
                disabled={tl.length === 0}
            />
            <SpeedPicker />
            {status && <div className="player-status">{status}</div>}
        </div>
    );
}

interface LiveControlsProps {
    running: boolean;
    onToggle: () => void;
    /** Advance by one step while paused. */
    onStep?: () => void;
    stepTitle?: string;
    onReset?: () => void;
    resetTitle?: string;
    /** Extra buttons, e.g. "Train 100 episodes". */
    children?: ReactNode;
    status?: ReactNode;
}

/** Transport for a simulation that runs live: play/pause, step, reset and speed. */
export function LiveControls({running, onToggle, onStep, stepTitle = "Step once", onReset, resetTitle = "Start over", children, status}: LiveControlsProps) {
    useTransportKeys({toggle: onToggle, forward: onStep});
    return (
        <div className="player player-live">
            <div className="player-buttons">
                {onReset && (
                    <Button onClick={onReset} title={resetTitle} variant="subtle">
                        {Icon.reset}
                    </Button>
                )}
                <Button onClick={onToggle} title={running ? "Pause (space)" : "Play (space)"} variant="primary">
                    {running ? Icon.pause : Icon.play}
                </Button>
                {onStep && (
                    <Button onClick={onStep} title={`${stepTitle} (→)`} variant="subtle" disabled={running}>
                        {Icon.next}
                    </Button>
                )}
            </div>
            <div className="player-extra">{children}</div>
            <SpeedPicker />
            {status && <div className="player-status">{status}</div>}
        </div>
    );
}
