import {useCallback, useEffect, useLayoutEffect, useRef, useState} from "react";

/** Call `cb(dtSeconds)` every animation frame while `active`. */
export function useAnimationFrame(cb: (dt: number) => void, active: boolean) {
    const cbRef = useRef(cb);
    useEffect(() => {
        cbRef.current = cb;
    });
    useEffect(() => {
        if (!active) return;
        let raf = 0;
        let last = performance.now();
        const loop = (now: number) => {
            const dt = Math.min(0.1, (now - last) / 1000);
            last = now;
            cbRef.current(dt);
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [active]);
}

/** Whether the element is (roughly) on screen. Used to pause offscreen animations. */
export function useInView<T extends Element>(margin = "100px"): [React.RefObject<T | null>, boolean] {
    const ref = useRef<T>(null);
    const [visible, setVisible] = useState(false);
    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const io = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), {rootMargin: margin});
        io.observe(el);
        return () => io.disconnect();
    }, [margin]);
    return [ref, visible];
}

export interface Timeline {
    /** Continuous time in "steps"; integer values are recorded states. */
    t: number;
    length: number;
    playing: boolean;
    atEnd: boolean;
    play: () => void;
    pause: () => void;
    toggle: () => void;
    stepForward: () => void;
    stepBack: () => void;
    reset: () => void;
    seek: (t: number) => void;
    speed: number;
    setSpeed: (s: number) => void;
}

type Mode = "idle" | "play" | "step";

/**
 * Drives a scrubbable animation over `length` discrete steps. Time glides
 * toward a target, so a single "step" is a short tween rather than a jump —
 * you get to watch each quantity move.
 */
export function useTimeline(length: number, initialSpeed = 2): Timeline {
    const [t, setTState] = useState(0);
    const [speed, setSpeed] = useState(initialSpeed);
    const [mode, setMode] = useState<Mode>("idle");
    const tRef = useRef(0);
    const targetRef = useRef(0);
    const len = Math.max(0, length);
    // Read through a ref so callbacks invoked right after the run changes
    // (e.g. "compute a new run, then play") see the new length.
    const lenRef = useRef(len);
    useLayoutEffect(() => {
        lenRef.current = len;
    });

    const setT = useCallback((v: number) => {
        tRef.current = v;
        setTState(v);
    }, []);

    useAnimationFrame((dt) => {
        const cur = Math.min(tRef.current, lenRef.current);
        const target = Math.min(targetRef.current, lenRef.current);
        const rate = mode === "step" ? Math.max(speed, 3) : speed;
        const next = cur < target ? Math.min(target, cur + dt * rate) : Math.max(target, cur - dt * rate);
        setT(next);
        if (next === target) setMode("idle");
    }, mode !== "idle");


    const play = useCallback(() => {
        // Defer one frame so a run computed in the same event is committed.
        requestAnimationFrame(() => {
            if (tRef.current >= lenRef.current) setT(0);
            targetRef.current = lenRef.current;
            setMode("play");
        });
    }, [setT]);

    const pause = useCallback(() => {
        targetRef.current = tRef.current;
        setMode("idle");
    }, []);

    const stepForward = useCallback(() => {
        targetRef.current = Math.min(lenRef.current, Math.floor(tRef.current + 1e-6) + 1);
        setMode("step");
    }, []);

    const stepBack = useCallback(() => {
        targetRef.current = Math.max(0, Math.ceil(Math.min(tRef.current, lenRef.current) - 1e-6) - 1);
        setMode("step");
    }, []);

    const reset = useCallback(() => {
        setT(0);
        targetRef.current = 0;
        setMode("idle");
    }, [setT]);

    const seek = useCallback(
        (v: number) => {
            const c = Math.max(0, Math.min(lenRef.current, v));
            setT(c);
            targetRef.current = c;
            setMode("idle");
        },
        [setT]
    );

    const playing = mode === "play";
    const toggle = useCallback(() => (playing ? pause() : play()), [playing, pause, play]);

    const tc = Math.min(t, len);
    return {t: tc, length: len, playing, atEnd: tc >= len, play, pause, toggle, stepForward, stepBack, reset, seek, speed, setSpeed};
}

/** Smooth ease-in-out on [0, 1]. */
export function ease(t: number): number {
    const x = Math.min(1, Math.max(0, t));
    return x * x * (3 - 2 * x);
}

export function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

export function clamp(v: number, lo: number, hi: number): number {
    return Math.min(hi, Math.max(lo, v));
}

/** Format a number compactly for readouts. */
export function fmt(v: number, digits = 2): string {
    if (!Number.isFinite(v)) return v > 0 ? "∞" : "—";
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(1);
    return v.toFixed(digits);
}
