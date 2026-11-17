import {useEffect, useLayoutEffect, useRef, useState} from "react";
import {Painter, type View} from "../lib/painter";

export interface PlotPointer {
    type: "down" | "move" | "up" | "leave";
    x: number;
    y: number;
    /** Pixels per data unit (for hit-testing in screen space). */
    kx: number;
    ky: number;
    dragging: boolean;
    shift: boolean;
    button: number;
}

interface PlotProps {
    view: View;
    draw: (p: Painter) => void;
    /** Height / width. Defaults to the view's aspect so 1 unit is square. */
    aspect?: number;
    maxHeight?: number;
    /** Fixed CSS height in pixels (width still fills the container). */
    height?: number;
    onPointer?: (e: PlotPointer) => void;
    cursor?: string;
    label: string;
    className?: string;
}

/**
 * A responsive, retina-aware canvas that redraws whenever it re-renders.
 * Pointer events are reported in data coordinates.
 */
export function Plot({view, draw, aspect, maxHeight, height: fixedHeight, onPointer, cursor, label, className}: PlotProps) {
    const wrapRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [width, setWidth] = useState(0);
    const dragging = useRef(false);

    const ratio = aspect ?? (view.ymax - view.ymin) / (view.xmax - view.xmin);
    let height = fixedHeight ?? Math.round(width * ratio);
    let drawWidth = width;
    if (!fixedHeight && maxHeight && height > maxHeight) {
        height = maxHeight;
        drawWidth = Math.round(height / ratio);
    }

    useEffect(() => {
        const el = wrapRef.current;
        if (!el) return;
        const ro = new ResizeObserver((entries) => {
            const w = Math.floor(entries[0].contentRect.width);
            setWidth((prev) => (prev === w ? prev : w));
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    useLayoutEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || drawWidth <= 0 || height <= 0) return;
        const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
        const pw = Math.round(drawWidth * dpr);
        const ph = Math.round(height * dpr);
        if (canvas.width !== pw || canvas.height !== ph) {
            canvas.width = pw;
            canvas.height = ph;
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        draw(new Painter(ctx, drawWidth, height, view));
    });

    const toData = (e: React.PointerEvent): PlotPointer | null => {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        const rect = canvas.getBoundingClientRect();
        const kx = rect.width / (view.xmax - view.xmin);
        const ky = rect.height / (view.ymax - view.ymin);
        return {
            type: "move",
            x: view.xmin + (e.clientX - rect.left) / kx,
            y: view.ymax - (e.clientY - rect.top) / ky,
            kx,
            ky,
            dragging: dragging.current,
            shift: e.shiftKey || e.altKey || e.metaKey,
            button: e.button,
        };
    };

    const handlers = onPointer
        ? {
              onPointerDown: (e: React.PointerEvent<HTMLCanvasElement>) => {
                  const p = toData(e);
                  if (!p) return;
                  dragging.current = true;
                  e.currentTarget.setPointerCapture(e.pointerId);
                  onPointer({...p, type: "down", dragging: true});
              },
              onPointerMove: (e: React.PointerEvent<HTMLCanvasElement>) => {
                  const p = toData(e);
                  if (p) onPointer({...p, type: "move"});
              },
              onPointerUp: (e: React.PointerEvent<HTMLCanvasElement>) => {
                  const p = toData(e);
                  dragging.current = false;
                  if (p) onPointer({...p, type: "up", dragging: false});
              },
              onPointerLeave: (e: React.PointerEvent<HTMLCanvasElement>) => {
                  if (dragging.current) return;
                  const p = toData(e);
                  if (p) onPointer({...p, type: "leave"});
              },
              onContextMenu: (e: React.MouseEvent) => e.preventDefault(),
          }
        : {};

    return (
        <div ref={wrapRef} className={`plot ${className ?? ""}`}>
            <canvas
                ref={canvasRef}
                role="img"
                aria-label={label}
                style={{
                    width: drawWidth,
                    height,
                    cursor: cursor ?? (onPointer ? "crosshair" : "default"),
                    touchAction: onPointer ? "none" : "auto",
                }}
                {...handlers}
            />
        </div>
    );
}
