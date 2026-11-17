import {useRef, useState} from "react";
import {useAnimationFrame} from "../lib/hooks";
import type {View} from "../lib/painter";
import {Scene3D, type Camera} from "../lib/scene3d";
import {Plot, type PlotPointer} from "./Plot";

const VIEW: View = {xmin: -1.75, xmax: 1.75, ymin: -1.25, ymax: 1.3};

interface Plot3DProps {
    label: string;
    draw: (s: Scene3D) => void;
    initial?: Partial<Camera>;
    /** Slowly spin while nobody is dragging. */
    autoRotate?: boolean;
    maxHeight?: number;
    /** Click (not drag) on the canvas: screen position in view coordinates plus the camera. */
    onPick?: (u: number, v: number, cam: Camera) => void;
}

/** A canvas you can orbit: drag to rotate, click to pick. */
export function Plot3D({label, draw, initial, autoRotate = true, maxHeight = 600, onPick}: Plot3DProps) {
    const [cam, setCam] = useState<Camera>({yaw: 0.7, pitch: 0.55, zoom: 1, ...initial});
    const drag = useRef<{x: number; y: number; moved: number} | null>(null);
    const [idle, setIdle] = useState(true);
    const [dragging, setDragging] = useState(false);

    useAnimationFrame((dt) => setCam((c) => ({...c, yaw: c.yaw + dt * 0.12})), autoRotate && idle);

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") {
            drag.current = {x: e.x, y: e.y, moved: 0};
            setIdle(false);
            setDragging(true);
        }
        if (e.type === "move" && drag.current) {
            const dx = e.x - drag.current.x;
            const dy = e.y - drag.current.y;
            drag.current = {x: e.x, y: e.y, moved: drag.current.moved + Math.abs(dx) + Math.abs(dy)};
            setCam((c) => ({...c, yaw: c.yaw - dx * 1.4, pitch: Math.max(0.05, Math.min(1.55, c.pitch - dy * 1.4))}));
        }
        if (e.type === "up") {
            if (drag.current && drag.current.moved < 0.02 && onPick) onPick(e.x, e.y, cam);
            drag.current = null;
            setDragging(false);
        }
    };

    return (
        <div className="plot3d" onMouseLeave={() => setIdle(true)}>
            <Plot
                label={label}
                view={VIEW}
                maxHeight={maxHeight}
                cursor={dragging ? "grabbing" : "grab"}
                onPointer={onPointer}
                draw={(p) => {
                    p.clear();
                    const s = new Scene3D(p, cam);
                    draw(s);
                    s.flush();
                }}
            />
        </div>
    );
}
