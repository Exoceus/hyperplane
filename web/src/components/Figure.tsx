import type {ReactNode} from "react";

/** A labeled panel over a plot (e.g. "data space" / "weight space"). */
export function Panel({label, children, right}: {label?: ReactNode; children: ReactNode; right?: ReactNode}) {
    return (
        <div className="panel">
            {(label || right) && (
                <div className="panel-label">
                    <span>{label}</span>
                    {right && <span className="panel-label-right">{right}</span>}
                </div>
            )}
            {children}
        </div>
    );
}
